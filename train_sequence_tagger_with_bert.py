import os
import sys
import csv
import pandas as pd
import numpy as np
from tqdm import tqdm, trange
from conlleval import evaluate
import time

import ast

import torch
from torch.optim import Adam
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from pytorch_pretrained_bert import BertTokenizer, BertConfig
from pytorch_pretrained_bert import BertForTokenClassification, BertAdam


EPOCHS = 15
MAX_GRAD_NORM = 1.0

MAX_LEN = 150
BS = 16
FULL_FINETUNING = True


def define_torch_seed(seed=3):
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def split_dev_train_and_test_sets(df, subtype, train_size: float):
    def add_col_for_prediction(row, subtype):
        if row['is_modal'] == 'O':
            return 'O'
        else:
            if subtype == 'coarse':
                return row['is_modal'][0] + '-' + row['modal_type'].split(':')[0]
            elif subtype == 'fine':
                return row['is_modal'][0] + '-' + row['modal_type'].split(':')[1]
            elif subtype == 'yes/no':
                return 'M'
            else:
                return row['is_modal'][0]

    dev_size = (1 - train_size) / 2
    sent_numbers = df['sentence_number'].unique()
    dev_sents, train_sents, test_sents = sent_numbers[1:(int(len(sent_numbers) * dev_size))], sent_numbers[(int(
        len(sent_numbers) * dev_size)):(int(len(sent_numbers) * train_size))], sent_numbers[
                                                                               (int(len(sent_numbers) * train_size)):]
    dev_set, train_set, test_set = df[df['sentence_number'].isin(dev_sents)], df[
        df['sentence_number'].isin(train_sents)], df[df['sentence_number'].isin(test_sents)]

    for df in [dev_set, train_set, test_set]:
        df['subtype'] = df.apply(lambda x: add_col_for_prediction(x, subtype), axis=1)

    return dev_set, train_set, test_set


class SentenceGetter(object):
    def __init__(self, dataframe, max_sent=None):
        self.df = dataframe
        self.tags = self.df['subtype'].unique().tolist()
        self.tags.insert(0, 'PAD')

        self.index = 0
        self.max_sent = max_sent
        self.tokens = dataframe['token']
        self.modal_tags = dataframe['subtype']

    def get_tokens_and_tags_by_sentences(self):
        sent = []
        counter = 0
        for token, tag in zip(self.tokens, self.modal_tags):
            sent.append((token, tag))
            if token.strip() in ['.', '?', '!'] and (len(sent) > 2):
                yield sent
                sent = []
                counter += 1
            if self.max_sent is not None and counter >= self.max_sent:
                return

    def get_tag2idx(self):
        return {tag: idx for idx, tag in enumerate(self.tags)}

    def get_idx2tag(self):
        return {idx: tag for idx, tag in enumerate(self.tags)}

    def get_2Dlist_of_sentences(self):
        return [[token for token, tag in sent] for sent in self.get_tokens_and_tags_by_sentences()]

    def get_2Dlist_of_tags(self):
        return [[tag for token, tag in sent] for sent in self.get_tokens_and_tags_by_sentences()]


class BertTrainer(object):


    def __init__(self, dev_df, train_df, test_df, pre_trained='bert-base-cased', bs=BS, max_len=MAX_LEN):
        self.pre_trained = pre_trained
        self.dev_df = dev_df
        self.train_df = train_df
        self.test_df = test_df
        self.bs = bs
        self.max_len = max_len

        self.dev_getter = SentenceGetter(self.dev_df)
        self.train_getter = SentenceGetter(self.train_df)
        self.test_getter = SentenceGetter(self.test_df)
        self.tag2idx, self.idx2tag = self.get_tag2idx_and_idx2tag()
        self.device, self.n_gpu = self.set_cuda()

    #         self.train_sentence = self.train_getter.get_2Dlist_of_sentences()
    #         self.train_tags = self.get_2Dlist_of_tags()

    def get_tag2idx_and_idx2tag(self):
        tag2idx = {**self.dev_getter.get_tag2idx(), **self.train_getter.get_tag2idx(), **self.test_getter.get_tag2idx()}
        idx2tag = {**self.dev_getter.get_idx2tag(), **self.train_getter.get_idx2tag(), **self.test_getter.get_idx2tag()}
        return tag2idx, idx2tag

    def set_parameters(self, max_len=MAX_LEN, bs=BS, full_finetuning=FULL_FINETUNING):
        self.MAX_LEN = max_len
        self.bs = bs
        self.FULL_FINETUNING = full_finetuning

    def set_cuda(self):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        n_gpu = torch.cuda.device_count()
        return device, n_gpu

    def tokenize(self, sentences, orig_labels, tokenizer):
        tokenized_texts = []
        labels = []
        sents, tags_li = [], []
        for sent, sent_labels in zip(sentences, orig_labels):
            bert_tokens = []
            bert_labels = []
            for orig_token, orig_label in zip(sent, sent_labels):
                b_tokens = tokenizer.tokenize(orig_token)
                bert_tokens.extend(b_tokens)
                for b_token in b_tokens:
                    bert_labels.append(orig_label)
            if b_tokens:
                tokenized_texts.append(bert_tokens)
                labels.append(bert_labels)
            assert len(bert_tokens) == len(bert_labels)
        return tokenized_texts, labels

    def pad_sentences_and_labels(self, tokenized_texts, labels, tokenizer):
        input_ids = pad_sequences([tokenizer.convert_tokens_to_ids(txt) for txt in tokenized_texts],
                                  maxlen=self.max_len, dtype="int", truncating="post", padding="post")
        try:
            tags = pad_sequences([[self.tag2idx.get(l) for l in lab] for lab in labels],
                                 maxlen=self.max_len, value=self.tag2idx['PAD'], padding="post",
                                 dtype="int", truncating="post")
        except TypeError:
            raise Exception('tokenized_texts{} \n, labels {}'.format(tokenized_texts, labels))

        attention_masks = [[float(i > 0) for i in ii] for ii in input_ids]
        return input_ids, tags, attention_masks

    def get_train_dataloader(self, input_ids, tags, attention_masks):
        tr_inputs = torch.tensor(input_ids, dtype=torch.long)
        tr_tags = torch.tensor(tags, dtype=torch.long)
        tr_masks = torch.tensor(attention_masks, dtype=torch.long)

        train_data = TensorDataset(tr_inputs, tr_masks, tr_tags)
        train_dataloader = DataLoader(train_data, batch_size=self.bs, shuffle=True)
        return train_dataloader

    def get_model(self, pre_trained):
        model = BertForTokenClassification.from_pretrained(pre_trained, num_labels=len(self.tag2idx))
        model.cuda()

        return model

    def define_optimizer_grouped_parameters(self, modelname, full_finetuning):
        if full_finetuning:
            param_optimizer = list(modelname.named_parameters())
            no_decay = ['bias', 'gamma', 'beta']
            optimizer_grouped_parameters = [
                {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)],
                 'weight_decay_rate': 0.01},
                {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)],
                 'weight_decay_rate': 0.0}
            ]
        else:
            param_optimizer = list(modelname.classifier.named_parameters())
            optimizer_grouped_parameters = [{"params": [p for n, p in param_optimizer]}]
        return optimizer_grouped_parameters

    def train_model(self, model, epochs, max_grad_norm, optimizer):
        epNum = 1
        for _ in trange(epochs, desc="Epoch"):
            # TRAIN loop
            model.train()
            tr_loss = 0
            nb_tr_examples, nb_tr_steps = 0, 0
            for step, batch in enumerate(train_dataloader):
                # add batch to gpu
                batch = tuple(t.to(self.device) for t in batch)
                b_input_ids, b_input_mask, b_labels = batch
                loss = model(b_input_ids, token_type_ids=None,
                             attention_mask=b_input_mask, labels=b_labels)
                # backward pass
                loss.backward()
                # track train loss
                tr_loss += loss.item()
                nb_tr_examples += b_input_ids.size(0)
                nb_tr_steps += 1
                # gradient clipping
                torch.nn.utils.clip_grad_norm_(parameters=model.parameters(), max_norm=max_grad_norm)
                # update parameters
                optimizer.step()
                model.zero_grad()
            # print train loss per epoch
            print("Epoch number: {} \t Train loss: {}".format(epNum, (tr_loss / nb_tr_steps)))
            epNum += 1
        return loss


if __name__ == '__main__':
    script, model_filename, modality_resolution = sys.argv

    define_torch_seed(3)
    gme_df = pd.read_csv('./data/tokenized_and_tagged_gme_coarse_grained.csv', sep='\t', keep_default_na=False)
    dev_df, train_df, test_df = split_dev_train_and_test_sets(gme_df, modality_resolution, 0.8)

    bert = BertTrainer(dev_df, train_df, test_df, pre_trained='../resources/wwm_cased_L-24_H-1024_A-16/')
    train_sentences, train_tags = bert.train_getter.get_2Dlist_of_sentences(), bert.train_getter.get_2Dlist_of_tags()

    dev_sentences, dev_tags = bert.dev_getter.get_2Dlist_of_sentences(), bert.dev_getter.get_2Dlist_of_tags()
    test_sentences, test_tags = bert.test_getter.get_2Dlist_of_sentences(), bert.test_getter.get_2Dlist_of_tags()

    tokenizer = BertTokenizer.from_pretrained(bert.pre_trained, do_lower_case=False)
    train_tokenized_texts, train_tokenized_labels = bert.tokenize(train_sentences, train_tags, tokenizer=tokenizer)
    input_ids, tags, attention_masks = bert.pad_sentences_and_labels(train_tokenized_texts, train_tokenized_labels,
                                                                     tokenizer=tokenizer)

    train_dataloader = bert.get_train_dataloader(input_ids, tags, attention_masks)
    model = bert.get_model('../resources/bert-base-cased/')
    optimizer_grouped_parameters = bert.define_optimizer_grouped_parameters(model, FULL_FINETUNING)
    optimizer = Adam(optimizer_grouped_parameters, lr=3e-5)

    device, n_gpu = bert.set_cuda()

    loss = bert.train_model(model, EPOCHS, MAX_GRAD_NORM, optimizer)

    torch.save({
        'epoch': EPOCHS,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss
    },
        './models/{}.pth'.format(model_filename))