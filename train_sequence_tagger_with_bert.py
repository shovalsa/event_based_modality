import os
import sys
import csv
import pandas as pd
import numpy as np
from tqdm import tqdm, trange
from conlleval import evaluate
import json
import time
import logging
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


def get_logger(filename):
    logging.basicConfig(filename=filename,
                                filemode='a',
                                format='%(asctime)s,%(msecs)d %(name)s %(levelname)s %(message)s',
                                datefmt='%H:%M:%S',
                                level=logging.INFO)
    logger = logging.getLogger('modality_trainer')
    return logger

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
            elif subtype == 'yes_no':
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
            logger.info("Epoch number: {} \t Train loss: {}".format(epNum, (tr_loss / nb_tr_steps)))
            epNum += 1
        return loss


class Evaluator():

    def __init__(self, trainer, tokenizer):
        self.trainer = trainer
        self.tokenizer = tokenizer
        self.results = dict()

    def add_network_parameters(self, epochs, max_grad_norm, max_len, bs, full_finetuning, device, n_gpu, loss, optimizer):
        self.results['network_parameters'] = {'epochs': epochs,
                        'max_grad_norm': max_grad_norm,
                        'max_len': max_len,
                        'bs': bs,
                        'full_finetuning': full_finetuning,
                        'device': device,
                        'n_gpu': n_gpu,
                        'loss': loss,
                        'optimizer': optimizer}

    def flat_accuracy(self, preds, labels):
        pred_flat = np.argmax(preds, axis=2).flatten()
        labels_flat = labels.flatten()
        return np.sum(pred_flat == labels_flat) / len(labels_flat)

    def aggregate_tokens(self, split_tokens: list):
        aggregated_tokens = []
        for token in split_tokens:
            if token.startswith("##"):
                aggregated_tokens[-1] = aggregated_tokens[-1] + token[2:]
            else:
                aggregated_tokens.append(token)
        return aggregated_tokens

    def delete_pads_from_preds(self, predicted_tags, test_tags):
        clean_predicted = []
        clean_test = []

        for ix in range(0, len(test_tags)):
            if test_tags[ix] != 'PAD':
                clean_predicted.append(predicted_tags[ix])
                clean_test.append(test_tags[ix])

        return clean_predicted, clean_test

    def calculate_accuracy_per_sent(self, prediction_dict: dict):
        """ here precision and recall are defined in terms of O/not-O"""
        numOfCorrectPredictions = 0
        numOfCorrectNon_O = 0
        precision, recall, f1 = evaluate(prediction_dict['test_tag'], prediction_dict['predicted_tag'], verbose=False)
        total_labeled = len([x for x in prediction_dict['test_tag'] if x != 'O'])
        for pred, orig in zip(prediction_dict['predicted_tag'], prediction_dict['test_tag']):
            if orig == pred:
                numOfCorrectPredictions += 1
                if orig != 'O':
                    numOfCorrectNon_O += 1
        agg_tokens = self.aggregate_tokens(prediction_dict['word'])
        evaluation = {
            'sentence': " ".join(agg_tokens),
            'tags': prediction_dict['test_tag'],
            'length of bert-tokenized sentence': len(prediction_dict['word']),
            'length of aggregated sentence': len(agg_tokens),
            'predictions': prediction_dict['predicted_tag'],
            'accuracy for all tags': numOfCorrectPredictions / len(prediction_dict['test_tag']),
            'accuracy for only labeled': numOfCorrectPredictions / len(prediction_dict['test_tag']),
            'precision': precision,
            'recall': recall,
            'modal': total_labeled != 0,
            'f1': f1}
        return evaluation

    def test_model(self, model, tok_sent, tok_labels):
        input_ids, tags, attention_masks = self.trainer.pad_sentences_and_labels([tok_sent], [tok_labels],
                                                                         tokenizer=self.tokenizer)
        val_inputs = torch.tensor(input_ids, dtype=torch.long)
        val_tags = torch.tensor(tags, dtype=torch.long)
        val_masks = torch.tensor(attention_masks, dtype=torch.long)

        test_data = TensorDataset(val_inputs, val_masks, val_tags)
        test_sampler = SequentialSampler(test_data)
        test_dataloader = DataLoader(test_data, sampler=test_sampler, batch_size=self.trainer.bs)

        eval_loss, eval_accuracy = 0, 0
        nb_eval_steps, nb_eval_examples = 0, 0
        predictions, true_labels = [], []
        for batch in test_dataloader:
            b_input_ids, b_input_mask, b_labels = tuple(t.to(self.trainer.device) for t in batch)

            with torch.no_grad():
                tmp_eval_loss = model(b_input_ids,
                                      token_type_ids=None,
                                      attention_mask=b_input_mask,
                                      labels=b_labels)
                logits = model(b_input_ids, token_type_ids=None,
                               attention_mask=b_input_mask)
            logits = logits.detach().cpu().numpy()
            label_ids = b_labels.to('cpu').numpy()
            predictions.append([list(p) for p in np.argmax(logits, axis=2)])

            true_labels.append(label_ids)
            tmp_eval_accuracy = self.flat_accuracy(logits, label_ids)

            eval_loss += tmp_eval_loss.mean().item()
            eval_accuracy += tmp_eval_accuracy

            nb_eval_examples += b_input_ids.size(0)
            nb_eval_steps += 1

        pred_tags = [self.trainer.idx2tag[p_ii] for p in predictions for p_i in p for p_ii in p_i]
        test_tags = [self.trainer.idx2tag[l_ii] for l in true_labels for l_i in l for l_ii in l_i]

        clean_predicted, clean_test = self.delete_pads_from_preds(pred_tags, test_tags)
        tmp = {'word': tok_sent, 'orig_label': tok_labels, 'predicted_tag': clean_predicted, 'test_tag': clean_test}

        return tmp

    def collect_accuracies(self, model, dev_sentences, dev_tags):
        sentence_num = 1
        dev_tokenized_texts, dev_tokenized_labels = self.trainer.tokenize(dev_sentences, dev_tags, self.tokenizer)
        for sent, label, tok_sent, tok_label in zip(dev_sentences, dev_tags, dev_tokenized_texts, dev_tokenized_labels):
            try:
                predictions = self.test_model(model, tok_sent, tok_label)
                sentence_num += 1
                if len(predictions['test_tag']) > 0:
                    self.results[sentence_num] = self.calculate_accuracy_per_sent(predictions)
            except ValueError:
                raise Exception(
                    'sent: {}, \n label{}, \n tok_sent{}, \n tok_label{}, \n'.format(sent, label, tok_sent, tok_label))

    def calculate_dataset_accuracy(self):
        all_tags = [tag for sentence in [details['tags'] for sent, details in self.results.items()] for tag in sentence]
        all_predictions = [pred for sentence in [details['predictions'] for sent, details in self.results.items()] for
                           pred in sentence]
        total_precision, total_recall, total_f1 = evaluate(all_tags, all_predictions, verbose=True)
        self.results['total_dataset_accuracy'] = {'total_precision': total_precision,
                                                   'total_recall': total_recall,
                                                   'total_f1': total_f1}




if __name__ == '__main__':
    script, model_filename, modality_resolution = sys.argv
    logger = get_logger('./logs/{}.log'.format(model_filename))
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
    logger.info('idx2tag: {} \t tag2idx: {}'.format(bert.idx2tag,bert.tag2idx))
    logger.info('device: {}\t n_gpu{}'.format(device, n_gpu))
    loss = bert.train_model(model, EPOCHS, MAX_GRAD_NORM, optimizer)
    logger.info('run training with parameters: epochs: {} \n loss: {}'.format(EPOCHS, loss))

    torch.save({
        'epoch': EPOCHS,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss
    },
        './models/{}.pth'.format(model_filename))



    # evaluations
    eval = Evaluator(bert)
    eval.collect_accuracies(model, dev_sentences, dev_tags, tokenizer)
    eval.calculate_dataset_accuracy()
    eval.add_network_parameters(EPOCHS, MAX_GRAD_NORM, MAX_LEN,BS, FULL_FINETUNING, device, n_gpu, loss, optimizer)
    results = eval.results

    with open('./results/{}.json'.format(model_filename), 'w') as outfile:
        json.dump(results, outfile, indent=4)

