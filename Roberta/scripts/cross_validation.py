import os
import io
import pandas as pd
import subprocess
from subprocess import Popen, PIPE
import csv
from random import shuffle
from itertools import zip_longest
import time
from conlleval import Eval
import argparse

def create_test_sets():
    with open("../../data/GME/cross_validation/test_coarse.bmes", "w") as test_coarse_f:
        with open("../../data/GME/cross_validation/test_fine.bmes", "w") as test_fine_f:
            with open("../../data/GME/cross_validation/test_coarse_mv.bmes", "w") as test_coarse_mv_f:
                test_rows = gme[gme["sentence_number"].isin(test_sentence_numbers)].iterrows()
                next(test_rows)
                for i, row in test_rows:
                    if "sentence_text =" in row["token"]:
                        test_coarse_f.write("\n")
                        test_fine_f.write("\n")
                        test_coarse_mv_f.write("\n")
                    else:
                        if row["is_modal"] != "O":
                            test_coarse_f.write("{}\t{}-{}\n".format(row["token"],
                                                                     row["is_modal"][0],
                                                                     row["modal_type"].split(":")[0]))
                            test_fine_f.write("{}\t{}\n".format(row["token"], row["is_modal"]))
                            if row["token"] in ["can", "could", "shall", "should", "may", "must"]:
                                test_coarse_mv_f.write("{}\t{}-{}\n".format(row["token"],
                                                                            row["is_modal"][0],
                                                                            row["modal_type"].split(":")[0]))
                            else:
                                test_coarse_mv_f.write("{}\tO\n".format(row["token"]))
                        else:
                            test_coarse_f.write("{}\tO\n".format(row["token"]))
                            test_fine_f.write("{}\tO\n".format(row["token"]))
                            test_coarse_mv_f.write("{}\tO\n".format(row["token"]))
                            

def create_train_and_dev_files(sentence_numbers, batch_type, fold):
    for x in range(fold+1):
        if not os.path.isdir(f'../../data/GME/cross_validation/{x}'):
            os.mkdir(f'../../data/GME/cross_validation/{x}')
        with open("../../data/GME/cross_validation/{}/{}_coarse.bmes".format(x, batch_type), "w") as coarse_f:
            with open("../../data/GME/cross_validation/{}/{}_fine.bmes".format(x, batch_type), "w") as fine_f:
                with open("../../data/GME/cross_validation/{}/{}_coarse_mv.bmes".format(x, batch_type), "w") as coarse_mv_f:
                    rows = gme[gme["sentence_number"].isin(sentence_numbers)].iterrows()
                    next(rows)
                    for i, row in rows:
                        if "sentence_text =" in row["token"]:
                            coarse_f.write("\n")
                            fine_f.write("\n")
                            coarse_mv_f.write("\n")
                        else:
                            if row["is_modal"] != "O":
                                coarse_f.write("{}\t{}-{}\n".format(row["token"],
                                                                    row["is_modal"][0],
                                                                    row["modal_type"].split(":")[0]))
                                fine_f.write("{}\t{}\n".format(row["token"],
                                                               row["is_modal"]))
                                if row["token"] in ["can", "could", "shall", "should", "may", "must"]:
                                    coarse_mv_f.write("{}\t{}-{}\n".format(row["token"],
                                                                           row["is_modal"][0], 
                                                                           row["modal_type"].split(":")[0]))
                                else:
                                    coarse_mv_f.write("{}\tO\n".format(row["token"]))
                            else:
                                coarse_f.write("{}\tO\n".format(row["token"]))
                                fine_f.write("{}\tO\n".format(row["token"]))
                                coarse_mv_f.write("{}\tO\n".format(row["token"]))
                                
def run_k_fold(k, train_dataset, device, vec="crawl-300d-2M-subword", files_already_exist=True):

    # note that the last chunk is probably too small   
    
    if files_already_exist == False:
        shuffle(train_dataset)
        chunk_size = int(len(train_dataset)/k)
        chunks = list(zip_longest(*[iter(train_dataset)]*chunk_size, fillvalue=None)) 
        for x in range(k):
            dev = chunks.pop(0)
            create_tmp_files(dev, "dev", fold=k)
            train = [item for sublist in chunks for item in sublist]
            create_tmp_files(train, "train", fold=k)
            chunks.append(dev)

    for x in range(k):
        start = time.time()
        print(f"starting fold {x}")
        run_ncrfpp(x, "fine", vec , device)
        run_ncrfpp(x, "coarse", vec, device)        
        run_ncrfpp(x, "coarse_mv", vec, device) 
        end = time.time() - start
        print(f"end fold {x} in {end} second")

def eval_biose_dataset(goldpath, predpath):
    with open(predpath, "r") as predfile, open(goldpath, "r") as goldfile:
        pred_tags = [line.split()[1] for line in predfile.readlines() if (
            line.strip() and not line.startswith("#"))]
        gold_tags = [line.split()[1] for line in goldfile.readlines() if line.strip()]
        if len(pred_tags) == len(gold_tags):
            ev = Eval()
            ev.evaluate(true_seqs=gold_tags, pred_seqs=pred_tags, verbose=True)
    return gold_tags, pred_tags
        
        
def run_ncrfpp(fold, condition, vec, device):
    train_config = f"""### use # to comment out the configure item

    ### I/O ###

    train_dir=/home/nlp/shovalsa/modality/data/GME/cross_validation/{fold}/train_{condition}.bmes
    dev_dir=/home/nlp/shovalsa/modality/data/GME/cross_validation/{fold}/dev_{condition}.bmes
    test_dir=/home/nlp/shovalsa/modality/data/GME/cross_validation/test_{condition}.bmes

    model_dir=/home/nlp/shovalsa/modality/modality_NN/models/ncrf-models/cv_modality.lstmcrf
    word_emb_dir=/home/nlp/shovalsa/modality/embeddings/{vec}.vec


    norm_word_emb=False
    norm_char_emb=False
    number_normalized=True
    seg=True
    word_emb_dim=300
    char_emb_dim=30

    ###NetworkConfiguration###
    use_crf=True
    use_char=True
    word_seq_feature=LSTM
    char_seq_feature=CNN
    #feature=[POS] emb_size=20
    #feature=[Cap] emb_size=20
    #nbest=1

    ###TrainingSetting###
    status=train
    optimizer=Adam
    iteration=1
    batch_size=10
    ave_batch_loss=False

    ###Hyperparameters###
    cnn_layer=4
    char_hidden_dim=50
    hidden_dim=300
    dropout=0.5
    lstm_layer=1
    bilstm=True
    learning_rate=0.015
    lr_decay=0.05
    momentum=0
    l2=1e-8
    gpu=True
    #clip="""
#     config_file = io.StringIO(config)
    process = Popen(["python", "/home/nlp/shovalsa/NCRFpp/main.py", "--config", train_config, "--device", device], 
                    stdout=PIPE, stderr=PIPE)
#     process.wait()
    stdout, stderr = process.communicate()
    print(f"\n ncrf fold {fold} conditon {condition}\n", stderr)
    for line in str(stdout).split("\n"):
        print(line)
    
    print("\n starting decode...")
    decode_config = f"""### Decode ###
    status=decode
    raw_dir=/home/nlp/shovalsa/modality/data/GME/cross_validation/test_{condition}.bmes
    nbest=10
    decode_dir=/home/nlp/shovalsa/modality/data/GME/cross_validation/decoded_include_non_modals_{fold}.bmes
    dset_dir=/home/nlp/shovalsa/modality/modality_NN/models/ncrf-models/cv_modality.lstmcrf.dset
    load_model_dir=/home/nlp/shovalsa/modality/modality_NN/models/ncrf-models/cv_modality.lstmcrf.0.model
    """
    process = Popen(["python", "/home/nlp/shovalsa/NCRFpp/main.py", "--config", decode_config, "--device", device],
                    stdout=PIPE, stderr=PIPE)
    process.wait()
    
    stdout, stderr = process.communicate()
    print(f"ncrf fold {fold} conditon {condition}\n", stderr)
    goldpath = f"/home/nlp/shovalsa/modality/data/GME/cross_validation/test_{condition}.bmes"
    predpath=f"/home/nlp/shovalsa/modality/data/GME/cross_validation/decoded_include_non_modals_{fold}.bmes"
    gold_tags, pred_tags = eval_biose_dataset(goldpath=goldpath,predpath=predpath)
    return gold_tags, pred_tags


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Cross validation on a specific method')
    # parser.add_argument('--status', choices=['train', 'decode'], help='update algorithm', default='train')
    help_for_method = """
    Choose the method number one of the following:
    (1) Sequence tagging with NCRF++
    (2) Sequence tagging with BERT
    (3) Sentence classification with AllenNLP CNN
    (4) Sentence classification with AllenNLP RNN
    """
    dataset_help = """dataset in csv format. 
    Required columns: token, is_modal, sentence_number, set"""
    
    parser.add_argument('--method', help=help_for_method, default='1')
    parser.add_argument('--k', help="Number of folds to run", default='5')
    parser.add_argument('--files_exist', help="1 if you already ran this script for the currect k, 0 otherwise", default='1')
    parser.add_argument('--device', help="Number of folds to run", default='2')
    parser.add_argument('--dataset', help=dataset_help, default="/home/nlp/shovalsa/modality/data/annotated_gme.csv")
    
    
    
    
    args = parser.parse_args()
    if args.files_exist == "0":
        gme = pd.read_csv(args.dataset, sep="\t", keep_default_na=False)

        test_sentence_numbers = [x for x in gme[gme["set"] == "test"]["sentence_number"].unique().tolist() if x.isdigit()]
        train_sentence_numbers = [x for x in gme[gme["set"] != "test"]["sentence_number"].unique().tolist() if x.isdigit()]

        run_k_fold(int(args.k), train_sentence_numbers, args.device, files_already_exist=False)
    else:
        run_k_fold(int(args.k), None, args.device, files_already_exist=True)
    