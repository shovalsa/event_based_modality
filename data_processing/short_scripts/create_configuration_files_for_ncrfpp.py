import pandas as pd
import os


ROOT_DIR = os.path.split(os.path.dirname(os.path.abspath(os.getcwd())))[0]
PROJECT_DIR = os.path.dirname(os.path.abspath(os.getcwd()))



def create_resolutions_confs(resolution, fold, cuda, ncrf_path, w2v_path=None):
    if not os.path.isdir(os.path.join(ncrf_path, "config", resolution)):
        os.mkdir(os.path.join(conf_basepath, "config", resolution))
    if not os.path.isdir(os.path.join(ncrf_path, "logs", resolution)):
        os.mkdir(os.path.join(conf_basepath, "logs", resolution))
    conf_path = os.path.join(ncrf_path, "config", resolution, f"modality_{resolution}_fold_{fold}.cuda_{cuda}.train.config")
    train_dir = os.path.join(ROOT_DIR, "data", "GME", resolution, fold, f"train_{resolution}_space.bmes")
    dev_dir = os.path.join(ROOT_DIR, "data", "GME", resolution, fold, f"dev_{resolution}_space.bmes")
    test_dir = os.path.join(ROOT_DIR, "data", "GME", resolution, f"test_{resolution}_space.bmes")
    model_dir = os.path.join(PROJECT_DIR, "models", "ncrf-models", f"{resolution}_fold_{fold}.lstmcrf")
    decode_dir = os.path.join(ROOT_DIR, "data", "GME", resolution, fold, f"decode_{resolution}_space.bmes")
    if not w2v_path:
        w2v_path = os.path.join(ROOT_DIR, "embeddings", "GoogleNews-vectors-negative300.vec")
    base_conf = f"""
train_dir={train_dir}
dev_dir={dev_dir}
test_dir={test_dir}
model_dir={model_dir}
decode_dir={decode_dir}
word_emb_dir={w2v_path}
word_seq_feature=LSTM
word_emb_dim=300
char_emb_dim=30
iteration=40
bilstm=True
norm_word_emb=False
norm_char_emb=False
ave_batch_loss=False
l2=1e-8
number_normalized=False
nbest=1
lstm_layer=2
seg=True
status=train
cnn_layer=4
lr_decay=0.05
momentum=0
gpu=True
hidden_dim=200
dropout=0.7
char_hidden_dim=50
optimizer=SGD
use_char=False
use_crf=True
batch_size=10
learning_rate=0.015
    """

    with open(conf_path) as f:
        f.write(base_conf)
        
def write_decode_config_of_best_epoch(ncrf_path, resolution):
    logs_path = os.path.join(conf_basepath, "logs", "resolutions", resolution)
    for root, dirs, files in os.walk(logs_path):
        for file in files:
            if ("train" in file) and (resolution in file):
                with open(os.path.join(root, file), "r") as f:
                    best_model = "0"
                    for line in f.readlines():
                        if "Save current best model" in line:
                            line = line.split(":")
                            best_model = line[1].strip()
                    if best_model != "0":
                        filename = file.split(".")
                        fold = filename[0][-1]
                        cuda = filename[1][-1]
                        
                        dset = ".".join(best_model.split(".")[:-2]) + ".dset"
                        conf_name = file.replace(".train.config.log", ".decode.config")
                        decode_path = os.path.join(conf_basepath, "config", "resolutions", resolution, conf_name)
                        raw_dir = os.path.join(ROOT_DIR, "data", "GME", resolution, f"test_{resolution}_space.bmes")
                        decoded_file = os.path.join(ROOT_DIR, "data", "GME", resolution, fold, f"decode_{res}_bio.bmes")
                        with open(decode_path, "w") as fw:
                            conf = f"""
### Decode ###
status=decode
raw_dir={raw_dir}
nbest=5
decode_dir={decoded_file}
dset_dir={dset}
load_model_dir={best_model}
        """
                            fw.write(conf)
        
        
if __name__ == "__main__":
    resolution = "plausibility_others"
    #change the conf_basepath to the location where NCRF++ is installed. This script assumes config and logs directories inside.
    ncrf_path = "/home/nlp/shovalsa/NCRFpp"
    train = True
    decode = False
    
    if train:
        for fold, cuda in zip(range(5), ["0", "1", "2", "3", "0"]):
            create_resolutions_confs(resolution, fold, cuda, ncrf_path)        
    if decode:
        write_decode_config_of_best_epoch(ncrf_path, resolution)
