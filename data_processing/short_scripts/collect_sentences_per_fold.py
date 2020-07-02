import pandas as pd
import os


ROOT_DIR = os.path.split(os.path.dirname(os.path.abspath(os.getcwd())))[0]
PROJECT_DIR = os.path.dirname(os.path.abspath(os.getcwd()))
# GME = pd.read_csv(os.path.join(ROOT_DIR, "data", "annotated_gme.csv"), sep="\t", keep_default_na=False)

def collect_sentences_from_mnm(dest_path, mnm_path): 
    with open(os.path.join(mnm_path, f"test_modal_not_modal_space.bmes"), "r") as tag_f:
    sentences = []
    sentence = ""
    for line in tag_f.readlines():
        if line.strip():
            token = line.split()[0]
            sentence += token + " "
        else:
            sentences.append(sentence+"\n")
            sentence = ""
    with open(os.path.join(dest_path, "test.txt"), "w") as sent_f:
        for sent in sentences:
            sent_f.write(sent)
        
        
    for fold in range(5):
        for dset in ["train", "dev"]:
            filepath = os.path.join(mnm_path, fold, f"{dset}_modal_not_modal_space.bmes")
            with open(filepath, "r") as tag_f:
                sentences = []
                sentence = ""
                for line in tag_f.readlines():
                    if line.strip():
                        token = line.split()[0]
                        sentence += token + " "
                    else:
                        sentences.append(sentence+"\n")
                        sentence = ""
            destination_path = os.path.join(dest_path, f"{dset}_{fold}.txt")
            with open(destination_path, "w") as sent_f:
                for sent in sentences:
                    sent_f.write(sent)

                    
                    


                
if __name__ == "__main__":
    dest_path = os.path.join(ROOT_DIR, "data", "GME", "cross_validation", "sentences")
    mnm_path = os.path.join(ROOT_DIR, "data", "GME", "modal_not_modal")
    
    if not os.path.isdir(dest_path):
        os.mkdir(dest_path)
    collect_sentences_from_mnm(dest_path, mnm_path)
    print("DONE!")