import pandas as pd
import os


ROOT_DIR = os.path.split(os.path.dirname(os.path.abspath(os.getcwd())))[0]
PROJECT_DIR = os.path.dirname(os.path.abspath(os.getcwd()))
GME = pd.read_csv(os.path.join(ROOT_DIR, "data", "annotated_gme.csv"), sep="\t", keep_default_na=False)

def collect_texts_and_annotations_from_df(df):

    tokens = []
    labels = []
    prej = []
    sentences = dict()
    text = ""
    for i, row in df.iterrows():
        if any(x in row["is_modal"] for x in ["O", "S-", "I", "B", "E"]):
            text += row["token"] + " "
            tokens.append(row["token"])
            labels.append(row["is_modal"])
            prej.append(row["span"])
        else:
            if text:
                text += "."
                tokens.append(".")
                labels.append("O")
                prej.append("_")
                sentences[text.strip()] = {"tokens": tokens, "labels":labels, "prej":prej}
                tokens = []
                labels = []
                prej = []
                text = ""                    

    return sentences

def write_new_scheme_files(df_sentences, ds_sentence, fpath, bio=False, binary=True):
    with open(fpath, "w") as eventf:
        for sentence in ds_sentence:
    #             clean_sent = " ".join([tok for tok in sentence.split() if tok.isalpha()])
            if sentence in df_sentences.keys():
                tokens = df_sentences[sentence]["tokens"]
                labels = df_sentences[sentence]["labels"]
                spans = df_sentences[sentence]["prej"]

                for t in range(len(tokens)):
                    token = tokens[t]
                    if not bio:
                        if labels[t] != "O":
                            tag = "T-event"
                            if binary:
                                if any(x in labels[t] for x in ["teleological", "deontic", "priority", "buletic"]):
                                    tag = "R-event"
                                elif any(x in labels[t] for x in ["epistemic", "circumstantial", "ability"]):
                                     tag = "L-event"
                            eventf.write(f"{token} {tag}\n")
                        else:
                            if spans[t] != "_":
                                if spans[t-1] != spans[t]:
                                    eventf.write(f"{token} B-event\n")
                                else:
                                    if spans[t-1] == spans[t]:
                                        eventf.write(f"{token} I-event\n")
                            else:
                                eventf.write(f"{token} O\n")
                    else:
                        if spans[t] != "_":
                            if spans[t-1] == "_":
                                eventf.write(f"{token} B-event\n")
                            else:
                                eventf.write(f"{token} I-event\n")
                        else:
                            eventf.write(f"{token} O\n")
                eventf.write("\n")

def write_prej_to_file(fold):
    basepath_dev = os.path.join(ROOT_DIR, "data", "GME", "prejacent", scheme, fold, "dev_prejacent_bio.bmes")
    basepath_train = os.path.join(ROOT_DIR, "data", "GME", "prejacent", scheme, fold, "train_prejacent_bio.bmes")
    sentences_path = os.path.join(ROOT_DIR, "data", "GME", "cross_validation", "sentences")

    with open(os.path.join(sentences_path, f"dev_{fold}.txt"), "r") as devf, 
        with open(os.path.join(sentences_path, f"train_{fold}.txt", "r") as trainf:

            dev_sents = [sent.strip() for sent in devf.readlines()]
    #         all_dev_sents = reverse_create_test_set(annotated_gme, bio=False, binary=True)
            train_sents = [sent.strip() for sent in trainf.readlines()]
    #         all_train_sents = reverse_create_test_set(annotated_gme,bio=False, binary=True)
            write_new_scheme_files(all_gme_sentences, dev_sents, basepath_dev, bio, binary)        
            write_new_scheme_files(all_gme_sentences, train_sents, basepath_train, bio, binary)                
                
            
if __name__ == "__main__":
    scheme = "with_T"
    bio = False
    binary = False
    
    
    all_gme_sentences = collect_texts_and_annotations_from_df(GME)

    test_path = os.path.join(ROOT_DIR, "data", "GME", "prejacent", scheme, "test_prejacent_bio.bmes")
    sentences_path = os.path.join(ROOT_DIR, "data", "GME", "cross_validation", "sentences")

    
    with open(os.path.join(sentences_path, "test.txt")) as testf:
        test_sents = [sent.strip() for sent in testf.readlines()]
        write_new_scheme_files(all_gme_sentences, test_sents, test_path, bio, binary)


    with Pool(5) as p:
        p.map(write_prej_to_file, range(5)) 