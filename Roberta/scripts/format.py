import codecs
import jsonlines


def convert(filename, outfilename):
    #     WORD###TAG [TAB] WORD###TAG [TAB] ..... \n
    infile = codecs.open(filename, 'r')
    outfile = codecs.open(outfilename, 'w')
    outlist = []
    for line in infile.readlines():
        out = line.strip().split()
        outlist.append('###'.join(out))
        if len(line.strip())==0:
            outfile.write('\t'.join(outlist)+'\n')
            outlist = []

def convert_to_eval_format(filename, outfilename, bottom_up=False):
    outfile = codecs.open(outfilename, 'w')
    with jsonlines.open(filename) as reader:
        for obj in reader:
            pred = obj["tags"]
            gold = obj["gold_labels"]
            words = obj["words"]
            for p, g, w in zip(pred, gold, words):
                if bottom_up:
                    # g = convert_pred_to_bottom_up(g)
                    # p = convert_pred_to_bottom_up(p)
                    g = convert_tags_to_mnm(g)
                    p = convert_tags_to_mnm(p)
                    outfile.write(w + '\t' + g + '\t' + p + '\n')
                else:
                    outfile.write(w+'\t'+g+'\t'+p+'\n')
            outfile.write('\n')
    outfile.close()

def convert_pred_to_bottom_up(tag):
    converted = []
    if len(tag.strip()) > 1:
        tag = tag.split("-")
        prefix = tag[0]
        suffix = tag[1]
        if suffix in ["deontic", "buletic", "teleological", "intentional", "buletic_teleological"]:
            converted.append(f"{prefix}-priority")
        elif suffix in ["epistemic", "ability", "circumstantial", "opportunity", "ability_circumstantial",
                       "epistemic_circumstantial"]:
            converted.append(f"{prefix}-plausibility")
        else:
            converted.append(f"{prefix}-{suffix}")
    else:
        converted.append(tag)
    return converted[0]

def convert_tags_to_mnm(tag):
    converted = []
    if len(tag.strip()) > 1:
        tag = tag.split("-")
        prefix = tag[0]
        suffix = tag[1]
        converted.append(f"{prefix}-modal")
    else:
        converted.append(tag)
    return converted[0]

#convert('/Users/vale/PycharmProjects/Modality/data/with_T/0/train_prejacent_bio_1.bmes', '/Users/vale/PycharmProjects/Modality/data/0/train_prejacent_naive.txt')
convert_to_eval_format('/Users/vale/PycharmProjects/Modality/data/predictions/prejacent_target_naive4', '/Users/vale/PycharmProjects/Modality/data/predictions/prejacent_target_naive4_eval', bottom_up=False)