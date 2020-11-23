import codecs
import jsonlines


def create_correct_bio_fine(binary=True):
    infile = codecs.open('data/only_bio/0/train_prejacent_bio_1.bmes', 'r')
    target_file = codecs.open('data/predictions/all_unrolled_no_ambiguities/0/train_all_unrolled_no_ambiguities_space.bmes', 'r')
    outfile = codecs.open('data/0/train_prejacent_fine_fixed.txt', 'w')
    outlist = []
    for inline, tline in zip(infile.readlines(), target_file.readlines()):
        print(inline)
        inline = inline.strip().split()
        tline = tline.strip().split()
        if len(inline) == 0:
            outfile.write('\t'.join(outlist) + '\n')
            outlist = []
        else:
            print('start')
            print(inline)
            print(tline)
            print('bla')
            if tline[1] != 'O':
                tag = inline[-1].split('-')[0]+'-'+tline[1][0]
                inline[-1] = tag
            outlist.append('###'.join(inline))

def create_correct_bio_naive(binary=True):
    infile = codecs.open('data/only_bio/test_prejacent_bio_1.bmes', 'r')
    target_file = codecs.open('data/with_T/test_prejacent_bio_1.bmes', 'r')
    outfile = codecs.open('data/full/test_prejacent_naive_fixed.txt', 'w')
    outlist = []
    for inline, tline in zip(infile.readlines(), target_file.readlines()):
        print(inline)
        inline = inline.strip().split()
        tline = tline.strip().split()
        if len(inline) == 0:
            outfile.write('\t'.join(outlist) + '\n')
            outlist = []
        else:
            print('start')
            print(inline)
            print(tline)
            if tline[1].startswith('T'):
                tag = inline[-1].split('-')[0]+'-T'
                inline[-1] = tag
            print(inline)
            outlist.append('###'.join(inline))

def create_correct_bio(binary=True):
    infile = codecs.open('data/only_bio/0/train_prejacent_bio_1.bmes', 'r')
    target_file = codecs.open('data/binary_prej/4/dev_prejacent_bio_1.bmes', 'r')
    outfile = codecs.open('data/4/dev_prejacent_binary_fixed.txt', 'w')
    outlist = []
    for inline, tline in zip(infile.readlines(), target_file.readlines()):
        print(inline)
        inline = inline.strip().split()
        tline = tline.strip().split()
        if len(inline) == 0:
            outfile.write('\t'.join(outlist) + '\n')
            outlist = []
        else:
            print(tline)
            if tline[1].startswith('R'):
                tag = inline[-1].split('-')[0]+'-R'
                inline[-1] = tag
            elif tline[1].startswith('L'):
                tag = inline[-1].split('-')[0]+'-L'
                inline[-1] = tag
            print(inline)
            outlist.append('###'.join(inline))

def create_correct_bio_five(binary=True):
    infile = codecs.open('data/only_bio/0/dev_prejacent_bio_1.bmes', 'r')
    target_file = codecs.open('data/all_unrolled_no_ambiguities/0/dev_all_unrolled_no_ambiguities_space.bmes', 'r')
    outfile = codecs.open('data/0/dev_prejacent_five.txt', 'w')
    outlist = []
    for inline, tline in zip(infile.readlines(), target_file.readlines()):
        print('new')
        inline = inline.strip().split()
        tline = tline.strip().split()
        print(inline)
        print(tline)
        if len(inline) == 0:
            outfile.write('\t'.join(outlist) + '\n')
            outlist = []
        else:
            if len(tline)<=1:
                pass
            elif 'ability' in tline[1]:
                tag = inline[-1].split('-')[0]+'-A'
                inline[-1] = tag
            elif 'deontic' in tline[1]:
                tag = inline[-1].split('-')[0]+'-D'
                inline[-1] = tag
            elif 'circu' in tline[1]:
                tag = inline[-1].split('-')[0] + '-C'
                inline[-1] = tag
            elif 'epist' in tline[1]:
                tag = inline[-1].split('-')[0] + '-E'
                inline[-1] = tag
            elif 'inte' in tline[1]:
                tag = inline[-1].split('-')[0] + '-I'
                inline[-1] = tag
            outlist.append('###'.join(inline))

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
                    if w not in ['T_S', 'T_E', 'R_S', 'R_E', 'L_S', 'L_E', 'Y_S', 'Y_E', 'X_S', 'X_E', 'V_S', 'V_E']:
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

#convert('data/with_T/0/train_prejacent_bio_1.bmes', 'data/0/train_prejacent_naive.txt')
convert_to_eval_format('data/predictions/new_prej/prejacent_binary_fixed_0', 'data/predictions/new_prej/prejacent_binary_fixed_0_eval', bottom_up=False)
#create_correct_bio_fine()
#create_correct_bio_five()