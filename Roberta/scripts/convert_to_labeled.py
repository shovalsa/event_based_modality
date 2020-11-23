import codecs

def change(label, prev_label):
    if label == 'O':
        return label
    elif 'event' in label:
        return 'O'
    else:
        if prev_label == 'O':
            label = label.split('-')[1]
            label = 'B-'+label
        else:
            label = label.split('-')[1]
            label = 'I-'+label
        return label

def convert():
    infile = codecs.open('/Users/valentinapyatkin/PycharmProjects/Modality/data/predictions/new_prej/prejacent_binary_fixed_0_eval', 'r')
    outfile = codecs.open('/Users/valentinapyatkin/PycharmProjects/Modality/data/predictions/new_prej/prejacent_binary_fixed_0_eval_lab','w')
    prev1 = 'O'
    prev2 = 'O'
    for line in infile.readlines():
        line = line.strip().split()
        if len(line)>0:
            word = line[0]
            label1 = line[1]
            label2 = line[2]
            label1 = change(label1, prev1)
            prev1 = label1
            label2 = change(label2, prev2)
            prev2 = label2
            outfile.write(word+'\t'+label1+'\t'+label2+'\n')
    outfile.close()

convert()
