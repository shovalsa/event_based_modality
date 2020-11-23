
import codecs

infile = codecs.open('data/predictions/bla/prejacent_correct_naive0_eval', 'r')
outfile = codecs.open('data/predictions/bla/prejacent_correct_naive0_eval_bla', 'w')

prev1 = 'O'
prev2 = 'O'
for line in infile.readlines():
    if line.strip() == '':
        pass
    else:
        word, l1, l2 = line.strip().split()

        if l1[0] in ['T', 'R', 'L', 'I'] and prev1 == 'O':
            pass
        elif l1[0] in ['T', 'R', 'L', 'I'] and prev1[0] in ['T', 'R', 'L', 'I']:
            l1 = 'I-event'
        if l2[0] in ['T', 'R', 'L', 'I']  and prev2 == 'O':
            pass
            #l2 = 'B-event'
        elif l1[0] in ['T', 'R', 'L', 'I'] and prev1[0] in ['T', 'R', 'L', 'I']:
            l2 = 'I-event'
        prev1 = l1
        prev2 = l2
        outfile.write(word+'\t'+l1+'\t'+l2+'\n')
outfile.close()