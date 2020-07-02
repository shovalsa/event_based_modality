# import codecs
#
#
#
# infile = codecs.open('/Users/vale/PycharmProjects/Modality/data/only_bio/0/train_prejacent_bio_1.bmes', 'r')
# target_file = codecs.open('/Users/vale/PycharmProjects/Modality/data/binary_prej/0/train_prejacent_bio_1.bmes', 'r')
# outfile = codecs.open('/Users/vale/PycharmProjects/Modality/data/0/train_prejacent_target_binary.txt', 'w')
# outlist = []
# for inline, tline in zip(infile.readlines(), target_file.readlines()):
#     print(inline)
#     inline = inline.strip().split()
#     tline = tline.strip().split()
#     if len(inline) == 0:
#         outfile.write('\t'.join(outlist) + '\n')
#         outlist = []
#     else:
#         print(tline)
#         if tline[1].startswith('R'):
#             inline.append('1')
#         elif tline[1].startswith('L'):
#             inline.append('2')
#         else:
#             inline.append('0')
#         outlist.append('###'.join(inline))


import codecs




target_file = codecs.open('/Users/vale/PycharmProjects/Modality/data/predictions/classifier_modal_not_modal_basic_fold4_test_eval', 'r')
infile = codecs.open('/Users/vale/PycharmProjects/Modality/data/only_bio/test_prejacent_bio_1.bmes', 'r')
outfile = codecs.open('/Users/vale/PycharmProjects/Modality/data/full/test_prejacent_target_naive_predicted.txt', 'w')
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
        if tline[1].startswith('S'):
            inline.append('1')
        else:
            inline.append('0')
        outlist.append('###'.join(inline))