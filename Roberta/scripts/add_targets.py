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
from collections import defaultdict

def get_sent_anno_map(infile_id):
    infile = codecs.open(infile_id)
    sent_anno_map = defaultdict(lambda : [])
    inner_list = []
    sent_list = []
    for line in infile.readlines():
        line = line.strip()
        if len(line)>0:
            inner_list.append(line)
            sent_list.append(line.split()[0])
        else:
            sent_anno_map[' '.join(sent_list)] = inner_list
            inner_list = []
            sent_list = []
    return sent_anno_map

def get_sent_anno_map2(infile_id):
    infile = codecs.open(infile_id)
    sent_anno_map = defaultdict(lambda : [])
    for line in infile.readlines():
        inner_list = []
        sent_list = []
        line = line.strip().split()
        for token in line:
            token = '\t'.join(token.split('###'))
            inner_list.append(token)
            sent_list.append(token.split()[0])
        sent_anno_map[' '.join(sent_list)] = inner_list
    return sent_anno_map


def merge():
    t_list = []
    sent_anno_map1 = get_sent_anno_map('$(user.home)/PycharmProjects/Modality/data/predictions/new_target/predicted_five_4_eval')
    sent_anno_map2 = get_sent_anno_map2('$(user.home)/PycharmProjects/Modality/data/full/test_prejacent_bio_new.txt')
    outfile = codecs.open('$(user.home)/PycharmProjects/Modality/data/full/test_prejacent_target_five_predicted.txt', 'w')
    for key, value in sent_anno_map2.items():
        anno1 = sent_anno_map1[key]
        print(anno1)
        outlist = []
        for inline, tline in zip(value, anno1):
            inline = inline.strip().split()
            tline = tline.strip().split()
            t_list.append(tline[1])
            if 'plau' in tline[2]:
                inline.append('2')
            elif 'prio' in tline[2]:
                inline.append('1')
            elif 'modal' in tline[2]:
                inline.append('1')
            elif 'inte' in tline[2]:
                inline.append('1')
            elif 'epist' in tline[2]:
                inline.append('2')
            elif 'circu' in tline[2]:
                inline.append('3')
            elif 'deontic' in tline[2]:
                inline.append('4')
            elif 'ability' in tline[2]:
                inline.append('5')
            else:
                inline.append('0')
            outlist.append('###'.join(inline))
        if len(outlist)>0:
            outfile.write('\t'.join(outlist) + '\n')
    print(set(t_list))

def merge_tags():
    t_list = []
    sent_anno_map1 = get_sent_anno_map2('$(user.home)/PycharmProjects/Modality/data/all_unrolled_no_ambiguities/test_all_unrolled_no_ambiguities_space.txt')
    sent_anno_map2 = get_sent_anno_map2('$(user.home)/PycharmProjects/Modality/data/full/test_prejacent_bio_new.txt')
    outfile = codecs.open('$(user.home)/PycharmProjects/Modality/data/full/test_prejacent_five.txt', 'w')
    for key, value in sent_anno_map2.items():
        anno1 = sent_anno_map1[key]
        print(anno1)
        outlist = []
        for inline, tline in zip(value, anno1):
            inline = inline.strip().split()
            tline = tline.strip().split()
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
        if len(outlist)>0:
            outfile.write('\t'.join(outlist) + '\n')
    print(set(t_list))

merge()
# target_file = codecs.open('$(user.home)/PycharmProjects/Modality/data/predictions/classifier_modal_not_modal_basic_fold4_test_eval', 'r')
# infile = codecs.open('$(user.home)/PycharmProjects/Modality/data/only_bio/test_prejacent_bio_1.bmes', 'r')
# outfile = codecs.open('$(user.home)/PycharmProjects/Modality/data/full/test_prejacent_target_naive_predicted.txt', 'w')
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
#         if tline[1].startswith('S'):
#             inline.append('1')
#         else:
#             inline.append('0')
#         outlist.append('###'.join(inline))
