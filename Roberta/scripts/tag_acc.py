import jsonlines

def eval(filename):
    TP = 0.0
    FP = 0.0
    FN = 0.0
    with jsonlines.open(filename) as reader:
        for obj in reader:
            pred = obj["tags"]
            gold = obj["gold_labels"]
            for p, g in zip(pred, gold):
                if p != 'O' and p == g:
                    print('yay')
                    print(pred)
                    print(gold)
                    TP += 1
                elif p != 'O':
                    print('lol')
                    print(pred)
                    print(gold)
                    FP += 1
                elif g != 'O':
                    FN += 1
                    print('lol2')
                    print(pred)
                    print(gold)
    precision = TP / (TP + FP)
    recall = TP / (TP + FN)
    f1 = 2.0 / ((1 / recall) + (1 / precision))
    print(f1)

eval("/Users/vale/PycharmProjects/Modality/data/predictions/classifier_roberta_basic_fold2")