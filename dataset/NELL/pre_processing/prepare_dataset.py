import os
import json


def main():
    # read NELL ground truth
    with open('../raw_data/NELL_Mturk', 'r') as f:
        groundTruth = f.readlines()

    # read NELL beliefs
    with open('../raw_data/beliefs', 'r') as f:
        beliefs = f.readlines()

    # associate ID w/ label
    id2label = {}
    for gt in groundTruth:
        i, label = gt.strip().split('\t')
        id2label[i] = int(label)

    # associate ID w/ triple (if found in id2label)
    id2triple = {}
    for belief in beliefs:
        i, s, p, o = belief.strip().split('\t')
        if i in id2label:  # ID found within id2label
            id2triple[i] = (s, p, o)

    # create data dir if not exists
    os.makedirs('../data/', exist_ok=True)
    # store NELL dataset (kg + gt)
    with open('../data/gt.json', 'w') as out:
        json.dump(id2label, out)
    with open('../data/kg.json', 'w') as out:
        json.dump(id2triple, out)


if __name__ == "__main__":
    main()
