import os
import json


def main():
    # read DISGENET beliefs
    with open('../raw_data/gda_triples.tsv', 'r') as f:
        beliefs = f.readlines()

    # read DISGENET scores
    with open('../raw_data/gda_scores.tsv', 'r') as f:
        scores = f.readlines()

    # associate ID w/ triple
    id2triple = {}
    for belief in beliefs[1:]:  # skip first row -- header
        i, s, p, o = belief.strip().split('\t')
        id2triple[i] = (s, p, o)

    # associate ID w/ score
    id2score = {}
    for score in scores[1:]:  # skip first row -- header
        i, v = score.strip().split('\t')
        id2score[i] = float(v)

    # create data dir if not exists
    os.makedirs('../data/', exist_ok=True)
    # store DISGENET dataset (kg + ts)
    with open('../data/kg.json', 'w') as out:
        json.dump(id2triple, out)
    with open('../data/ts.json', 'w') as out:
        json.dump(id2score, out)


if __name__ == "__main__":
    main()
