import os
import json


def main():
    # read DISGENET beliefs
    with open('../raw_data/gda_triples.tsv', 'r') as f:
        beliefs = f.readlines()

    # associate ID w/ triple
    id2triple = {}
    for belief in beliefs[1:]:  # skip first row -- header
        i, s, p, o = belief.strip().split('\t')
        id2triple[i] = (s, p, o)

    # create data dir if not exists
    os.makedirs('../data/', exist_ok=True)
    # store DISGENET KG
    with open('../data/kg.json', 'w') as out:
        json.dump(id2triple, out)


if __name__ == "__main__":
    main()
