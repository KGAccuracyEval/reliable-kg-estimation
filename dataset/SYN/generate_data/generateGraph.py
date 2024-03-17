import numpy as np
import argparse
import random
import json
import os

from scipy import stats
from tqdm import tqdm


parser = argparse.ArgumentParser()

parser.add_argument('--numTriples', default=100000000, type=int, help='Number of triples in the KG (total).')
parser.add_argument('--clusterMean', default=20, type=int, help='The (mean) cluster size of entity clusters in the KG.')
parser.add_argument('--clusterDev', default=15, type=int, help='The deviation from cluster (mean) size of entity clusters in the KG.')
parser.add_argument('--seed', default=42, type=int, help='Random seed.')

args = parser.parse_args()


def main():
    # set seed
    random.seed(args.seed)
    np.random.seed(args.seed)

    # compute the number of subject entities required to have a mean cluster size = args.clusterMean
    numSubj = args.numTriples // args.clusterMean
    print('The number of subject entities is {}'.format(numSubj))

    # generate IDs for subject entities
    subjIDs = list(range(numSubj))

    # set normal distribution to generate cluster sizes w/ loc=args.clusterMean and scale=args.clusterDev
    norm = stats.norm(loc=args.clusterMean, scale=args.clusterDev)

    # set params to store KG
    kg = {}
    tID = 1
    # iterate over subject entities and store args.clusterSize triples associated w/ subject
    for s in tqdm(subjIDs):
        # sample cluster size from normal distribution and require min size to be (at least) 1
        clusterSize = norm.rvs(size=1).astype(int)[0]
        clusterSize = max(1, clusterSize)
        # sample clusterSize object entities to generate triples
        objIDs = random.sample(range(args.numTriples), k=clusterSize)
        # iterate over object entities and store triple
        for o in objIDs:
            # store triple
            kg[str(tID)] = [str(s), 'p', str(o)]
            # update triple ID
            tID += 1

    # create data dir if not exists
    os.makedirs('../data/', exist_ok=True)
    # store KG as JSON
    with open('../data/kg.json', 'w') as out:
        json.dump(kg, out)


if __name__ == "__main__":
    main()
