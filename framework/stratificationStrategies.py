import numpy as np


def stratifyCSRF(stratFeature, numStrata):
    """
    Perform stratification w/ Cumulative Square Root of Frequency (CSRF) based on stratification feature

    :param stratFeature: target stratification feature (must represent the entire population)
    :param numStrata: number of strata
    :return: feature indices divided into numStrata strata
    """

    # compute CSRF
    unique, counts = np.unique(stratFeature, return_counts=True)
    sqrt_counts = np.sqrt(counts)
    csrf = np.cumsum(sqrt_counts)
    csrf2unique = dict(zip(csrf, unique))

    # define boundaries for strata (intervals)
    strataSize = csrf[-1]/numStrata
    boundaries = [-1]
    boundaries += [strataSize * (i+1) for i in range(numStrata-1)]

    # sanity check
    assert len(boundaries) == numStrata

    # assign features to strata based on CSRF intervals
    strata = []
    for b in range(numStrata):
        if b == numStrata-1:
            intervals = [csrf2unique[c] for c in csrf if c > boundaries[b]]
        else:
            intervals = [csrf2unique[c] for c in csrf if (c > boundaries[b]) and (c < boundaries[b+1])]
        # store feature indices within stratum
        stratum = [i for i in range(len(stratFeature)) if stratFeature[i] in intervals]
        if stratum:  # remove empty stratum
            strata += [stratum]
    return strata
