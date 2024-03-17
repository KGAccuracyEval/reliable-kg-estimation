from scipy import stats


def clusterCostFunction(heads, triples, c1=45, c2=25):
    """
    Compute the cluster-based annotation cost function (in hours)

    :param heads: num of heads (clusters)
    :param triples: num of triples
    :param c1: average cost for Entity Identification (EI)
    :param c2: average cost for Fact Verification (FV)
    :return: the annotation cost function (in hours)
    """

    return (heads * c1 + triples * c2) / 3600


def computeCoverage(acc, ci):
    """
    Compute empirical coverage probability given (true) accuracy and estimator confidence intervals

    :param acc: (true) KG accuracy
    :param ci: estimator confidence intervals
    :return: the coverage probability
    """

    # count how many times the true KG accuracy lies within the estimated CIs
    counts = [lB <= acc <= uB for lB, uB in ci]
    # compute coverage
    coverage = sum(counts)/len(counts)
    stat = stats.binomtest(k=sum(counts), n=len(counts), p=0.95)
    bounds = stat.proportion_ci()
    width = (bounds[1]-bounds[0])/2
    # return coverage
    return coverage, width
