import random
import numpy as np
import networkx as ntx

from tqdm import tqdm
from scipy import stats
from .utils import clusterCostFunction
from .stratificationStrategies import stratifyCSRF


class SRSSampler(object):
    """
    This class represents the Simple Random Sampling (SRS) scheme used to perform KG accuracy evaluation.
    The SRS estimator is an unbiased estimator.
    """

    def __init__(self, alpha=0.05):
        """
        Initialize the sampler and set confidence level plus Normal critical value z with right-tail probability αlpha/2

        :param alpha: the user defined confidence level
        """

        # confidence level
        self.alpha = alpha
        self.z = stats.norm.isf(self.alpha/2)

    def estimate(self, sample):
        """
        Estimate the KG accuracy based on sample

        :param sample: input sample (i.e., set of triples) used for estimation
        :return: KG accuracy estimate
        """

        return sum(sample)/len(sample)

    def computeVar(self, sample):
        """
        Compute the sample variance

        :param sample: input sample (i.e., set of triples) used for estimation
        :return: sample variance
        """

        ae = self.estimate(sample)

        # count number of triples in sample
        n = len(sample)

        if n*(n-1) != 0:  # compute variance
            var = (1/(n*(n-1))) * sum([(t - ae) ** 2 for t in sample])
        else:  # set variance to inf
            var = np.inf
        return var

    def computeMoE(self, sample):
        """
        Compute the Margin of Error (MoE) based on the sample and the Normal critical value z with right-tail probability α/2

        :param sample: input sample (i.e., set of triples) used for estimation
        :return: the MoE value
        """

        # compute sample variance
        var = self.computeVar(sample)
        # compute the margin of error (i.e., z * sqrt(var))
        moe = self.z * (var ** 0.5)
        return moe

    def run(self, kg, groundTruth, minSample=30, thrMoE=0.05, c1=45, c2=25, iters=1000):
        """
        Run the evaluation procedure on KG w/ SRS and stop when MoE <= thr
        :param kg: the target KG
        :param groundTruth: the ground truth associated w/ target KG
        :param minSample: the min sample size required to trigger the evaluation procedure
        :param thrMoE: the user defined MoE threshold
        :param c1: average cost for Entity Identification (EI)
        :param c2: average cost for Fact Verification (FV)
        :param iters: number of times the estimation is performed
        :return: evaluation statistics
        """

        estimates = []
        for _ in tqdm(range(iters)):  # perform iterations
            # set params
            moe = 1
            heads = {}
            sample = []

            while moe > thrMoE:  # stop when MoE gets lower than threshold
                # perform SRS over the KG
                id_, triple = random.choices(population=kg, k=1)[0]
                if triple[0] not in heads:  # found new head (cluster) -- increase the num of clusters within sample
                    heads[triple[0]] = 1
                # get annotations for triples within sample
                sample += [groundTruth[id_]]

                if len(sample) >= minSample:  # compute MoE
                    moe = self.computeMoE(sample)

            # compute cost function (cluster based)
            cost = clusterCostFunction(len(heads), len(sample), c1, c2)
            # compute KG accuracy estimate
            estimate = self.estimate(sample)

            # store stats
            estimates += [[len(sample), cost, estimate, moe]]
        # return stats
        return estimates


class TWCSSampler(object):
    """
    This class represents the Two-stage Weighted Cluster Sampling (TWCS) scheme used to perform KG accuracy evaluation.
    The TWCS estimator is an unbiased estimator.
    """

    def __init__(self, alpha=0.05):
        """
        Initialize the sampler and set confidence level plus Normal critical value z with right-tail probability α/2

        :param alpha: the user defined confidence level
        """

        # confidence level
        self.alpha = alpha
        self.z = stats.norm.isf(self.alpha/2)

    def estimate(self, sample):
        """
        Estimate the KG accuracy based on the sample

        :param sample: input sample (i.e., clusters of triples) used for estimation
        :return: KG accuracy estimate
        """

        # compute, for each cluster, the cluster accuracy estimate
        cae = [sum(cluster)/len(cluster) for cluster in sample]
        # compute estimate
        return sum(cae)/len(cae)

    def computeVar(self, sample):
        """
        Compute the sample variance

        :param sample: input sample (i.e., set of triples) used for estimation
        :return: sample variance
        """

        # compute, for each cluster, the cluster accuracy estimate
        cae = [sum(cluster) / len(cluster) for cluster in sample]
        # compute estimate
        ae = sum(cae) / len(cae)

        # count number of clusters in sample
        n = len(sample)

        if n*(n-1) != 0:  # compute variance
            var = (1/(n*(n-1)))*sum([(cae[i] - ae) ** 2 for i in range(n)])
        else:  # set variance to inf
            var = np.inf
        return var

    def computeMoE(self, sample):
        """
        Compute the Margin of Error (MoE) based on the sample and the Normal critical value z with right-tail probability α/2

        :param sample: input sample (i.e., set of triples) used for estimation
        :return: the MoE value
        """

        # compute sample variance
        var = self.computeVar(sample)
        # compute the margin of error (i.e., z * sqrt(var))
        moe = self.z * (var ** 0.5)
        return moe

    def run(self, kg, groundTruth, stageTwoSize=5, minSample=30, thrMoE=0.05, c1=45, c2=25, iters=1000):
        """
        Run the evaluation procedure on KG w/ TWCS and stop when MoE < thr
        :param kg: the target KG
        :param groundTruth: the ground truth associated w/ target KG
        :param stageTwoSize: second-stage sample size.
        :param minSample: the min sample size required to trigger the evaluation procedure
        :param thrMoE: the user defined MoE threshold
        :param c1: average cost for Entity Identification (EI)
        :param c2: average cost for Fact Verification (FV)
        :param iters: number of times the estimation is performed
        :return: evaluation statistics
        """

        # prepare (head) clusters
        clusters = {}
        for id_, triple in kg:  # iterate over KG triples and make clusters
            if triple[0] in clusters:  # cluster found -- add triple (id)
                clusters[triple[0]] += [id_]
            else:  # cluster not found -- create cluster and add triple (id)
                clusters[triple[0]] = [id_]

        # get cluster heads
        heads = list(clusters.keys())
        # get cluster sizes
        sizes = [len(clusters[s]) for s in heads]
        # compute cluster weights based on cluster sizes
        weights = [sizes[i]/sum(sizes) for i in range(len(sizes))]

        estimates = []
        for _ in tqdm(range(iters)):  # perform iterations
            # set params
            moe = 1
            numC = 0
            numT = 0
            sample = []

            while moe > thrMoE:  # stop when MoE gets lower than threshold
                # perform TWCS over clusters
                head = random.choices(population=heads, weights=weights, k=1)[0]
                # increase heads number
                numC += 1

                # second-stage sampling
                pool = clusters[head]
                stageTwo = random.sample(pool, min(stageTwoSize, len(pool)))

                # get annotations for triples within sample
                sample += [[groundTruth[triple] for triple in stageTwo]]
                # increase triples number
                numT += len(stageTwo)

                if numT >= minSample:  # compute MoE
                    moe = self.computeMoE(sample)

            # compute cost function (cluster based)
            cost = clusterCostFunction(numC, numT, c1, c2)
            # compute KG accuracy estimate
            estimate = self.estimate(sample)

            # store stats
            estimates += [[numT, cost, estimate, moe]]
        # return stats
        return estimates


class STWCSSampler(object):
    """
    This class represents the Stratified Two-stage Weighted Cluster Sampling (STWCS) scheme used to perform KG accuracy evaluation.
    The STWCS estimator is an unbiased estimator.
    """

    def __init__(self, alpha=0.05):
        """
        Initialize the sampler and set confidence level plus Normal critical value z with right-tail probability α/2

        :param alpha: the user defined confidence level
        """

        # confidence level
        self.alpha = alpha
        self.z = stats.norm.isf(self.alpha/2)

        # instantiate the TWCS sampling method
        self.twcs = TWCSSampler(self.alpha)

    def estimate(self, strataSamples, strataWeights):
        """
        Estimate the KG accuracy based on the sample

        :param strataSamples: strata samples (i.e., clusters of triples) used for estimation
        :param strataWeights: strata weights
        :return: KG accuracy estimate
        """

        # compute, for each stratum sample, the TWCS based accuracy estimate
        sae = [self.twcs.estimate(stratumSample) if stratumSample else None for stratumSample in strataSamples]
        # compute estimate
        return sum([sae[i] * strataWeights[i] for i in range(len(strataSamples)) if sae[i]])

    def computeVar(self, strataSamples, strataWeights):
        """
        Compute the sample variance

        :param strataSamples: strata samples (i.e., clusters of triples) used for estimation
        :param strataWeights: strata weights
        :return: sample standard deviation
        """

        # compute, for each stratum, the TWCS estimated variance
        strataVars = [self.twcs.computeVar(stratumSample) if stratumSample else None for stratumSample in strataSamples]
        # compute variance
        return sum([(strataVars[i]) * (strataWeights[i] ** 2) for i in range(len(strataSamples)) if strataVars[i]])

    def computeMoE(self, strataSamples, strataWeights):
        """
        Compute the Margin of Error (MoE) based on the sample and the Normal critical value z with right-tail probability α/2

        :param strataSamples: strata samples (i.e., clusters of triples) used for estimation
        :param strataWeights: strata weights
        :return: the MoE value
        """

        # compute sample variance
        var = self.computeVar(strataSamples, strataWeights)

        # compute the margin of error (i.e., z * sqrt(var))
        moe = self.z * (var ** 0.5)
        return moe

    def run(self, kg, groundTruth, numStrata=5, stratFeature='degree', stageTwoSize=5, minSample=30, thrMoE=0.05, c1=45, c2=25, iters=1000):
        """
        Run the evaluation procedure on KG w/ TWCS and stop when MoE <= thr
        :param kg: the target KG
        :param groundTruth: the ground truth associated w/ target KG
        :param numStrata: number of considered strata
        :param stratFeature: target stratification feature.
        :param stageTwoSize: second-stage sample size
        :param minSample: the min sample size required to trigger the evaluation procedure
        :param thrMoE: the user defined MoE threshold
        :param c1: average cost for Entity Identification (EI)
        :param c2: average cost for Fact Verification (FV)
        :param iters: number of times the estimation is performed
        :return: evaluation statistics
        """

        # prepare (head) clusters
        clusters = {}
        for id_, triple in kg:  # iterate over KG triples and make clusters
            if triple[0] in clusters:  # cluster found -- add triple (id)
                clusters[triple[0]] += [id_]
            else:  # cluster not found -- create cluster and add triple (id)
                clusters[triple[0]] = [id_]

        # get cluster heads
        heads = list(clusters.keys())
        g = ntx.DiGraph()

        for id, triple in kg:
            s, p, o = triple
            g.add_node(s)
            g.add_node(o)
            g.add_edge(s, o, label=p)

        # compute degree centrality
        dCent = ntx.degree_centrality(g)

        # get cluster sizes and degree centralities
        sizes = [len(clusters[s]) for s in heads]
        centrs = [dCent[h] for h in heads]

        # perform stratification based on stratFeature
        assert stratFeature in ['degree']
        strata = stratifyCSRF(centrs, numStrata)

        # compute strata weights
        strataWeights = [sum([sizes[i] for i in stratum])/len(kg) for stratum in strata]

        # partition data by stratum
        headsXstratum = [[heads[i] for i in stratum] for stratum in strata]
        sizesXstratum = [[sizes[i] for i in stratum] for stratum in strata]
        weightsXstratum = [[size/sum(stratumSizes) for size in stratumSizes] for stratumSizes in sizesXstratum]

        estimates = []
        for _ in tqdm(range(iters)):  # perform iterations
            # set params
            moe = 1
            strataC = [0 for _ in range(numStrata)]
            strataT = [0 for _ in range(numStrata)]
            strataSamples = [[] for _ in range(numStrata)]

            while moe > thrMoE:  # stop when MoE gets lower than threshold
                ix = random.choices(population=range(numStrata), weights=strataWeights, k=1)[0]
                # perform TWCS over stratum ix
                head = random.choices(population=headsXstratum[ix], weights=weightsXstratum[ix], k=1)[0]
                # increase heads number
                strataC[ix] += 1

                # second-stage sampling
                pool = clusters[head]
                stageTwo = random.sample(pool, min(stageTwoSize, len(pool)))

                # get annotations for triples within sample
                strataSamples[ix] += [[groundTruth[triple] for triple in stageTwo]]
                # increase triples number
                strataT[ix] += len(stageTwo)

                if sum(strataT) >= minSample:  # compute MoE
                    moe = self.computeMoE(strataSamples, strataWeights)

            # compute cost function (cluster based)
            cost = clusterCostFunction(sum(strataC), sum(strataT), c1, c2)
            # compute KG accuracy estimate
            estimate = self.estimate(strataSamples, strataWeights)

            # store stats
            estimates += [[sum(strataT), cost, estimate, moe]]
        # return stats
        return estimates
