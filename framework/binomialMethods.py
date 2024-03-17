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

        # possible confidence intervals
        self.computeMoE = {'wilson': self.computeWCI, 'wilsonCC': self.computeCCWCI, 'agresti-coull': self.computeACCI}

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

        # estimate mean
        ae = self.estimate(sample)
        # count number of triples in sample
        n = len(sample)
        # compute variance
        var = (1/n) * (ae * (1-ae))
        return var

    def computeWCI(self, sample):
        """
        Compute the Wilson Confidence Interval (CI)

        :param sample: input sample (i.e., set of triples) used for estimation
        :return: the CI as (lowerBound, upperBound)
        """

        # compute mean estimate
        ae = self.estimate(sample)

        n = len(sample)
        x = sum(sample)

        # compute the adjusted sample size
        n_ = n + self.z ** 2
        # compute the adjusted number of successes
        x_ = x + (self.z ** 2) / 2
        # compute the adjusted mean estimate
        ae_ = x_ / n_

        # compute the margin of error
        moe = ((self.z * (n ** 0.5)) / n_) * (((ae * (1 - ae)) + ((self.z ** 2) / (4 * n))) ** 0.5)

        if (n <= 50 and x in [1, 2]) or (n >= 51 and x in [1, 2, 3]):
            lowerB = 0.5 * stats.chi2.isf(q=1 - self.alpha, df=2 * x) / n
        else:
            lowerB = ae_ - moe

        if (n <= 50 and x in [n - 1, n - 2]) or (n >= 51 and x in [n - 1, n - 2, n - 3]):
            upperB = 1 - (0.5 * stats.chi2.isf(q=1 - self.alpha, df=2 * (n - x))) / n
        else:
            upperB = ae_ + moe

        # return CI as (lowerBound, upperBound)
        return lowerB, upperB

    def computeCCWCI(self, sample):
        """
        Compute the Continuity-Corrected Wilson Confidence Interval (CI)

        :param sample: input sample (i.e., set of triples) used for estimation
        :return: the CI as (lowerBound, upperBound)
        """

        # compute mean estimate
        ae = self.estimate(sample)
        _ae = 1 - ae

        n = len(sample)
        x = sum(sample)

        denom = 2 * (n + self.z ** 2)
        center = (2 * n * ae + self.z ** 2) / denom

        if x == 0:
            lowerB = 0.0
        else:
            dlo = (1 + self.z * (self.z ** 2 - 2 - 1 / n + 4 * ae * (n * _ae + 1))**0.5) / denom
            lowerB = center - dlo

        if x == n:
            upperB = 1.0
        else:
            dup = (1 + self.z * (self.z ** 2 + 2 - 1 / n + 4 * ae * (n * _ae - 1))**0.5) / denom
            upperB = center + dup

        # return CI as (lowerBound, upperBound)
        return lowerB, upperB

    def computeACCI(self, sample):
        """
        Compute the Agresti-Coull Confidence Interval (CI)

        :param sample: input sample (i.e., set of triples) used for estimation
        :return: the CI as (lowerBound, upperBound)
        """

        # compute the adjusted sample size
        n_ = len(sample) + self.z ** 2
        # compute the adjusted number of successes
        x_ = sum(sample) + (self.z ** 2)/2
        # compute the adjusted mean estimate
        ae_ = x_/n_
        # compute the margin of error
        moe = self.z * (((1/n_) * (ae_ * (1-ae_))) ** 0.5)
        # return CI as (lowerBound, upperBound)
        return ae_ - moe, ae_ + moe

    def run(self, kg, groundTruth, minSample=30, thrMoE=0.05, ciMethod='wilson', c1=45, c2=25, iters=1000):
        """
        Run the evaluation procedure on KG w/ SRS and stop when MoE < thr
        :param kg: the target KG
        :param groundTruth: the ground truth associated w/ target KG
        :param minSample: the min sample size required to trigger the evaluation procedure
        :param thrMoE: the user defined MoE threshold
        :param ciMethod: the method used to compute Confidence Intervals (CIs)
        :param c1: average cost for Entity Identification (EI)
        :param c2: average cost for Fact Verification (FV)
        :param iters: number of times the estimation is performed
        :return: evaluation statistics
        """

        estimates = []
        for _ in tqdm(range(iters)):  # perform iterations
            # set params
            lowerB = 0.0
            upperB = 1.0
            heads = {}
            sample = []

            while (upperB-lowerB)/2 > thrMoE:  # stop when MoE gets lower than threshold
                # perform SRS over the KG
                id_, triple = random.choices(population=kg, k=1)[0]
                if triple[0] not in heads:  # found new head (cluster) -- increase the num of clusters within sample
                    heads[triple[0]] = 1
                # get annotations for triples within sample
                sample += [groundTruth[id_]]

                if len(sample) >= minSample:  # compute CI
                    lowerB, upperB = self.computeMoE[ciMethod](sample)

            # compute cost function (cluster based)
            cost = clusterCostFunction(len(heads), len(sample), c1, c2)
            # compute KG accuracy estimate
            estimate = self.estimate(sample)

            # store stats
            estimates += [[len(sample), cost, estimate, lowerB, upperB]]
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

        # possible confidence intervals
        self.computeMoE = {'wilson': self.computeWCI, 'wilsonCC': self.computeCCWCI, 'agresti-coull': self.computeACCI}

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

    def computeESS(self, sample, numT):
        """
        Compute the Effective Sample Size

        :param sample: input sample (i.e., set of triples) used for estimation
        :param numT: total number of triples in the sample
        :return: the effective sample size
        """

        # compute clusters mean size
        meanSize = np.mean([len(cluster) for cluster in sample])

        N = len(sample)
        M = sum([len(c) for c in sample])

        x_ = 1 / M * sum([sum(c) for c in sample])

        cSizes = [len(c) for c in sample]
        maxSize = max(cSizes)

        s = 1 / (M - 1) * (sum([sum([(c[i] - x_) ** 2 for c in sample if i < len(c)]) for i in range(maxSize)]))
        if s == 0:
            return numT

        # compute icc
        icc = meanSize / (meanSize - 1) * (1 / (s * N)) * (sum([(sum(c) / len(c) - x_) ** 2 for c in sample])) - 1 / (meanSize - 1)

        # compute design effect
        dEffect = 1 + ((meanSize - 1) * icc)

        # compute the Effective Sample Size (ESS)
        if dEffect > 0:
            ess = (numT / dEffect)
        else:
            ess = numT

        # return effective sample size
        return ess

    def computeWCI(self, sample, numT):
        """
        Compute the Wilson Confidence Interval (CI)

        :param sample: input sample (i.e., set of triples) used for estimation
        :param numT: total number of triples in the sample
        :return: the CI as (lowerBound, upperBound)
        """

        # compute mean estimate
        ae = self.estimate(sample)

        # compute the effective sample size
        n = self.computeESS(sample, numT)
        if n == numT:  # effective sample size equal to actual sample size
            x = sum([sum(c) for c in sample])
        else:
            x = n*ae
        # compute the adjusted sample size
        n_ = n + self.z ** 2
        # compute the adjusted number of successes
        x_ = x + (self.z ** 2)/2
        # compute the adjusted mean estimate
        ae_ = x_/n_

        # compute the margin of error
        moe = ((self.z * (n ** 0.5)) / n_) * (((ae * (1-ae)) + ((self.z ** 2) / (4 * n))) ** 0.5)

        if (n <= 50 and x in [1, 2]) or (n >= 51 and x in [1, 2, 3]):
            lowerB = 0.5 * stats.chi2.isf(q=1-self.alpha, df=2 * x) / n
        else:
            lowerB = ae_ - moe

        if (n <= 50 and x in [n - 1, n - 2]) or (n >= 51 and x in [n - 1, n - 2, n - 3]):
            upperB = 1 - (0.5 * stats.chi2.isf(q=1-self.alpha, df=2 * (n-x))) / n
        else:
            upperB = ae_ + moe

        # return CI as (lowerBound, upperBound)
        return lowerB, upperB

    def computeCCWCI(self, sample, numT):
        """
        Compute the Continuity-Corrected Wilson Confidence Interval (CI)

        :param sample: input sample (i.e., set of triples) used for estimation
        :param numT: total number of triples in the sample
        :return: the CI as (lowerBound, upperBound)
        """

        # compute mean estimate
        ae = self.estimate(sample)
        _ae = 1 - ae

        # compute the effective sample size
        n = self.computeESS(sample, numT)
        if n == numT:  # effective sample size equal to actual sample size
            x = sum([sum(c) for c in sample])
        else:  # effective sample size smaller than actual sample size
            x = n * ae

        denom = 2 * (n + self.z ** 2)
        center = (2 * n * ae + self.z ** 2) / denom

        if x == 0:
            lowerB = 0.0
        else:
            dlo = (1 + self.z * (self.z ** 2 - 2 - 1 / n + 4 * ae * (n * _ae + 1))**0.5) / denom
            lowerB = center - dlo

        if x == n:
            upperB = 1.0
        else:
            dup = (1 + self.z * (self.z ** 2 + 2 - 1 / n + 4 * ae * (n * _ae - 1))**0.5) / denom
            upperB = center + dup

        # return CI as (lowerBound, upperBound)
        return lowerB, upperB

    def computeACCI(self, sample, numT):
        """
        Compute the Agresti-Coull Confidence Interval (CI)

        :param sample: input sample (i.e., set of triples) used for estimation
        :param numT: total number of triples in the sample
        :return: the CI as (lowerBound, upperBound)
        """

        # compute mean estimate
        ae = self.estimate(sample)

        # compute the effective sample size
        n = self.computeESS(sample, numT)
        if n == numT:  # effective sample size equal to actual sample size
            x = sum([sum(c) for c in sample])
        else:  # effective sample size smaller than actual sample size
            x = n * ae
        # compute the adjusted sample size
        n_ = n + self.z ** 2
        # compute the adjusted number of successes
        x_ = x + (self.z ** 2) / 2
        # compute the adjusted mean estimate
        ae_ = x_ / n_

        # compute the margin of error
        moe = self.z * (((1 / n_) * (ae_ * (1 - ae_))) ** 0.5)
        # return CI as (lowerBound, upperBound)
        return ae_ - moe, ae_ + moe

    def run(self, kg, groundTruth, stageTwoSize=5, minSample=30, thrMoE=0.05, ciMethod='wilson', c1=45, c2=25, iters=1000):
        """
        Run the evaluation procedure on KG w/ TWCS and stop when MoE <= thr
        :param kg: the target KG
        :param groundTruth: the ground truth associated w/ target KG
        :param stageTwoSize: second-stage sample size.
        :param minSample: the min sample size required to trigger the evaluation procedure
        :param thrMoE: the user defined MoE threshold
        :param ciMethod: the method used to compute Confidence Intervals (CIs)
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
            lowerB = 0.0
            upperB = 1.0
            numC = 0
            numT = 0
            sample = []

            while (upperB-lowerB)/2 > thrMoE:  # stop when MoE gets lower than threshold
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
                    lowerB, upperB = self.computeMoE[ciMethod](sample, numT)

            # compute cost function (cluster based)
            cost = clusterCostFunction(numC, numT, c1, c2)
            # compute KG accuracy estimate
            estimate = self.estimate(sample)

            # store stats
            estimates += [[numT, cost, estimate, lowerB, upperB]]
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

        # possible confidence intervals
        self.computeMoE = {'wilson': self.computeWCI, 'wilsonCC': self.computeCCWCI, 'agresti-coull': self.computeACCI}

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
        strataVars = [self.twcs.computeVar(stratumSample) for stratumSample in strataSamples]
        # compute variance
        return sum([(strataVars[i]) * (strataWeights[i] ** 2) for i in range(len(strataSamples))])

    def computeESS(self, strataSamples, strataWeights, strataT, numC, numS):
        """
        Compute the Effective Sample Size adjusted for the design degrees of freedom

        :param strataSamples: strata samples (i.e., clusters of triples) used for estimation
        :param strataWeights: strata weights
        :param strataT: per-stratum number of triples in the sample
        :param numC: total number of clusters in the sample
        :param numS: total number of strata
        :return: the effective sample size
        """

        # compute sample size (num of triples across strata)
        numT = sum(strataT)
        # set vars to store ICCs and strata (sample) variances
        iccs = []
        strataSVars = []
        # set vars to store design effect based on the boundary conditions
        strataEVars = []
        strataEVarsV1 = []
        strataEVarsV2 = []

        for ix, sample in enumerate(strataSamples):
            if not sample:
                continue
            # compute clusters mean size
            meanSize = np.mean([len(cluster) for cluster in sample])

            N = len(sample)
            M = sum([len(c) for c in sample])

            x_ = (1 / M) * sum([sum(c) for c in sample])

            cSizes = [len(c) for c in sample]
            maxSize = max(cSizes)

            if M > 1:
                s = 1 / (M - 1) * (sum([sum([(c[i] - x_) ** 2 for c in sample if i < len(c)]) for i in range(maxSize)]))
            else:
                s = 0

            if s == 0 or meanSize == 1:
                icc = 0
            else:
                icc = meanSize / (meanSize - 1) * (1 / (s * N)) * (sum([(sum(c) / len(c) - x_) ** 2 for c in sample])) - 1 / (meanSize - 1)

            iccs.append(icc)
            strataEVars.append((strataWeights[ix] ** 2 * (1 + (meanSize - 1) * icc)) * ((x_ * (1 - x_)) / M))
            strataEVarsV1.append((strataWeights[ix] ** 2 * (numT / strataT[ix])) * (1 + ((meanSize - 1) * icc)))
            strataEVarsV2.append(strataWeights[ix] ** 2 * ((x_ * (1 - x_)) / M))
            strataSVars.append((x_ * (1 - x_)))

        if len(strataSVars) > 1 and abs(strataSVars[0]-strataSVars[1]) < 0.01:
            dEffect = sum(strataEVarsV1)
        elif len(strataSVars) > 1 and abs(iccs[0]-iccs[1]) < 0.01:
            icc = np.mean(iccs)
            mSize = np.mean([len(cluster) for sample in strataSamples for cluster in sample])
            srsAE = sum([sum(c) for sample in strataSamples for c in sample]) / numT
            srsEVar = (srsAE * (1 - srsAE)) / numT
            dEffect = (1 + (mSize - 1) * icc) * (sum(strataEVarsV2) / srsEVar)
        elif len(strataSVars) > 1 and abs(iccs[0]-iccs[1]) < 0.01 and abs(strataSVars[0]-strataSVars[1]) < 0.01:
            icc = np.mean(iccs)
            mSize = np.mean([len(cluster) for sample in strataSamples for cluster in sample])
            dEffect = 1 + (mSize - 1) * icc
        elif len(strataSVars) == 1:
            icc = np.mean(iccs)
            mSize = np.mean([len(cluster) for sample in strataSamples for cluster in sample])
            dEffect = 1 + (mSize - 1) * icc
        else:
            srsAE = sum([sum(c) for sample in strataSamples for c in sample]) / numT
            srsEVar = (srsAE * (1 - srsAE)) / numT
            if srsEVar == 0:
                dEffect = 0
            else:
                dEffect = sum(strataEVars) / srsEVar

        # compute design factor
        dFactor = (self.z / stats.t.isf(self.alpha / 2, df=numC - numS)) ** 2

        # compute the Effective Sample Size (ESS)
        if dEffect > 0:
            ess = (numT / dEffect) * dFactor
            if ess == 0:
                ess = numT
        else:
            ess = numT

        # return effective sample size
        return ess

    def computeWCI(self, strataSamples, strataWeights, strataT, strataC):
        """
        Compute the Wilson Confidence Interval (CI)

        :param strataSamples: strata samples (i.e., clusters of triples) used for estimation
        :param strataWeights: strata weights
        :param strataT: per-stratum number of triples in the sample
        :param strataC: per-stratum number of clusters in the sample
        :return: the CI as (lowerBound, upperBound)
        """

        # compute mean estimate
        ae = self.estimate(strataSamples, strataWeights)

        # get number of sample triples and clusters, as well as number of strata
        numT = sum(strataT)
        numC = sum(strataC)
        numS = len(strataSamples)

        n = self.computeESS(strataSamples, strataWeights, strataT, numC, numS)
        if n == numT:  # effective sample size equal to actual sample size
            x = sum([sum(c) for sample in strataSamples for c in sample])
        else:  # effective sample size smaller than actual sample size
            x = n * ae
        # compute the adjusted sample size
        n_ = n + self.z ** 2
        # compute the adjusted number of successes
        x_ = x + (self.z ** 2) / 2
        # compute the adjusted mean estimate
        ae_ = x_ / n_

        # compute the margin of error
        moe = ((self.z * (n ** 0.5)) / n_) * (((ae * (1 - ae)) + ((self.z ** 2) / (4 * n))) ** 0.5)

        if (n <= 50 and x in [1, 2]) or (n >= 51 and x in [1, 2, 3]):
            lowerB = 0.5 * stats.chi2.isf(q=1 - self.alpha, df=2 * x) / n
        else:
            lowerB = ae_ - moe

        if (n <= 50 and x in [n - 1, n - 2]) or (n >= 51 and x in [n - 1, n - 2, n - 3]):
            upperB = 1 - (0.5 * stats.chi2.isf(q=1 - self.alpha, df=2 * (n - x))) / n
        else:
            upperB = ae_ + moe

        # return CI as (lowerBound, upperBound)
        return lowerB, upperB

    def computeCCWCI(self, strataSamples, strataWeights, strataT, strataC):
        """
        Compute the Continuity-Corrected Wilson Confidence Interval (CI)

        :param strataSamples: strata samples (i.e., clusters of triples) used for estimation
        :param strataWeights: strata weights
        :param strataT: per-stratum number of triples in the sample
        :param strataC: per-stratum number of clusters in the sample
        :return: the CI as (lowerBound, upperBound)
        """

        # compute mean estimate
        ae = self.estimate(strataSamples, strataWeights)
        _ae = 1 - ae

        # get number of sample triples and clusters, as well as number of strata
        numT = sum(strataT)
        numC = sum(strataC)
        numS = len(strataSamples)

        n = self.computeESS(strataSamples, strataWeights, strataT, numC, numS)
        if n == numT:  # effective sample size equal to actual sample size
            x = sum([sum(c) for sample in strataSamples for c in sample])
        else:  # effective sample size smaller than actual sample size
            x = n * ae

        denom = 2 * (n + self.z ** 2)
        center = (2 * n * ae + self.z ** 2) / denom

        if x == 0:
            lowerB = 0.0
        else:
            dlo = (1 + self.z * (self.z ** 2 - 2 - 1 / n + 4 * ae * (n * _ae + 1))**0.5) / denom
            lowerB = center - dlo

        if x == n:
            upperB = 1.0
        else:
            dup = (1 + self.z * (self.z ** 2 + 2 - 1 / n + 4 * ae * (n * _ae - 1))**0.5) / denom
            upperB = center + dup

        # return CI as (lowerBound, upperBound)
        return lowerB, upperB

    def computeACCI(self, strataSamples, strataWeights, strataT, strataC):
        """
        Compute the Agresti-Coull Confidence Interval (CI)

        :param strataSamples: strata samples (i.e., clusters of triples) used for estimation
        :param strataWeights: strata weights
        :param strataT: per-stratum number of triples in the sample
        :param strataC: per-stratum number of clusters in the sample
        :return: the CI as (lowerBound, upperBound)
        """

        # compute mean estimate
        ae = self.estimate(strataSamples, strataWeights)

        # get number of sample triples and clusters, as well as number of strata
        numT = sum(strataT)
        numC = sum(strataC)
        numS = len(strataSamples)

        n = self.computeESS(strataSamples, strataWeights, strataT, numC, numS)
        if n == numT:  # effective sample size equal to actual sample size
            x = sum([sum(c) for sample in strataSamples for c in sample])
        else:  # effective sample size smaller than actual sample size
            x = n * ae
        # compute the adjusted sample size
        n_ = n + self.z ** 2
        # compute the adjusted number of successes
        x_ = x + (self.z ** 2) / 2
        # compute the adjusted mean estimate
        ae_ = x_ / n_

        # compute the margin of error
        moe = self.z * (((1 / n_) * (ae_ * (1 - ae_))) ** 0.5)
        # return CI as (lowerBound, upperBound)
        return ae_ - moe, ae_ + moe

    def run(self, kg, groundTruth, numStrata=5, stratFeature='degree', stageTwoSize=5, minSample=30, thrMoE=0.05, ciMethod='wilson', c1=45, c2=25, iters=1000):
        """
        Run the evaluation procedure on KG w/ TWCS and stop when MoE <= thr
        :param kg: the target KG
        :param groundTruth: the ground truth associated w/ target KG
        :param numStrata: number of considered strata
        :param stratFeature: target stratification feature.
        :param stageTwoSize: second-stage sample size
        :param minSample: the min sample size required to trigger the evaluation procedure
        :param thrMoE: the user defined MoE threshold
        :param ciMethod: the method used to compute Confidence Intervals (CIs)
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
            lowerB = 0.0
            upperB = 1.0
            strataC = [0 for _ in range(numStrata)]
            strataT = [0 for _ in range(numStrata)]
            strataSamples = [[] for _ in range(numStrata)]

            while (upperB-lowerB)/2 > thrMoE:  # stop when MoE gets lower than threshold
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

                if sum(strataT) >= minSample:  # compute CI
                    lowerB, upperB = self.computeMoE[ciMethod](strataSamples, strataWeights, strataT, strataC)

            # compute cost function (cluster based)
            cost = clusterCostFunction(sum(strataC), sum(strataT), c1, c2)
            # compute KG accuracy estimate
            estimate = self.estimate(strataSamples, strataWeights)

            # store stats
            estimates += [[sum(strataT), cost, estimate, lowerB, upperB]]
        # return stats
        return estimates
