import numpy as np


class TripleErrorModel(object):
    """
    The Triple Error Model (TEM) labels the triples in the KG using a fixed error rate in [0, 1]
    """

    def annotateKG(self, kg, errorP):
        """
        Annotate the KG based on the TEM fixed error rate

        :param kg: target KG
        :param errorP: fixed error rate
        :return: computed ground truth for the KG
        """

        # generate labels where prob(0) = errorP and prob(1) = 1-errorP
        labels = np.random.choice([0, 1], size=len(kg), p=[errorP, 1-errorP])

        # associate labels w/ triple IDs to create the ground truth
        groundTruth = {kg[i][0]: labels[i] for i in range(len(kg))}
        return groundTruth


class ClusterErrorModel(object):
    """
    The Cluster Error Model (CEM) labels clusters in the KG using a Binomial Mixture Model (BMM)
    """

    def computeProb(self, cSize, k, c, eps):
        """
        Compute the correctness probability of the BMM based on error term and cluster size

        :param cSize: the cluster size
        :param k: the cluster size threshold -- lower than k: use prob=0.5+eps
        :param c: the scaling factor for the influence of cluster size on cluster accuracy
        :param eps: the error term based on normal distribution w/ mean=0 and sd=sigma
        :return: the correctness probability based on BMM
        """

        if cSize < k:
            prob = 0.5 + eps
        else:
            prob = 1/(1+np.exp(-c*(cSize-k))) + eps
        return max(0.0, min(1.0, prob))

    def annotateKG(self, kg, k=3, c=0.01, sigma=0.1):
        """
        Annotate the KG based on the CEM fixed error rate

        :param kg: target KG
        :param k: the cluster size threshold -- lower than k: use prob=0.5+eps
        :param c: the scaling factor for the influence of cluster size on cluster accuracy
        :param sigma: the standard deviation of the error (normal) distribution
        :return: computed ground truth for the KG
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

        # compute error term based on N(0, sigma)
        eps = np.random.normal(loc=0.0, scale=sigma)

        # compute number of (correct) labels for each cluster based on BMM
        trues = [np.random.binomial(n=cSize, p=self.computeProb(cSize, k, c, eps)) for cSize in sizes]
        # generate cluster labels as [1*true+0*(size-true)]
        labels = [[1]*true + [0]*(size-true) for true, size in zip(trues, sizes)]

        # associate labels w/ triple IDs to create the ground truth
        groundTruth = {id_: label for i in range(len(heads)) for id_, label in zip(clusters[heads[i]], labels[i])}
        return groundTruth


class ScoreErrorModel(object):
    """
    The Score Error Model (SEM) labels the triples in the KG using the triple confidence score
    """

    def annotateKG(self, kg, ts):
        """
        Annotate the KG based on triple confidence scores

        :param kg: target KG
        :param ts: triple confidence scores
        :return: computed ground truth for the KG
        """

        # generate len(kg) labels where prob(1) = ts -- w/ n=1 binomial == bernoulli
        labels = np.random.binomial(n=1, p=ts, size=len(kg))

        # associate labels w/ triple IDs to create the ground truth
        groundTruth = {kg[i][0]: labels[i] for i in range(len(kg))}
        return groundTruth
