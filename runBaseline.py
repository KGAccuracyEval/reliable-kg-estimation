import numpy as np
import pandas as pd
import argparse
import random
import json
import os

from framework import baselineMethods, labelGenerators, utils


parser = argparse.ArgumentParser()

################################
###### Dataset parameters ######
################################

parser.add_argument('--dataset', default='YAGO', choices=['YAGO', 'NELL', 'DISGENET', 'SYN'], help='Target dataset.')
parser.add_argument('--generator', default='', choices=['TEM', 'CEM', ''], help='Synthetic label generation model. Default to none.')
parser.add_argument('--errorP', default=0.1, type=float, help='Fixed error rate for synthetic label generation. Required by Triple Error Model.')
parser.add_argument('--cSizeThr', default=3, type=int, help='Cluster size threshold. Required by Cluster Error Model.')
parser.add_argument('--scaleF', default=0.001, type=float, help='Scaling factor for the influence of cluster size on cluster accuracy. Required by Cluster Error Model.')
parser.add_argument('--errorSD', default=0.01, type=float, help='Standard deviation of the error (normal) distribution. Required by Cluster Error Model.')

###############################
###### Method parameters ######
###############################

parser.add_argument('--method', default='SRS', choices=['SRS', 'TWCS', 'STWCS'], help='Method of choice.')
parser.add_argument('--minSample', default=30, type=int, help='Min sample size required to perform eval.')
parser.add_argument('--stageTwoSize', default=3, type=int, help='Second-stage sample size. Required by two-stage sampling methods.')

#######################################
###### Stratification parameters ######
#######################################

parser.add_argument('--numStrata', default=2, type=int, help='Number of strata considered by stratification based sampling methods.')
parser.add_argument('--stratFeature', default='degree', choices=['degree'], help='Stratification feature of choice.')

###################################
###### Estimation parameters ######
###################################

parser.add_argument('--confLevel', default=0.05, type=float, help='Estimator confidence level (1-confLevel).')
parser.add_argument('--thrMoE', default=0.05, type=float, help='Threshold for Margin of Error (MoE).')

###################################
###### Annotation parameters ######
###################################

parser.add_argument('--c1', default=45, type=int, help='Average cost for Entity Identification (EI).')
parser.add_argument('--c2', default=25, type=int, help='Average cost for Fact Verification (FV).')

##################################
###### Computing parameters ######
##################################

parser.add_argument('--iterations', default=1000, type=int, help='Number of iterations for computing estimates.')
parser.add_argument('--seed', default=42, type=int, help='Random seed.')

args = parser.parse_args()


def labelGenerator(method):
    """

    :param method: synthetic label generation model
    :return: instance of specified synthetic label generation model
    """

    return {
        'TEM': lambda: labelGenerators.TripleErrorModel(),
        'CEM': lambda: labelGenerators.ClusterErrorModel()
    }[method]()


def samplingMethod(method, confLevel):
    """
    Instantiate the specific sampling method

    :param method: sampling method
    :param confLevel: estimator confidence level (1-confLevel).
    :return: instance of specified sampling method
    """

    return {
        'SRS': lambda: baselineMethods.SRSSampler(confLevel),
        'TWCS': lambda: baselineMethods.TWCSSampler(confLevel),
        'STWCS': lambda: baselineMethods.STWCSSampler(confLevel)
    }[method]()


def main():
    # set seed
    random.seed(args.seed)
    np.random.seed(args.seed)

    print('Load {} dataset.'.format(args.dataset))
    # get target dataset
    with open('./dataset/' + args.dataset + '/data/kg.json', 'r') as f:
        id2triple = json.load(f)

    # set KG as [(id, triple), ...]
    kg = list(id2triple.items())

    if args.generator:  # generate synthetic labels
        # set generator to label the KG
        print('Set {} generator to label {} KG'.format(args.generator, args.dataset))
        generator = labelGenerator(args.generator)
        # set params to generate labels
        gParams = {'kg': kg}
        if args.generator == 'TEM':  # TEM generator requires fixed error rate
            print('{} w/ error rate = {}'.format(args.generator, args.errorP))
            gParams['errorP'] = args.errorP
        if args.generator == 'CEM':  # CEM generator requires size threshold, scaling factor, and error standard dev
            print('{} w/ size thr={}, scaling factor={}, and error SD={}'.format(args.generator, args.cSizeThr, args.scaleF, args.errorSD))
            gParams['k'] = args.cSizeThr
            gParams['c'] = args.scaleF
            gParams['sigma'] = args.errorSD
        # annotate KG w/ generator
        gt = generator.annotateKG(**gParams)
    else:  # target dataset has ground truth
        with open('./dataset/' + args.dataset + '/data/gt.json', 'r') as f:  # get ground truth
            gt = json.load(f)

    # compute KG (real) accuracy
    acc = sum(gt.values())/len(gt)
    print('KG (real) accuracy: {}'.format(acc))

    # set efficient KG accuracy estimator w/ confidence level 1-args.confLevel
    print('Set {} estimator with confidence level {}%.'.format(args.method, 1 - args.confLevel))
    estimator = samplingMethod(args.method, args.confLevel)

    # set params to perform evaluation
    eParams = {'kg': kg, 'groundTruth': gt, 'minSample': args.minSample, 'thrMoE': args.thrMoE, 'iters': args.iterations}
    if (args.method == 'TWCS') or (args.method == 'STWCS'):  # two-stage sampling methods require second-stage sample size parameter
        eParams['stageTwoSize'] = args.stageTwoSize
    if args.method == 'STWCS':  # stratified sampling methods require stratification feature and warmup number of triples
        eParams['numStrata'] = args.numStrata
        eParams['stratFeature'] = args.stratFeature
    # all the considered methods work w/ cluster based cost function
    eParams['c1'] = args.c1
    eParams['c2'] = args.c2

    # perform the evaluation procedure for args.iterations times and compute estimates
    print('Perform KG accuracy evaluation for {} times and stop at each iteration when MoE <= {}'.format(args.iterations, args.thrMoE))
    estimates = estimator.run(**eParams)
    # convert estimates to pandas and store them
    estimates = pd.DataFrame(estimates, columns=['annotTriples', 'annotCost', 'estimatedAcc', 'MoE'])
    print(u'estimated accuracy: {}\u00B1{}'.format(round(estimates['estimatedAcc'].mean(), 2), round(estimates['estimatedAcc'].std(), 2)))
    print(u'annotated triples: {}\u00B1{}'.format(round(estimates['annotTriples'].mean()), round(estimates['annotTriples'].std())))
    print(u'annotation cost (hours): {}\u00B1{}'.format(round(estimates['annotCost'].mean(), 2), round(estimates['annotCost'].std(), 2)))
    # reformat CIs as list of pairs
    lowerBound = (estimates['estimatedAcc'] - estimates['MoE']).tolist()
    upperBound = (estimates['estimatedAcc'] + estimates['MoE']).tolist()
    ci = list(zip(lowerBound, upperBound))
    # compute empirical coverage probability
    coverage, width = utils.computeCoverage(acc, ci)
    print(u'empirical coverage: {}\u00B1{}'.format(round(coverage, 2), round(width, 2)))

    # create dir (if not exists) where storing estimates
    dname = './results/'+args.dataset+'/'
    if args.generator:
        dname += args.generator+'/wald/'
        if args.generator == 'TEM':
            dname += str(args.errorP)+'/'
        if args.generator == 'CEM':
            dname += 'k='+str(args.cSizeThr)+'c='+str(args.scaleF)+'sigma='+str(args.errorSD)+'/'
    else:
        dname += 'wald/'
    os.makedirs(dname, exist_ok=True)
    # set file name
    fname = args.method + '_batch=' + str(args.minSample)
    if args.method in ['TWCS', 'STWCS']:
        fname += '_stage2=' + str(args.stageTwoSize)
    if args.method == 'STWCS':
        fname += '_feature=' + args.stratFeature + '_strata=' + str(args.numStrata)
    # add file type
    fname += '.tsv'
    # store estimates
    estimates.to_csv(dname+fname, sep='\t', index=False)
    print('Estimates stored in {}{}'.format(dname, fname))


if __name__ == "__main__":
    main()
