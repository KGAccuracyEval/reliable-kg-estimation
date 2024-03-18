# Efficient and Reliable Estimation of Knowledge Graph Accuracy
Data accuracy is a central dimension of data quality, especially when dealing with Knowledge Graphs (KGs). Auditing the accuracy of KGs is essential to make informed decisions in entity-oriented services or applications. However, manually evaluating the accuracy of large-scale KGs is prohibitively expensive, and research is focused on developing efficient sampling techniques for estimating KG accuracy. This work addresses the limitations of current KG accuracy estimation methods, which rely on the Wald method to build confidence intervals, addressing reliability issues such as zero-width and overshooting intervals. Our solution, rooted in the Wilson method and tailored for complex sampling designs, overcomes these limitations and ensures applicability across various evaluation scenarios.

## Contents

This repository contains the source code to estimate KG accuracy in an efficient and reliable manner. <br>
Instructions on how to acquire the data used for the experiments are reported below.

## Installation 

Clone this repository

```bash
git clone https://github.com/KGAccuracyEval/reliable-kg-estimation.git
```

Install all the requirements:

```bash
pip install -r requirements.txt
```

## Datasets

The KGs used in the exeperiments are four: YAGO, NELL, DisGeNET, and SYN 100M. <br>
For each KG, we report how to acquire and process the raw data before running the methods to estimate KG accuracy.

### YAGO & NELL

#### Acquisition

The YAGO and NELL datasets can be obtained using the following command:

```bash
wget https://aclanthology.org/attachments/D17-1183.Attachment.zip
```

From within the downloaded repository (i.e., ```supplementary```), the datasets and corresponding ground-truths can be obtained as follows.

For YAGO, take ```supplementary/kgeval_data/YAGO/data/beliefs``` and ```supplementary/kgeval_data/Mturk_data/YAGO_Mturk``` and move them into ```reliable-kg-estimation/dataset/YAGO/raw_data/```.

For NELL, take ```supplementary/kgeval_data/NELL/data/beliefs``` and ```supplementary/kgeval_data/Mturk_data/NELL_Mturk``` and move them into ```reliable-kg-estimation/dataset/NELL/raw_data/```.

#### Preparation

To prepare YAGO and NELL data for experiments, use the following commands.

For YAGO, move to ```/reliable-kg-estimation/dataset/YAGO/pre_processing/``` and run:

```bash
python prepare_dataset.py
```

For NELL, move to ```/reliable-kg-estimation/dataset/NELL/pre_processing/``` and run:

```bash
python prepare_dataset.py
```

