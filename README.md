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

From the  

#### Processing
