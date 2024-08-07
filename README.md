# Efficient and Reliable Estimation of KG Accuracy
Data accuracy is a central dimension of data quality, especially when dealing with Knowledge Graphs (KGs). Auditing the accuracy of KGs is essential to make informed decisions in entity-oriented services or applications. However, manually evaluating the accuracy of large-scale KGs is prohibitively expensive, and research is focused on developing efficient sampling techniques for estimating KG accuracy. This work addresses the limitations of current KG accuracy estimation methods, which rely on the Wald method to build confidence intervals, addressing reliability issues such as zero-width and overshooting intervals. Our solution, rooted in the Wilson method and tailored for complex sampling designs, overcomes these limitations and ensures applicability across various evaluation scenarios.

## Contents

This repository contains the source code to estimate KG accuracy in an efficient and reliable manner. <br>
Instructions on installation, acquisition and preparation of the data used for the experiments, and deployment of baseline and proposed methods are reported below.

## Installation 

Clone this repository

```bash
git clone https://github.com/KGAccuracyEval/reliable-kg-estimation.git
```

Install Python 3.10 (preferably in a virtual environment). <br>
Install all the requirements:

```bash
pip install -r requirements.txt
```

## Datasets

The KGs used in the experiments are four: YAGO, NELL, DisGeNET, and SYN 100M. <br>
For each KG, we report how to acquire and process the raw data before running the methods to estimate KG accuracy.

### YAGO & NELL

#### Acquisition

The YAGO and NELL datasets can be obtained using the following command:

```bash
wget https://aclanthology.org/attachments/D17-1183.Attachment.zip
```

Then, use the following command to extract the ```supplementary``` repository from ```D17-1183.Attachment.zip```:

```bash
unzip D17-1183.Attachment.zip
```

From within the downloaded repository (i.e., ```supplementary```), the datasets and corresponding ground-truths can be obtained as follows.

For YAGO, take ```supplementary/kgeval_data/YAGO/data/beliefs``` and ```supplementary/kgeval_data/Mturk_data/YAGO_Mturk``` and move them into ```reliable-kg-estimation/dataset/YAGO/raw_data/```.

For NELL, take ```supplementary/kgeval_data/NELL/data/beliefs``` and ```supplementary/kgeval_data/Mturk_data/NELL_Mturk``` and move them into ```reliable-kg-estimation/dataset/NELL/raw_data/```.

#### Preparation

To prepare YAGO and NELL for experiments, use the following commands.

For YAGO, move to ```/reliable-kg-estimation/dataset/YAGO/pre_processing/``` and run:

```bash
python prepare_dataset.py
```

For NELL, move to ```/reliable-kg-estimation/dataset/NELL/pre_processing/``` and run:

```bash
python prepare_dataset.py
```

### DisGeNET

#### Acquisition

For DisGeNET, we used the rdf version ```v7.0.0```, which can be obtained using the following command:

```bash
wget -r -np -nH --cut-dirs=2 -A .ttl.gz -P /path/to/DisGeNET_RDF_v7 http://rdf.disgenet.org/download/v7.0.0/
```

Together with DisGeNET data, use the following command to obtain the DisGeNET ontology:

```bash
wget -P /path/to/DisGeNET_RDF_v7 http://semanticscience.org/ontology/sio.owl
```

Then, import the contents stored in ```/path/to/DisGeNET_RDF_v7``` into a graph database of choice (e.g., GraphDB or Virtuoso). Once DisGeNET has been imported into a graph database, use the following SPARQL query to fetch the required data:

```bash
PREFIX dcterms: <http://purl.org/dc/terms/>
PREFIX sio: <http://semanticscience.org/resource/>
PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>
PREFIX ncit: <http://ncicb.nci.nih.gov/xml/owl/EVS/Thesaurus.owl#>
PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
PREFIX wi: <http://purl.org/ontology/wi/core#>

SELECT DISTINCT ?gdaID ?geneID ?associationType ?diseaseID
WHERE {
  ?gda dcterms:identifier ?gdaID ;
       dcterms:description ?sentence ;
       sio:SIO_000628 ?geneIRI, ?diseaseIRI ;
       sio:SIO_000216 ?scoreIRI ;
       sio:SIO_000253 ?sourceIRI ;
       sio:SIO_000772 ?pmidIRI ;
       rdf:type ?associationTypeIRI .

  ?geneIRI rdf:type ncit:C16612 ;
           dcterms:identifier ?geneID .
    
  ?associationTypeIRI rdfs:label ?associationType .
    
  ?diseaseIRI rdf:type ncit:C7057 ;
              dcterms:identifier ?diseaseID .

  ?scoreIRI sio:SIO_000300 ?gdaScore .

  ?sourceIRI dcterms:title ?source ;
             wi:evidence ?sourceTypeIRI .

  ?sourceTypeIRI rdfs:label ?sourceType .

  ?pmidIRI dcterms:identifier ?pmid .
}
```

Finally, save the results of the SPARQL query as ```gda_triples.tsv``` and store them into ```reliable-kg-estimation/dataset/DISGENET/raw_data/```. <b>CAVEAT:</b> store ```gda_triples.tsv``` with a header containing column names.

#### Preparation

To prepare DisGeNET for experiments, move to ```/reliable-kg-estimation/dataset/DISGENET/pre_processing/``` and run:

```bash
python prepare_dataset.py
```

### SYN 100M

#### Generation

To generate the SYN 100M KG, move to ```/reliable-kg-estimation/dataset/SYN/generate_data/``` and run:

```bash
python generateGraph.py
```

## Methods 

We provide both the methods based on the Wald interval (baseline) and those based on binomial intervals (proposed) -- that is, Wilson, continuity-corrected Wilson, and Agresti-Coull. The methods based on Wald can be deployed via ```runBaseline.py```, whereas those based on binomial intervals via ```runEval.py```.

For instance, to use TWCS with Wald interval on YAGO, run:

```bash
python runBaseline.py --dataset YAGO --method TWCS --stageTwoSize 3
```

Similarly, to use TWCS with Wilson interval on NELL, run:

```bash
python runEval.py --dataset NELL --method TWCS --stageTwoSize 3 --ciMethod wilson
```

When working on KGs w/o ground-truth (i.e., DisGeNET and SYN 100M), the synthetic label generation model must be specified as well.

For instance, to use SRS with Agresti-Coull interval on DisGeNET TEM ($\varepsilon_{T}=0.5$), run:

```bash
python runEval.py --dataset DISGENET --generator TEM --errorP 0.5 --method SRS --ciMethod agresti-coull
```

Similarly, to use STWCS with Wilson interval on DisGeNET CEM, run:

```bash
python runEval.py --dataset DISGENET --generator CEM --method STWCS --stageTwoSize 5 --stratFeature degree --numStrata 2 --ciMethod wilson
``` 

The description of all the available arguments can be obtained by running ```python runBaseline.py --help```, for baseline methods, and ```python runEval.py --help```, for proposed methods. 

## Acknowledgments
The work is partially supported by the HEREDITARY project, as part of the EU Horizon Europe research and innovation programme under Grant Agreement No GA 101137074.

## Reference
If you use or extend our work, please cite the following:

```
@article{marchesin_silvello-vldb2024,
  title = {Efficient and Reliable Estimation of Knowledge Graph Accuracy},
  author = {S. Marchesin and G. Silvello},
  journal = {Proc. {VLDB} Endow.},
  volume = {17},
  number = {9},
  pages = {2392--2404},
  year = {2024},
  doi = {10.14778/3665844.3665865},
  url = {https://www.vldb.org/pvldb/vol17/p2392-marchesin.pdf}
}
```
