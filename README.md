# Ranked List Truncation for Re-ranking (RLT4Reranking)
Supplementary materials for the paper titled "_Ranked List Truncation: From Retrieval to Re-ranking_". 

In this paper, we reproduce a comprehensive ranked list truncation (RLT) methods, originally designed for optimizing retrieval, in a "retrieve-then-re-rank" setup; we seek to examine to what extent established findings on RLT for retrieval are generalizable to the ``retrieve-then-re-rank'' setup.

This repository is structured into five distinct parts:
1. Prerequisites
2. Data preparation,
3. Feature generation,
4. Train and infer RLT methods,
5. Evaluation.
6  Plots
6. Results on Robust04


Note that for ease of reproducibility, we already uploaded the predicted performance files for all QPP methods reported in our paper. See here.

## 1. Prerequisites
We recommend executing all processes in a Linux environment.
```bash
pip install -r requirements.txt
```

## 1. Data preparation
For ease of reproducibility, 

### Download raw data

#### MS MARCO V1 passage ranking
```bash
# Download queries and qrels for TREC-DL 19 and 20, as well as the MS MARCO V1 passage ranking collection:
mkdir datasets/msmarco-v1-passage/queries 
wget -P ./datasets/msmarco-v1-passage/queries/ https://msmarco.z22.web.core.windows.net/msmarcoranking/msmarco-test2019-queries.tsv.gz
wget -P ./datasets/msmarco-v1-passage/queries/ https://msmarco.z22.web.core.windows.net/msmarcoranking/msmarco-test2020-queries.tsv.gz
gzip -d ./datasets/msmarco-v1-passage/queries/*.tsv.gz
mv ./datasets/msmarco-v1-passage/queries/msmarco-test2019-queries.tsv ./datasets/msmarco-v1-passage/queries/dl-19-passage.queries-original.tsv
mv ./datasets/msmarco-v1-passage/queries/msmarco-test2020-queries.tsv ./datasets/msmarco-v1-passage/queries/dl-20-passage.queries-original.tsv 

mkdir datasets/msmarco-v1-passage/qrels
wget -P ./datasets/msmarco-v1-passage/qrels/ ./datasets/msmarco-v1-passage/qrels https://trec.nist.gov/data/deep/2019qrels-pass.txt
wget -P ./datasets/msmarco-v1-passage/qrels/ ./datasets/msmarco-v1-passage/qrels https://trec.nist.gov/data/deep/2020qrels-pass.txt
mv ./datasets/msmarco-v1-passage/qrels/2019qrels-pass.txt ./datasets/msmarco-v1-passage/qrels/dl-19-passage.qrels.txt
mv ./datasets/msmarco-v1-passage/qrels/2020qrels-pass.txt ./datasets/msmarco-v1-passage/qrels/dl-20-passage.qrels.txt

mkdir datasets/
mkdir datasets/msmarco-v1-passage/ 
mkdir datasets/msmarco-v1-passage/corpus  
wget -P ./datasets/msmarco-v1-passage/corpus/ https://msmarco.z22.web.core.windows.net/msmarcoranking/collection.tar.gz --no-check-certificate
tar -zxvf  ./datasets/msmarco-v1-passage/corpus/collection.tar.gz  -C ./datasets/msmarco-v1-passage/corpus/
```

#### Robust04
```bash

```

```bash

### Fetch retrieved list

#### BM25 
```bash
# TREC-DL 19
python -m pyserini.search.lucene \
  --threads 16 --batch-size 128 \
  --index msmarco-v1-passage-slim \
  --topics datasets/msmarco-v1-passage/queries/dl-19-passage.queries-original.tsv \
  --output datasets/msmarco-v1-passage/runs/dl-19-passage.run-original-bm25-1000.txt \
  --bm25 --k1 0.9 --b 0.4 --hits 1000

# TREC-DL 20
python -m pyserini.search.lucene \
  --threads 16 --batch-size 128 \
  --index msmarco-v1-passage-slim \
  --topics datasets/msmarco-v1-passage/queries/dl-20-passage.queries-original.tsv \
  --output datasets/msmarco-v1-passage/runs/dl-20-passage.run-original-bm25-1000.txt \
  --bm25 --k1 0.9 --b 0.4 --hits 1000

# Roubust04
python -m pyserini.search.lucene \
  --threads 16 --batch-size 128 \
  --index /gpfs/work3/0/guse0654/cache/index/lucene-index.beir-v1.0.0-robust04.flat.20221116.505594 \
  --topics ./datasets/robust04/queries/robust04.queries-original.tsv \
  --output ./datasets/robust04/runs/robust04.run-original-bm25-flat-1000.txt \
  --output-format trec \
  --hits 1000 --bm25 --remove-query

```bash


#### SPLADE++
```bash
# TREC-DL 19
python -m pyserini.search.lucene \
  --threads 16 --batch-size 128 \
  --index msmarco-v1-passage-splade-pp-ed \
  --topics ./datasets/msmarco-v1-passage/queries/dl-19-passage.queries-original.tsv \
  --encoder naver/splade-cocondenser-ensembledistil \
  --output ./datasets/msmarco-v1-passage/runs/dl-19-passage.run-original-splade-pp-ed-pytorch-1000.txt \
  --hits 1000 --impact

# TREC-DL 20
python -m pyserini.search.lucene \
  --threads 16 --batch-size 128 \
  --index msmarco-v1-passage-splade-pp-ed \
  --topics dl20 \
  --encoder naver/splade-cocondenser-ensembledistil \
  --output run.msmarco-v1-passage.splade-pp-ed-pytorch.dl20.txt \
  --hits 1000 --impact
```

#### RepLLaMA
```bash
# TREC-DL 19
wget https://www.dropbox.com/scl/fi/byty1lk2um36imz0788yd/run.repllama.psg.dl19.txt?rlkey=615ootx2mia42cxdilp4tvqzh -O /datasets/msmarco-v1-passage/runs/run.repllama.psg.dl19.txt

python -u format \
--input_path /datasets/msmarco-v1-passage/runs/run.repllama.psg.dl19.txt \
--output_path /datasets/msmarco-v1-passage/runs/dl-19-passage.run-original-repllama-1000.txt \
--ranker_name

# TREC-DL 20
wget https://www.dropbox.com/scl/fi/drgg9vj8mxe3qwayggj9o/run.repllama.psg.dl20.txt?rlkey=22quuq5wzvn6ip0c5ml6ad5cs -O /datasets/msmarco-v1-passage/runs/run.repllama.psg.dl20.txt

python -u format \
--input_path /datasets/msmarco-v1-passage/runs/run.repllama.psg.dl20.txt \
--output_path /datasets/msmarco-v1-passage/runs/dl-20-passage.run-original-repllama-1000.txt \
--ranker_name
```

### Fetch re-ranked lists

#### BM25--RankLLaMA 
```bash

```

#### SPLADE++--RankLLaMA 
```bash

```

#### RepLLaMA--RankLLaMA 
```bash

```

#### BM25--MonoT5 
```bash

```

### Training label generation 

###  Feature generation

## 3. Train and infer RLT methods

## 4. Evaluation
