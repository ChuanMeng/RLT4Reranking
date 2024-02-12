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


Note that for ease of reproducibility, we already uploaded the predicted performance files for all methods reported in our paper. See here.

## 1. Prerequisites
We recommend executing all processes in a Linux environment.
```bash
pip install -r requirements.txt
```

## 1. Data preparation
For ease of reproducibility, 

### Download raw data

#### MS MARCO V1 passage ranking
Download queries and qrels for TREC-DL 19 and 20, as well as the MS MARCO V1 passage ranking collection:
```bash
# queries
mkdir datasets/msmarco-v1-passage/queries 
wget -P ./datasets/msmarco-v1-passage/queries/ https://msmarco.z22.web.core.windows.net/msmarcoranking/msmarco-test2019-queries.tsv.gz
wget -P ./datasets/msmarco-v1-passage/queries/ https://msmarco.z22.web.core.windows.net/msmarcoranking/msmarco-test2020-queries.tsv.gz
gzip -d ./datasets/msmarco-v1-passage/queries/*.tsv.gz
mv ./datasets/msmarco-v1-passage/queries/msmarco-test2019-queries.tsv ./datasets/msmarco-v1-passage/queries/dl-19-passage.queries-original.tsv
mv ./datasets/msmarco-v1-passage/queries/msmarco-test2020-queries.tsv ./datasets/msmarco-v1-passage/queries/dl-20-passage.queries-original.tsv 

# qrels
mkdir datasets/msmarco-v1-passage/qrels
wget -P ./datasets/msmarco-v1-passage/qrels/ ./datasets/msmarco-v1-passage/qrels https://trec.nist.gov/data/deep/2019qrels-pass.txt
wget -P ./datasets/msmarco-v1-passage/qrels/ ./datasets/msmarco-v1-passage/qrels https://trec.nist.gov/data/deep/2020qrels-pass.txt
mv ./datasets/msmarco-v1-passage/qrels/2019qrels-pass.txt ./datasets/msmarco-v1-passage/qrels/dl-19-passage.qrels.txt
mv ./datasets/msmarco-v1-passage/qrels/2020qrels-pass.txt ./datasets/msmarco-v1-passage/qrels/dl-20-passage.qrels.txt

# collection
mkdir datasets/
mkdir datasets/msmarco-v1-passage/ 
mkdir datasets/msmarco-v1-passage/collection
wget -P ./datasets/msmarco-v1-passage/collection/ https://msmarco.z22.web.core.windows.net/msmarcoranking/collection.tar.gz --no-check-certificate
tar -zxvf  ./datasets/msmarco-v1-passage/collection/collection.tar.gz  -C ./datasets/msmarco-v1-passage/collection/
```

#### Robust04
```bash

```

### Obtain retrieved lists
We consider three retrievers: BM25, SPLADE++ ("EnsembleDistil") and RepLLaMA (7B).
We use [Pyserini](https://github.com/castorini/pyserini) to get the retrieved lists returned by BM25 and SPLADE++.
For RepLLaMA, we use the retrieved lists shared by the original author.

#### BM25 
Use the following commands to get BM25 ranking results on TREC-DL 19, TREC-DL 20 and Robust04:
```bash
# TREC-DL 19
python -m pyserini.search.lucene \
  --threads 16 --batch-size 128 \
  --index msmarco-v1-passage-full \
  --topics datasets/msmarco-v1-passage/queries/dl-19-passage.queries-original.tsv \
  --output datasets/msmarco-v1-passage/runs/dl-19-passage.run-original-bm25-1000.txt \
  --bm25 --k1 0.9 --b 0.4 --hits 1000

# TREC-DL 20
python -m pyserini.search.lucene \
  --threads 16 --batch-size 128 \
  --index msmarco-v1-passage-full \
  --topics datasets/msmarco-v1-passage/queries/dl-20-passage.queries-original.tsv \
  --output datasets/msmarco-v1-passage/runs/dl-20-passage.run-original-bm25-1000.txt \
  --bm25 --k1 0.9 --b 0.4 --hits 1000

# Robust04
python -m pyserini.search.lucene \
  --threads 16 --batch-size 128 \
  --index /gpfs/work3/0/guse0654/cache/index/lucene-index.beir-v1.0.0-robust04.flat.20221116.505594 \
  --topics ./datasets/robust04/queries/robust04.queries-original.tsv \
  --output ./datasets/robust04/runs/robust04.run-original-bm25-flat-1000.txt \
  --output-format trec \
  --hits 1000 --bm25 --remove-query
```

#### SPLADE++
Use the following commands to get SPLADE++ ranking results on TREC-DL 19 and 20:
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
  --topics ./datasets/msmarco-v1-passage/queries/dl-20-passage.queries-original.tsv \
  --encoder naver/splade-cocondenser-ensembledistil \
  --output ./datasets/msmarco-v1-passage/runs/dl-20-passage.run-original-splade-pp-ed-pytorch-1000.txt \
  --hits 1000 --impact
```

#### RepLLaMA
Use the following commands to get RepLLaMA ranking results on TREC-DL 19 and 20:
```bash
# TREC-DL 19
wget https://www.dropbox.com/scl/fi/byty1lk2um36imz0788yd/run.repllama.psg.dl19.txt?rlkey=615ootx2mia42cxdilp4tvqzh -O ./datasets/msmarco-v1-passage/runs/run.repllama.psg.dl19.txt

python -u format.py \
--input_path ./datasets/msmarco-v1-passage/runs/run.repllama.psg.dl19.txt \
--output_path ./datasets/msmarco-v1-passage/runs/dl-19-passage.run-original-repllama-1000.txt \
--ranker_name repllama

# TREC-DL 20
wget https://www.dropbox.com/scl/fi/drgg9vj8mxe3qwayggj9o/run.repllama.psg.dl20.txt?rlkey=22quuq5wzvn6ip0c5ml6ad5cs -O ./datasets/msmarco-v1-passage/runs/run.repllama.psg.dl20.txt

python -u format.py \
--input_path ./datasets/msmarco-v1-passage/runs/run.repllama.psg.dl20.txt \
--output_path ./datasets/msmarco-v1-passage/runs/dl-20-passage.run-original-repllama-1000.txt \
--ranker_name repllama
```

### Obtain re-ranked lists
We consider RankLLaMA (7B) and MonoT5 as re-rankers.
We use [Tevatron](https://github.com/texttron/tevatron/tree/main/examples/rankllama) to perform RankLLaMA.
We need the source code of Tevatron, so please first clone it: 

```bash
git clone https://github.com/texttron/tevatron.git
mkdir ./datasets/msmarco-v1-passage/runs/rankllama_input/
```

#### BM25--RankLLaMA 
Use the following commands to use RankLLaMA to re-rank the retrieved list returned by BM25 on TREC-DL 19 and 20:
```bash
# TREC-DL 19
python -u ./tevatron/examples/rankllama/prepare_rerank_file.py \
--query_data_name Tevatron/msmarco-passage \
--query_data_split dl19 \
--corpus_data_name Tevatron/msmarco-passage-corpus \
--retrieval_results ./datasets/msmarco-v1-passage/runs/dl-19-passage.run-original-bm25-1000.txt \
--output_path ./datasets/msmarco-v1-passage/runs/rankllama_input/rerank_input.dl-19-passage.run-original-bm25-1000.jsonl \
--depth 1000

python -u ./tevatron/examples/rankllama/reranker_inference.py \
  --output_dir=temp \
  --model_name_or_path castorini/rankllama-v1-7b-lora-passage \
  --tokenizer_name meta-llama/Llama-2-7b-hf \
  --encode_in_path ./datasets/msmarco-v1-passage/runs/rankllama_input/rerank_input.dl-19-passage.run-original-bm25-1000.jsonl \
  --fp16 \
  --per_device_eval_batch_size 64 \
  --q_max_len 32 \
  --p_max_len 164 \
  --dataset_name json \
  --encoded_save_path ./datasets/msmarco-v1-passage/runs/dl-19-passage.run-original-bm25-1000-rankllama-1000.txt

# TREC-DL 20
python -u ./tevatron/examples/rankllama/prepare_rerank_file.py \
--query_data_name Tevatron/msmarco-passage \
--query_data_split dl20 \
--corpus_data_name Tevatron/msmarco-passage-corpus \
--retrieval_results ./datasets/msmarco-v1-passage/runs/dl-20-passage.run-original-bm25-1000.txt \
--output_path ./datasets/msmarco-v1-passage/runs/rankllama_input/rerank_input.dl-20-passage.run-original-bm25-1000.jsonl \
--depth 1000

python -u ./tevatron/examples/rankllama/reranker_inference.py \
  --output_dir=temp \
  --model_name_or_path castorini/rankllama-v1-7b-lora-passage \
  --tokenizer_name meta-llama/Llama-2-7b-hf \
  --encode_in_path ./datasets/msmarco-v1-passage/runs/rankllama_input/rerank_input.dl-20-passage.run-original-bm25-1000.jsonl \
  --fp16 \
  --per_device_eval_batch_size 64 \
  --q_max_len 32 \
  --p_max_len 164 \
  --dataset_name json \
  --encoded_save_path ./datasets/msmarco-v1-passage/runs/dl-20-passage.run-original-bm25-1000-rankllama-1000.txt
```

#### SPLADE++—-RankLLaMA 
Use the following commands to use RankLLaMA to re-rank the retrieved list returned by SPLADE++ on TREC-DL 19 and 20:
```bash
# TREC-DL 19
python -u ./tevatron/examples/rankllama/prepare_rerank_file.py \
--query_data_name Tevatron/msmarco-passage \
--query_data_split dl19 \
--corpus_data_name Tevatron/msmarco-passage-corpus \
--retrieval_results ./datasets/msmarco-v1-passage/runs/dl-19-passage.run-original-splade-pp-ed-pytorch-1000.txt \
--output_path ./datasets/msmarco-v1-passage/runs/rankllama_input/rerank_input.dl-19-passage.run-original-splade-pp-ed-pytorch-1000.jsonl \
--depth 1000

python -u ./tevatron/examples/rankllama/reranker_inference.py \
  --output_dir=temp \
  --model_name_or_path castorini/rankllama-v1-7b-lora-passage \
  --tokenizer_name meta-llama/Llama-2-7b-hf \
  --encode_in_path ./datasets/msmarco-v1-passage/runs/rankllama_input/rerank_input.dl-19-passage.run-original-splade-pp-ed-pytorch-1000.jsonl \
  --fp16 \
  --per_device_eval_batch_size 64 \
  --q_max_len 32 \
  --p_max_len 164 \
  --dataset_name json \
  --encoded_save_path ./datasets/msmarco-v1-passage/runs/dl-19-passage.run-original-splade-pp-ed-pytorch-1000-rankllama-1000.txt

# TREC-DL 20
python -u ./tevatron/examples/rankllama/prepare_rerank_file.py \
--query_data_name Tevatron/msmarco-passage \
--query_data_split dl20 \
--corpus_data_name Tevatron/msmarco-passage-corpus \
--retrieval_results ./datasets/msmarco-v1-passage/runs/dl-20-passage.run-original-splade-pp-ed-pytorch-1000.txt \
--output_path ./datasets/msmarco-v1-passage/runs/rankllama_input/rerank_input.dl-20-passage.run-original-splade-pp-ed-pytorch-1000.jsonl \
--depth 1000

python ./tevatron/examples/rankllama/reranker_inference.py \
  --output_dir=temp \
  --model_name_or_path castorini/rankllama-v1-7b-lora-passage \
  --tokenizer_name meta-llama/Llama-2-7b-hf \
  --encode_in_path ./datasets/msmarco-v1-passage/runs/rankllama_input/rerank_input.dl-20-passage.run-original-splade-pp-ed-pytorch-1000.jsonl \
  --fp16 \
  --per_device_eval_batch_size 64 \
  --q_max_len 32 \
  --p_max_len 164 \
  --cache_dir /gpfs/work3/0/guse0654/cache/ \
  --dataset_name json \
  --encoded_save_path ./datasets/msmarco-v1-passage/runs/dl-20-passage.run-original-splade-pp-ed-pytorch-1000-rankllama-1000.txt
```

#### RepLLaMA--RankLLaMA 
Use the following commands to use RankLLaMA to re-rank the retrieved list returned by RepLLaMA on TREC-DL 19 and 20:
```bash
# TREC-DL 19
python -u ./tevatron/examples/rankllama/prepare_rerank_file.py \
--query_data_name Tevatron/msmarco-passage \
--query_data_split dl19 \
--corpus_data_name Tevatron/msmarco-passage-corpus \
--retrieval_results ./datasets/msmarco-v1-passage/runs/dl-19-passage.run-original-repllama-1000.txt \
--output_path rerank_input.dl-19-passage.run-original-repllama-1000.jsonl \
--depth 1000

python -u ./tevatron/examples/rankllama/reranker_inference.py \
  --output_dir=temp \
  --model_name_or_path castorini/rankllama-v1-7b-lora-passage \
  --tokenizer_name meta-llama/Llama-2-7b-hf \
  --encode_in_path rerank_input.dl-19-passage.run-original-repllama-1000.jsonl \
  --fp16 \
  --per_device_eval_batch_size 64 \
  --q_max_len 32 \
  --p_max_len 164 \
  --dataset_name json \
  --encoded_save_path dl-19-passage.run-original-repllama-1000-rankllama-1000.txt

# TREC-DL 20
python -u ./tevatron/examples/rankllama/prepare_rerank_file.py \
--query_data_name Tevatron/msmarco-passage \
--query_data_split dl20 \
--corpus_data_name Tevatron/msmarco-passage-corpus \
--retrieval_results ./datasets/msmarco-v1-passage/runs/dl-20-passage.run-original-repllama-1000.txt \
--output_path rerank_input.dl-20-passage.run-original-repllama-1000.jsonl \
--depth 1000

python -u ./tevatron/examples/rankllama/reranker_inference.py \
  --output_dir=temp \
  --model_name_or_path castorini/rankllama-v1-7b-lora-passage \
  --tokenizer_name meta-llama/Llama-2-7b-hf \
  --encode_in_path rerank_input.dl-20-passage.run-original-repllama-1000.jsonl \
  --fp16 \
  --per_device_eval_batch_size 64 \
  --q_max_len 32 \
  --p_max_len 164 \
  --dataset_name json \
  --encoded_save_path dl-20-passage.run-original-repllama-1000-rankllama-1000.txt
```

#### BM25--MonoT5 
We use MonoT5 from [PyGaggle](https://github.com/castorini/pygaggle). 
Please first install [PyGaggle](https://github.com/castorini/pygaggle). Note that PyGaggle requires earlier versions of packages (i.e., Pyserini), so we suggest installing PyGaggle in a separate conda environment.
Note that using MonoT5 to re-rank the retrieved list returned by RepLLaMA and Splade++ yields worse results; hence we only consider the pipeline of BM25--MonoT5.

Use the following commands to use MonoT5 to re-rank BM25 results on TREC-DL 19 and 20:
```bash
# TREC-DL 19
python -u monoT5.py \
--query_path  ./datasets/msmarco-v1-passage/queries/dl-19-passage.queries-original.tsv \
--run_path ./datasets/msmarco-v1-passage/runs/dl-19-passage.run-original-bm25-1000.txt \
--index_path msmarco-v1-passage-full \
--model castorini/monot5-base-msmarco \
--k 1000

# TREC-DL 20
python -u monoT5.py \
--query_path  ./datasets/msmarco-v1-passage/queries/dl-20-passage.queries-original.tsv \
--run_path ./datasets/msmarco-v1-passage/runs/dl-20-passage.run-original-bm25-1000.txt \
--index_path msmarco-v1-passage-full \
--model castorini/monot5-base-msmarco \
--k 1000
```

### Training label generation 
RLT methods (especially supervised ones) need the re-ranking quality in terms of a specific IR evaluation metric across all re-ranking cut-off candidates. 
However, only considering the re-ranking quality would ignore efficiency.
Thus, to quantify different effectiveness/efficiency trade-offs in re-ranking, we use [the efficiency-effectiveness trade-off (EET) metric](https://dl.acm.org/doi/abs/10.1145/1835449.1835475) values to score all re-ranking cut-off candidates; each re-ranking cut-off candidate would have a different score under each effectiveness/efficiency trade-off specified by EET.


#### BM25--RankLLaMA
Use the following commands to generate the training labels on TREC-DL 19 and 20:
```bash
# TREC-DL 19
python  -u rlt/reranking_labels.py \
--retrieval_run_path datasets/msmarco-v1-passage/runs/dl-19-passage.run-original-bm25-1000.txt \
--reranking_run_path datasets/msmarco-v1-passage/runs/dl-19-passage.run-original-bm25-1000-rankllama-1000.txt \
--qrels_path datasets/msmarco-v1-passage/qrels/dl-19-passage.qrels.txt \
--metric ndcg@10 \
--seq_len 1000 \
--output_path datasets/msmarco-v1-passage/labels

# TREC-DL 20
python  -u rlt/reranking_labels.py \
--retrieval_run_path datasets/msmarco-v1-passage/runs/dl-20-passage.run-original-bm25-1000.txt \
--reranking_run_path datasets/msmarco-v1-passage/runs/dl-20-passage.run-original-bm25-1000-rankllama-1000.txt \
--qrels_path datasets/msmarco-v1-passage/qrels/dl-20-passage.qrels.txt \
--metric ndcg@10 \
--seq_len 1000 \
--output_path datasets/msmarco-v1-passage/labels

```

#### SPLADE++--RankLLaMA
Use the following commands to generate the training labels on TREC-DL 19 and 20:
```bash
# TREC-DL 19
python  -u rlt/reranking_labels.py \
--retrieval_run_path datasets/msmarco-v1-passage/runs/dl-19-passage.run-original-splade-pp-ed-pytorch-1000.txt \
--reranking_run_path datasets/msmarco-v1-passage/runs/dl-19-passage.run-original-splade-pp-ed-pytorch-1000-rankllama-1000.txt \
--qrels_path datasets/msmarco-v1-passage/qrels/dl-19-passage.qrels.txt \
--metric ndcg@10 \
--seq_len 1000 \
--output_path datasets/msmarco-v1-passage/labels

# TREC-DL 20
python  -u rlt/reranking_labels.py \
--retrieval_run_path datasets/msmarco-v1-passage/runs/dl-20-passage.run-original-splade-pp-ed-pytorch-1000.txt \
--reranking_run_path datasets/msmarco-v1-passage/runs/dl-20-passage.run-original-splade-pp-ed-pytorch-1000-rankllama-1000.txt \
--qrels_path datasets/msmarco-v1-passage/qrels/dl-20-passage.qrels.txt \
--metric ndcg@10 \
--seq_len 1000 \
--output_path datasets/msmarco-v1-passage/labels
```

#### RepLLaMA--RankLLaMA
Use the following commands to generate the training labels on TREC-DL 19 and 20:
```bash
python  -u rlt/reranking_labels.py \
--retrieval_run_path datasets/msmarco-v1-passage/runs/dl-19-passage.run-original-repllama-1000.txt \
--reranking_run_path datasets/msmarco-v1-passage/runs/dl-19-passage.run-original-repllama-1000-rankllama-1000.txt \
--qrels_path datasets/msmarco-v1-passage/qrels/dl-19-passage.qrels.txt \
--metric ndcg@10 \
--seq_len 1000 \
--output_path datasets/msmarco-v1-passage/labels

python  -u rlt/reranking_labels.py \
--retrieval_run_path datasets/msmarco-v1-passage/runs/dl-20-passage.run-original-repllama-1000.txt \
--reranking_run_path datasets/msmarco-v1-passage/runs/dl-20-passage.run-original-repllama-1000-rankllama-1000.txt \
--qrels_path datasets/msmarco-v1-passage/qrels/dl-20-passage.qrels.txt \
--metric ndcg@10 \
--seq_len 1000 \
--output_path datasets/msmarco-v1-passage/labels
```

#### BM25--MonoT5
Use the following commands to generate the training labels on TREC-DL 19 and 20:
```bash
python  -u rlt/reranking_labels.py \
--retrieval_run_path datasets/msmarco-v1-passage/runs/dl-19-passage.run-original-bm25-1000.txt \
--reranking_run_path datasets/msmarco-v1-passage/runs/dl-19-passage.run-original-bm25-1000-monot5-1000.txt \
--qrels_path datasets/msmarco-v1-passage/qrels/dl-19-passage.qrels.txt \
--metric ndcg@10 \
--seq_len 1000 \
--output_path datasets/msmarco-v1-passage/labels

python  -u rlt/reranking_labels.py \
--retrieval_run_path datasets/msmarco-v1-passage/runs/dl-20-passage.run-original-bm25-1000.txt \
--reranking_run_path datasets/msmarco-v1-passage/runs/dl-20-passage.run-original-bm25-1000-monot5-1000.txt \
--qrels_path datasets/msmarco-v1-passage/qrels/dl-20-passage.qrels.txt \
--metric ndcg@10 \
--seq_len 1000 \
--output_path datasets/msmarco-v1-passage/labels
```

###  Feature generation
We need to first build tf-idf and doc2vec models for collections, and then infer features for retrieved lists.

#### Build a tf-idf model for collections:

Use the following commands to build tf-idf models for MS MARCO V1 passage ranking and Robust04 collections:
```bash
# MS MARCO V1 passage ranking
python -u ./rlt/features.py \
--index_path ./datasets/msmarco-v1-passage/collection/collection.tsv \
--output_path ./datasets/msmarco-v1-passage/features/ \
--mode tfidf 

# Robust04
```

#### Build doc2vec models for collections:
Use the following commands to train doc2vec models for MS MARCO V1 passage ranking and Robust04 collections:
```bash
# MS MARCO V1 passage ranking
python -u ./rlt/features.py \
--index_path ./datasets/msmarco-v1-passage/collection/collection.tsv \
--output_path ./datasets/msmarco-v1-passage/features/ \
--mode doc2vec --vector_size 128 

# Robust04


```
#### Fetch embedding from RepLLaMA


#### Generate features for BM25 ranking results
Use the following commands to generate features for BM25 ranking results on 
```bash
# TREC-DL 19
python -u ./rlt/features.py \
--index_path ./datasets/msmarco-v1-passage/collection/collection.tsv \
--output_path ./datasets/msmarco-v1-passage/features/ \
--run_path ./datasets/msmarco-v1-passage/runs/dl-19-passage.run-original-bm25-1000.txt \
--qrels_path ./datasets/msmarco-v1-passage/qrels/dl-19-passage.qrels.txt \
--mode infer 

# TREC-DL 20
python -u ./rlt/document_features.py \
--index_path ./datasets/msmarco-v1-passage/collection/collection.tsv \
--output_path ./datasets/msmarco-v1-passage/features/ \
--run_path ./datasets/msmarco-v1-passage/runs/dl-20-passage.run-original-bm25-1000.txt \
--qrels_path ./datasets/msmarco-v1-passage/qrels/dl-20-passage.qrels.txt \
--mode infer 
```

#### Generate features for SPLADE++ ranking results

```bash
# TREC-DL 19
python -u ./rlt/features.py \
--index_path ./datasets/msmarco-v1-passage/collection/collection.tsv \
--output_path ./datasets/msmarco-v1-passage/features/ \
--run_path ./datasets/msmarco-v1-passage/runs/dl-19-passage.run-original-splade-pp-ed-pytorch-1000.txt \
--qrels_path ./datasets/msmarco-v1-passage/qrels/dl-19-passage.qrels.txt \
--mode infer 

# TREC-DL 20
python -u ./rlt/features.py \
--index_path ./datasets/msmarco-v1-passage/collection/collection.tsv \
--output_path ./datasets/msmarco-v1-passage/features/ \
--run_path ./datasets/msmarco-v1-passage/runs/dl-20-passage.run-original-splade-pp-ed-pytorch-1000.txt \
--qrels_path ./datasets/msmarco-v1-passage/qrels/dl-20-passage.qrels.txt \
--mode infer
``` 

#### Generate features for RepLLaMA ranking results


```bash
# TREC-DL 19
python -u ./rlt/features.py \
--index_path ./datasets/msmarco-v1-passage/collection/collection.tsv \
--output_path ./datasets/msmarco-v1-passage/features/ \
--run_path ./datasets/msmarco-v1-passage/runs/dl-19-passage.run-original-repllama-1000.txt \
--qrels_path ./datasets/msmarco-v1-passage/qrels/dl-19-passage.qrels.txt \
--mode infer 

# TREC-DL 20
python -u ./rlt/features.py \
--index_path ./datasets/msmarco-v1-passage/collection/collection.tsv \
--output_path ./datasets/msmarco-v1-passage/features/ \
--run_path ./datasets/msmarco-v1-passage/runs/dl-20-passage.run-original-repllama-1000.txt \
--qrels_path ./datasets/msmarco-v1-passage/qrels/dl-20-passage.qrels.txt \
--mode infer 
```


```bash
# TREC-DL 19
python -u ./rlt/embedding.py \
--feature_path ./datasets/msmarco-v1-passage/statistics/dl-19-passage.feature-original-repllama-1000 \
--qrels_path ./datasets/msmarco-v1-passage/qrels/dl-19-passage.qrels.txt \
--output_path ./datasets/msmarco-v1-passage/statistics/ \
--encoder repllama \
--split dl19 \
--query_path Tevatron/msmarco-passage \
--index_path Tevatron/msmarco-passage-corpus \
--fp16 \
--q_max_len=512 \
--p_max_len=512

# TREC-DL 20
python -u ./rlt/embedding.py \
--feature_path ./datasets/msmarco-v1-passage/statistics/dl-20-passage.feature-original-repllama-1000 \
--qrels_path ./datasets/msmarco-v1-passage/qrels/dl-20-passage.qrels.txt \
--output_path ./datasets/msmarco-v1-passage/statistics/ \
--encoder repllama \
--split dl20 \
--query_path Tevatron/msmarco-passage \
--index_path Tevatron/msmarco-passage-corpus \
--fp16 \
--q_max_len=512 \
--p_max_len=512
```


## 3. Train and infer RLT methods

### Unsupervised RLT methods

#### Train and infer Bicut
```bash
retrievers=("original-bm25-1000" "original-splade-pp-ed-pytorch-1000" "original-repllama-1000")
alphas=(0.4 0.5 0.6)

# train a model on dl19, and infer it on dl20
for retriever in "${retrievers[@]}"
do
	for alpha in "${alphas[@]}"
	do 
	# training
	python -u ./rlt/main.py \
	--name bicut \
	--checkpoint_path ./checkpoint_rlt/ \
	--feature_path ./datasets/msmarco-v1-passage/statistics/dl-19-passage.feature-${retriever} \
	--qrels_path ./datasets/msmarco-v1-passage/qrels/dl-19-passage.qrels.txt \
	--epoch_num 100 \
	--alpha ${alpha} \
	--interval 1 \
	--seq_len 1000 \
	--batch_size 64 \
	--binarise_qrels \

	#inference
	python -u ./rlt/main.py \
	--name bicut \
	--checkpoint_path ./checkpoint_rlt/ \
	--feature_path ./datasets/msmarco-v1-passage/statistics/dl-20-passage.feature-${retriever} \
	--qrels_path ./datasets/msmarco-v1-passage/qrels/dl-20-passage.qrels.txt \
	--epoch_num 100 \
	--alpha ${alpha} \
	--interval 1 \
	--seq_len 1000 \
	--batch_size 64 \
	--binarise_qrels \
	--checkpoint_name dl-19-passage.${retriever}.bicut.alpha${alpha} \
	--output_path ./output_rlt \
	--infer
	done
done

# train a model on dl19, and infer it on dl20
for retriever in "${retrievers[@]}"
do
	for alpha in "${alphas[@]}"
	do 
	# training
	python -u ./rlt/main.py \
	--name bicut \
	--checkpoint_path ./checkpoint_rlt/ \
	--feature_path ./datasets/msmarco-v1-passage/statistics/dl-20-passage.feature-${retriever} \
	--qrels_path ./datasets/msmarco-v1-passage/qrels/dl-20-passage.qrels.txt \
	--epoch_num 100 \
	--alpha ${alpha} \
	--interval 1 \
	--seq_len 1000 \
	--batch_size 64 \
	--binarise_qrels \

	# inference
	python -u ./rlt/main.py \
	--name bicut \
	--checkpoint_path ./checkpoint_rlt/ \
	--feature_path ./datasets/msmarco-v1-passage/statistics/dl-19-passage.feature-${retriever} \
	--qrels_path ./datasets/msmarco-v1-passage/qrels/dl-19-passage.qrels.txt \
	--epoch_num 100 \
	--alpha ${alpha} \
	--interval 1 \
	--seq_len 1000 \
	--batch_size 64 \
	--binarise_qrels \
	--checkpoint_name dl-20-passage.${retriever}.bicut.alpha${alpha} \
	--output_path ./output_rlt \
	--infer
	done
done
```

#### Train and infer Choppy, AttnCut and MtCut 
```bash
retrievers=("original-splade-pp-ed-pytorch-1000" "original-repllama-1000" "original-bm25-1000")
metrics=("rankllama-1000-ndcg@10-eet-alpha-0.001-beta0" "rankllama-1000-ndcg@10-eet-alpha-0.001-beta1" "rankllama-1000-ndcg@10-eet-alpha-0.001-beta2")
models=("choppy" "attncut" "mmoecut")

# train a model on dl19, and infer it on dl20
for retriever in "${retrievers[@]}"
do
	for metric in "${metrics[@]}"
	do 

		for model in "${models[@]}"
		do
		# training
		python -u ./rlt/main.py \
		--name ${model} \
		--checkpoint_path ./checkpoint_rlt/ \
		--feature_path ./datasets/msmarco-v1-passage/statistics/dl-19-passage.feature-${retriever} \
		--label_path ./datasets/msmarco-v1-passage/labels/dl-19-passage.label-${retriever}.${metric}.json \
		--qrels_path ./datasets/msmarco-v1-passage/qrels/dl-19-passage.qrels.txt \
		--epoch_num 100 \
		--interval 1 \
		--seq_len 1000 \
		--batch_size 64 \
		--binarise_qrels \

		# inference
		python -u ./rlt/main.py \
		--name ${model} \
		--checkpoint_path ./checkpoint_rlt/ \
		--feature_path ./datasets/msmarco-v1-passage/statistics/dl-20-passage.feature-${retriever} \
		--label_path ./datasets/msmarco-v1-passage/labels/dl-20-passage.label-${retriever}.${metric}.json \
		--qrels_path ./datasets/msmarco-v1-passage/qrels/dl-20-passage.qrels.txt \
		--epoch_num 100 \
		--interval 1 \
		--seq_len 1000 \
		--batch_size 64 \
		--binarise_qrels \
		--checkpoint_name dl-19-passage.${retriever}.${model}.${metric} \
		--output_path ./output_rlt \
		--infer

		done
	done
done

# train a model on dl19, and infer it on dl20
for retriever in "${retrievers[@]}"
do
	for metric in "${metrics[@]}"
	do 
		for model in "${models[@]}"
		do
		# training
		python -u ./rlt/main.py \
		--name ${model} \
		--checkpoint_path ./checkpoint_rlt/ \
		--feature_path ./datasets/msmarco-v1-passage/statistics/dl-20-passage.feature-${retriever} \
		--label_path ./datasets/msmarco-v1-passage/labels/dl-20-passage.label-${retriever}.${metric}.json \
		--qrels_path ./datasets/msmarco-v1-passage/qrels/dl-20-passage.qrels.txt \
		--epoch_num 100 \
		--interval 1 \
		--seq_len 1000 \
		--batch_size 64 \
		--binarise_qrels \
	
		# inference
		python -u ./rlt/main.py \
		--name ${model} \
		--checkpoint_path ./checkpoint_rlt/ \
		--feature_path ./datasets/msmarco-v1-passage/statistics/dl-19-passage.feature-${retriever} \
		--label_path ./datasets/msmarco-v1-passage/labels/dl-19-passage.label-${retriever}.${metric}.json \
		--qrels_path ./datasets/msmarco-v1-passage/qrels/dl-19-passage.qrels.txt \
		--epoch_num 100 \
		--interval 1 \
		--seq_len 1000 \
		--batch_size 64 \
		--binarise_qrels \
		--checkpoint_name dl-20-passage.${retriever}.${model}.${metric} \
		--output_path ./output_rlt \
		--infer
		done
	done
done
```


## 4. Evaluation
