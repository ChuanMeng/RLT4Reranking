# Ranked List Truncation for Re-ranking (RLT4Reranking)
Supplementary materials for the paper titled "_Ranked List Truncation: From Retrieval to Re-ranking_". 
In this paper, we reproduce a comprehensive ranked list truncation (RLT) methods, originally designed for optimizing retrieval, in a "retrieve-then-re-rank" setup; we seek to examine to what extent established findings on RLT for retrieval are generalizable to the ``retrieve-then-re-rank'' setup.

**This repository enables anyone to replicate all numerical results and recreate all visual plots as presented in the paper.**
`plots.ipynb` can recreate all plots in the paper.

This repository is structured into four distinct parts:
1. Prerequisites
2. Data preparation
   * 2.1 Download raw data
   * 2.2 Obtain retrieved lists
   * 2.3 Obtain re-ranked lists
   * 2.4 Feature generation
   * 2.5 Training label generation
3. Reproducing results
   * 3.1 Unsupervised RLT methods
   * 3.2 Supervised RLT methods
   * 3.3 Evaluation
4. Reproducing plots
5. Robust04 results

## 1. Prerequisites
We recommend executing all processes in a Linux environment.
```bash
pip install -r requirements.txt
```

## 2. Data preparation
We conduct experiments on TREC 2019 and 2020 deep learning (TREC-DL) and TREC 2004 Robust (Robust04) tracks. 

### 2.1 Download raw data
All raw data would be stored in the `./datesets` directory.

#### 2.1.1 TREC-DL 19 and 20
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
wget -P ./datasets/msmarco-v1-passage/qrels/ https://trec.nist.gov/data/deep/2019qrels-pass.txt
wget -P ./datasets/msmarco-v1-passage/qrels/ https://trec.nist.gov/data/deep/2020qrels-pass.txt
mv ./datasets/msmarco-v1-passage/qrels/2019qrels-pass.txt ./datasets/msmarco-v1-passage/qrels/dl-19-passage.qrels.txt
mv ./datasets/msmarco-v1-passage/qrels/2020qrels-pass.txt ./datasets/msmarco-v1-passage/qrels/dl-20-passage.qrels.txt

# collection
mkdir datasets/
mkdir datasets/msmarco-v1-passage/ 
mkdir datasets/msmarco-v1-passage/collection
wget -P ./datasets/msmarco-v1-passage/collection/ https://msmarco.z22.web.core.windows.net/msmarcoranking/collection.tar.gz --no-check-certificate
tar -zxvf  ./datasets/msmarco-v1-passage/collection/collection.tar.gz  -C ./datasets/msmarco-v1-passage/collection/
mv ./datasets/msmarco-v1-passage/collection/collection.tsv ./datasets/msmarco-v1-passage/collection/msmarco.tsv  
```

#### 2.1.2 Robust04
We follow `ir_datasets` to fetch Robust04 queries and the collection; please follow finish what the [instruction](https://ir-datasets.com/trec-robust04.html#trec-robust04) requires and before executing the following commands:
```bash
# queries & collection
mkdir datasets/robust04/
mkdir datasets/robust04/collection
mkdir datasets/robust04/queries

python -u process_robust04.py \
--mode download
--query_output_path ./datasets/robust04/queries/robust04.query-title.tsv \
--collection_output_path ./datasets/robust04/collection/robust04.json

# qrels
mkdir datasets/robust04/qrels
wget -P ./datasets/robust04/qrels/ https://trec.nist.gov/data/robust/qrels.robust2004.txt
mv ./datasets/robust04/qrels/qrels.robust2004.txt ./datasets/robust04/qrels/robust04.qrels.txt
```
 
### 2.2 Obtain retrieved lists
We consider three retrievers: BM25, SPLADE++ ("EnsembleDistil") and RepLLaMA (7B).
We use [Pyserini](https://github.com/castorini/pyserini) to get the retrieved lists returned by BM25 and SPLADE++.
For RepLLaMA, we use the retrieved lists shared by the original author.
Note that we rely on publicly available indexes to increase our paper's reproducibility; for Robust04, we only consider BM25 because RepLLaMA and SPLADE++'s indexes are not publicly available at the time of writing.

All retrieved lists would be stored in the directory `datasets/msmarco-v1-passage/runs` or `datasets/robust04/runs`.

#### 2.2.1 BM25 
Use the following commands to get BM25 ranking results on TREC-DL 19, TREC-DL 20 and [Robust04](https://github.com/castorini/pyserini/blob/2154e79a63de0287578d4a3b1239e9a729e1c415/docs/experiments-robust04.md):
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
  --topics ./datasets/robust04/queries/robust04.query-title.tsv \
  --index robust04 \
  --output ./datasets/robust04/runs/robust04.run-title-bm25-1000.txt \
  --hits 1000 --bm25
```

#### 2.2.2 SPLADE++
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

#### 2.2.3 RepLLaMA
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

### 2.3 Obtain re-ranked lists
We consider RankLLaMA (7B) and MonoT5 as re-rankers.
We use [Tevatron](https://github.com/texttron/tevatron/tree/main/examples/rankllama) to perform RankLLaMA.
We already put the source code of [Tevatron](https://github.com/texttron/tevatron/tree/main/examples/rankllama) in the current directory.
So please install [Tevatron](https://github.com/texttron/tevatron/tree/main/examples/rankllama) by its source code:
```bash
cd tevatron
pip install --editable .
cd ..
```

We use MonoT5 from [PyGaggle](https://github.com/castorini/pygaggle). 
Please first install it by following the [PyGaggle documentation](https://github.com/castorini/pygaggle).
Make sure to clone PyGaggle in the current directory:
```
git clone --recursive https://github.com/castorini/pygaggle.git
```
Note that PyGaggle requires earlier versions of packages (i.e., Pyserini), so we suggest installing PyGaggle in a separate conda environment.
Note that using MonoT5 to re-rank the retrieved list returned by RepLLaMA and Splade++ yields worse results; hence we only consider the pipeline of BM25--MonoT5.

All re-ranked lists would be stored in the directory `datasets/msmarco-v1-passage/runs` or `datasets/robust04/runs`.

Note that we recommend using GPU to execute all commands in this section.

#### 2.3.1 BM25--RankLLaMA
Note that Robust04 is a document-based corpus, so we use RankLLaMA's checkpoint ("castorini/rankllama-v1-7b-lora-doc") trained on the MS MARCO v1 document corpus, and set the max length of a document to 2048.
Use the following commands to use RankLLaMA to re-rank the retrieved list returned by BM25 on TREC-DL 19 and 20 and Robust04:
```bash
# TREC-DL 19 
mkdir ./datasets/msmarco-v1-passage/runs/rankllama_input/

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

# Robust04
mkdir ./datasets/robust04/runs/rankllama_input/

python -u ./tevatron/examples/rankllama/prepare_rerank_file.py \
--query_path ./datasets/robust04/queries/robust04.query-title.tsv \
--corpus_path ./datasets/robust04/collection/robust04.json \
--retrieval_results ./datasets/robust04/runs/robust04.run-title-bm25-1000.txt \
--output_path ./datasets/robust04/runs/rankllama_input/rerank_input.robust04.run-title-bm25-1000.jsonl \
--depth 1000

python -u ./tevatron/examples/rankllama/reranker_inference.py \
  --output_dir=temp \
  --model_name_or_path castorini/rankllama-v1-7b-lora-doc \
  --tokenizer_name meta-llama/Llama-2-7b-hf \
  --encode_in_path ./datasets/robust04/runs/rankllama_input/rerank_input.robust04.run-title-bm25-1000.jsonl \
  --fp16 \
  --per_device_eval_batch_size 8 \
  --q_max_len 32 \
  --p_max_len 2048 \
  --dataset_name json \
  --encoded_save_path ./datasets/robust04/runs/robust04.run-title-bm25-1000-rankllama-doc-2048-1000.txt
```

#### 2.3.2 SPLADE++—-RankLLaMA 
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

#### 2.3.3 RepLLaMA--RankLLaMA 
Use the following commands to use RankLLaMA to re-rank the retrieved list returned by RepLLaMA on TREC-DL 19 and 20:
```bash
# TREC-DL 19
python -u ./tevatron/examples/rankllama/prepare_rerank_file.py \
--query_data_name Tevatron/msmarco-passage \
--query_data_split dl19 \
--corpus_data_name Tevatron/msmarco-passage-corpus \
--retrieval_results ./datasets/msmarco-v1-passage/runs/dl-19-passage.run-original-repllama-1000.txt \
--output_path ./datasets/msmarco-v1-passage/runs/rankllama_input/rerank_input.dl-19-passage.run-original-repllama-1000.jsonl \
--depth 1000

python -u ./tevatron/examples/rankllama/reranker_inference.py \
  --output_dir=temp \
  --model_name_or_path castorini/rankllama-v1-7b-lora-passage \
  --tokenizer_name meta-llama/Llama-2-7b-hf \
  --encode_in_path ./datasets/msmarco-v1-passage/runs/rankllama_input/rerank_input.dl-19-passage.run-original-repllama-1000.jsonl \
  --fp16 \
  --per_device_eval_batch_size 64 \
  --q_max_len 32 \
  --p_max_len 164 \
  --dataset_name json \
  --encoded_save_path ./datasets/msmarco-v1-passage/runs/dl-19-passage.run-original-repllama-1000-rankllama-1000.txt

# TREC-DL 20
python -u ./tevatron/examples/rankllama/prepare_rerank_file.py \
--query_data_name Tevatron/msmarco-passage \
--query_data_split dl20 \
--corpus_data_name Tevatron/msmarco-passage-corpus \
--retrieval_results ./datasets/msmarco-v1-passage/runs/dl-20-passage.run-original-repllama-1000.txt \
--output_path ./datasets/msmarco-v1-passage/runs/rankllama_input/rerank_input.dl-20-passage.run-original-repllama-1000.jsonl \
--depth 1000

python -u ./tevatron/examples/rankllama/reranker_inference.py \
  --output_dir=temp \
  --model_name_or_path castorini/rankllama-v1-7b-lora-passage \
  --tokenizer_name meta-llama/Llama-2-7b-hf \
  --encode_in_path ./datasets/msmarco-v1-passage/runs/rankllama_input/rerank_input.dl-20-passage.run-original-repllama-1000.jsonl \
  --fp16 \
  --per_device_eval_batch_size 64 \
  --q_max_len 32 \
  --p_max_len 164 \
  --dataset_name json \
  --encoded_save_path ./datasets/msmarco-v1-passage/runs/dl-20-passage.run-original-repllama-1000-rankllama-1000.txt
```

#### 2.3.4 BM25--MonoT5 

Note that to deal with the long documents in Robust04, MonoT5 uses the [MaxP technique](https://aclanthology.org/2020.findings-emnlp.63/).
Use the following commands to use MonoT5 to re-rank BM25 results on TREC-DL 19 and 20, as well as [Robust04](https://github.com/castorini/pygaggle/blob/master/docs/experiments-robust04-monot5-gpu.md):
```bash
# TREC-DL 19
python -u monot5.py \
--query_path  ./datasets/msmarco-v1-passage/queries/dl-19-passage.queries-original.tsv \
--run_path ./datasets/msmarco-v1-passage/runs/dl-19-passage.run-original-bm25-1000.txt \
--index_path msmarco-v1-passage-full \
--model castorini/monot5-base-msmarco \
--k 1000

# TREC-DL 20
python -u monot5.py \
--query_path  ./datasets/msmarco-v1-passage/queries/dl-20-passage.queries-original.tsv \
--run_path ./datasets/msmarco-v1-passage/runs/dl-20-passage.run-original-bm25-1000.txt \
--index_path msmarco-v1-passage-full \
--model castorini/monot5-base-msmarco \
--k 1000

# Robust04
wget -P ./datasets/robust04/collection/ https://storage.googleapis.com/castorini/robust04/trec_disks_4_and_5_concat.txt --no-check-certificate

python ./pygaggle/pygaggle/run/robust04_reranker_pipeline_gpu.py \
--queries ./datasets/robust04/queries/04.testset \
--run ./datasets/robust04/runs/robust04.run-title-bm25-1000.txt \
--corpus ./datasets/robust04/collection/trec_disks_4_and_5_concat.txt \
--output_monot5 ./datasets/robust04/runs/robust04.run-title-bm25-1000-monot5-1000.txt
```

### 2.4 Training label generation 
RLT methods (especially supervised ones) need the re-ranking quality in terms of a specific IR evaluation metric across all re-ranking cut-off candidates.
However, only considering the re-ranking quality would ignore efficiency.
Thus, to quantify different effectiveness/efficiency trade-offs in re-ranking, we use [the efficiency-effectiveness trade-off (EET) metric](https://dl.acm.org/doi/abs/10.1145/1835449.1835475) values to score all re-ranking cut-off candidates; each re-ranking cut-off candidate would have a different score under each effectiveness/efficiency trade-off specified by EET.

EET has two hypeparamters, i.e., α and β. We consider α=-0.001, and β=0 (only effectiveness), 1 (balance effectiveness and efficiency) and 2 (more efficiency).

For the target IR valuation metric, we use follow [Craswell et al., 2019](https://trec.nist.gov/pubs/trec28/papers/OVERVIEW.DL.pdf) and [Craswell et al., 2020](https://trec.nist.gov/pubs/trec29/papers/OVERVIEW.DL.pdf) to use nDCG@10 on TREC-DL 19 and 20, and follow [Dai et al., 2019](https://dl.acm.org/doi/abs/10.1145/3331184.3331303) to use nDCG@10 on Robust04.

Please first create the folder where label files would be produced.
```bash
mkdir datasets/msmarco-v1-passage/labels
mkdir datasets/robust04/labels
```

#### 2.4.1 BM25--RankLLaMA
Use the following commands to generate the training labels on TREC-DL 19 and 20:
```bash
# TREC-DL 19
python -u rlt/reranking_labels.py \
--retrieval_run_path datasets/msmarco-v1-passage/runs/dl-19-passage.run-original-bm25-1000.txt \
--reranking_run_path datasets/msmarco-v1-passage/runs/dl-19-passage.run-original-bm25-1000-rankllama-1000.txt \
--qrels_path datasets/msmarco-v1-passage/qrels/dl-19-passage.qrels.txt \
--metric ndcg@10 \
--seq_len 1000 \
--output_path datasets/msmarco-v1-passage/labels

# TREC-DL 20 
python -u rlt/reranking_labels.py \
--retrieval_run_path datasets/msmarco-v1-passage/runs/dl-20-passage.run-original-bm25-1000.txt \
--reranking_run_path datasets/msmarco-v1-passage/runs/dl-20-passage.run-original-bm25-1000-rankllama-1000.txt \
--qrels_path datasets/msmarco-v1-passage/qrels/dl-20-passage.qrels.txt \
--metric ndcg@10 \
--seq_len 1000 \
--output_path datasets/msmarco-v1-passage/labels

# Robust04
python -u ./process_robust04.py \
--mode split_run \
--run_path ./datasets/robust04/runs/robust04.run-title-bm25-1000-rankllama-doc-2048-1000.txt

python -u rlt/reranking_labels.py \
--retrieval_run_path datasets/robust04/runs/robust04-fold1.run-title-bm25-1000.txt \
--reranking_run_path datasets/robust04/runs/robust04-fold1.run-title-bm25-1000-rankllama-doc-2048-1000.txt \
--qrels_path datasets/robust04/qrels/robust04.qrels.txt \
--metric ndcg@20 \
--seq_len 1000 \
--output_path datasets/robust04/labels

python -u rlt/reranking_labels.py \
--retrieval_run_path datasets/robust04/runs/robust04-fold2.run-title-bm25-1000.txt \
--reranking_run_path datasets/robust04/runs/robust04-fold2.run-title-bm25-1000-rankllama-doc-2048-1000.txt \
--qrels_path datasets/robust04/qrels/robust04.qrels.txt \
--metric ndcg@20 \
--seq_len 1000 \
--output_path datasetsrobust04/labels

python -u rlt/reranking_labels.py \
--retrieval_run_path datasets/robust04/runs/robust04-fold3.run-title-bm25-1000.txt \
--reranking_run_path datasets/robust04/runs/robust04-fold3.run-title-bm25-1000-rankllama-doc-2048-1000.txt \
--qrels_path datasets/robust04/qrels/robust04.qrels.txt \
--metric ndcg@20 \
--seq_len 1000 \
--output_path datasets/robust04/labels

python -u rlt/reranking_labels.py \
--retrieval_run_path datasets/robust04/runs/robust04-fold4.run-title-bm25-1000.txt \
--reranking_run_path datasets/robust04/runs/robust04-fold4.run-title-bm25-1000-rankllama-doc-2048-1000.txt \
--qrels_path datasets/robust04/qrels/robust04.qrels.txt \
--metric ndcg@20 \
--seq_len 1000 \
--output_path datasets/robust04/labels

python -u rlt/reranking_labels.py \
--retrieval_run_path datasets/robust04/runs/robust04-fold5.run-title-bm25-1000.txt \
--reranking_run_path datasets/robust04/runs/robust04-fold5.run-title-bm25-1000-rankllama-doc-2048-1000.txt \
--qrels_path datasets/robust04/qrels/robust04.qrels.txt \
--metric ndcg@20 \
--seq_len 1000 \
--output_path datasets/robust04/labels
```

#### 2.4.2 SPLADE++--RankLLaMA
Use the following commands to generate the training labels on TREC-DL 19 and 20:
```bash
# TREC-DL 19
python -u rlt/reranking_labels.py \
--retrieval_run_path datasets/msmarco-v1-passage/runs/dl-19-passage.run-original-splade-pp-ed-pytorch-1000.txt \
--reranking_run_path datasets/msmarco-v1-passage/runs/dl-19-passage.run-original-splade-pp-ed-pytorch-1000-rankllama-1000.txt \
--qrels_path datasets/msmarco-v1-passage/qrels/dl-19-passage.qrels.txt \
--metric ndcg@10 \
--seq_len 1000 \
--output_path datasets/msmarco-v1-passage/labels

# TREC-DL 20
python -u rlt/reranking_labels.py \
--retrieval_run_path datasets/msmarco-v1-passage/runs/dl-20-passage.run-original-splade-pp-ed-pytorch-1000.txt \
--reranking_run_path datasets/msmarco-v1-passage/runs/dl-20-passage.run-original-splade-pp-ed-pytorch-1000-rankllama-1000.txt \
--qrels_path datasets/msmarco-v1-passage/qrels/dl-20-passage.qrels.txt \
--metric ndcg@10 \
--seq_len 1000 \
--output_path datasets/msmarco-v1-passage/labels
```

#### 2.4.3 RepLLaMA--RankLLaMA
Use the following commands to generate the training labels on TREC-DL 19 and 20:
```bash
# TREC-DL 19
python -u rlt/reranking_labels.py \
--retrieval_run_path datasets/msmarco-v1-passage/runs/dl-19-passage.run-original-repllama-1000.txt \
--reranking_run_path datasets/msmarco-v1-passage/runs/dl-19-passage.run-original-repllama-1000-rankllama-1000.txt \
--qrels_path datasets/msmarco-v1-passage/qrels/dl-19-passage.qrels.txt \
--metric ndcg@10 \
--seq_len 1000 \
--output_path datasets/msmarco-v1-passage/labels

# TREC-DL 20
python -u rlt/reranking_labels.py \
--retrieval_run_path datasets/msmarco-v1-passage/runs/dl-20-passage.run-original-repllama-1000.txt \
--reranking_run_path datasets/msmarco-v1-passage/runs/dl-20-passage.run-original-repllama-1000-rankllama-1000.txt \
--qrels_path datasets/msmarco-v1-passage/qrels/dl-20-passage.qrels.txt \
--metric ndcg@10 \
--seq_len 1000 \
--output_path datasets/msmarco-v1-passage/labels
```

#### 2.4.4 BM25--MonoT5
Use the following commands to generate the training labels on TREC-DL 19 and 20:
```bash
# TREC-DL 19
python -u rlt/reranking_labels.py \
--retrieval_run_path datasets/msmarco-v1-passage/runs/dl-19-passage.run-original-bm25-1000.txt \
--reranking_run_path datasets/msmarco-v1-passage/runs/dl-19-passage.run-original-bm25-1000-monot5-1000.txt \
--qrels_path datasets/msmarco-v1-passage/qrels/dl-19-passage.qrels.txt \
--metric ndcg@10 \
--seq_len 1000 \
--output_path datasets/msmarco-v1-passage/labels

# TREC-DL 20
python -u rlt/reranking_labels.py \
--retrieval_run_path datasets/msmarco-v1-passage/runs/dl-20-passage.run-original-bm25-1000.txt \
--reranking_run_path datasets/msmarco-v1-passage/runs/dl-20-passage.run-original-bm25-1000-monot5-1000.txt \
--qrels_path datasets/msmarco-v1-passage/qrels/dl-20-passage.qrels.txt \
--metric ndcg@10 \
--seq_len 1000 \
--output_path datasets/msmarco-v1-passage/labels

# Robust04
python -u ./process_robust04.py \
--mode split_run \
--run_path ./datasets/robust04/runs/robust04.run-title-bm25-1000-monot5-1000.txt

python -u rlt/reranking_labels.py \
--retrieval_run_path datasets/robust04/runs/robust04-fold1.run-title-bm25-1000.txt \
--reranking_run_path datasets/robust04/runs/robust04-fold1.run-title-bm25-1000-monot5-1000.txt \
--qrels_path datasets/robust04/qrels/robust04.qrels.txt \
--metric ndcg@20 \
--seq_len 1000 \
--output_path datasets/robust04/labels

python -u rlt/reranking_labels.py \
--retrieval_run_path datasets/robust04/runs/robust04-fold2.run-title-bm25-1000.txt \
--reranking_run_path datasets/robust04/runs/robust04-fold2.run-title-bm25-1000-monot5-1000.txt \
--qrels_path datasets/robust04/qrels/robust04.qrels.txt \
--metric ndcg@20 \
--seq_len 1000 \
--output_path datasets/robust04/labels

python -u rlt/reranking_labels.py \
--retrieval_run_path datasets/robust04/runs/robust04-fold3.run-title-bm25-1000.txt \
--reranking_run_path datasets/robust04/runs/robust04-fold3.run-title-bm25-1000-monot5-1000.txt \
--qrels_path datasets/robust04/qrels/robust04.qrels.txt \
--metric ndcg@20 \
--seq_len 1000 \
--output_path datasets/robust04/labels

python -u rlt/reranking_labels.py \
--retrieval_run_path datasets/robust04/runs/robust04-fold4.run-title-bm25-1000.txt \
--reranking_run_path datasets/robust04/runs/robust04-fold4.run-title-bm25-1000-monot5-1000.txt \
--qrels_path datasets/robust04/qrels/robust04.qrels.txt \
--metric ndcg@20 \
--seq_len 1000 \
--output_path datasets/robust04/labels

python -u rlt/reranking_labels.py \
--retrieval_run_path datasets/robust04/runs/robust04-fold5.run-title-bm25-1000.txt \
--reranking_run_path datasets/robust04/runs/robust04-fold5.run-title-bm25-1000-monot5-1000.txt \
--qrels_path datasets/robust04/qrels/robust04.qrels.txt \
--metric ndcg@20 \
--seq_len 1000 \
--output_path datasets/robust04/labels
```

### 2.5 Feature generation
We need first to build tf-idf and doc2vec models for collections, and then to infer features for retrieved lists.

Please first create the folder where feature files would be produced.
```bash
mkdir datasets/msmarco-v1-passage/features
mkdir datasets/robust04/features
```

#### 2.5.1 Build tf-idf models for collections:

Use the following commands to build tf-idf models for MS MARCO V1 passage ranking and Robust04 collections:
```bash
# MS MARCO V1 passage ranking
python -u ./rlt/features.py \
--index_path ./datasets/msmarco-v1-passage/collection/msmarco.tsv \
--output_path ./datasets/msmarco-v1-passage/features/ \
--mode tfidf 

# Robust04
python -u ./rlt/features.py \
--index_path ./datasets/robust04/collection/robust04.json \
--output_path ./datasets/robust04/features/ \
--mode tfidf 
```

#### 2.5.2 Build doc2vec models for collections:
Use the following commands to train doc2vec models for MS MARCO V1 passage ranking and Robust04 collections:
```bash
# MS MARCO V1 passage ranking
python -u ./rlt/features.py \
--index_path ./datasets/msmarco-v1-passage/collection/msmarco.tsv \
--output_path ./datasets/msmarco-v1-passage/features/ \
--mode doc2vec --vector_size 128 

# Robust04
python -u ./rlt/features.py \
--index_path ./datasets/robust04/collection/robust04.json \
--output_path ./datasets/robust04/features/ \
--mode doc2vec --vector_size 128 
```
#### 2.5.3 Generate features for BM25 ranking results
Use the following commands to generate features for BM25 ranking results on TREC-DL 19 and 20, as well as Robust04:
```bash
# TREC-DL 19
python -u ./rlt/features.py \
--output_path ./datasets/msmarco-v1-passage/features/ \
--run_path ./datasets/msmarco-v1-passage/runs/dl-19-passage.run-original-bm25-1000.txt \
--qrels_path ./datasets/msmarco-v1-passage/qrels/dl-19-passage.qrels.txt \
--seq_len 1000 \
--mode infer 

# TREC-DL 20
python -u ./rlt/document_features.py \
--output_path ./datasets/msmarco-v1-passage/features/ \
--run_path ./datasets/msmarco-v1-passage/runs/dl-20-passage.run-original-bm25-1000.txt \
--qrels_path ./datasets/msmarco-v1-passage/qrels/dl-20-passage.qrels.txt \
--seq_len 1000 \
--mode infer 

# Robust04
python -u ./process_robust04.py \
--mode split_run \
--run_path ./datasets/robust04/runs/robust04.run-title-bm25-1000.txt


fold_ids=("1" "2" "3" "4" "5")
for fold_id in "${fold_ids[@]}"
do
	python -u ./rlt/features.py \
	--output_path ./datasets/robust04/features/ \
	--run_path ./datasets/robust04/runs/robust04-fold${fold_id}.run-title-bm25-1000.txt \
	--qrels_path ./datasets/robust04/qrels/robust04.qrels.txt \
	--seq_len 1000 \
	--mode infer 
done

python -u ./process_robust04.py \
--mode merge \
--fold_one_path ./datasets/robust04/features/robust04-fold1.feature-title-bm25-1000.json


python -u ./process_robust04.py \
--mode merge \
--fold_one_path ./datasets/robust04/labels/robust04-fold1.label-title-bm25-1000.monot5-1000-ndcg@20.json

python -u ./process_robust04.py \
--mode merge \
--fold_one_path ./datasets/robust04/labels/robust04-fold1.label-title-bm25-1000.monot5-1000-ndcg@20-eet-alpha-0.001-beta0.json

python -u ./process_robust04.py \
--mode merge \
--fold_one_path ./datasets/robust04/labels/robust04-fold1.label-title-bm25-1000.monot5-1000-ndcg@20-eet-alpha-0.001-beta1.json

python -u ./process_robust04.py \
--mode merge \
--fold_one_path ./datasets/robust04/labels/robust04-fold1.label-title-bm25-1000.monot5-1000-ndcg@20-eet-alpha-0.001-beta2.json


python -u ./process_robust04.py \
--mode merge \
--fold_one_path ./datasets/robust04/labels/robust04-fold1.label-title-bm25-1000.rankllama-doc-2048-1000-ndcg@20.json

python -u ./process_robust04.py \
--mode merge \
--fold_one_path ./datasets/robust04/labels/robust04-fold1.label-title-bm25-1000.rankllama-doc-2048-1000-ndcg@20-eet-alpha-0.001-beta0.json

python -u ./process_robust04.py \
--mode merge \
--fold_one_path ./datasets/robust04/labels/robust04-fold1.label-title-bm25-1000.rankllama-doc-2048-1000-ndcg@20-eet-alpha-0.001-beta1.json

python -u ./process_robust04.py \
--mode merge \
--fold_one_path ./datasets/robust04/labels/robust04-fold1.label-title-bm25-1000.rankllama-doc-2048-1000-ndcg@20-eet-alpha-0.001-beta2.json

```

#### 2.5.4 Generate features for SPLADE++ ranking results
Use the following commands to generate features for SPLADE++ ranking results on TREC-DL 19 and 20:
```bash
# TREC-DL 19
python -u ./rlt/features.py \
--output_path ./datasets/msmarco-v1-passage/features/ \
--run_path ./datasets/msmarco-v1-passage/runs/dl-19-passage.run-original-splade-pp-ed-pytorch-1000.txt \
--qrels_path ./datasets/msmarco-v1-passage/qrels/dl-19-passage.qrels.txt \
--seq_len 1000 \
--mode infer 

# TREC-DL 20
python -u ./rlt/features.py \
--output_path ./datasets/msmarco-v1-passage/features/ \
--run_path ./datasets/msmarco-v1-passage/runs/dl-20-passage.run-original-splade-pp-ed-pytorch-1000.txt \
--qrels_path ./datasets/msmarco-v1-passage/qrels/dl-20-passage.qrels.txt \
--seq_len 1000 \
--mode infer
``` 

#### 2.5.5 Generate features for RepLLaMA ranking results
Use the following commands to generate features for RepLLaMA ranking results on TREC-DL 19 and 20:
```bash
# TREC-DL 19
python -u ./rlt/features.py \
--output_path ./datasets/msmarco-v1-passage/features/ \
--run_path ./datasets/msmarco-v1-passage/runs/dl-19-passage.run-original-repllama-1000.txt \
--qrels_path ./datasets/msmarco-v1-passage/qrels/dl-19-passage.qrels.txt \
--seq_len 1000 \
--mode infer 

# TREC-DL 20
python -u ./rlt/features.py \
--output_path ./datasets/msmarco-v1-passage/features/ \
--run_path ./datasets/msmarco-v1-passage/runs/dl-20-passage.run-original-repllama-1000.txt \
--qrels_path ./datasets/msmarco-v1-passage/qrels/dl-20-passage.qrels.txt \
--seq_len 1000 \
--mode infer 
```
Note that the supervised RLT method LeCut needs to be fed with the query-item embeddings from the given neural retriever.
We need to fetch embeddings from RepLLaMA and merge the embeddings with the features generated in the above step.
We recommend using GPU to execute the following commands:
```bash
# TREC-DL 19
python -u ./rlt/embedding.py \
--feature_path ./datasets/msmarco-v1-passage/features/dl-19-passage.feature-original-repllama-1000 \
--qrels_path ./datasets/msmarco-v1-passage/qrels/dl-19-passage.qrels.txt \
--output_path ./datasets/msmarco-v1-passage/features/ \
--encoder repllama \
--split dl19 \
--query_path Tevatron/msmarco-passage \
--index_path Tevatron/msmarco-passage-corpus \
--fp16 \
--q_max_len=512 \
--p_max_len=512

# TREC-DL 20
python -u ./rlt/embedding.py \
--feature_path ./datasets/msmarco-v1-passage/features/dl-20-passage.feature-original-repllama-1000 \
--qrels_path ./datasets/msmarco-v1-passage/qrels/dl-20-passage.qrels.txt \
--output_path ./datasets/msmarco-v1-passage/features/ \
--encoder repllama \
--split dl20 \
--query_path Tevatron/msmarco-passage \
--index_path Tevatron/msmarco-passage-corpus \
--fp16 \
--q_max_len=512 \
--p_max_len=512
```

## 3. Reproducing results

All checkpoints would be stored in the `./checkpoint` directory.
Inference outputs would be stored in the `./output/{dataset name}.{retriever name}` directory; an output file; each file has the same number of lines as queries in the test set; each line is composed of "query id\tpredicted cut-off".

###  3.1 Unsupervised RLT methods
We consider 3 unsupervised methods, i.e., Fixed-k, Greedy-k, [Suprise](https://dl.acm.org/doi/abs/10.1145/3539618.3592066).
We also consider Oracle here.

#### 3.1.1 Fixed-k

Run the following commands to perform Fixed-k on TREC-DL 19 and 20:
```bash
retrievers=("original-bm25-1000" "original-splade-pp-ed-pytorch-1000" "original-repllama-1000")

# TREC-DL 19
for retriever in "${retrievers[@]}"
do
python -u ./rlt/unsupervised_rlt.py \
--name fixed \
--feature_path ./datasets/msmarco-v1-passage/features/dl-19-passage.feature-${retriever}.json \
--output_path ./output
done

# TREC-DL 20
for retriever in "${retrievers[@]}"
do
python -u ./rlt/unsupervised_rlt.py \
--name fixed \
--feature_path ./datasets/msmarco-v1-passage/features/dl-20-passage.feature-${retriever}.json \
--output_path ./output
done

# Robust04
retrievers=("title-bm25-1000")
folds=("1" "2" "3" "4" "5")
for retriever in "${retrievers[@]}"
do
	for fold in "${folds[@]}"
	do
	python -u ./rlt/unsupervised_rlt.py \
	--name fixed \
	--feature_path ./datasets/robust04/features/robust04-fold${fold}.feature-${retriever}.json \
	--output_path ./output
	done
done
```

#### 3.1.2 Greedy-k
Run the following commands to perform Greedy-k on TREC-DL 19 and 20:
```bash
retrievers=("original-bm25-1000" "original-splade-pp-ed-pytorch-1000" "original-repllama-1000")
metrics=("rankllama-1000-ndcg@10-eet-alpha-0.001-beta0" "rankllama-1000-ndcg@10-eet-alpha-0.001-beta1" "rankllama-1000-ndcg@10-eet-alpha-0.001-beta2" "monot5-1000-ndcg@10-eet-alpha-0.001-beta0" "monot5-1000-ndcg@10-eet-alpha-0.001-beta1" "monot5-1000-ndcg@10-eet-alpha-0.001-beta2")

# TREC-DL 19
for retriever in "${retrievers[@]}"
do
	for metric in "${metrics[@]}"
	do
	python -u ./rlt/unsupervised_rlt.py \
	--name greedy \
	--feature_path ./datasets/msmarco-v1-passage/features/dl-19-passage.feature-${retriever}.json \
	--train_labels_path ./datasets/msmarco-v1-passage/labels/dl-20-passage.label-${retriever}.${metric}.json \
	--output_path ./output
	done	
done

# TREC-DL 20
for retriever in "${retrievers[@]}"
do
	for metric in "${metrics[@]}"
	do
	python -u ./rlt/unsupervised_rlt.py \
	--name greedy \
	--feature_path ./datasets/msmarco-v1-passage/features/dl-20-passage.feature-${retriever}.json \
	--train_labels_path ./datasets/msmarco-v1-passage/labels/dl-19-passage.label-${retriever}.${metric}.json \
	--output_path ./output
	done	
done

# Robust04
retrievers=("title-bm25-1000")
metrics=("monot5-1000-ndcg@20" "monot5-1000-ndcg@20-eet-alpha-0.001-beta0" "monot5-1000-ndcg@20-eet-alpha-0.001-beta1" "monot5-1000-ndcg@20-eet-alpha-0.001-beta2" "rankllama-doc-2048-1000-ndcg@20" "rankllama-doc-2048-1000-ndcg@20-eet-alpha-0.001-beta0" "rankllama-doc-2048-1000-ndcg@20-eet-alpha-0.001-beta1" "rankllama-doc-2048-1000-ndcg@20-eet-alpha-0.001-beta2")
folds_training=("2345" "1345" "1245" "1235" "1234")
folds_inference=("1" "2" "3" "4" "5")

# train a model
for retriever in "${retrievers[@]}"
do
	for metric in "${metrics[@]}"
	do
		for ((i=0; i<${#folds_training[@]}; i++));
		do
		python -u ./rlt/unsupervised_rlt.py \
		--name greedy \
		--feature_path ./datasets/robust04/features/robust04-fold${folds_inference[$i]}.feature-${retriever}.json \
		--train_labels_path ./datasets/robust04/labels/robust04-fold${folds_training[$i]}.label-${retriever}.${metric}.json \
		--output_path ./output
		done
	done
done
```

#### 3.1.3 Surprise
Note that Surprise only depends on retrieval scores and uses a score threshold to truncate a ranked list; Suprise cannot be tuned for re-rankers because the score threshold is set based on Cramer-von-Mises statistic testings and the threshold is not a tunable hyperparameter.
Run the following commands to perform Surprise on TREC-DL 19 and 20:
```bash
retrievers=("original-bm25-1000" "original-splade-pp-ed-pytorch-1000" "original-repllama-1000")

# TREC-DL 19
for retriever in "${retrievers[@]}"
do
python -u ./rlt/unsupervised_rlt.py \
--name surprise \
--feature_path ./datasets/msmarco-v1-passage/features/dl-19-passage.feature-${retriever}.json \
--output_path ./output
done

# TREC-DL 20
for retriever in "${retrievers[@]}"
do
python -u ./rlt/unsupervised_rlt.py \
--name surprise \
--feature_path ./datasets/msmarco-v1-passage/features/dl-20-passage.feature-${retriever}.json \
--output_path ./output
done

# Robust04
retrievers=("title-bm25-1000")
folds=("1" "2" "3" "4" "5")
for retriever in "${retrievers[@]}"
do
	for fold in "${folds[@]}"
	do
	python -u ./rlt/unsupervised_rlt.py \
	--name surprise \
	--feature_path ./datasets/robust04/features/robust04-fold${fold}.feature-${retriever}.json \
	--output_path ./output
	done
done
```

#### 3.1.4 Oracle
Run the following commands to perform Oracle on TREC-DL 19 and 20:
```bash
retrievers=("original-bm25-1000" "original-splade-pp-ed-pytorch-1000" "original-repllama-1000")
metrics=("rankllama-1000-ndcg@10" "monot5-1000-ndcg@10")

# TREC-DL 19
for retriever in "${retrievers[@]}"
do
	for metric in "${metrics[@]}"
	do
	python -u ./unsupervised_rlt.py \
	--name oracle \
	--test_labels_path ./datasets/msmarco-v1-passage/labels/dl-19-passage.label-${retriever}.${metric}.json \
	--output_path ./output
	done
done

# TREC-DL 20
for retriever in "${retrievers[@]}"
do
	for metric in "${metrics[@]}"
	do
	python -u ./unsupervised_rlt.py \
	--name oracle \
	--test_labels_path ./datasets/msmarco-v1-passage/labels/dl-20-passage.label-${retriever}.${metric}.json \
	--output_path ./output
	done
done

# Robust04
retrievers=("title-bm25-1000")
metrics=("monot5-1000-ndcg@20" "rankllama-doc-2048-1000-ndcg@20")
folds=("1" "2" "3" "4" "5")

for retriever in "${retrievers[@]}"
do
	for metric in "${metrics[@]}"
	do
		for fold in "${folds[@]}"
		do
		python -u ./rlt/unsupervised_rlt.py \
		--name oracle \
		--test_labels_path ./datasets/robust04/labels/robust04-fold${fold}.label-${retriever}.${metric}.json \
		--output_path ./output
		done
	done
done
```
 
### 3.2. Supervised RLT methods
We consider 5 supervised methods, i.e., [BiCut](https://dl.acm.org/doi/abs/10.1145/3341981.3344234), [Choppy](https://dl.acm.org/doi/10.1145/3397271.3401188), [AttnCut](https://ojs.aaai.org/index.php/AAAI/article/view/16572), [MtCut](https://dl.acm.org/doi/abs/10.1145/3488560.3498466) and [LeCut](https://dl.acm.org/doi/abs/10.1145/3477495.3531998).

We recommend using GPU to execute all commands in this section.

### 3.2.1 Train and infer BiCut
Note that the training of BiCut is independent of re-ranking. 
As shown in our paper, BiCut uses a hyperparameter "η" to control trade-offs between effectiveness and efficiency.
Run the following commands to train BiCut on TREC-DL 19 (TREC-DL 20) and then infer it on TREC-DL 20 (TREC-DL 19):
```bash
retrievers=("original-bm25-1000" "original-splade-pp-ed-pytorch-1000" "original-repllama-1000")
alphas=(0.4 0.5 0.6) # the symbol "alpha" used here corresponds to "η" as denoted in the paper.

# train a model on TREC-DL 19, and infer it on TREC-DL 20
for retriever in "${retrievers[@]}"
do
	for alpha in "${alphas[@]}"
	do 
	# training
	python -u ./rlt/supervised_rlt/main.py \
	--name bicut \
	--checkpoint_path ./checkpoint/ \
	--feature_path ./datasets/msmarco-v1-passage/features/dl-19-passage.feature-${retriever}.json \
	--qrels_path ./datasets/msmarco-v1-passage/qrels/dl-19-passage.qrels.txt \
	--epoch_num 100 \
	--alpha ${alpha} \
	--interval 1 \
	--seq_len 1000 \
	--batch_size 64 \
	--binarise_qrels \

	#inference
	python -u ./rlt/supervised_rlt/main.py \
	--name bicut \
	--checkpoint_path ./checkpoint/ \
	--feature_path ./datasets/msmarco-v1-passage/features/dl-20-passage.feature-${retriever}.json \
	--qrels_path ./datasets/msmarco-v1-passage/qrels/dl-20-passage.qrels.txt \
	--epoch_num 100 \
	--alpha ${alpha} \
	--interval 1 \
	--seq_len 1000 \
	--batch_size 64 \
	--binarise_qrels \
	--checkpoint_name dl-19-passage.${retriever}.bicut.alpha${alpha} \
	--output_path ./output \
	--infer
	done
done

# train a model on TREC-DL 20, and infer it on TREC-DL 19
for retriever in "${retrievers[@]}"
do
	for alpha in "${alphas[@]}"
	do 
	# training
	python -u ./rlt/supervised_rlt/main.py \
	--name bicut \
	--checkpoint_path ./checkpoint/ \
	--feature_path ./datasets/msmarco-v1-passage/features/dl-20-passage.feature-${retriever}.json \
	--qrels_path ./datasets/msmarco-v1-passage/qrels/dl-20-passage.qrels.txt \
	--epoch_num 100 \
	--alpha ${alpha} \
	--interval 1 \
	--seq_len 1000 \
	--batch_size 64 \
	--binarise_qrels \

	# inference
	python -u ./rlt/supervised_rlt/main.py \
	--name bicut \
	--checkpoint_path ./checkpoint/ \
	--feature_path ./datasets/msmarco-v1-passage/features/dl-19-passage.feature-${retriever}.json \
	--qrels_path ./datasets/msmarco-v1-passage/qrels/dl-19-passage.qrels.txt \
	--epoch_num 100 \
	--alpha ${alpha} \
	--interval 1 \
	--seq_len 1000 \
	--batch_size 64 \
	--binarise_qrels \
	--checkpoint_name dl-20-passage.${retriever}.bicut.alpha${alpha} \
	--output_path ./output \
	--infer
	done
done

# Robust04
retrievers=("title-bm25-1000")
alphas=(0.4 0.5 0.6) # the symbol "alpha" used here corresponds to "η" as denoted in the paper.
folds_training=("2345" "1345" "1245" "1235" "1234")
folds_inference=("1" "2" "3" "4" "5")

# train a model
for retriever in "${retrievers[@]}"
do
	for alpha in "${alphas[@]}"
	do
		for ((i=0; i<${#folds_training[@]}; i++));
		do
		# training
		python -u ./rlt/supervised_rlt/main.py \
		--name bicut \
		--checkpoint_path ./checkpoint/ \
		--feature_path ./datasets/robust04/features/robust04-fold${folds_training[$i]}.feature-${retriever}.json \
		--qrels_path ./datasets/robust04/qrels/robust04.qrels.txt \
		--epoch_num 100 \
		--alpha ${alpha} \
		--interval 1 \
		--seq_len 1000 \
		--batch_size 64 \

		#inference
		python -u ./rlt/supervised_rlt/main.py \
		--name bicut \
		--checkpoint_path ./checkpoint/ \
		--feature_path ./datasets/robust04/features/robust04-fold${folds_inference[$i]}.feature-${retriever}.json \
		--qrels_path ./datasets/robust04/qrels/robust04.qrels.txt \
		--epoch_num 100 \
		--alpha ${alpha} \
		--interval 1 \
		--seq_len 1000 \
		--batch_size 64 \
		--checkpoint_name robust04-fold${folds_training[$i]}.${retriever}.bicut.alpha${alpha} \
		--output_path ./output \
		--infer
		done
	done
done
```

#### 3.2.2 Train and infer Choppy, AttnCut and MtCut 
Run the following commands to train Choppy, AttnCut and MtCut on TREC-DL 19 (TREC-DL 20) and then infer it on TREC-DL 20 (TREC-DL 19):
```bash
retrievers=("original-bm25-1000" "original-splade-pp-ed-pytorch-1000" "original-repllama-1000" )
metrics=("rankllama-1000-ndcg@10-eet-alpha-0.001-beta0" "rankllama-1000-ndcg@10-eet-alpha-0.001-beta1" "rankllama-1000-ndcg@10-eet-alpha-0.001-beta2" "monot5-1000-ndcg@10-eet-alpha-0.001-beta0" "monot5-1000-ndcg@10-eet-alpha-0.001-beta1" "monot5-1000-ndcg@10-eet-alpha-0.001-beta2")
models=("choppy" "attncut" "mmoecut")

# train a model on TREC-DL 19, and infer it on TREC-DL 20
for retriever in "${retrievers[@]}"
do
	for metric in "${metrics[@]}"
	do
		for model in "${models[@]}"
		do
		# training
		python -u ./rlt/supervised_rlt/main.py \
		--name ${model} \
		--checkpoint_path ./checkpoint/ \
		--feature_path ./datasets/msmarco-v1-passage/features/dl-19-passage.feature-${retriever}.json \
		--label_path ./datasets/msmarco-v1-passage/labels/dl-19-passage.label-${retriever}.${metric}.json \
		--qrels_path ./datasets/msmarco-v1-passage/qrels/dl-19-passage.qrels.txt \
		--epoch_num 100 \
		--interval 1 \
		--seq_len 1000 \
		--batch_size 64 \
		--binarise_qrels \

		# inference
		python -u ./rlt/supervised_rlt/main.py \
		--name ${model} \
		--checkpoint_path ./checkpoint/ \
		--feature_path ./datasets/msmarco-v1-passage/features/dl-20-passage.feature-${retriever}.json \
		--label_path ./datasets/msmarco-v1-passage/labels/dl-20-passage.label-${retriever}.${metric}.json \
		--qrels_path ./datasets/msmarco-v1-passage/qrels/dl-20-passage.qrels.txt \
		--epoch_num 100 \
		--interval 1 \
		--seq_len 1000 \
		--batch_size 64 \
		--binarise_qrels \
		--checkpoint_name dl-19-passage.${retriever}.${model}.${metric} \
		--output_path ./output \
		--infer

		done
	done
done

# train a model on TREC-DL 20, and infer it on TREC-DL 19
for retriever in "${retrievers[@]}"
do
	for metric in "${metrics[@]}"
	do 
		for model in "${models[@]}"
		do
		# training
		python -u ./rlt/supervised_rlt/main.py \
		--name ${model} \
		--checkpoint_path ./checkpoint/ \
		--feature_path ./datasets/msmarco-v1-passage/features/dl-20-passage.feature-${retriever}.json \
		--label_path ./datasets/msmarco-v1-passage/labels/dl-20-passage.label-${retriever}.${metric}.json \
		--qrels_path ./datasets/msmarco-v1-passage/qrels/dl-20-passage.qrels.txt \
		--epoch_num 100 \
		--interval 1 \
		--seq_len 1000 \
		--batch_size 64 \
		--binarise_qrels \
	
		# inference
		python -u ./rlt/supervised_rlt/main.py \
		--name ${model} \
		--checkpoint_path ./checkpoint/ \
		--feature_path ./datasets/msmarco-v1-passage/features/dl-19-passage.feature-${retriever}.json \
		--label_path ./datasets/msmarco-v1-passage/labels/dl-19-passage.label-${retriever}.${metric}.json \
		--qrels_path ./datasets/msmarco-v1-passage/qrels/dl-19-passage.qrels.txt \
		--epoch_num 100 \
		--interval 1 \
		--seq_len 1000 \
		--batch_size 64 \
		--binarise_qrels \
		--checkpoint_name dl-20-passage.${retriever}.${model}.${metric} \
		--output_path ./output \
		--infer
		done
	done
done

# Robust04
retrievers=("title-bm25-1000")
metrics=("monot5-1000-ndcg@20-eet-alpha-0.001-beta0" "monot5-1000-ndcg@20-eet-alpha-0.001-beta1" "monot5-1000-ndcg@20-eet-alpha-0.001-beta2" "rankllama-doc-2048-1000-ndcg@20-eet-alpha-0.001-beta0" "rankllama-doc-2048-1000-ndcg@20-eet-alpha-0.001-beta1" "rankllama-doc-2048-1000-ndcg@20-eet-alpha-0.001-beta2")
models=("choppy" "attncut" "mmoecut")
folds_training=("2345" "1345" "1245" "1235" "1234")
folds_inference=("1" "2" "3" "4" "5")

for retriever in "${retrievers[@]}"
do
	for metric in "${metrics[@]}"
	do 
		for model in "${models[@]}"
		do
			for ((i=0; i<${#folds_training[@]}; i++));
			do
			# training
			python -u ./rlt/supervised_rlt/main.py \
			--name ${model} \
			--checkpoint_path ./checkpoint/ \
			--feature_path ./datasets/robust04/features/robust04-fold${folds_training[$i]}.feature-${retriever}.json \
			--label_path ./datasets/robust04/labels/robust04-fold${folds_training[$i]}.label-${retriever}.${metric}.json \
			--qrels_path ./datasets/robust04/qrels/robust04.qrels.txt \
			--epoch_num 100 \
			--interval 1 \
			--seq_len 1000 \
			--batch_size 64 \
	
			# inference
			python -u ./rlt/supervised_rlt/main.py \
			--name ${model} \
			--checkpoint_path ./checkpoint/ \
			--feature_path ./datasets/robust04/features/robust04-fold${folds_inference[$i]}.feature-${retriever}.json \
			--label_path ./datasets/robust04/labels/robust04-fold${folds_inference[$i]}.label-${retriever}.${metric}.json \
			--qrels_path ./datasets/robust04/qrels/robust04.qrels.txt \
			--epoch_num 100 \
			--interval 1 \
			--seq_len 1000 \
			--batch_size 64 \
			--checkpoint_name robust04-fold${folds_training[$i]}.${retriever}.${model}.${metric} \
			--output_path ./output \
			--infer
			done
		done
	done
done
```

#### 3.2.3 Train and infer LeCut
Note that LeCut can only work for RepLLaMA.
Run the following commands to train LeCut on TREC-DL 19 (TREC-DL 20) and then infer it on TREC-DL 20 (TREC-DL 19):
```bash
retrievers=("original-repllama-1000")
metrics=("rankllama-1000-ndcg@10-eet-alpha-0.001-beta0" "rankllama-1000-ndcg@10-eet-alpha-0.001-beta1" "rankllama-1000-ndcg@10-eet-alpha-0.001-beta2")
models=("lecut")

# train a model on TREC-DL 19, and infer it on TREC-DL 20
for retriever in "${retrievers[@]}"
do
	for metric in "${metrics[@]}"
	do 
		for model in "${models[@]}"
		do
		# training
		python -u ./rlt/supervised_rlt/main.py \
		--name ${model} \
		--checkpoint_path ./checkpoint/ \
		--feature_path ./datasets/msmarco-v1-passage/features/dl-19-passage.feature-${retriever}.embed-repllama.json \
		--label_path ./datasets/msmarco-v1-passage/labels/dl-19-passage.label-${retriever}.${metric}.json \
		--qrels_path ./datasets/msmarco-v1-passage/qrels/dl-19-passage.qrels.txt \
		--epoch_num 100 \
		--interval 1 \
		--seq_len 1000 \
		--batch_size 64 \
		--binarise_qrels \
	
		# inference
		python -u ./rlt/supervised_rlt/main.py \
		--name ${model} \
		--checkpoint_path ./checkpoint/ \
		--feature_path ./datasets/msmarco-v1-passage/features/dl-20-passage.feature-${retriever}.embed-repllama.json \
		--label_path ./datasets/msmarco-v1-passage/labels/dl-20-passage.label-${retriever}.${metric}.json \
		--qrels_path ./datasets/msmarco-v1-passage/qrels/dl-20-passage.qrels.txt \
		--epoch_num 100 \
		--interval 1 \
		--seq_len 1000 \
		--batch_size 64 \
		--binarise_qrels \
		--checkpoint_name dl-19-passage.${retriever}.${model}-embed-repllama.${metric} \
		--output_path ./output \
		--infer
		done
	done
done

# train a model on TREC-DL 20, and infer it on TREC-DL 19
for retriever in "${retrievers[@]}"
do
	for metric in "${metrics[@]}"
	do 
		for model in "${models[@]}"
		do
		# training
		python -u ./rlt/supervised_rlt/main.py \
		--name ${model} \
		--checkpoint_path ./checkpoint/ \
		--feature_path ./datasets/msmarco-v1-passage/features/dl-20-passage.feature-${retriever}.embed-repllama.json \
		--label_path ./datasets/msmarco-v1-passage/labels/dl-20-passage.label-${retriever}.${metric}.json \
		--qrels_path ./datasets/msmarco-v1-passage/qrels/dl-20-passage.qrels.txt \
		--epoch_num 100 \
		--interval 1 \
		--seq_len 1000 \
		--batch_size 64 \
		--binarise_qrels \
	
		# inference
		python -u ./rlt/supervised_rlt/main.py \
		--name ${model} \
		--checkpoint_path ./checkpoint/ \
		--feature_path ./datasets/msmarco-v1-passage/features/dl-19-passage.feature-${retriever}.embed-repllama.json \
		--label_path ./datasets/msmarco-v1-passage/labels/dl-19-passage.label-${retriever}.${metric}.json \
		--qrels_path ./datasets/msmarco-v1-passage/qrels/dl-19-passage.qrels.txt \
		--epoch_num 100 \
		--interval 1 \
		--seq_len 1000 \
		--batch_size 64 \
		--binarise_qrels \
		--checkpoint_name dl-20-passage.${retriever}.${model}-embed-repllama.${metric} \
		--output_path ./output \
		--infer
		done
	done
done
```
### 3.3 Evaluation
A file that shows the results (e.g., re-ranking results using the predicted cut-offs) would be generated in the corresponding `./output/{dataset name}.{retriever name}" directory.

Use the following commands to evaluate RLT methods w.r.t the pipeline of BM25--RankLLaMA:
```bash
# TREC-DL 19
python -u ./evaluation.py \
--pattern './output/dl-19-passage.original-bm25-1000/dl-19-passage.original-bm25-1000.*' \
--reranking_labels_path ./datasets/msmarco-v1-passage/labels/dl-19-passage.label-original-bm25-1000.rankllama-1000-ndcg@10.json \

# TREC-DL 20
python -u ./evaluation.py \
--pattern './output/dl-20-passage.original-bm25-1000/dl-20-passage.original-bm25-1000.*' \
--reranking_labels_path ./datasets/msmarco-v1-passage/labels/dl-20-passage.label-original-bm25-1000.rankllama-1000-ndcg@10.json \

# Robust04
python -u ./rlt/evaluation.py \
--pattern './output/robust04-fold1.title-bm25-1000/robust04-fold1.*' \
--reranking_labels_path ./datasets/robust04/labels/robust04-fold1.label-title-bm25-1000.rankllama-doc-2048-1000-ndcg@20.json

python -u ./rlt/evaluation.py \
--pattern './output/robust04-fold2.title-bm25-1000/robust04-fold2.*' \
--reranking_labels_path ./datasets/robust04/labels/robust04-fold2.label-title-bm25-1000.rankllama-doc-2048-1000-ndcg@20.json

python -u ./rlt/evaluation.py \
--pattern './output/robust04-fold3.title-bm25-1000/robust04-fold3.*' \
--reranking_labels_path ./datasets/robust04/labels/robust04-fold3.label-title-bm25-1000.rankllama-doc-2048-1000-ndcg@20.json

python -u ./rlt/evaluation.py \
--pattern './output/robust04-fold4.title-bm25-1000/robust04-fold4.*' \
--reranking_labels_path ./datasets/robust04/labels/robust04-fold4.label-title-bm25-1000.rankllama-doc-2048-1000-ndcg@20.json

python -u ./rlt/evaluation.py \
--pattern './output/robust04-fold5.title-bm25-1000/robust04-fold5.*' \
--reranking_labels_path ./datasets/robust04/labels/robust04-fold5.label-title-bm25-1000.rankllama-doc-2048-1000-ndcg@20.json
```

Use the following commands to evaluate RLT methods w.r.t the pipeline of SPLADE++--RankLLaMA:
```bash
# TREC-DL 19
python -u ./evaluation.py \
--pattern './output/dl-19-passage.original-splade-pp-ed-pytorch-1000/dl-19-passage.original-splade-pp-ed-pytorch-1000.*' \
--reranking_labels_path ./datasets/msmarco-v1-passage/labels/dl-19-passage.label-original-splade-pp-ed-pytorch-1000.rankllama-1000-ndcg@10.json \

# TREC-DL 20
python -u ./evaluation.py \
--pattern './output/dl-20-passage.original-splade-pp-ed-pytorch-1000/dl-20-passage.original-splade-pp-ed-pytorch-1000.*' \
--reranking_labels_path ./datasets/msmarco-v1-passage/labels/dl-20-passage.label-original-splade-pp-ed-pytorch-1000.rankllama-1000-ndcg@10.json \
```

Use the following commands to evaluate RLT methods w.r.t the pipeline of RepLLaMA--RankLLaMA:
```bash
# TREC-DL 19
python -u ./evaluation.py \
--pattern './output/dl-19-passage.original-repllama-1000/dl-19-passage.original-repllama-1000.*' \
--reranking_labels_path ./datasets/msmarco-v1-passage/labels/dl-19-passage.label-original-repllama-1000.rankllama-1000-ndcg@10.json \

# TREC-DL 20
python -u ./evaluation.py \
--pattern './output/dl-20-passage.original-repllama-1000/dl-20-passage.original-repllama-1000.*' \
--reranking_labels_path ./datasets/msmarco-v1-passage/labels/dl-20-passage.label-original-repllama-1000.rankllama-1000-ndcg@10.json \
```

Use the following commands to evaluate RLT methods w.r.t the pipeline of BM25--MonoT5:
```bash
# TREC-DL 19
python -u ./evaluation.py \
--pattern './output/dl-19-passage.original-bm25-1000/dl-19-passage.original-bm25-1000.*' \
--reranking_labels_path ./datasets/msmarco-v1-passage/labels/dl-19-passage.label-original-bm25-1000.monot5-1000-ndcg@10.json \

# TREC-DL 20
python -u ./evaluation.py \
--pattern './output/dl-20-passage.original-bm25-1000/dl-20-passage.original-bm25-1000.*' \
--reranking_labels_path ./datasets/msmarco-v1-passage/labels/dl-20-passage.label-original-bm25-1000.monot5-1000-ndcg@10.json \

# Robust04
python -u ./rlt/evaluation.py \
--pattern './output/robust04-fold1.title-bm25-1000/robust04-fold1.*' \
--reranking_labels_path ./datasets/robust04/labels/robust04-fold1.label-title-bm25-1000.monot5-1000-ndcg@20.json

python -u ./rlt/evaluation.py \
--pattern './output/robust04-fold2.title-bm25-1000/robust04-fold2.*' \
--reranking_labels_path ./datasets/robust04/labels/robust04-fold2.label-title-bm25-1000.monot5-1000-ndcg@20.json

python -u ./rlt/evaluation.py \
--pattern './output/robust04-fold3.title-bm25-1000/robust04-fold3.*' \
--reranking_labels_path ./datasets/robust04/labels/robust04-fold3.label-title-bm25-1000.monot5-1000-ndcg@20.json

python -u ./rlt/evaluation.py \
--pattern './output/robust04-fold4.title-bm25-1000/robust04-fold4.*' \
--reranking_labels_path ./datasets/robust04/labels/robust04-fold4.label-title-bm25-1000.monot5-1000-ndcg@20.json

python -u ./rlt/evaluation.py \
--pattern './output/robust04-fold5.title-bm25-1000/robust04-fold5.*' \
--reranking_labels_path ./datasets/robust04/labels/robust04-fold5.label-title-bm25-1000.monot5-1000-ndcg@20.json



python -u ./process_robust04.py \
--mode merge_k \
--fold_one_pattern './output/robust04-fold1.title-bm25-1000/robust04-fold1.*'


python -u ./rlt/evaluation.py \
--pattern './output/robust04.title-bm25-1000/robust04.*' \
--reranking_labels_path ./datasets/robust04/labels/robust04.label-title-bm25-1000.monot5-1000-ndcg@20.json

python -u ./rlt/evaluation.py \
--pattern './output/robust04.title-bm25-1000/robust04.*' \
--reranking_labels_path ./datasets/robust04/labels/robust04.label-title-bm25-1000.rankllama-doc-2048-1000-ndcg@20.json
```

## 4. Reproducing plots
Run `plots.ipynb` can recreate all plots represented in the paper.
The recreated plots would be stored in the `./plots` directory.


## 5. Robust04 results

Table: A comparison of RLT methods, optimized for re-ranking effectiveness/efficiency tradeoffs, in predicting re-ranking cut-off points for the BM25–RankLLaMA pipeline on Robust04. 
| Method |Avg. k | nDCG@20|
|---|---|---|
| w/o re-ranking | - | 0.413 |       
| Fixed-k (10)   | 10 | 0.430 |         
| Fixed-k (20)   | 20 | 0.435 |   
| Fixed-k (100)  | 100 | 0.467 |   
| Fixed-k (200)  | 200 | 0.465 |  
| Fixed-k (500)  | 500 | 0.453 |      
| Fixed-k (1000) | 1000 | 0.451 |   
| Surprise       | 721.91 | 0.449 |  
| Greedy-k (β=0) | 398.85 | 0.455 |
| BiCut (η=0.40) | 341.05 | 0.461 | 
| Choppy (β=0)   | 495.03 | 0.455 | 
| AttnCut (β=0)  | 771.01 | 0.452 | 
| MtCut (β=0)    | 590.67 | 0.457 | 
| Greedy-k (β=1) | 136.62 | 0.468 |
| BiCut (η=0.50) | 243.79 | 0.463 | 
| Choppy (β=1)   | 480.02 | 0.456 | 
| AttnCut (β=1)  | 237.37 | 0.462 | 
| MtCut (β=1)    | 223.32 | 0.464 |
| Greedy-k (β=2) | 121.34 | 0.468 |
| BiCut (η=0.60) | 166.22 | 0.464 | 
| Choppy (β=2)   | 487.69 | 0.453 | 
| AttnCut (β=2)  | 121.27 | 0.465 | 
| MtCut (β=2)    | **125.71** | **0.469** |
| Oracle         | 131.42 | 0.559 |


Table: A comparison of RLT methods, optimized for re-ranking effectiveness/efficiency tradeoffs, in predicting re-ranking cut-off points for the BM25–MonoT5 pipeline on Robust04. 
| Method |Avg. k | nDCG@20|
|---|---|---|
| w/o re-ranking | - | 0.413 |       
| Fixed-k (10)   | 10 | 0.440 |         
| Fixed-k (20)   | 20 | 0.452 |   
| Fixed-k (100)  | 100 | 0.543 |   
| Fixed-k (200)  | 200  | 0.556 | 
| Fixed-k (500)  | 500 | 0.556 |      
| Fixed-k (1000) | 1000 | 0.556 |   
| Surprise       | 721.91 | 0.556 |  
| Greedy-k (β=0) | 795.41 | 0.556 |
| BiCut (η=0.40) | 341.05 | 0.555 | 
| Choppy (β=0)   | 489.28 | 0.538 | 
| AttnCut (β=0)  | **799.28** | **0.560** | 
| MtCut (β=0)    | **754.96** | **0.560** | 
| Greedy-k (β=1) | 209.23 | 0.556  |
| BiCut (η=0.50) | **243.79**  | **0.557** | 
| Choppy (β=1)   | 512.17 | 0.555 | 
| AttnCut (β=1)  | 261.03 | 0.554 | 
| MtCut (β=1)    | 266.57 | 0.558 |
| Greedy-k (β=2) | 142.37 | 0.545 |
| BiCut (η=0.60) | 166.22 | 0.549 | 
| Choppy (β=2)   | 484.45 | 0.550 | 
| AttnCut (β=2)  | 131.48 | 0.539 | 
| MtCut (β=2)    | 147.32 | 0.544 |
| Oracle         | 212.35 | 0.635 |








