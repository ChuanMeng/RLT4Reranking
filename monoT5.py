import argparse
import json
from tqdm import tqdm
import os
from trec_car import read_data
from pygaggle.rerank.base import Query, Text
from pygaggle.rerank.transformer import MonoT5
import pytrec_eval
from pyserini.search.lucene import LuceneSearcher
import sys
from transformers import T5ForConditionalGeneration

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--query_path', type=str, required=True)
    parser.add_argument('--run_path', type=str, required=True)
    parser.add_argument('--index_path', type=str, required=True)
    parser.add_argument('--model', type=str, required=True) # castorini/monot5-base-msmarco
    parser.add_argument('--k', type=int, required=True)
    args = parser.parse_args()

    model = T5ForConditionalGeneration.from_pretrained(args.model)
    reranker = MonoT5(model=model)

    searcher = LuceneSearcher.from_prebuilt_index(args.index_path)

    queries = {}
    query_reader = open(args.query_path, 'r').readlines()
    for line in query_reader:
        qid, qtext = line.split('\t')
        queries[qid] = qtext.strip()

    with open(args.run_path, 'r') as f_run:
        run = pytrec_eval.parse_run(f_run)

    run_path_=args.run_path.removesuffix(".txt")
    w = open(f"{run_path_}-monot5-{args.k}.txt", "w")

    for idx, qid in enumerate(run.keys()):
        if idx==0 or idx % 10==0 or len(run[qid])!=args.k:
            print(f"{idx+1}/{len(run.keys())}, qid: {qid}, len ranked list: {len(run[qid])}")
            sys.stdout.flush()

        query = Query(queries[qid])

        run[qid] = sorted(run[qid].items(), key=lambda x: x[1], reverse=True)[0:args.k]


        if "v1" in args.index_path:
            passages = [[entry[0],json.loads(searcher.doc(entry[0]).raw())['contents']]  for entry in run[qid]]
        elif "v2" in args.index_path:
            passages = [[entry[0], json.loads(searcher.doc(entry[0]).raw())['passage']] for entry in run[qid]]


        texts = [Text(p[1], {'docid': p[0]}, 0) for p in passages]

        reranked = reranker.rerank(query, texts)

        assert len(reranked)==len(run[qid])
        for i in range(len(reranked)):
            #print(f'{i + 1:2} {reranked[i].metadata["docid"]:15} {reranked[i].score:.5f} {reranked[i].text}')
            docid = reranked[i].metadata["docid"]
            w.write(f"{qid} Q0 {docid} {i+1} {reranked[i].score} monoT5-{args.model}\n")

    w.close()





