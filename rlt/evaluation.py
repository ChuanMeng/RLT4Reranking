import argparse
import json
import os
import numpy as np
from collections import defaultdict
import glob
import math
import logging
import scipy
import pytrec_eval

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

def evaluation(k_path, reranking_labels_path):
    logging.info("*"*10+"Evaluation begins"+"*"*10)
    logging.info(k_path)

    # for reranking
    q2k = {}
    with open(k_path, 'r') as r:
        for line in r:
            qid, k = line.rstrip().split("\t")
            q2k[qid] = int(k)

    with open(reranking_labels_path, 'r') as r:
        q2result = json.loads(r.read())

    print(len(q2k),len(q2result))
    assert len(q2k)==len(q2result)

    q_num =len(q2k)
    avg_k = sum(list(q2k.values()))/q_num

    k_result =[]
    qids = []

    for qid in q2k.keys():
        qids.append(qid)
        if q2k[qid]==0:
            q2k[qid]=1
        assert q2k[qid]-1>=0
        k_result.append(q2result[qid][q2k[qid]-1])

    for fixed_k in [1, 10, 20, 100, 200, 1000]:
        fixed_k_result =[]
        for qid in qids:
            fixed_k_result.append(q2result[qid][fixed_k-1])

        _, pvalue = scipy.stats.ttest_rel(k_result, fixed_k_result)

        logging.info(
            f'sanity check: fixed-k {fixed_k}, re-ranking quality w.r.t ndcg@10 is {round(sum(fixed_k_result) / len(fixed_k_result), 4)}; pvalue: {pvalue}')


    result_dict = {"avg. k": round(avg_k,2),
                   "ndcg@10": round(sum(k_result)/q_num,3),
                   "# q": q_num,
                   }

    logging.info(result_dict)

    return result_dict

def evaluation_glob(pattern, reranking_labels_path):
    for k_path in sorted(glob.glob(pattern)):

        name = k_path.split("/")[-1]
        epoch = name.split("-")[-1]
        output_path ="/".join(pattern.split("/")[:-1])
        pattern_name = pattern.split("/")[-1]
        reranker_metric = reranking_labels_path.split("/")[-1].split(".")[2]

        result_dict = evaluation(k_path, reranking_labels_path)

        with open(f"{output_path}/result.{pattern_name}", 'a+', encoding='utf-8') as w:
            name_=f"{name}.{reranker_metric}:"
            w.write(f"{name_.ljust(85, ' ')} {str(result_dict)}{os.linesep}")


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("--pattern", type=str, default=None)
    parser.add_argument("--k_path", type=str, default=None)
    parser.add_argument("--reranking_labels_path", type=str, default=None)
    args = parser.parse_args()

    if args.pattern is not None:
        evaluation_glob(args.pattern, args.reranking_labels_path)
    else:
        result = evaluation(args.k_path, args.reranking_labels_path)

