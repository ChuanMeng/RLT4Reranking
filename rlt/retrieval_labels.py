import argparse
import json
import os
import numpy as np
from collections import defaultdict
import glob
import pytrec_eval
import logging
import math

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s %(name)s %(levelname)s %(message)s",
                    datefmt = '%Y-%m-%d  %H:%M:%S %a'
                    )

def f1(ranked_list: list, k: int) -> float:
    count, N_D = sum(ranked_list[:k]), sum(ranked_list)
    p_k = count / k
    r_k = (count / N_D) if N_D != 0 else 0
    return (2 * p_k * r_k / (p_k + r_k)) if p_k + r_k != 0 else 0

def dcg(ranked_list: list, k: int, penalty=-1) -> float:
    value = 0
    for i in range(k):
        value += (1 / math.log(i + 2, 2)) if ranked_list[i] else (penalty / math.log(i + 2, 2))
    return value

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--feature_path", type=str)
    parser.add_argument("--qrels_path", type=str)
    parser.add_argument("--binarise_qrels",action='store_true')  # only turn on for datasets having graded relevance judgments
    parser.add_argument("--metric", type=str,default="f1")
    parser.add_argument("--seq_len", type=int, required=True)  # specify
    parser.add_argument('--output_path', type=str)

    args = parser.parse_args()

    args.dataset_class = args.feature_path.split("/")[-3]
    args.dataset_name = args.feature_path.split("/")[-1].split(".")[0]
    args.retriever= "-".join(args.feature_path.split("/")[-1].split(".")[1].split("-")[1:])

    args.setup = f"{args.dataset_name}.label-{args.retriever}.{args.metric}.json"

    cal_metric = {
        "f1": f1,
        "dcg": dcg,
    }

    # for retrieval
    with open(args.feature_path, 'r') as r:
        features = json.load(r)

    with open(args.qrels_path, 'r') as r:
        qrels = pytrec_eval.parse_qrel(r)

    q2result={}
    for qid in features.keys():
        rel = []
        for docid in features[qid]["docid"]:
            rel_grade = qrels[qid][docid] if docid in qrels[qid] else 0

            if args.binarise_qrels:
                if rel_grade >= 2:
                    rel_grade = 1
                else:
                    rel_grade = 0

            rel.append(rel_grade)

        q2result[qid] = []
        for idx in range(args.seq_len):
            k=idx+1
            q2result[qid].append(cal_metric[args.metric](rel,k))

    # analysis
    q_result_matrix=[]
    q2optimalresult = {}
    q2optimalk = {}

    for qid in features.keys():
        q_result_matrix.append(q2result[qid])

        q_optimalresult = max(q2result[qid])
        q2optimalresult[qid] = q_optimalresult
        q2optimalk[qid] = q2result[qid].index(q_optimalresult)+1

    q_result_matrix = np.array(q_result_matrix)
    result_over_k = np.mean(np.array(q_result_matrix), axis=0).tolist()

    logging.info(f"oracle {args.metric}: {round(np.mean(list(q2optimalresult.values())),3)}")
    logging.info(f"oracle {args.metric}, avg. k={round(np.mean(list(q2optimalk.values())),2)}")

    optimalresult = max(result_over_k)
    optimalk = result_over_k.index(optimalresult)+1

    logging.info(f"optimal fixed-k {args.metric}: {round(optimalresult,3)}")
    logging.info(f"optimal fixed-k {args.metric}, fixed-k={round(optimalk,2)}")

    for fixed_k in [1, 5, 10, 20, 50, 100, 200, 1000]:
        idx = fixed_k-1
        logging.info(f"fixed k={fixed_k}, {args.metric}: {round(result_over_k[idx],3)}")

    if args.output_path is not None:
        f = open(f"{args.output_path}/{args.setup}", 'w')
        f.write(json.dumps(q2result))
        f.close()

