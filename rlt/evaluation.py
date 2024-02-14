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

def mrr(labels: np.array, k=10):
    rr_k =[]
    for i in range(len(labels)):
        label = labels[i,:k].tolist()
        if 1 not in label:
            rr_k.append(0)
        else:
            rr_k.append(1/(label.index(1)+1))
    return np.mean(rr_k)

def precision(labels: np.array, k=10):
    p_k =[]
    for i in range(len(labels)):
        hit_num = np.sum(labels[i, :k])
        p_k.append(hit_num/k)
    return np.mean(p_k)

def f1(labels: np.array, k_s: list):
    N_D = np.sum(labels, axis=1)
    p_k, r_k, results = [], [], []
    for i in range(len(labels)):
        count = np.sum(labels[i, :k_s[i]]) # hit num
        p_k.append((count / k_s[i]))
        r_k.append((count / N_D[i]) if N_D[i] != 0 else 0)
        results.append((2 * p_k[-1] * r_k[-1] / (p_k[-1] + r_k[-1])) if p_k[-1] + r_k[-1] != 0 else 0)
    return np.mean(results), results

def dcg(labels: np.array, k_s: list, penalty=-1):
    results = []
    for i in range(len(labels)):
        value, x = 0, labels[i]
        for j in range(k_s[i]):
            value += (1 / math.log2(j + 2)) if x[j] else (penalty / math.log2(j + 2))
        results.append(value)
    return np.mean(results), results

def evaluation(k_path, reranking_labels_path, feature_path, qrels_path, binarise_qrels=False):
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

    # for retrieval
    with open(feature_path, 'r') as r:
        features = json.load(r)

    with open(qrels_path, 'r') as r:
        qrels = pytrec_eval.parse_qrel(r)

    rels = []
    ks = []

    for qid in features.keys():

        ks.append(q2k[qid])
        rel = []
        for docid in features[qid]["docid"]:
            rel_grade = qrels[qid][docid] if docid in qrels[qid] else 0

            if binarise_qrels:
                if rel_grade >= 2:
                    rel_grade = 1
                else:
                    rel_grade = 0

            rel.append(rel_grade)
        rels.append(rel)

    # for retrieval
    rels = np.array(rels)

    logging.info('sanity check: retrieval quality w.r.t precision@10 is {:.3f}'.format(precision(rels,k=10)))
    logging.info('sanity check: retrieval quality w.r.t mrr@10 is {:.3f}'.format(mrr(rels, k=10)))

    f1_overall, _ = f1(rels, ks)
    dcg_overall, _ = dcg(rels,ks)

    for fixed_k in [1, 10, 20, 100, 200, 1000]:
        fixed_ks = [fixed_k for _ in rels]
        logging.info(f'sanity check: fixed k={fixed_k},f1={f1(rels, fixed_ks)[0]}, dcg={dcg(rels, fixed_ks)[0]}')


    result_dict = {"avg. k": round(avg_k,2),
                   "ndcg@10": round(sum(k_result)/q_num,3),
                   "f1": round(f1_overall,3),
                   "dcg": round(dcg_overall,3),
                   "# q": q_num,
                   }

    logging.info(result_dict)

    return result_dict


def evaluation_glob(pattern, reranking_labels_path, feature_path, qrels_path, binarise_qrels=False):
    for k_path in sorted(glob.glob(pattern)):

        name = k_path.split("/")[-1]
        epoch = name.split("-")[-1]
        output_path ="/".join(pattern.split("/")[:-1])
        pattern_name = pattern.split("/")[-1]
        reranker_metric = reranking_labels_path.split("/")[-1].split(".")[2]

        result_dict = evaluation(k_path, reranking_labels_path, feature_path, qrels_path, binarise_qrels=binarise_qrels)

        with open(f"{output_path}/result.{pattern_name}", 'a+', encoding='utf-8') as w:
            name_=f"{name}.{reranker_metric}:"
            w.write(f"{name_.ljust(85, ' ')} {str(result_dict)}{os.linesep}")

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("--pattern", type=str, default=None)
    parser.add_argument("--k_path", type=str, default=None)
    parser.add_argument("--reranking_labels_path", type=str, default=None)
    parser.add_argument("--feature_path", type=str)
    parser.add_argument("--qrels_path", type=str)
    parser.add_argument("--binarise_qrels",action='store_true')  # only turn on for datasets having graded relevance judgments
    args = parser.parse_args()

    if args.pattern is not None:
        evaluation_glob(args.pattern, args.reranking_labels_path, args.feature_path, args.qrels_path, binarise_qrels=args.binarise_qrels)
    else:
        result = evaluation(args.k_path, args.reranking_labels_path, args.feature_path, args.qrels_path, binarise_qrels=args.binarise_qrels)

