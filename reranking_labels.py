import argparse
import json
import os
import numpy as np
from collections import defaultdict
import glob
import pytrec_eval
import logging
import math
import scipy

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s %(name)s %(levelname)s %(message)s",
                    datefmt = '%Y-%m-%d  %H:%M:%S %a'
                    )

def eet(retrieval_result, reranking_result, k, alpha=None, beta=None):
  effectivenss = reranking_result - retrieval_result # [0, 1]
  if effectivenss<0:
      effectivenss=0.
  efficiency = math.exp(alpha*k)
  numerator = (1+beta**2)*(efficiency*effectivenss)
  denominator = (beta**2)*effectivenss+efficiency
  return numerator/denominator

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("--retrieval_run_path", type=str)
    parser.add_argument("--reranking_run_path", type=str)
    parser.add_argument("--qrels_path", type=str)
    parser.add_argument("--metric", type=str,default=None)
    parser.add_argument("--binarise_qrels", action='store_true') # Turn on only when using trec 19 & 20 and MRR@10
    parser.add_argument("--seq_len", type=int, required=True)
    parser.add_argument('--output_path', type=str, default=None)

    args = parser.parse_args()

    args.dataset_class = args.reranking_run_path.split("/")[-3]
    args.dataset_name = args.reranking_run_path.split("/")[-1].split(".")[0]
    args.retriever_name = "-".join(args.retrieval_run_path.split("/")[-1].split(".")[1].split("-")[1:])
    args.pipeline_name = "-".join(args.reranking_run_path.split("/")[-1].split(".")[1].split("-")[1:])
    assert args.retriever_name in args.pipeline_name
    args.reranker_name = args.pipeline_name.replace(args.retriever_name+"-","")

    args.setup = f"{args.dataset_name}.label-{args.retriever_name}.{args.reranker_name}-{args.metric}.json"

    if args.metric=="ndcg@10":
        args.metric = "ndcg_cut_10"
    elif args.metric=="mrr@10":
        args.metric ='recip_rank'
    elif args.metric=="map@100":
        args.metric ="map_cut_100"
    else:
        raise NotImplementedError

    if args.metric=="ndcg_cut_10":
        assert args.binarise_qrels == False
    elif (args.metric=="recip_rank" or args.metric=="map@100")  and "dl" in args.dataset_name:
        assert args.binarise_qrels == True
    elif (args.metric=="recip_rank" or args.metric=="map@100") and ("dev" in args.dataset_name or "train" in args.dataset_name):
        assert args.binarise_qrels == False

    with open(args.retrieval_run_path, 'r') as r:
        retrieval_run = pytrec_eval.parse_run(r)

    with open(args.reranking_run_path, 'r') as r:
        reranking_run = pytrec_eval.parse_run(r)

    with open(args.qrels_path, 'r') as r_qrel:
        qrels = pytrec_eval.parse_qrel(r_qrel)

    if args.binarise_qrels:
        for q_id, pid_rel in qrels.items():
            for p_id, rel in pid_rel.items():
                if int(rel) == 0:
                    qrels[q_id][p_id] = 0
                elif int(rel) == 1:
                    qrels[q_id][p_id] = 0
                elif int(rel) >= 2:
                    qrels[q_id][p_id] = 1
                else:
                    raise Exception

    logging.info(f"# retrieval_run: {len(retrieval_run)}")
    logging.info(f"# reranking_run: {len(reranking_run)}")
    logging.info(f"# qrels: {len(qrels)}")

    q2result = {}
    q2list = {}
    for qid, docid2score in retrieval_run.items():

        if qid not in qrels:
            continue

        # score normalisation
        retrieval_list = [(docid, score) for docid, score in sorted(docid2score.items(), key=lambda item: item[1], reverse=True)]
        max_retrieval_score = retrieval_list[0][1]  # retrieval scores maybe very large; so make them negative
        retrieval_list = [(docid, score - max_retrieval_score - 1) for (docid, score) in retrieval_list]

        reranking_list = [(docid, score) for docid, score in sorted(reranking_run[qid].items(), key=lambda item: item[1], reverse=True)]
        min_reranking_score = reranking_list[-1][1]  # reranking scores maybe negative
        for docid in reranking_run[qid].keys():
            reranking_run[qid][docid] = reranking_run[qid][docid]+abs(min_reranking_score) + 1

        q2result[qid]=[]
        q2list[qid]={}

        for idx in range(args.seq_len):
            k = idx + 1  # [1, 1000]

            q2list[qid][k]=[]

            # 1. merging run
            partial_reranking_list = []
            for docid, _ in retrieval_list[:k]:
                partial_reranking_list.append((docid, reranking_run[qid][docid]))

            partial_reranking_list = [(docid, score) for docid, score in sorted(partial_reranking_list, key=lambda item: item[1], reverse=True)]

            merged_list = partial_reranking_list + retrieval_list[k:]

            q2list[qid][k]=merged_list

            assert len(merged_list) == len(retrieval_list)==len(reranking_list)

            # 2. evaluation
            if args.metric == 'recip_rank':
                merged_list = merged_list[:10]

            evaluator = pytrec_eval.RelevanceEvaluator(qrels, {args.metric})
            q2metric2result = evaluator.evaluate({qid:dict(merged_list)})  # {qid:{ndcg@10:xx,...},...}
            #aggregated_result = pytrec_eval.compute_aggregated_measure(metric,[metric2result[metric] for metric2result in q2metric2result.values()])
            q2result[qid].append(q2metric2result[qid][args.metric])

    # analysis
    q_result_matrix=[]
    q2optimalresult = {}
    q2optimalk = {}

    rates = []

    for qid in retrieval_run.keys():
        if qid not in qrels:
            continue

        q_result_matrix.append(q2result[qid])

        q_optimalresult = max(q2result[qid])
        q2optimalresult[qid] = q_optimalresult
        q2optimalk[qid] = q2result[qid].index(q_optimalresult)+1

        rate = []
        for (docid, socre) in q2list[qid][q2optimalk[qid]][:10]:
            rate.append(1 if docid in qrels[qid] else 0)
        rates.append(sum(rate)/10)

    assert len(rates)==len(qrels)
    rates_avg = sum(rates)/len(rates)


    q_result_matrix = np.array(q_result_matrix)
    result_over_k = np.mean(np.array(q_result_matrix), axis=0).tolist()


    logging.info(f"oracle {args.metric}: {round(np.mean(list(q2optimalresult.values())),3)}")
    logging.info(f"oracle {args.metric}, avg. k={round(np.mean(list(q2optimalk.values())),2)}, judging rate: {rates_avg}")

    optimalresult = max(result_over_k)
    optimalk = result_over_k.index(optimalresult)+1

    logging.info(f"optimal fixed-k {args.metric}: {round(optimalresult,3)}")
    logging.info(f"optimal fixed-k {args.metric}, fixed-k={round(optimalk,2)}")\

    qids = list(q2result.keys())

    for fixed_k in [1, 5, 10, 20, 50, 100, 200, 1000]:
        fixed_k_result = []
        idx = fixed_k-1

        _, pvalue = scipy.stats.ttest_rel([q2optimalresult[qid] for qid in qids], [q2result[qid][idx] for qid in qids])

        rates = []
        for qid in qids:
            rate = []
            for (docid, socre) in q2list[qid][fixed_k][:10]:
                rate.append(1 if docid in qrels[qid] else 0)
            rates.append(sum(rate)/10)
        rates_avg = sum(rates)/len(qids)

        logging.info(f"fixed k={fixed_k}, {args.metric}: {round(result_over_k[idx],3)}; pvalue: {pvalue}; judging rate: {rates_avg}")

    if args.output_path is not None:
        f = open(f"{args.output_path}/{args.setup}", 'w')
        f.write(json.dumps(q2result))
        f.close()

        for beta in [0, 0.5, 1 , 2]:
            for alpha in [-0.05, -0.01, -0.005, -0.001]:
                q2eet={}
                for qid in q2result.keys():
                    q2eet[qid]=[]
                    for idx,_ in enumerate(q2result[qid]):
                        k=idx+1
                        q2eet[qid].append(eet(q2result[qid][0], q2result[qid][idx], k, alpha=alpha, beta=beta))

                setup_ = args.setup.rstrip(".json")
                f = open(f"{args.output_path}/{setup_}-eet-alpha{alpha}-beta{beta}.json", 'w')
                f.write(json.dumps(q2eet))
                f.close()
