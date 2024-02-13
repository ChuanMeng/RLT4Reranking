import argparse
import json
import os
import math
import logging
import scipy
import pytrec_eval
import numpy as np
import attr
from scipy.stats import genpareto
import random
import math

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

@attr.s
class Gpd(object):
    k = attr.ib()
    a = attr.ib()

    def cdf(self, x):
        if self.k == 0:
            return 1 - np.exp(-x / self.a)
        else:
            return 1 - np.power(1 - self.k * x / self.a, 1. / self.k)

    def pdf(self, x):
        if self.k == 0:
            pass
        else:
            return 1. / self.a * np.power(1 - self.k * x / self.a, (1. - self.k) / self.k)

    @classmethod
    def fit(cls, vals):
        eps = 1e-6 / np.mean(vals)
        c, _, scale = genpareto.fit(vals, loc=0)
        theta = min(-c / scale, -eps)
        k = -1. / len(vals) * np.sum(np.log(1 - theta * vals))
        a = k / theta
        return cls(k=k, a=a)


def surprise_truncation(scores):
    assert len(scores)==1000

    scores = sorted(scores)
    x = [scores[i] / max(scores) for i in range(len(scores))]
    all_scores = x.copy()
    all_scores = sorted(all_scores)
    j_hat = len(all_scores)
    for j_hat in range(len(all_scores), 0, -1):
        i_hat = 0
        scores = scores[:j_hat]
        excess = [all_scores[i] - all_scores[i_hat] for i in range(i_hat, j_hat - i_hat)]

        tmp_scores = excess.copy()
        tmp_scores = np.array(tmp_scores)

        gpd_object = Gpd(k=len(all_scores) - j_hat, a=scores[i_hat])
        gpd_fitted = gpd_object.fit((tmp_scores))

        cdf_f = gpd_fitted.cdf(tmp_scores)
        m = j_hat - i_hat
        w2 = 0
        for i in range(i_hat, j_hat):
            w2 += math.pow(cdf_f[i] - (2 * (i + 1) - 1) / (2 * m), 2)
        w2 += 1 / (12 * m)

        if j_hat == len(all_scores) and i_hat == 0:
            previous_value = w2
        elif w2 > (previous_value):
            break
        else:
            j_hat -= 1

    for i_hat in range(0, j_hat):
        excess = [all_scores[i] - all_scores[i_hat] for i in range(i_hat, j_hat)]

        tmp_scores = excess.copy()
        tmp_scores = np.array(tmp_scores)

        gpd_object = Gpd(k=len(all_scores) - j_hat, a=scores[i_hat])
        gpd_fitted = gpd_object.fit((tmp_scores))

        cdf_f = gpd_fitted.cdf(tmp_scores)
        m = j_hat - i_hat
        w2 = 0
        for i in range(0, len(tmp_scores)):
            w2 += math.pow(cdf_f[i] - (2 * (i + 1) - 1) / (2 * m), 2)
        w2 += 1 / (12 * m)

        if j_hat == len(all_scores) and i_hat == 0:
            previous_value = w2
        if w2 > (previous_value):
            break
        else:
            i_hat += 1

    gpd_fitted = Gpd(k=len(all_scores) - j_hat, a=all_scores[i_hat])
    revised_surprise = []
    gpd_fitted = gpd_object.fit((tmp_scores))
    cdf_f = gpd_fitted.cdf(tmp_scores)

    for k in range(i_hat, len(all_scores)):
        revised_surprise.append(- math.log(1 - gpd_fitted.cdf(all_scores[k] - all_scores[i_hat])))

    cut_off = i_hat
    for i in range(len(revised_surprise)):
        if revised_surprise[i] < all_scores[i_hat]:
            cut_off += 1
        else:
            break

    return (len(scores) - cut_off)


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("--name", type=str, required=True) # [surprise, fixed, greedy, oracle]
    parser.add_argument("--feature_path", type=str)
    parser.add_argument("--train_labels_path", type=str, default=None)
    parser.add_argument("--test_labels_path", type=str, default=None)
    parser.add_argument("--output_path", type=str)
    args = parser.parse_args()

    if args.feature_path is not None:
        args.dataset_class = args.feature_path.split("/")[-3]
        args.dataset_name = args.feature_path.split("/")[-1].split(".")[0]
        args.retriever = "-".join(args.feature_path.split("/")[-1].split(".")[1].split("-")[1:])
        args.output_path_ = f"{args.output_path}/{args.dataset_name}.{args.retriever}"
    else:
        args.dataset_class = args.test_labels_path.split("/")[-3]
        args.dataset_name = args.test_labels_path.split("/")[-1].split(".")[0]
        args.retriever = "-".join(args.test_labels_path.split("/")[-1].split(".")[1].split("-")[1:])
        args.metric = args.test_labels_path.split("/")[-1].split(".")[2]

        args.output_path_ = f"{args.output_path}/{args.dataset_name}.{args.retriever}"


    if not os.path.exists(args.output_path):
        os.makedirs(args.output_path)

    if not os.path.exists(args.output_path_):
        os.makedirs(args.output_path_)

    if args.name == "fixed":
        fixed_ks =[1, 5, 10, 20, 100, 200, 500, 1000]

        with open(args.feature_path, 'r') as r:
            features = json.load(r)

        for k in fixed_ks:
            args.setup = f"{args.dataset_name}.{args.retriever}.{args.name}-{k}"
            q2p_w = open(args.output_path_ + "/" + args.setup, 'w')
            for qid in features.keys():
                q2p_w.write(qid + '\t' + str(k) + '\n')
            q2p_w.close()

    elif args.name=="surprise":
        args.setup = f"{args.dataset_name}.{args.retriever}.{args.name}"

        with open(args.feature_path, 'r') as r:
            features = json.load(r)

        q2p = {}
        for qid in features.keys():
            assert len(features[qid]["score"]) == 1000
            q2p[qid] = surprise_truncation(features[qid]["score"])

        q2p_w = open(args.output_path_ + "/" + args.setup, 'w')
        for index, qid in enumerate(list(q2p.keys())):
            q2p_w.write(qid + '\t' + str(q2p[qid]) + '\n')
        q2p_w.close()

    elif args.name=="oracle":

        args.setup = f"{args.dataset_name}.{args.retriever}.{args.name}-{args.metric}"

        if not os.path.exists(args.output_path_):
            os.makedirs(args.output_path_)

        with open(args.test_labels_path, 'r') as r:
            q2result = json.load(r)

        q_result_matrix = []
        q2optimalresult = {}
        q2optimalk = {}

        for qid in q2result.keys():
            q_result_matrix.append(q2result[qid])
            q_optimalresult = max(q2result[qid])
            q2optimalresult[qid] = q_optimalresult
            q2optimalk[qid] = q2result[qid].index(q_optimalresult) + 1

        q2p_w = open(args.output_path_ + "/" + args.setup, 'w')
        for index, qid in enumerate(list(q2optimalk.keys())):
            q2p_w.write(qid + '\t' + str(q2optimalk[qid]) + '\n')
        q2p_w.close()

    elif args.name=="greedy":

        args.dataset_class_train = args.train_labels_path.split("/")[-3]
        args.dataset_name_train = args.train_labels_path.split("/")[-1].split(".")[0]
        args.retriever_train = "-".join(args.train_labels_path.split("/")[-1].split(".")[1].split("-")[1:])
        args.metric_train = ".".join(args.train_labels_path.split("/")[-1].split(".")[2:-1])


        args.setup = f"{args.dataset_name}.{args.retriever}.{args.name}-ckpt-{args.dataset_name_train}.{args.retriever_train}.{args.metric_train}"

        with open(args.train_labels_path, 'r') as r:
            q2result = json.load(r)

        q_result_matrix = []
        q2optimalresult = {}
        q2optimalk = {}

        for qid in q2result.keys():
            q_result_matrix.append(q2result[qid])
            q_optimalresult = max(q2result[qid])
            q2optimalresult[qid] = q_optimalresult
            q2optimalk[qid] = q2result[qid].index(q_optimalresult) + 1

        q_result_matrix = np.array(q_result_matrix)
        result_over_k = np.mean(np.array(q_result_matrix), axis=0).tolist()

        optimalresult = max(result_over_k)
        optimalk = result_over_k.index(optimalresult) + 1

        print("training set:", args.train_labels_path)
        print(f"training: k:{optimalk}, result: {optimalresult}")

        with open(args.feature_path, 'r') as r:
            features = json.load(r)

        q2p_w = open(args.output_path_ + "/" + args.setup, 'w')
        for index, qid in enumerate(list(features.keys())):
            q2p_w.write(qid + '\t' + str(optimalk) + '\n')
        q2p_w.close()

        """""
        with open(args.test_labels_path, 'r') as r:
            q2result = json.load(r)

        q_result_matrix = []
        q2optimalresult = {}
        q2optimalk = {}

        for qid in q2result.keys():
            q_result_matrix.append(q2result[qid])
            q_optimalresult = max(q2result[qid])
            q2optimalresult[qid] = q_optimalresult
            q2optimalk[qid] = q2result[qid].index(q_optimalresult) + 1

        q_result_matrix = np.array(q_result_matrix)
        result_over_k = np.mean(np.array(q_result_matrix), axis=0).tolist()

        print("test set:", args.test_labels_path)
        print(f"testing: k:{optimalk}, result: {result_over_k[optimalk - 1]}")
        """""
    else:
        raise NotImplementedError