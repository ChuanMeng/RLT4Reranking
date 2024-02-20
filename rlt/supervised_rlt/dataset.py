import numpy as np
import torch
import random
from transformers import BertTokenizer
from pyserini.index.lucene import IndexReader
import pytrec_eval
import more_itertools
import json
import os
from scipy import stats
import logging
import math

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s %(name)s %(levelname)s %(message)s",
                    datefmt = '%Y-%m-%d  %H:%M:%S %a'
                    )


class Dataset(torch.utils.data.Dataset):
    def __init__(self, args):
        super(Dataset, self).__init__()

        self.args = args
        #self.index_reader = IndexReader(args.index_path)
        self.input = []
        self.r = None
        self.d_embd = None

        self.load()


    def load(self):

        with open(self.args.feature_path, 'r') as r:
            features = json.load(r)

        if self.args.label_path is not None:
            with open(self.args.label_path, 'r') as r:
                labels = json.load(r)
        else:
            assert self.args.name =="bicut"
            labels = {}

        with open(self.args.qrels_path, 'r') as r:
            qrels = pytrec_eval.parse_qrel(r)

        num_q = len(list(features))

        num_q_real=0

        rel_dataset=[]

        for qid in features.keys():
            if qid not in qrels:
                logging.info(f"query {qid} is not in qrels. Throw it away.")
                continue

            if len(features[qid]["pos"])!=self.args.seq_len:
                logging.info(f"query {qid} has less than {self.args.seq_len} retrieved items. Throw it away.")
                continue

            num_q_real += 1

            pos = features[qid]["pos"]

            # relevance judgments
            rel=[]

            for docid in features[qid]["docid"]:
                rel_grade = qrels[qid][docid] if docid in qrels[qid] else 0

                if self.args.binarise_qrels:
                    if rel_grade >= 2:
                        rel_grade = 1
                    else:
                        rel_grade = 0
                else:
                    if rel_grade >= 1:
                        rel_grade = 1
                    else:
                        rel_grade = 0

                rel.append(rel_grade)

            assert len(rel)==len(features[qid]["docid"])

            rel_dataset+=rel

            score = np.array(features[qid]["score"])
            doc_len = np.array(features[qid]["doc_len"])
            unique_num = np.array(features[qid]["unique_num"])
            tfidf_sim = np.array(features[qid]["tfidf_sim"])
            doc2vec_sim = np.array(features[qid]["doc2vec_sim"])

            if self.args.name == "bicut":
                feature = np.column_stack((score,doc_len,unique_num,tfidf_sim,doc2vec_sim))

            elif self.args.name == "choppy":
                feature = np.column_stack((score,doc_len,unique_num,tfidf_sim,doc2vec_sim))

            elif self.args.name == "attncut":
                feature = np.column_stack((score,doc_len,unique_num,tfidf_sim,doc2vec_sim))

            elif self.args.name == "mmoecut":
                feature = np.column_stack((score,doc_len,unique_num,tfidf_sim,doc2vec_sim))

            elif self.args.name == "lecut":
                self.d_embd = len(features[qid]["embedding"][0])

                cos_avg_embedding = np.array(features[qid]["cos_avg_embedding"])  # [S]
                embedding = np.array(features[qid]["embedding"]) # [S, D]

                #self.d_embd= embeddings[qid][features[qid]["docid"][0]].size()[0]
                #embedding = np.array([embeddings[qid][docid] for docid in features[qid]["docid"]])
                #cos_avg_embedding = [0]+ [cos(embedding[i], np.sum(softmax(score[:i], -1)[0].reshape(-1, 1) * embedding[:i], 0)) for i in range(1, self.args.seq_len)]

                feature = np.column_stack((score, doc_len, unique_num, tfidf_sim, doc2vec_sim, cos_avg_embedding, embedding))

            else:
                raise NotImplementedError

            assert len(feature)==self.args.seq_len

            if self.args.label_path is None:
                assert self.args.name == "bicut"
                labels[qid]=rel

            self.input.append([qid, torch.Tensor(pos).long(), torch.Tensor(feature), torch.Tensor(rel), torch.Tensor(labels[qid])])

        self.r = sum(rel_dataset)/len(rel_dataset)

        logging.info(f"# relevant items: {sum(rel_dataset)}; # total items: {len(rel_dataset)}; r:{self.r}")
        if self.d_embd is not None:
            logging.info(f"d_embd: {self.d_embd}")

        logging.info(f"process {num_q_real} out of {num_q} queries.")

    def __getitem__(self, index):
        return self.input[index]

    def __len__(self):
        return len(self.input)

def collate_fn(data):
    qid, pos, feature, rel, label  = zip(*data)


    return {'qid': qid,  #  tuple(B)
            'pos': torch.stack(pos),  # [B,S]
            'feature': torch.stack(feature),  # [B,S,feature_dim]
            'rel': torch.stack(rel),  # [B,S]
            'label': torch.stack(label)  # [B,S]
            }


