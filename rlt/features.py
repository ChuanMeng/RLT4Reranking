import argparse
import json
import os
import numpy as np
import gensim
from tqdm import tqdm
import logging
import pytrec_eval
from collections import defaultdict
import torch

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)


def cos_sim(x,y):
    num = x.dot(y.T)
    denom = np.linalg.norm(x) * np.linalg.norm(y)
    sim = (num / denom) if denom != 0 else 0
    return sim if not np.isnan(sim) else 0

def neighbor_sim(vec_list):
    sim_list = [[0, cos_sim(np.array(vec_list[0]), np.array(vec_list[1]))]]
    for i in range(1, len(vec_list)-1):
        sim_list.append([cos_sim(np.array(vec_list[i]), np.array(vec_list[i-1])), cos_sim(np.array(vec_list[i]), np.array(vec_list[i+1]))])
    sim_list.append([cos_sim(np.array(vec_list[-2]), np.array(vec_list[-1])), 0])
    return sim_list

def dense(i2v, vocab_len):
    dense = [0] * vocab_len
    for (i,v) in i2v:
        dense[i] = v
    return dense

def preprocess_doc(text):
    text = text.replace("\t", " ").replace("\n", " ").replace("\r", " ")
    text = gensim.parsing.preprocessing.strip_punctuation(text)
    text = gensim.parsing.preprocessing.remove_stopwords(text)
    text = gensim.parsing.preprocessing.stem(text)
    tokens = [token for token in text.split()]
    #tokens = gensim.utils.simple_preprocess(text, deacc=True) # this function will remove numbers in the text; do not use it
    return tokens

def preproces_corpus(index_path):
    frequency = defaultdict(int)
    preprocessed_corpus ={}
    d_count= 0
    with open(index_path, "r") as r:
        for line in tqdm(r):
            docid, text = line.strip().split("\t")
            preprocessed_doc = preprocess_doc(text)
            for token in preprocessed_doc:
                frequency[token]+=1
            preprocessed_corpus[docid] = preprocessed_doc
            d_count+=1

    print(f"# doc: {d_count}")
    print(f"# vocab (initial): {len(frequency)}")

    # filter out tokens only appearing once
    for docid in tqdm(preprocessed_corpus.keys()):
        preprocessed_corpus[docid] = [token for token in preprocessed_corpus[docid] if frequency[token] > 1]

    return preprocessed_corpus


def bow(preprocessed_corpus):
    dictionary = gensim.corpora.Dictionary(list(preprocessed_corpus.values()))
    bow_corpus={}
    for docid, preprocessed_doc in tqdm(preprocessed_corpus.items()):
        bow_corpus[docid]=dictionary.doc2bow(preprocessed_doc)
    return bow_corpus, dictionary

def tfidf(bow_corpus):
    #tfidf_corpus ={}
    tfidf_model = gensim.models.TfidfModel(list(bow_corpus.values()))
    #for docid, bow_doc in tqdm(bow_corpus.items()):
        #tfidf_corpus[docid]=tfidf_model[bow_doc]
        #tfidf_corpus[docid] = dense(tfidf_model[bow_doc], len(dictionary.keys()))
        #print([[id, dictionary[id], np.around(freq, decimals=2)] for id, freq in tfidf_corpus[docid]])

    #return tfidf_corpus, tfidf_model
    return tfidf_model

def train_doc2vec(preprocessed_corpus, vector_size, min_count, epochs):
    tag =0
    train_corpus ={}
    for docid, preprocessed_doc in tqdm(preprocessed_corpus.items()):
        train_corpus[docid]=gensim.models.doc2vec.TaggedDocument(preprocessed_doc, [tag])
        tag += 1

    model = gensim.models.doc2vec.Doc2Vec(vector_size=vector_size, min_count=min_count, epochs=epochs)
    model.build_vocab(list(train_corpus.values()))
    model.train(list(train_corpus.values()), total_examples=model.corpus_count, epochs=model.epochs)

    return model

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", type=str) # [tfidf,doc2vec,infer]
    parser.add_argument("--vector_size", type=int, default= 128)
    parser.add_argument("--epoch", type=int, default= 40)
    parser.add_argument("--min_count", type=int, default= 2)
    parser.add_argument("--index_path", type=str)
    parser.add_argument("--query_path", type=str)
    parser.add_argument("--run_path", type=str)
    parser.add_argument("--qrels_path", type=str)
    parser.add_argument("--output_path", type=str)
    parser.add_argument("--cache_dir", type=str)
    parser.add_argument("--token", type=str)

    args = parser.parse_args()

    if args.mode == "infer":
        args.dataset_class = args.run_path.split("/")[-3]
        args.dataset_name = args.run_path.split("/")[-1].split(".")[0]
        args.retriever = "-".join(args.run_path.split("/")[-1].split(".")[1].split("-")[1:])
        args.setup = f"{args.dataset_name}.feature-{args.retriever}"

        with open(os.path.join(args.output_path, "preprocessed_corpus.json"), 'r') as r:
            preprocessed_corpus = json.loads(r.read())

        with open(args.run_path, 'r') as r:
            run = pytrec_eval.parse_run(r)

        with open(args.qrels_path, 'r') as r:
            qrels = pytrec_eval.parse_qrel(r)

        dictionary = gensim.corpora.Dictionary().load(os.path.join(args.output_path, "dictionary.dic"))
        vocab_len = len(dictionary.keys())
        tfidf_model = gensim.models.TfidfModel.load(os.path.join(args.output_path, "tfidf_model"))
        doc2vec_model = gensim.models.doc2vec.Doc2Vec.load(os.path.join(args.output_path, f"doc2vec_model_{args.vector_size}"))

        feature = {}

        for qid in tqdm(run.keys()):

            if qid not in qrels:
                continue

            feature[qid]={}

            pos_list = []
            docid_list = []
            score_list = []

            doc_len_list =[]
            unique_num_list =[]

            doc2vec_list =[]
            tfidf_list = []

            ranked_list = sorted(run[qid].items(), key=lambda x: x[1], reverse=True)

            for idx, (docid, score) in enumerate(ranked_list):
                preprocessed_doc = preprocessed_corpus[docid]
                bow_doc = dictionary.doc2bow(preprocessed_doc)
                tfidf_doc = tfidf_model[bow_doc]
                doc2vec_doc = doc2vec_model.infer_vector(preprocessed_doc).tolist()

                #bow_dense = dense(bow_doc, vocab_len)

                assert len(preprocessed_doc) == sum([f for voc_idx, f in bow_doc])
                assert len(bow_doc) == len(np.unique(preprocessed_doc))

                pos_list.append(idx)
                docid_list.append(docid)
                score_list.append(score)

                doc_len_list.append(len(preprocessed_doc))
                unique_num_list.append(len(bow_doc))

                tfidf_list.append(dense(tfidf_doc, vocab_len))
                doc2vec_list.append(doc2vec_doc)


            feature[qid]["pos"]=pos_list
            feature[qid]["docid"]=docid_list
            feature[qid]["score"]=score_list

            feature[qid]["doc_len"]=doc_len_list
            feature[qid]["unique_num"]=unique_num_list

            feature[qid]["tfidf_sim"]=neighbor_sim(tfidf_list) # [[left, right],...]
            feature[qid]["doc2vec_sim"]=neighbor_sim(doc2vec_list) # [[left, right],...]


        with open(os.path.join(args.output_path, args.setup)+".json", "w") as w:
            w.write(json.dumps(feature))

    elif args.mode == "tfidf":

        if os.path.exists(os.path.join(args.output_path, "preprocessed_corpus.json")):
            with open(os.path.join(args.output_path, "preprocessed_corpus.json"), 'r') as r:
                preprocessed_corpus = json.loads(r.read())
        else:
            preprocessed_corpus = preproces_corpus(args.index_path)
            with open(os.path.join(args.output_path, "preprocessed_corpus.json"), "w") as w:
                w.write(json.dumps(preprocessed_corpus))

        if os.path.exists(os.path.join(args.output_path, "dictionary.dic")) and os.path.exists(os.path.join(args.output_path, "bow_corpus.json")):
            dictionary = gensim.corpora.Dictionary().load(os.path.join(args.output_path, "dictionary.dic"))
            with open(os.path.join(args.output_path, "bow_corpus.json"), 'r') as r:
                bow_corpus = json.loads(r.read())
        else:
            bow_corpus, dictionary = bow(preprocessed_corpus)
            print(dictionary)

            dictionary.save(os.path.join(args.output_path, "dictionary.dic"))

            with open(os.path.join(args.output_path, "bow_corpus.json"), "w") as w:
                w.write(json.dumps(bow_corpus))

        if os.path.exists(os.path.join(args.output_path, "tfidf_model")):
            gensim.models.TfidfModel.load(os.path.join(args.output_path, "tfidf_model"))

            #with open(os.path.join(args.output_path, "tfidf_corpus.json"), 'r') as r:
                #tfidf_corpus = json.loads(r.read())
        else:
            #tfidf_corpus, tfidf_model = tfidf(bow_corpus)
            tfidf_model = tfidf(bow_corpus)
            tfidf_model.save(os.path.join(args.output_path, "tfidf_model"))

            #with open(os.path.join(args.output_path, "tfidf_corpus.json"), "w") as w:
                #w.write(json.dumps(tfidf_corpus))

    elif args.mode == "doc2vec":

        if os.path.exists(os.path.join(args.output_path, "preprocessed_corpus.json")):
            with open(os.path.join(args.output_path, "preprocessed_corpus.json"), 'r') as r:
                preprocessed_corpus = json.loads(r.read())
        else:
            preprocessed_corpus = preproces_corpus(args.index_path)
            with open(os.path.join(args.output_path, "preprocessed_corpus.json"), "w") as w:
                w.write(json.dumps(preprocessed_corpus))

        if os.path.exists(os.path.join(args.output_path, f"doc2vec_model_{args.vector_size}")):
            doc2vec_model = gensim.models.doc2vec.Doc2Vec.load(
                os.path.join(args.output_path, f"doc2vec_model_{args.vector_size}"))
        else:
            doc2vec_model = train_doc2vec(preprocessed_corpus, args.vector_size, args.min_count, args.epoch)
            doc2vec_model.save(os.path.join(args.output_path, f"doc2vec_model_{args.vector_size}"))

        print(doc2vec_model.infer_vector(['only', 'you', 'can', 'prevent', 'forest', 'fires'][:10]))

    else:
        raise NotImplementedError
