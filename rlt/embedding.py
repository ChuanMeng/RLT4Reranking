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
from datasets import load_dataset

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

def cos(x,y):
    assert len(x) == len(y)
    xy=0.0
    x2=0.0
    y2=0.0
    for i in range(len(x)):
        xy += x[i]*y[i]
        x2 += x[i]**2
        y2 += y[i]**2
    return xy/((x2*y2)**0.5)


def softmax(x, axis=1):
    row_max = x.max(axis=axis)
    row_max = row_max.reshape(-1, 1)
    x = x - row_max
    x_exp = np.exp(x)  # [len, len]
    x_sum = np.sum(x_exp, axis=axis, keepdims=True)  # [len, 1]
    s = x_exp / x_sum  # [len, dis]
    return s

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("--encoder", type=str)
    parser.add_argument("--q_max_len", type=int)
    parser.add_argument("--p_max_len", type=int)
    parser.add_argument("--index_path", type=str)
    parser.add_argument("--query_path", type=str)
    parser.add_argument("--feature_path", type=str)
    parser.add_argument("--qrels_path", type=str)
    parser.add_argument("--split", type=str)
    parser.add_argument("--output_path", type=str)
    parser.add_argument("--cache_model_dir", type=str)
    parser.add_argument("--cache_data_dir", type=str)
    parser.add_argument("--token", type=str)
    parser.add_argument("--fp16", action='store_true')

    args = parser.parse_args()

    args.dataset_class = args.feature_path.split("/")[-3]
    args.dataset_name = args.feature_path.split("/")[-1].split(".")[0]
    args.retriever= "-".join(args.feature_path.split("/")[-1].split(".")[1].split("-")[1:])

    args.setup = f"{args.dataset_name}.feature-{args.retriever}.embed-{args.encoder}"

    with open(args.feature_path, 'r') as r:
        features = json.load(r)

    with open(args.qrels_path, 'r') as r:
        qrels = pytrec_eval.parse_qrel(r)

    if args.encoder == "repllama":

        import torch
        from transformers import AutoModel, AutoTokenizer
        from peft import PeftModel, PeftConfig

        query_data = load_dataset(args.query_path, cache_dir=args.cache_data_dir)[args.split]
        corpus_data = load_dataset(args.index_path, cache_dir=args.cache_data_dir)['train']

        # {qid:qtext}
        query_id_map = {}
        for e in tqdm(query_data):
            query_id_map[e['query_id']] = e['query']

        logging.info(f"# query {len(query_id_map)}")

        # {docid: doctext}
        corpus_id_map = {}
        for e in tqdm(corpus_data):
            corpus_id_map[e['docid']] = e  # {"docid": 1, "title": 2, "text": 3}

        logging.info(f"# doc {len(corpus_id_map)}")


        tokenizer = AutoTokenizer.from_pretrained(
            'meta-llama/Llama-2-7b-hf',
            cache_dir=args.cache_model_dir,
            token=args.token
        )

        tokenizer.pad_token_id = tokenizer.unk_token_id
        tokenizer.pad_token = tokenizer.unk_token
        tokenizer.padding_side = "right"

        config = PeftConfig.from_pretrained('castorini/repllama-v1-7b-lora-passage')
        base_model = AutoModel.from_pretrained(
            config.base_model_name_or_path,
            cache_dir=args.cache_model_dir,
            token=args.token,
            device_map="auto",
        )
        model = PeftModel.from_pretrained(base_model, 'castorini/repllama-v1-7b-lora-passage')
        model = model.merge_and_unload()

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        model = model.to(device)
        model.eval()

        """
        queries = {}
        query_reader = open(args.query_path, 'r').readlines()
        for line in query_reader:
            qid, qtext = line.split('\t')
            queries[qid] = qtext

        corpus = {}
        with open(args.index_path, "r") as r:
            for line in tqdm(r):
                docid, text = line.strip().split("\t")
                corpus[docid] = text.replace("\t", " ").replace("\n", " ").replace("\r", " ")
        """


        for qid in tqdm(features.keys()):

            if qid not in qrels:
                continue

            #feature[qid] = {}

            query = query_id_map[qid]
            #query_input_ = tokenizer(f'query: {query}</s>', return_tensors='pt')
            query_input = tokenizer.encode('query: ' + query,
                             add_special_tokens=False,
                             max_length=args.q_max_len - 3,
                             truncation=True)

            query_input = tokenizer.prepare_for_model(query_input + [tokenizer.eos_token_id], max_length=args.q_max_len,
                                                      truncation='only_first', padding=False,
                                                      return_token_type_ids=False, return_tensors='pt')


            embedding=[]

            with torch.cuda.amp.autocast() if args.fp16 else nullcontext():
                with torch.no_grad():
                    query_input = {k: v.unsqueeze(0).to(device) for k, v in query_input.items()}
                    query_outputs = model(**query_input)
                    query_embedding = query_outputs.last_hidden_state[0][-1]
                    query_embedding = torch.nn.functional.normalize(query_embedding, p=2, dim=0)

            for idx, docid in enumerate(features[qid]["docid"]):
                title = corpus_id_map[docid]["title"]
                passage = corpus_id_map[docid]["text"]

                #passage_input_ = tokenizer(f'passage: {title} {passage}</s>', return_tensors='pt')
                text = title + ' ' + passage
                passage_input = tokenizer.encode('passage: ' + text,
                                     add_special_tokens=False,
                                     max_length=args.p_max_len-3,
                                     truncation=True)

                passage_input = tokenizer.prepare_for_model(passage_input + [tokenizer.eos_token_id],
                                                          max_length=args.p_max_len,
                                                          truncation='only_first', padding=False,
                                                          return_token_type_ids=False, return_tensors='pt')

                #print(passage_input)
                with torch.cuda.amp.autocast() if args.fp16 else nullcontext():
                    with torch.no_grad():
                        passage_input = {k: v.unsqueeze(0).to(device) for k, v in passage_input.items()}
                        passage_outputs = model(**passage_input)
                        passage_embeddings = passage_outputs.last_hidden_state[0][-1]
                        passage_embeddings = torch.nn.functional.normalize(passage_embeddings, p=2, dim=0)
                        q_doc_embed = torch.cat((query_embedding, passage_embeddings))

                        embedding.append(q_doc_embed.cpu().detach().numpy())

            score = np.array(features[qid]["score"])
            embedding = np.array(embedding)
            cos_avg_embedding = [0] + [
                cos(embedding[i], np.sum(softmax(score[:i], -1)[0].reshape(-1, 1) * embedding[:i], 0)) for i in
                range(1, len(features[qid]["docid"]))]

            features[qid]["embedding"] = embedding.tolist()
            features[qid]["cos_avg_embedding"] = cos_avg_embedding

            #print(embedding.shape)
            #print(cos_avg_embedding)
            #break

        #torch.save(feature,f"{os.path.join(args.output_path, args.setup)}.pkl")
        with open(f"{os.path.join(args.output_path, args.setup)}.json", "w") as w:
            w.write(json.dumps(features))

