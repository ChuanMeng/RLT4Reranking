import json
from argparse import ArgumentParser
from datasets import load_dataset
from tqdm import tqdm

def read_result(path):
    retrieval_results = {}
    with open(path) as f:
        for line in f:
            if len(line.rstrip().split()) == 3:
                qid, pid, score = line.rstrip().split()
            if len(line.rstrip().split()) == 4:
                qid, pid, _, score = line.rstrip().split()
            else:
                qid, _, pid, _, score, _ = line.rstrip().split()

            if qid not in retrieval_results:
                retrieval_results[qid] = []

            retrieval_results[qid].append((pid, float(score)))

    return retrieval_results

parser = ArgumentParser()
parser.add_argument('--query_data_name', type=str) # Tevatron/msmarco-passage
parser.add_argument('--query_data_split', type=str) # dl19
parser.add_argument('--corpus_data_name', type=str) # Tevatron/msmarco-passage-corpus

parser.add_argument('--query_path', type=str)
parser.add_argument('--corpus_path', type=str)

parser.add_argument('--retrieval_results', type=str, required=True) # run.repllama.psg.dl19.txt
parser.add_argument('--output_path', type=str, required=True) # rerank_input.repllama.psg.dl19.jsonl
parser.add_argument('--depth', type=int, default=1000, required=False)
parser.add_argument('--cache_dir', type=str, required=False)

args = parser.parse_args()

if args.query_path is None:
    query_data = load_dataset(args.query_data_name, cache_dir=args.cache_dir)[args.query_data_split]
    corpus_data = load_dataset(args.corpus_data_name, cache_dir=args.cache_dir)['train']

    # {qid:qtext}
    query_id_map = {}
    for e in tqdm(query_data):
        query_id_map[e['query_id']] = e['query']

    print(f"# query {len(query_id_map)}")

    # {docid: doctext}
    corpus_id_map = {}
    for e in tqdm(corpus_data):
        corpus_id_map[e['docid']] = e  # {"docid": 1, "title": 2, "text": 3}

    # {qid:[(pid, socre), (pid, socre), ...]}
    retrieval_results = read_result(args.retrieval_results)

    with open(args.output_path, 'w') as f:
        for qid in tqdm(retrieval_results):
            if qid not in query_id_map:
                continue
            query = query_id_map[qid]
            pid_and_scores = retrieval_results[qid]  # [(pid, socre), (pid, socre), ...]
            for item in pid_and_scores[:args.depth]:
                pid, score = item
                psg_info = corpus_id_map[pid]
                psg_info['score'] = score
                psg_info['query_id'] = qid
                psg_info['query'] = query
                f.write(json.dumps(psg_info) + '\n')
                # {"docid": "3620986",
                # "title": "Robert Grey (musician)",
                # "text": "Robert Grey (born 21 April 1951 in Marefield, Leicestershire) is an English musician best known as the drummer for Wire. In 1973, Grey joined his first band, an R&B group called the Snakes, as vocalist. The Snakes released one single: Teenage Head.. After the group folded, Grey began teaching himself to drum. He has been Wire 's regular drummer since their start in 1976, using the stage name 'Robert Gotobed'.",
                # "score": 0.861,
                # "query_id": "1037798",
                # "query": "who is robert gray"}
else:
    retrieval_results = read_result(args.retrieval_results)

    query = {}
    query_reader = open(args.query_path, 'r').readlines()
    for line in query_reader:
        qid, qtext = line.split('\t')
        query[qid] = qtext.replace("\t", "").replace("\n", "").replace("\r", "")

    with open(args.corpus_path, 'r') as r:
        corpus = json.loads(r.read())

    with open(args.output_path, 'w') as f:
        for qid in tqdm(retrieval_results):
            pid_and_scores = retrieval_results[qid]  # [(pid, socre), (pid, socre), ...]
            for item in pid_and_scores[:args.depth]:
                pid, score = item
                psg_info = corpus[pid]
                psg_info["docid"]= pid
                psg_info['score'] = score
                psg_info['query_id'] = qid
                psg_info['query'] = query[qid]
                f.write(json.dumps(psg_info) + '\n')
