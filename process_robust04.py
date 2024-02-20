import argparse
import ir_datasets
import os
import json
import glob
import tqdm

def read_result(path):
    retrieval_results = {}

    with open(path) as f:
        for line in f:
            qid, _, pid, _, score, _ = line.rstrip().split()

            if qid not in retrieval_results:
                retrieval_results[qid] = []

            retrieval_results[qid].append((pid, float(score)))

    with open(path) as f:
        for line in f:
            name = line.rstrip().split()[-1]
            break
    return retrieval_results, name

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', type=str, required=True)
    #
    parser.add_argument('--query_output_path', type=str)
    parser.add_argument('--collection_output_path', type=str)
    #
    parser.add_argument('--run_path', type=str)
    #
    parser.add_argument('--fold_one_path', type=str)
    #
    parser.add_argument('--fold_one_pattern', type=str)
    args = parser.parse_args()

    if args.mode == "download":
        dataset = ir_datasets.load("disks45/nocr/trec-robust-2004")

        q_w = open(args.query_output_path, "w")

        for tuple in dataset.queries_iter():
            qid = tuple[0]
            title = tuple[1]
            description = tuple[2]
            q_w.write(f"{qid}\t{title}\n")

        q_w.close()

        # d_w = open(args.collection_output_path, "w")
        dataset = ir_datasets.load("disks45/nocr/trec-robust-2004")
        corpus = {}
        for tuple in dataset.docs_iter():
            docid = tuple[0]
            title = tuple[1]
            body = tuple[2]
            marked_up_doc = tuple[3]

            corpus[docid] = {}
            # corpus[docid]["docid"] = docid
            corpus[docid]["title"] = title
            corpus[docid]["text"] = body
            # d_w.write(f"{docid}\t{title}. {body}\n")
        # d_w.close()

        with open(args.collection_output_path, "w") as w:
            w.write(json.dumps(corpus))

    elif args.mode =="split_run":
        args.dataset_name = args.run_path.split("/")[-1].split(".")[0]
        retrieval_results, name = read_result(args.run_path)

        for fold_id in [1, 2, 3, 4, 5]:
            count = 0
            run_path_ = args.run_path.replace(f"{args.dataset_name}.", f"{args.dataset_name}-fold{fold_id}.")
            w = open(run_path_, "w")
            dataset = ir_datasets.load(f"disks45/nocr/trec-robust-2004/fold{fold_id}")
            for tuple in dataset.queries_iter():
                qid = tuple[0]
                count += 1
                for idx, (docid, score) in enumerate(retrieval_results[qid]):
                    rank = idx + 1
                    w.write(f'{qid} Q0 {docid} {rank} {score} {name}\n')
            print(fold_id, count)
            w.close()

    elif args.mode == "merge":

        for fold_ids in [[2,3,4,5], [1,3,4,5], [1,2,4,5], [1,2,3,5], [1,2,3,4]]:
            folds={}
            for fold_id in fold_ids:
                with open(args.fold_one_path.replace("-fold1",f"-fold{fold_id}"), 'r') as r:
                    fold = json.load(r)
                folds.update(fold)

            print(fold_ids,len(folds))
            name = "".join(list(map(str,fold_ids)))
            with open(args.fold_one_path.replace("-fold1",f"-fold{name}"), "w") as w:
                w.write(json.dumps(folds))


        folds={}
        for fold_id in [1,2,3,4,5]:
            with open(args.fold_one_path.replace("-fold1",f"-fold{fold_id}"), 'r') as r:
                fold = json.load(r)
            folds.update(fold)

        print("all folds:", len(folds))

        with open(args.fold_one_path.replace("-fold1",""), "w") as w:
            w.write(json.dumps(folds))

    elif args.mode == "merge_k":

        fold_ids_inference = ["1", "2", "3", "4", "5"]
        fold_ids_training = ["2345", "1345", "1245", "1235", "1234"]


        for fold_one_path in tqdm.tqdm(sorted(glob.glob(args.fold_one_pattern))):
            q2k = {}
            # merging
            for fold_id_inference, fold_id_training in zip(fold_ids_inference, fold_ids_training):
                fold_one_path_ = fold_one_path.replace("-fold1", f"-fold{fold_id_inference}")
                #print(fold_one_path_)
                if "ckpt" in fold_one_path_:
                    fold_one_path_ = fold_one_path_.replace("ckpt-robust04-fold2345", f"ckpt-robust04-fold{fold_id_training}")

                #print(fold_one_path_,"\n\n")


                with open(fold_one_path_, 'r') as r:
                    for line in r:
                        qid, k = line.rstrip().split("\t")
                        q2k[qid] = int(k)


            # write
            assert len(q2k) == 238
            fold_one_path_ = fold_one_path.replace("-fold1", "")
            if "ckpt" in fold_one_path_:
                fold_one_path_ = fold_one_path_.replace("ckpt-robust04-fold2345", "ckpt-robust04")

            #print(fold_one_path_)
            q2p_w = open(fold_one_path_, 'w')
            for index, qid in enumerate(q2k.keys()):
                q2p_w.write(qid + '\t' + str(q2k[qid]) + '\n')
            q2p_w.close()

