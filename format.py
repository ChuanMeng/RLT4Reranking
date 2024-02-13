import argparse
import os

def translate(source, target, name):
    with open(source, "r") as r, open(target, "w") as w:
        for line in r.readlines():
            qid, did, rank, score = line.split()
            w.write(f"{qid} Q0 {did} {rank} {score} {name}\n")

    return None

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_path', type=str, required=True)
    parser.add_argument('--output_path', type=str, required=True)
    parser.add_argument('--ranker_name', type=str, required=True)
    args = parser.parse_args()

translate(args.input_path,args.output_path, args.ranker_name)


