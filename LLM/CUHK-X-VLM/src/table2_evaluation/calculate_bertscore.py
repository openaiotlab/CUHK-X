import pandas as pd
from bert_score import score
import argparse


def main(args):
    df = pd.read_csv(args.file_path)
    references = []
    candidates = []
    if args.type == "com-cap":
        references = df["ground_truth"].dropna().tolist()
        candidates = df["ground_truth"].dropna().tolist()
    elif args.type == "free-act":
        references = df["ground_truth"].dropna().tolist()
        candidates = df["free_act"].dropna().tolist()
    elif args.type == "instruct-act":
        references = df["ground_truth"].dropna().tolist()
        candidates = df["instruct_act"].dropna().tolist()

    P, R, F1 = score(candidates, references, lang="zh", verbose=True)
    print("P:", P.mean().item())
    print("R:", R.mean().item())
    print("F1:", F1.mean().item())

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--file_path", type=str, default="result.csv")
    parser.add_argument("--type", type=str, choices=["com-cap", "free-act", "instruct-act"])
    args = parser.parse_args()
    main(args)