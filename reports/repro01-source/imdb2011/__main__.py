import os
# Required by deterministic CUDA matrix multiplication; set before importing torch.
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description="Maas et al. 2011 IMDB reconstruction")
    sub = parser.add_subparsers(dest="command", required=True)
    download = sub.add_parser("download", help="Download official corpus and record SHA256")
    download.add_argument("--destination", default="data")
    train = sub.add_parser("train", help="Train word vectors without reading test data")
    train.add_argument("--data", default="data/aclImdb")
    train.add_argument("--config", default="configs/full_unsup.json")
    train.add_argument("--output", required=True)
    train.add_argument("--resume", action="store_true")
    fit = sub.add_parser("fit-svm", help="Select SVM C using training-only CV and refit")
    fit.add_argument("--data", default="data/aclImdb")
    fit.add_argument("--output", required=True)
    fit.add_argument("--features", choices=["bow", "delta", "vectors", "combined"], default="bow")
    fit.add_argument("--checkpoint")
    fit.add_argument("--bow-scope", choices=["5000", "all"], default="5000")
    fit.add_argument("--folds", type=int, default=5)
    fit.add_argument("--c-values", type=float, nargs="+", default=[0.01, 0.1, 1, 10, 100])
    fit.add_argument("--seed", type=int, default=42)
    evaluate = sub.add_parser("evaluate", help="Final official test evaluation; results persist")
    evaluate.add_argument("--data", default="data/aclImdb")
    evaluate.add_argument("--run", required=True)
    args = parser.parse_args()
    if args.command == "download":
        from .data import download as run
        print(run(args.destination))
    elif args.command == "train":
        from .train import train as run
        run(args.data, args.output, json.loads(Path(args.config).read_text(encoding="utf-8")), args.resume)
    elif args.command == "fit-svm":
        from .classify import fit_classifier
        fit_classifier(args.data, args.output, args.features, args.checkpoint,
                       args.bow_scope, args.folds, args.c_values, args.seed)
    else:
        from .classify import evaluate as run
        run(args.data, args.run)


if __name__ == "__main__":
    main()
