"""Cross-check saved predictions, metrics and frozen code provenance after a suite."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--prefix", required=True)
    parser.add_argument("--data", default="data/aclImdb", help="Official dataset directory")
    args = parser.parse_args()
    if not args.prefix.replace("-", "").replace("_", "").isalnum():
        parser.error("Invalid prefix")
    root = Path(__file__).resolve().parents[1]
    summary = json.loads((root / "reports" / f"{args.prefix}-results.json").read_text())
    if not summary["complete"]:
        raise RuntimeError("All eight final test results are required before auditing")
    frozen = json.loads((root / "reports" / f"{args.prefix}-source/source_hashes.json").read_text())
    for name, expected in frozen.items():
        assert digest(root / "reports" / f"{args.prefix}-source" / name) == expected, name
    for path in (root / "runs").glob(f"{args.prefix}-*/source_hashes.json"):
        observed = json.loads(path.read_text())
        assert all(observed.get(name) == value for name, value in frozen.items()), path
    labels = None
    test_hash = None
    checks = []
    for row in summary["results"]:
        path = root / row["test_evidence"]
        metric = json.loads(path.read_text())
        with np.load(path.parent / "test_predictions.npz") as saved:
            truth, pred = saved["labels"], saved["predictions"]
            assert truth.shape == pred.shape == (25000,), path
            assert np.array_equal(np.bincount(truth), [12500, 12500]), path
            assert np.isin(pred, [0, 1]).all(), path
            accuracy = float(np.mean(truth == pred))
            assert abs(accuracy - metric["accuracy"]) < 1e-12, path
            assert int(np.sum(truth == pred)) == metric["correct"], path
            matrix = [[int(np.sum((truth == a) & (pred == b))) for b in (0, 1)] for a in (0, 1)]
            assert matrix == metric["confusion_matrix"], path
            if labels is None:
                labels = truth.copy()
                test_hash = metric["test_sha256"]
            assert np.array_equal(labels, truth) and test_hash == metric["test_sha256"], path
            assert digest(path.parent / "classifier.joblib") == metric["classifier_sha256"], path
        checks.append({"name": row["name"], "accuracy": accuracy, "metrics_match_predictions": True})
    assert test_hash == digest(root / args.data / "test/labeledBow.feat")
    result = {"status": "PASS", "models_checked": len(checks), "n_test": 25000,
              "same_frozen_sources": True, "same_test_examples_and_labels": True,
              "checks": checks,
              "scope": "Artifact consistency only; does not prove equivalence to the author's implementation"}
    path = root / "reports" / f"{args.prefix}-audit.json"
    path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
