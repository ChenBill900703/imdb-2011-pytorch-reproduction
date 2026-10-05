import json
import warnings
from pathlib import Path

import joblib
import numpy as np
from scipy import sparse
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.exceptions import ConvergenceWarning
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import normalize
from sklearn.svm import LinearSVC
import torch

from .data import binary, fingerprint, load_split, select_vocabulary, sha256


class ReviewFeatures(TransformerMixin, BaseEstimator):
    def __init__(self, kind="bow", vocab_ids=None, vectors=None, bow_scope="5000"):
        self.kind = kind
        self.vocab_ids = vocab_ids
        self.vectors = vectors
        self.bow_scope = bow_scope

    def fit(self, x, y=None):
        # All-column BoW still excludes terms unseen in this training fold.
        if self.bow_scope == "all":
            self.bow_ids_ = np.flatnonzero(np.asarray(x.sum(axis=0)).ravel() > 0)
        else:
            self.bow_ids_ = self.vocab_ids
        if self.kind == "delta":
            b = binary(x[:, self.bow_ids_])
            n_pos, n_neg = (y == 1).sum(), (y == 0).sum()
            pos = np.asarray(b[y == 1].sum(axis=0)).ravel()
            neg = np.asarray(b[y == 0].sum(axis=0)).ravel()
            # Add-one smoothed log document-frequency ratio. Fit within CV fold.
            self.delta_ = np.log2(((pos + 1) / (n_pos + 1)) /
                                 ((neg + 1) / (n_neg + 1)))
        return self

    def transform(self, x):
        if self.kind in {"bow", "delta", "combined"}:
            b = binary(x[:, self.bow_ids_])
            if self.kind == "delta":
                b = b.multiply(self.delta_).tocsr()
            b = normalize(b, norm="l2")
            if self.kind in {"bow", "delta"}:
                return b
        if self.vectors is None:
            raise ValueError("Word vectors required")
        # bnn before Rv; normalize only the resulting document vector.
        v = normalize(binary(x[:, self.vocab_ids]) @ self.vectors, norm="l2")
        if self.kind == "vectors":
            return sparse.csr_matrix(v)
        # Each block is individually normalized; no extra joint normalization.
        return sparse.hstack([b, sparse.csr_matrix(v)], format="csr")


def fit_classifier(root, output, kind, checkpoint=None, bow_scope="5000",
                   folds=5, c_values=(0.01, 0.1, 1, 10, 100), seed=42):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    if (output / "classifier.joblib").exists():
        raise FileExistsError("Classifier exists; use a new output directory")
    x, ratings = load_split(root, "train")
    y = (ratings >= 7).astype(np.int64)
    vectors = None
    checkpoint_hash = None
    if checkpoint:
        state = torch.load(checkpoint, map_location="cpu", weights_only=True)
        expected = fingerprint(root, state["config"]["variant"] == "full-unsup")
        if state["fingerprint"] != expected:
            raise ValueError("Word checkpoint does not match this training dataset")
        ids = state["vocab_ids"].numpy()
        vectors = state["model"]["words"].numpy()
        checkpoint_hash = sha256(checkpoint)
    else:
        if kind in {"vectors", "combined"}:
            raise ValueError("--checkpoint is required for vector features")
        ids = select_vocabulary(x, 5000, 50)
    pipeline = Pipeline([
        ("features", ReviewFeatures(kind, ids, vectors, bow_scope)),
        ("svm", LinearSVC(dual="auto", loss="squared_hinge", penalty="l2",
                          max_iter=20000, tol=1e-5, random_state=seed)),
    ])
    cv = StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed)
    search = GridSearchCV(pipeline, {"svm__C": list(c_values)}, scoring="accuracy", cv=cv,
                          n_jobs=1, error_score="raise", refit=True)
    with warnings.catch_warnings():
        warnings.simplefilter("error", ConvergenceWarning)
        search.fit(x, y)
    report = {
        "features": kind, "bow_scope": bow_scope, "folds": folds, "seed": seed,
        "best_C": search.best_params_["svm__C"], "cv_accuracy": search.best_score_,
        "candidates": [{"C": params["svm__C"], "mean": float(mean), "std": float(std)}
                       for params, mean, std in zip(search.cv_results_["params"],
                       search.cv_results_["mean_test_score"], search.cv_results_["std_test_score"])],
        "checkpoint_sha256": checkpoint_hash, "fingerprint": fingerprint(root),
        "cv_scope": "Classifier-only CV with frozen word vectors. If vectors used labels, "
                    "CV scores are conditional and NOT an unbiased end-to-end validation estimate.",
    }
    joblib.dump(search.best_estimator_, output / "classifier.joblib")
    (output / "selection.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)
    return report


def evaluate(root, run):
    run = Path(run)
    result_path = run / "test_metrics.json"
    if result_path.exists():
        raise FileExistsError("Test result already exists; read it instead of repeatedly evaluating")
    selection = json.loads((run / "selection.json").read_text(encoding="utf-8"))
    if selection["fingerprint"] != fingerprint(root):
        raise ValueError("Training corpus changed since classifier selection")
    # Only this explicit command reads the official test split.
    x, ratings = load_split(root, "test")
    y = (ratings >= 7).astype(np.int64)
    classifier = joblib.load(run / "classifier.joblib")  # Load own trusted local artifacts only.
    pred = classifier.predict(x)
    result = {"accuracy": accuracy_score(y, pred), "n_test": len(y),
              "correct": int((y == pred).sum()), "confusion_matrix": confusion_matrix(y, pred).tolist(),
              "classification_report": classification_report(y, pred, output_dict=True, zero_division=0),
              "test_sha256": sha256(Path(root) / "test/labeledBow.feat"),
              "classifier_sha256": sha256(run / "classifier.joblib"),
              "selection": selection}
    result_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    np.savez_compressed(run / "test_predictions.npz", labels=y, predictions=pred)
    print(json.dumps(result, indent=2), flush=True)
    return result
