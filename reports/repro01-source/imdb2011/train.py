import json
import platform
import time
from pathlib import Path

import numpy as np
import scipy
from scipy import sparse
import torch

from .data import fingerprint, load_split, select_vocabulary, vocabulary
from .model import WordModel, sentiment_statistics


DEFAULTS = {
    "variant": "full-unsup", "dimension": 50, "vocab_size": 5000,
    "skip_top": 50, "vocab_source": "labeled", "lambda_theta": 1.0,
    "nu_words": 1.0, "sentiment_weight": 1.0, "outer_steps": 20,
    "theta_steps": 20, "word_steps": 10, "batch_size": 256,
    "history_size": 10, "relative_tolerance": 1e-5, "patience": 3,
    "seed": 42, "device": "auto", "dtype": "float32",
}


def validate_config(config):
    unknown = set(config) - set(DEFAULTS)
    if unknown:
        raise ValueError(f"Unknown config keys: {sorted(unknown)}")
    cfg = DEFAULTS | config
    if cfg["variant"] not in {"semantic", "full", "full-unsup"}:
        raise ValueError("variant must be semantic, full, or full-unsup")
    if cfg["vocab_source"] not in {"labeled", "all-training"}:
        raise ValueError("Invalid vocab_source")
    if cfg["dtype"] not in {"float32", "float64"}:
        raise ValueError("dtype must be float32 or float64")
    for key in ("dimension", "vocab_size", "outer_steps", "theta_steps", "word_steps",
                "batch_size", "history_size", "patience"):
        if not isinstance(cfg[key], int) or cfg[key] < 1:
            raise ValueError(f"{key} must be a positive integer")
    for key in ("skip_top", "lambda_theta", "nu_words", "sentiment_weight", "relative_tolerance"):
        if not np.isfinite(cfg[key]) or cfg[key] < 0:
            raise ValueError(f"Invalid {key}")
    if cfg["lambda_theta"] == 0:
        raise ValueError("lambda_theta must be positive for regularized MAP inference")
    if not isinstance(cfg["skip_top"], int):
        raise ValueError("skip_top must be an integer")
    if not isinstance(cfg["seed"], int) or not 0 <= cfg["seed"] < 2 ** 32:
        raise ValueError("seed must be an integer in [0, 2**32)")
    return cfg


def save_checkpoint(path, payload):
    path = Path(path)
    temporary = path.with_suffix(".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)


def train(root, output, config, resume=False):
    cfg = validate_config(config)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    checkpoint_path = output / "checkpoint.pt"
    if checkpoint_path.exists() and not resume:
        raise FileExistsError(f"{checkpoint_path} exists; use --resume or a new run directory")
    if resume and not checkpoint_path.exists():
        raise FileNotFoundError(checkpoint_path)
    device = cfg["device"]
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = getattr(torch, cfg["dtype"])
    torch.manual_seed(cfg["seed"])
    np.random.seed(cfg["seed"])
    torch.use_deterministic_algorithms(True)
    torch.set_float32_matmul_precision("highest")
    x, ratings = load_split(root, "train")
    n_labeled = x.shape[0]
    labeled_x = x
    if cfg["variant"] == "full-unsup":
        unlabeled_x, _ = load_split(root, "unsup")
        x = sparse.vstack([x, unlabeled_x], format="csr")
        ratings = np.concatenate([ratings, np.zeros(unlabeled_x.shape[0])])
    ids = select_vocabulary(labeled_x if cfg["vocab_source"] == "labeled" else x,
                            cfg["vocab_size"], cfg["skip_top"])
    x = x[:, ids].tocsr()
    provenance = fingerprint(root, cfg["variant"] == "full-unsup")
    n = x.shape[0]
    model = WordModel(len(ids), cfg["dimension"]).to(device=device, dtype=dtype)
    with torch.no_grad():
        frequency = torch.as_tensor(np.asarray(x.sum(axis=0)).ravel(), device=device, dtype=dtype)
        model.bias.copy_(torch.log(frequency.clamp_min(1) / frequency.sum()))
    theta = torch.randn(n, cfg["dimension"], dtype=dtype) * 0.01  # CPU, one row per document
    pos, neg = sentiment_statistics(x, ratings)
    pos = torch.as_tensor(pos, device=device, dtype=dtype)
    neg = torch.as_tensor(neg, device=device, dtype=dtype)
    sentiment_weight = 0.0 if cfg["variant"] == "semantic" else cfg["sentiment_weight"]
    start, previous, stale = 0, None, 0
    if resume:
        state = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        allowed_changes = {"outer_steps", "device"}
        if any(state["config"][k] != cfg[k] for k in cfg if k not in allowed_changes):
            raise ValueError("Resume config changed; only outer_steps and device may change")
        if state["fingerprint"] != provenance:
            raise ValueError("Training data changed since checkpoint")
        model.load_state_dict(state["model"])
        theta = state["theta"].to(dtype=dtype)
        start, previous, stale = state["iteration"], state["objective"], state["stale"]
    meta = {"config": cfg, "fingerprint": provenance, "labeled_rows": n_labeled,
            "semantic_rows": n, "empty_rows": int((np.asarray(x.sum(axis=1)).ravel() == 0).sum()),
            "python": platform.python_version(), "torch": str(torch.__version__),
            "numpy": np.__version__, "scipy": scipy.__version__, "device": str(device),
            "gpu": torch.cuda.get_device_name() if str(device).startswith("cuda") else None}
    (output / "metadata.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    words = vocabulary(root)
    (output / "vocabulary.json").write_text(json.dumps([words[i] for i in ids], ensure_ascii=False), encoding="utf-8")

    def batches():
        for offset in range(0, n, cfg["batch_size"]):
            end = min(offset + cfg["batch_size"], n)
            counts = torch.as_tensor(x[offset:end].toarray(), device=device, dtype=dtype)
            yield offset, end, counts

    for iteration in range(start, cfg["outer_steps"]):
        begun = time.perf_counter()
        # Word phase: accumulated FULL-corpus gradients; no per-batch Adam approximation.
        model.requires_grad_(True)
        optimizer = torch.optim.LBFGS(model.parameters(), max_iter=cfg["word_steps"],
                                     history_size=cfg["history_size"], line_search_fn="strong_wolfe",
                                     tolerance_grad=1e-7, tolerance_change=1e-9)
        def word_closure():
            optimizer.zero_grad()
            total = torch.zeros((), device=device, dtype=dtype)
            for lo, hi, counts in batches():
                loss = model.semantic_nll(counts, theta[lo:hi].to(device)) / n
                loss.backward()
                total += loss.detach()
            # Divide the ENTIRE objective by n, preserving Eq. (11)'s relative weights.
            loss = (cfg["nu_words"] * model.words.square().sum() +
                    sentiment_weight * model.sentiment_nll(pos, neg)) / n
            loss.backward()
            total += loss.detach()
            if not torch.isfinite(total):
                raise FloatingPointError("Non-finite word objective")
            return total
        optimizer.step(word_closure)

        # MAP phase: independent convex document problems, optimized in blocks.
        model.requires_grad_(False)
        for lo, hi, counts in batches():
            local_theta = theta[lo:hi].to(device).clone().requires_grad_()
            solver = torch.optim.LBFGS([local_theta], max_iter=cfg["theta_steps"],
                                      history_size=cfg["history_size"], line_search_fn="strong_wolfe",
                                      tolerance_grad=1e-7, tolerance_change=1e-9)
            def theta_closure():
                solver.zero_grad()
                loss = (model.semantic_nll(counts, local_theta) +
                        cfg["lambda_theta"] * local_theta.square().sum()) / (hi - lo)
                if not torch.isfinite(loss):
                    raise FloatingPointError("Non-finite MAP objective")
                loss.backward()
                return loss
            solver.step(theta_closure)
            theta[lo:hi] = local_theta.detach().cpu()
        with torch.no_grad():
            semantic = sum(float(model.semantic_nll(counts, theta[lo:hi].to(device)))
                           for lo, hi, counts in batches())
            sentiment = float(model.sentiment_nll(pos, neg))
            regularizer = (cfg["nu_words"] * float(model.words.square().sum()) +
                           cfg["lambda_theta"] * float(theta.square().sum()))
        objective = (semantic + sentiment_weight * sentiment + regularizer) / n
        if not np.isfinite(objective):
            raise FloatingPointError("Non-finite complete objective")
        relative = None if previous is None else (previous - objective) / max(abs(previous), 1e-12)
        if relative is not None and relative < -1e-5:
            raise RuntimeError("Alternating optimization increased loss; inspect numerical stability")
        stale = stale + 1 if relative is not None and abs(relative) < cfg["relative_tolerance"] else 0
        record = {"iteration": iteration + 1, "objective": objective, "semantic_nll": semantic,
                  "sentiment_nll": sentiment, "regularizer": regularizer,
                  "relative_improvement": relative, "seconds": time.perf_counter() - begun,
                  "converged": stale >= cfg["patience"]}
        with (output / "training.jsonl").open("a", encoding="utf-8") as log:
            log.write(json.dumps(record) + "\n")
        print(json.dumps(record), flush=True)
        save_checkpoint(checkpoint_path, {"model": {k: v.detach().cpu() for k, v in model.state_dict().items()},
                        "theta": theta, "vocab_ids": torch.tensor(ids), "config": cfg,
                        "fingerprint": provenance, "iteration": iteration + 1,
                        "objective": objective, "stale": stale, "converged": record["converged"]})
        previous = objective
        if record["converged"]:
            break
    return checkpoint_path
