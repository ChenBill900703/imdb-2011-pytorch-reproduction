"""Read the author's zero-based .feat files; never fit on test documents."""
import hashlib
import json
import tarfile
import urllib.request
from pathlib import Path

import numpy as np
from sklearn.datasets import load_svmlight_file

URL = "https://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz"


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def download(destination):
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=True)
    archive = destination / "aclImdb_v1.tar.gz"
    if not archive.exists():
        temporary = archive.with_suffix(".part")
        print(f"Downloading {URL}", flush=True)
        with urllib.request.urlopen(URL, timeout=120) as response, temporary.open("wb") as out:
            while block := response.read(1024 * 1024):
                out.write(block)
        temporary.replace(archive)
    root = (destination / "aclImdb").resolve()
    with tarfile.open(archive, "r:gz") as tar:
        # Extract only regular data files and directories under the expected root.
        for member in tar.getmembers():
            target = (destination / member.name).resolve()
            if not target.is_relative_to(root) or not (member.isfile() or member.isdir()):
                raise ValueError(f"Unsafe archive member: {member.name}")
        tar.extractall(destination, filter="data")
    manifest = {"url": URL, "sha256": sha256(archive)}
    (destination / "download.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return root


def vocabulary(root):
    return (Path(root) / "imdb.vocab").read_text(encoding="utf-8").splitlines()


def load_split(root, split):
    names = {"train": "train/labeledBow.feat", "unsup": "train/unsupBow.feat",
             "test": "test/labeledBow.feat"}
    path = Path(root) / names[split]
    x, ratings = load_svmlight_file(str(path), n_features=len(vocabulary(root)),
                                    zero_based=True, dtype=np.float32)
    x = x.tocsr()
    x.sum_duplicates()
    if not np.isfinite(x.data).all() or np.any(x.data < 0):
        raise ValueError("Invalid token counts")
    if split != "unsup" and not np.isin(ratings, [1, 2, 3, 4, 7, 8, 9, 10]).all():
        raise ValueError("Expected original 1–10 polarized ratings, not binary labels")
    return x, ratings


def select_vocabulary(x, size, skip):
    frequencies = np.asarray(x.sum(axis=0)).ravel()
    order = np.lexsort((np.arange(x.shape[1]), -frequencies))
    order = order[frequencies[order] > 0]
    ids = order[skip:skip + size]
    if len(ids) != size:
        raise ValueError(f"Only {len(ids)} eligible words; requested {size}")
    return ids


def binary(x):
    result = x.copy().tocsr()
    result.eliminate_zeros()
    result.data[:] = 1
    return result


def fingerprint(root, include_unsup=False):
    files = ["imdb.vocab", "train/labeledBow.feat"]
    if include_unsup:
        files.append("train/unsupBow.feat")
    return {name: sha256(Path(root) / name) for name in files}
