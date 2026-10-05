import os
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import json
import numpy as np
import pytest
from scipy import sparse
import torch
from torch.nn import functional as F

from imdb2011.data import select_vocabulary, load_split
from imdb2011.model import WordModel, sentiment_statistics
from imdb2011.classify import ReviewFeatures, fit_classifier, evaluate
from imdb2011.train import train, validate_config


def test_aggregated_objective_matches_explicit_tokens_and_gradients():
    torch.manual_seed(7)
    model = WordModel(3, 2).double()
    counts = sparse.csr_matrix([[2, 1, 0], [0, 2, 1], [1, 0, 2], [3, 1, 0]])
    ratings = np.array([10, 7, 1, 0])  # unequal class counts; unlabeled excluded
    theta = torch.randn(4, 2, dtype=torch.float64)
    pos, neg = sentiment_statistics(counts, ratings)
    aggregated = model.semantic_nll(torch.tensor(counts.toarray(), dtype=torch.float64), theta)
    aggregated = aggregated + model.sentiment_nll(torch.tensor(pos), torch.tensor(neg))
    explicit = torch.zeros((), dtype=torch.float64)
    for d in range(4):
        probs = F.log_softmax(theta[d] @ model.words.T + model.bias, dim=0)
        for word in range(3):
            for _ in range(counts[d, word]):
                explicit = explicit - probs[word]
                if ratings[d] > 0:
                    target = torch.tensor((ratings[d] - 1) / 9, dtype=torch.float64)
                    class_count = 2 if ratings[d] >= 7 else 1
                    explicit = explicit + F.binary_cross_entropy_with_logits(
                        model.words[word] @ model.sentiment + model.sentiment_bias, target) / class_count
    assert torch.allclose(explicit, aggregated, atol=1e-10)
    gradients_a = torch.autograd.grad(aggregated, tuple(model.parameters()), retain_graph=True)
    gradients_b = torch.autograd.grad(explicit, tuple(model.parameters()))
    for a, b in zip(gradients_a, gradients_b):
        assert torch.allclose(a, b, atol=1e-10)


def test_document_features_use_binary_counts_then_normalize():
    vectors = np.array([[1., 0.], [0., 2.], [1., 1.]])
    x = sparse.csr_matrix([[100., 1., 0.], [0., 0., 0.]])
    transformer = ReviewFeatures("vectors", np.arange(3), vectors).fit(x, np.array([1, 0]))
    result = transformer.transform(x).toarray()
    np.testing.assert_allclose(result[0], [1 / np.sqrt(5), 2 / np.sqrt(5)])
    np.testing.assert_equal(result[1], [0, 0])


def test_vocabulary_uses_frequency_and_excludes_top_terms():
    x = sparse.csr_matrix([[10, 5, 5, 1, 0]])
    np.testing.assert_equal(select_vocabulary(x, 2, 1), [1, 2])


def fixture_corpus(root):
    (root / "train").mkdir(parents=True)
    (root / "imdb.vocab").write_text("a\nb\nc\nd\n", encoding="utf-8")
    rows = ["10 0:4 1:1", "9 0:3 2:1", "8 0:2 1:2", "7 0:3 3:1",
            "1 1:4 2:1", "2 1:3 3:1", "3 1:2 2:2", "4 0:1 1:3"]
    (root / "train/labeledBow.feat").write_text("\n".join(rows) + "\n", encoding="utf-8")
    (root / "train/unsupBow.feat").write_text("0 2:3 3:1\n0 0:1 3:4\n", encoding="utf-8")
    return rows


def test_training_resume_classifier_and_test_isolation(tmp_path):
    data = tmp_path / "data"
    rows = fixture_corpus(data)
    config = {"vocab_size": 3, "skip_top": 0, "dimension": 2, "outer_steps": 2,
              "theta_steps": 8, "word_steps": 5, "batch_size": 4, "dtype": "float64", "device": "cpu"}
    run = tmp_path / "words"
    checkpoint = train(data, run, config)
    records = [json.loads(line) for line in (run / "training.jsonl").read_text().splitlines()]
    assert records[1]["objective"] <= records[0]["objective"] + 1e-8
    train(data, run, config | {"outer_steps": 3}, resume=True)
    assert torch.load(checkpoint, weights_only=True)["iteration"] == 3
    classifier_run = tmp_path / "classifier"
    fit_classifier(data, classifier_run, "combined", checkpoint, folds=2, c_values=[0.1, 1])
    # Training and selection succeeded while there was NO test directory.
    (data / "test").mkdir()
    (data / "test/labeledBow.feat").write_text("\n".join(rows) + "\n", encoding="utf-8")
    result = evaluate(data, classifier_run)
    assert result["n_test"] == 8
    assert result["accuracy"] >= 0.75  # pipeline smoke test, not a scientific result
    with pytest.raises(FileExistsError):
        evaluate(data, classifier_run)
    with pytest.raises(ValueError, match="Resume config changed"):
        train(data, run, config | {"dimension": 4}, resume=True)


def test_delta_statistics_fit_only_on_training_rows():
    x = sparse.csr_matrix([[1, 0], [1, 1], [0, 1], [0, 1]])
    y = np.array([1, 1, 0, 0])
    transformer = ReviewFeatures("delta", np.arange(2)).fit(x, y)
    expected = np.log2([3., 2. / 3.])
    np.testing.assert_allclose(transformer.delta_, expected)
    before = transformer.delta_.copy()
    transformer.transform(sparse.csr_matrix([[100, 10]]))
    np.testing.assert_equal(transformer.delta_, before)


def test_zero_based_official_features_and_rating_validation(tmp_path):
    fixture_corpus(tmp_path / "data")
    x, ratings = load_split(tmp_path / "data", "train")
    assert x.shape == (8, 4)
    assert x[0, 0] == 4 and ratings[0] == 10


@pytest.mark.parametrize("config", [{"skip_top": 1.5}, {"seed": -1}, {"lambda_theta": float("nan")}])
def test_invalid_config_fails_before_loading_data(config):
    with pytest.raises(ValueError):
        validate_config(config)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
def test_cuda_full_softmax_backward():
    model = WordModel(5000, 50).cuda()
    theta = torch.randn(16, 50, device="cuda", requires_grad=True)
    counts = torch.zeros(16, 5000, device="cuda")
    counts[:, :20] = 1
    loss = model.semantic_nll(counts, theta) + theta.square().sum()
    loss.backward()
    assert torch.isfinite(model.words.grad).all()
    assert torch.isfinite(theta.grad).all()
