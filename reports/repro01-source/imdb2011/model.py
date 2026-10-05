"""Exact vocabulary softmax and count-aggregated Eq. (11)."""
import torch
from torch import nn
from torch.nn import functional as F


class WordModel(nn.Module):
    def __init__(self, vocabulary_size, dimension):
        super().__init__()
        # Rows are word vectors: transpose of R in the paper.
        self.words = nn.Parameter(torch.randn(vocabulary_size, dimension) * 0.01)
        self.bias = nn.Parameter(torch.zeros(vocabulary_size))
        self.sentiment = nn.Parameter(torch.randn(dimension) * 0.01)
        self.sentiment_bias = nn.Parameter(torch.zeros(()))

    def semantic_nll(self, counts, theta):
        logits = theta @ self.words.T + self.bias
        return -(counts * F.log_softmax(logits, dim=1)).sum()

    def sentiment_nll(self, positive_mass, negative_mass):
        logits = self.words @ self.sentiment + self.sentiment_bias
        return (positive_mass * F.softplus(-logits) +
                negative_mass * F.softplus(logits)).sum()


def sentiment_statistics(counts, ratings):
    """Exactly aggregate sum_k sum_i BCE(s_k, p_i) / |S_k|.

    Unlabeled rows have rating 0 and make no contribution. Star targets are
    (rating - 1) / 9; class sizes count documents, not tokens.
    """
    import numpy as np
    labeled = ratings > 0
    positive = labeled & (ratings >= 7)
    negative = labeled & (ratings <= 4)
    weights = np.zeros(len(ratings), dtype=np.float64)
    for mask in (positive, negative):
        if mask.any():
            weights[mask] = 1.0 / mask.sum()
    targets = np.where(labeled, (ratings - 1) / 9, 0)
    return (np.asarray(counts.T @ (weights * targets)).ravel(),
            np.asarray(counts.T @ (weights * (1 - targets))).ravel())
