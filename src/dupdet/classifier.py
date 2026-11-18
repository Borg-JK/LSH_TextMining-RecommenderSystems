"""Logistic regression on the candidate pairs (Sect. 3.5)."""
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from .config import MIN_TRAIN_POS, PROB_GRID


def _f1(tp: int, n_pred: int, n_gold: int) -> float:
    """F1 with every gold pair that is not predicted counted as FN."""
    return 2 * tp / (n_pred + n_gold) if tp else 0.0


class PairClassifier:
    """Logistic regression on standardised features. The probability threshold
    maximises training F1 with the transitivity step applied (ties: the lowest
    threshold). Fallbacks: no positive training pair -> nothing is predicted;
    only positives -> everything is predicted."""

    def __init__(self):
        self.mode = "model"          # 'model', 'all' or 'none'
        self.threshold = 0.5

    def fit(self, X: np.ndarray, y: np.ndarray, n_gold: int, closure):
        """closure(prob, thresholds) -> ({t: (true pairs, predicted pairs)}, clusters)
        on the training pairs (transitivity.transitive_clusters)."""
        n_pos = int(y.sum())
        if n_pos < MIN_TRAIN_POS:
            self.mode = "none"
            return self
        if n_pos == len(y):
            self.mode = "all"
            return self
        self.scaler = StandardScaler().fit(X)
        self.model = LogisticRegression(max_iter=1000).fit(self.scaler.transform(X), y)
        prob = self.model.predict_proba(self.scaler.transform(X))[:, 1]
        scores, _ = closure(prob, PROB_GRID)
        best = -1.0
        for thr in PROB_GRID:
            f1 = _f1(*scores[float(thr)], n_gold)
            if f1 > best:
                best, self.threshold = f1, thr
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predicted probability per pair (fallbacks: 0 for 'none', 1 for 'all')."""
        if self.mode == "none" or len(X) == 0:
            return np.zeros(len(X))
        if self.mode == "all":
            return np.ones(len(X))
        return self.model.predict_proba(self.scaler.transform(X))[:, 1]
