import numpy as np


class StubXGB:
    """Score each row as its word_count divided by 100."""

    def predict_proba(self, X):
        p = X["word_count"].to_numpy() / 100.0
        return np.column_stack([1 - p, p])
