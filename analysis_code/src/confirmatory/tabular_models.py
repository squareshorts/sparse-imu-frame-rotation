from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.multioutput import MultiOutputRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


@dataclass
class TabularWrapper:
    family: str
    params: dict
    model: object | None = None

    def fit(self, X: np.ndarray, Y: np.ndarray):
        if self.family == "ridge":
            self.model = Pipeline([
                ("scale", StandardScaler()),
                ("model", Ridge(alpha=float(self.params["alpha"]))),
            ])
        elif self.family == "histgb":
            base = HistGradientBoostingRegressor(
                learning_rate=float(self.params["learning_rate"]),
                max_leaf_nodes=int(self.params["max_leaf_nodes"]),
                l2_regularization=float(self.params["l2_regularization"]),
                max_iter=int(self.params.get("max_iter", 300)),
                random_state=int(self.params.get("random_state", 0)),
            )
            self.model = MultiOutputRegressor(base, n_jobs=1)
        else:
            raise ValueError(f"Unknown tabular family {self.family}")
        self.model.fit(X, Y)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        return np.asarray(self.model.predict(X), dtype=np.float32)
