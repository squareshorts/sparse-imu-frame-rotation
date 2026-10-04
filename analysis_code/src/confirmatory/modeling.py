from __future__ import annotations

import itertools
from typing import Iterator

import numpy as np

from .features import engineered_features
from .tabular_models import TabularWrapper
from .tcn import TCNRegressor, prepare_sequence


def grid_ridge(cfg: dict) -> Iterator[dict]:
    for a in cfg["alpha"]:
        yield {"alpha": float(a)}


def grid_histgb(cfg: dict, seed: int) -> Iterator[dict]:
    for lr, leaves, l2 in itertools.product(
        cfg["learning_rate"], cfg["max_leaf_nodes"], cfg["l2_regularization"]
    ):
        yield {
            "learning_rate": float(lr),
            "max_leaf_nodes": int(leaves),
            "l2_regularization": float(l2),
            "max_iter": int(cfg.get("max_iter", 300)),
            "random_state": int(seed),
        }


def grid_tcn(cfg: dict, quick: bool = False) -> Iterator[dict]:
    for lr, wd in itertools.product(cfg["learning_rate"], cfg["weight_decay"]):
        p = dict(cfg)
        p["learning_rate"] = float(lr)
        p["weight_decay"] = float(wd)
        if quick:
            p["max_epochs"] = 3
            p["patience"] = 2
        yield p


def fit_predict_family(
    family: str,
    params: dict,
    X_train: np.ndarray,
    y_train: np.ndarray,
    dur_train: np.ndarray,
    speed_train: np.ndarray,
    X_test: np.ndarray,
    dur_test: np.ndarray,
    speed_test: np.ndarray,
    sensor_ids: tuple[int, ...],
    seed: int,
    val: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray] | None = None,
):
    if family in ("ridge", "histgb"):
        Ftr = engineered_features(X_train, sensor_ids, dur_train, speed_train)
        Fte = engineered_features(X_test, sensor_ids, dur_test, speed_test)
        model = TabularWrapper(family, params).fit(Ftr, y_train)
        return model.predict(Fte), model
    if family == "tcn":
        S_tr = prepare_sequence(X_train, sensor_ids)
        scal_tr = np.column_stack([dur_train, speed_train]).astype(np.float32)
        S_te = prepare_sequence(X_test, sensor_ids)
        scal_te = np.column_stack([dur_test, speed_test]).astype(np.float32)
        val_data = None
        if val is not None:
            Xv, yv, dv, sv, _ = val
            val_data = (
                prepare_sequence(Xv, sensor_ids),
                np.column_stack([dv, sv]).astype(np.float32),
                yv,
            )
        model = TCNRegressor(params, seed=seed).fit(S_tr, scal_tr, y_train, val_data=val_data)
        return model.predict(S_te, scal_te), model
    raise ValueError(f"Unknown family {family}")
