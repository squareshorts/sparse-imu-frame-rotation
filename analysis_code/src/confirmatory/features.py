from __future__ import annotations

import numpy as np


def engineered_features(
    X: np.ndarray,
    sensor_ids: tuple[int, ...],
    durations: np.ndarray,
    speeds: np.ndarray,
) -> np.ndarray:
    """
    Frozen engineered representation based on the exploratory pipeline:
    5-sample temporal decimation, per-channel moments/extrema/RMS, and
    acceleration/gyro magnitude summaries, plus gait-cycle duration and speed.
    """
    parts = [durations[:, None].astype(np.float32), speeds[:, None].astype(np.float32)]
    for sensor in sensor_ids:
        a = X[:, sensor].astype(np.float32)  # [N,101,6]
        decimated = a[:, ::5, :].reshape(len(a), -1)
        stats = np.concatenate([
            a.mean(axis=1),
            a.std(axis=1),
            a.min(axis=1),
            a.max(axis=1),
            np.sqrt(np.mean(a * a, axis=1)),
        ], axis=1)
        amag = np.linalg.norm(a[..., :3], axis=2)
        gmag = np.linalg.norm(a[..., 3:], axis=2)
        mag_stats = np.column_stack([
            amag.mean(1), amag.std(1), amag.max(1), np.percentile(amag, 90, axis=1),
            gmag.mean(1), gmag.std(1), gmag.max(1), np.percentile(gmag, 90, axis=1),
        ]).astype(np.float32)
        parts.extend([decimated, stats, mag_stats])
    F = np.concatenate(parts, axis=1)
    if not np.isfinite(F).all():
        raise ValueError("Engineered features contain non-finite values")
    return F


def magnitude_only_features(
    X: np.ndarray,
    sensor_ids: tuple[int, ...],
    durations: np.ndarray,
    speeds: np.ndarray,
) -> np.ndarray:
    parts = [durations[:, None].astype(np.float32), speeds[:, None].astype(np.float32)]
    for sensor in sensor_ids:
        a = X[:, sensor].astype(np.float32)
        amag = np.linalg.norm(a[..., :3], axis=2)
        gmag = np.linalg.norm(a[..., 3:], axis=2)
        z = np.stack([amag, gmag], axis=2)
        parts.append(z[:, ::5, :].reshape(len(a), -1))
        parts.append(np.column_stack([
            amag.mean(1), amag.std(1), amag.min(1), amag.max(1), np.percentile(amag, 90, axis=1),
            gmag.mean(1), gmag.std(1), gmag.min(1), gmag.max(1), np.percentile(gmag, 90, axis=1),
        ]).astype(np.float32))
    return np.concatenate(parts, axis=1)
