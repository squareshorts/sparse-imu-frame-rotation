from __future__ import annotations

import numpy as np


def axis_angle_to_matrix(axis: np.ndarray, angle_rad: float) -> np.ndarray:
    axis = np.asarray(axis, dtype=np.float64)
    norm = np.linalg.norm(axis)
    if not np.isfinite(norm) or norm == 0:
        raise ValueError("Rotation axis must have nonzero finite norm")
    x, y, z = axis / norm
    c, s = np.cos(angle_rad), np.sin(angle_rad)
    C = 1.0 - c
    return np.array([
        [c + x*x*C, x*y*C - z*s, x*z*C + y*s],
        [y*x*C + z*s, c + y*y*C, y*z*C - x*s],
        [z*x*C - y*s, z*y*C + x*s, c + z*z*C],
    ], dtype=np.float64)


def random_axis(rng: np.random.Generator) -> np.ndarray:
    a = rng.normal(size=3)
    return a / np.linalg.norm(a)


def rotate_sensor_block(block: np.ndarray, R: np.ndarray) -> np.ndarray:
    """Rotate local sensor axes for [N, T, 6] data, preserving vector magnitudes."""
    out = block.copy()
    out[..., :3] = block[..., :3] @ R.T
    out[..., 3:] = block[..., 3:] @ R.T
    return out


def perturb_by_subject(
    X: np.ndarray,
    subjects: np.ndarray,
    sensor_ids: tuple[int, ...],
    max_angle_deg: float,
    seed: int,
    exact_angle: bool,
):
    """
    Apply participant-consistent static mounting rotations.

    If exact_angle=True, every sensor rotation has magnitude max_angle_deg.
    Otherwise angle magnitude is sampled Uniform(0, max_angle_deg).
    Each participant and sensor gets an independent axis/angle.
    """
    out = X.copy()
    rng = np.random.default_rng(seed)
    records: list[dict] = []
    for sid in np.unique(subjects):
        idx = np.flatnonzero(subjects == sid)
        for sensor in sensor_ids:
            axis = random_axis(rng)
            angle_deg = float(max_angle_deg if exact_angle else rng.uniform(0.0, max_angle_deg))
            R = axis_angle_to_matrix(axis, np.deg2rad(angle_deg))
            out[idx, sensor] = rotate_sensor_block(out[idx, sensor], R)
            records.append({
                "subject": int(sid), "sensor": int(sensor), "seed": int(seed),
                "axis_x": float(axis[0]), "axis_y": float(axis[1]), "axis_z": float(axis[2]),
                "angle_deg": angle_deg,
            })
    return out, records


def make_augmented_training_set(
    X: np.ndarray,
    Y: np.ndarray,
    subjects: np.ndarray,
    durations: np.ndarray,
    speeds: np.ndarray,
    sensor_ids: tuple[int, ...],
    max_angle_deg: float,
    copies: int,
    seed: int,
):
    """Return clean data plus deterministic participant-consistent augmented copies."""
    if max_angle_deg <= 0 or copies <= 0:
        return X, Y, subjects, durations, speeds
    xs = [X]
    ys = [Y]
    gs = [subjects]
    ds = [durations]
    ss = [speeds]
    for k in range(copies):
        Xa, _ = perturb_by_subject(
            X, subjects, sensor_ids, max_angle_deg,
            seed=seed + 10007 * (k + 1), exact_angle=False,
        )
        xs.append(Xa)
        ys.append(Y)
        gs.append(subjects)
        ds.append(durations)
        ss.append(speeds)
    return (
        np.concatenate(xs), np.concatenate(ys), np.concatenate(gs),
        np.concatenate(ds), np.concatenate(ss),
    )


def assert_magnitude_preservation(original: np.ndarray, rotated: np.ndarray, atol: float = 1e-5) -> None:
    for sl in (slice(0, 3), slice(3, 6)):
        a = np.linalg.norm(original[..., sl], axis=-1)
        b = np.linalg.norm(rotated[..., sl], axis=-1)
        if not np.allclose(a, b, atol=atol, rtol=1e-5):
            raise AssertionError("Pure orientation rotation changed vector magnitudes")
