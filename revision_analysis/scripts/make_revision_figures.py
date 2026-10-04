"""Make revised vector Figures 2 and 3 from the submitted source tables.

Only legend placement and the Figure 3 panel definitions change. Source data and
submitted figures are read-only.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "submitted_final/analysis/source_data"
FIG = ROOT / "results/revision_2026_10/figures"

mpl.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 9.5,
    "axes.titlesize": 10.5,
    "axes.labelsize": 9.5,
    "legend.fontsize": 8.5,
    "xtick.labelsize": 8.5,
    "ytick.labelsize": 8.5,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "axes.spines.top": False,
    "axes.spines.right": False,
})

METHODS = ["Unaugmented", "10-deg augmentation", "Template registration", "Gait-functional frame"]
COLORS = {
    "Unaugmented": "#4d4d4d",
    "10-deg augmentation": "#0072B2",
    "Template registration": "#D55E00",
    "Gait-functional frame": "#009E73",
}
MARKERS = {
    "Unaugmented": "o",
    "10-deg augmentation": "s",
    "Template registration": "D",
    "Gait-functional frame": "^",
}
ENDPOINTS = {
    "ankle_dorsiflex_deg": "Ankle\ndorsiflexion",
    "ankle_plantarflex_deg": "Ankle\nplantarflexion",
    "knee_stance_flex_deg": "Knee stance\nflexion",
    "knee_swing_flex_deg": "Knee swing\nflexion",
}


def save(fig: plt.Figure, stem: str) -> None:
    FIG.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG / f"{stem}.pdf", bbox_inches="tight", pad_inches=0.04)
    fig.savefig(FIG / f"{stem}.png", dpi=350, bbox_inches="tight", pad_inches=0.04)
    plt.close(fig)


def figure2() -> None:
    w = pd.read_csv(SOURCE / "virtual_worst_endpoint_matrix.csv")
    dep = pd.read_csv(SOURCE / "virtual_deployment_rule.csv")
    caps = sorted(w.train_augmentation_max_deg.unique())
    tests = sorted(w.test_rotation_deg.unique())
    mat = w.pivot(index="train_augmentation_max_deg", columns="test_rotation_deg", values="worst_ci_high_deg").loc[caps, tests].to_numpy()
    assert mat.shape == (7, 7)

    fig, axs = plt.subplots(1, 2, figsize=(11.2, 4.9), gridspec_kw={"width_ratios": [1.55, 0.9]})
    ax = axs[0]
    edges = np.arange(8) - 0.5
    ax.pcolormesh(edges, edges, mat, cmap="RdYlBu_r", vmin=1.5, vmax=8.5,
                  shading="flat", edgecolors="none", rasterized=False)
    ax.set_xlim(-0.5, 6.5)
    ax.set_ylim(-0.5, 6.5)
    for i, cap in enumerate(caps):
        for j, test in enumerate(tests):
            value = mat[i, j]
            ax.text(j, i, f"{value:.2f}", ha="center", va="center", fontsize=7.5,
                    color="white" if value > 6.3 else "black")
    selected = {
        float(r.field_q95_rotation_deg): float(r.selected_train_augmentation_max_deg)
        for _, r in dep[dep.decision == "DEPLOY"].iterrows()
    }
    for bound, cap in selected.items():
        i, j = caps.index(cap), tests.index(bound)
        ax.add_patch(Rectangle((j - 0.48, i - 0.48), 0.96, 0.96,
                               fill=False, linewidth=2.1, edgecolor="black"))
    ax.contour(np.arange(7), np.arange(7), mat, levels=[4.5], colors="black",
               linewidths=1.2, linestyles="--")
    ax.set_xticks(range(7), [f"{x:g}" for x in tests])
    ax.set_yticks(range(7), [f"{x:g}" for x in caps])
    ax.set_xlabel("Exact test rotation per sensor (deg)")
    ax.set_ylabel("Training augmentation cap (deg)")
    ax.set_title("A. Worst endpoint: upper 95% CI of RMSE")

    ax = axs[1]
    xx = dep.field_q95_rotation_deg.to_numpy(float)
    ok = dep.decision.eq("DEPLOY").to_numpy()
    yy = dep.selected_train_augmentation_max_deg.to_numpy(float)
    ax.plot(xx[ok], yy[ok], "-o", color="#0072B2", lw=2, ms=6, label="Selected cap")
    ax.scatter(xx[~ok], np.full((~ok).sum(), 14.3), marker="x", s=65,
               color="#D55E00", lw=2, label="No tested cap passed")
    for x, y in zip(xx[ok], yy[ok]):
        ax.annotate(f"{y:g} deg", (x, y), xytext=(0, 8), textcoords="offset points",
                    ha="center", fontsize=8)
    ax.set_xlim(-0.6, 15.6)
    ax.set_ylim(-0.6, 15.8)
    ax.set_xticks(tests)
    ax.set_yticks(tests)
    ax.set_xlabel("Field rotation bound (deg)")
    ax.set_ylabel("Selected training cap (deg)")
    ax.set_title("B. Fixed operating rule")
    ax.grid(axis="both", alpha=0.2)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.24), ncol=2,
              frameon=False, columnspacing=0.8, handlelength=1.3)
    fig.suptitle("Full virtual rotation matrix defines an operating envelope",
                 fontsize=12, weight="bold", y=0.98)
    fig.subplots_adjust(left=0.08, right=0.98, bottom=0.25, top=0.83, wspace=0.28)
    save(fig, "fig2_virtual_rotation_matrix_revision")


def method_frame(angle: float) -> pd.DataFrame:
    matrix = pd.read_csv(SOURCE / "virtual_rotation_matrix_summary.csv")
    baseline = pd.read_csv(SOURCE / "virtual_orientation_baselines.csv")
    rows = []
    for label, cap in [("Unaugmented", 0.0), ("10-deg augmentation", 10.0)]:
        q = matrix[(matrix.train_augmentation_max_deg == cap) & (matrix.test_rotation_deg == angle)].copy()
        q["method"] = label
        q = q.rename(columns={"ci_low": "lo", "ci_high": "hi", "n_subjects": "n"})
        rows.append(q[["method", "target", "mean_participant_rmse_deg", "lo", "hi", "n"]])
    for method, label in [("template_registration", "Template registration"),
                          ("gait_functional_calibration", "Gait-functional frame")]:
        q = baseline[(baseline.method == method) & (baseline.test_rotation_deg == angle)].copy()
        q["method"] = label
        q = q.rename(columns={"ci95_low_deg": "lo", "ci95_high_deg": "hi", "n_participants": "n"})
        rows.append(q[["method", "target", "mean_participant_rmse_deg", "lo", "hi", "n"]])
    out = pd.concat(rows, ignore_index=True)
    assert len(out) == 16 and (out.n == 138).all()
    return out


def figure3() -> None:
    zero = method_frame(0.0)
    rotated = method_frame(10.0)
    fig, axs = plt.subplots(1, 2, figsize=(11.2, 4.85))
    offsets = np.array([-0.27, -0.09, 0.09, 0.27])

    # Because the same 138 participants occur in both conditions, the
    # difference of participant means equals the mean paired increase.
    # Individual paired bootstrap intervals were not shipped for every method.
    for k, method in enumerate(METHODS):
        a = zero[zero.method == method].set_index("target").loc[list(ENDPOINTS)]
        b = rotated[rotated.method == method].set_index("target").loc[list(ENDPOINTS)]
        x = np.arange(4) + offsets[k]
        inc = b.mean_participant_rmse_deg.to_numpy() - a.mean_participant_rmse_deg.to_numpy()
        axs[0].plot(x, inc, linestyle="none", marker=MARKERS[method], ms=6,
                    color=COLORS[method], markeredgecolor="black", markeredgewidth=0.35,
                    label=method)
        y, lo, hi = (b[col].to_numpy() for col in ("mean_participant_rmse_deg", "lo", "hi"))
        axs[1].errorbar(x, y, yerr=np.vstack([y - lo, hi - y]),
                        fmt=MARKERS[method], capsize=2.3, lw=1.2, ms=5.2,
                        label=method, color=COLORS[method],
                        markeredgecolor="black", markeredgewidth=0.35)

    for ax in axs:
        ax.set_xticks(range(4), list(ENDPOINTS.values()))
        ax.grid(axis="y", alpha=0.2)
    axs[0].axhline(0, color="#555555", lw=1)
    axs[0].set_ylim(-0.4, 8.2)
    axs[0].set_ylabel("Paired mean RMSE increase vs 0 deg (deg)")
    axs[0].set_title("A. Rotation increase: 0 to 10 deg")
    axs[1].axhline(4.5, color="#555555", ls="--", lw=1)
    axs[1].set_ylim(0, 9.0)
    axs[1].set_ylabel("Absolute mean participant RMSE (deg)")
    axs[1].set_title("B. Absolute error at 10 deg")
    axs[1].legend(loc="upper center", bbox_to_anchor=(0.5, -0.24), ncol=2,
                  frameon=False, columnspacing=1.0, handlelength=1.3)
    fig.suptitle("Virtual accuracy and robustness under frame rotation",
                 fontsize=12, weight="bold", y=0.98)
    fig.subplots_adjust(left=0.08, right=0.98, bottom=0.25, top=0.83, wspace=0.24)
    save(fig, "fig3_virtual_orientation_methods_revision")


if __name__ == "__main__":
    figure2()
    figure3()
    print("Wrote revised vector Figures 2 and 3 to", FIG)
