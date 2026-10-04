from __future__ import annotations

import argparse

import matplotlib.pyplot as plt
import pandas as pd

from .utils import project_root


def main():
    ap=argparse.ArgumentParser();ap.parse_args();root=project_root()
    stats=root/"results/confirmatory/statistics/robustness_summary.csv"
    if not stats.exists():raise SystemExit("Run statistics.py first")
    df=pd.read_csv(stats);out=root/"figures/confirmatory";out.mkdir(parents=True,exist_ok=True)
    for task in ("knee_two","ankle_two"):
        sub=df[df.task==task]
        fig,ax=plt.subplots(figsize=(8,5))
        for (regime,target),g in sub.groupby(["training_regime","target"]):
            g=g.sort_values("test_angle_deg")
            ax.plot(g.test_angle_deg,g.rmse,marker="o",label=f"{regime} | {target}")
        ax.set_xlabel("Test mounting misalignment (degrees)")
        ax.set_ylabel("Participant-level RMSE (degrees)")
        ax.set_title(f"Orientation robustness: {task}")
        ax.legend(fontsize=8)
        fig.tight_layout();fig.savefig(out/f"orientation_robustness_{task}.png",dpi=220);plt.close(fig)

    s=pd.read_csv(root/"results/confirmatory/statistics/augmentation_rescue_primary.csv")
    fig,ax=plt.subplots(figsize=(8,5))
    labels=s.target.str.replace("_deg","",regex=False)
    y=s.mean_rescue_rmse_deg
    yerr=[y-s.ci_low,s.ci_high-y]
    ax.errorbar(range(len(s)),y,yerr=yerr,fmt="o",capsize=4)
    ax.axhline(0,linewidth=1)
    ax.set_xticks(range(len(s)));ax.set_xticklabels(labels,rotation=30,ha="right")
    ax.set_ylabel("Augmentation rescue (RMSE degrees; positive is better)")
    fig.tight_layout();fig.savefig(out/"augmentation_rescue.png",dpi=220);plt.close(fig)

if __name__=="__main__":main()
