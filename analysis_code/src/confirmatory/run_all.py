from __future__ import annotations
import argparse, subprocess, sys

STEPS=["integrity","freeze_protocol","select_models","run_family_benchmarks","run_clean_oof","run_orientation_augmentation","statistics","controls","run_secondary_grf","make_figures"]

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--quick",action="store_true")
    ap.add_argument("--skip-freeze",action="store_true")
    args=ap.parse_args()
    for step in STEPS:
        if step=="freeze_protocol" and args.skip_freeze:
            continue
        cmd=[sys.executable,"-m",f"src.confirmatory.{step}"]
        if args.quick and step in {"select_models","run_family_benchmarks","run_clean_oof","run_orientation_augmentation","statistics","run_secondary_grf"}:
            cmd.append("--quick")
        print("RUN"," ".join(cmd),flush=True)
        subprocess.run(cmd,check=True)
if __name__=="__main__":main()
