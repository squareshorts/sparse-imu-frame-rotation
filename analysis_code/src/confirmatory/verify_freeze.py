from __future__ import annotations

from .utils import project_root, sha256_file


def main():
    root=project_root();freeze=root/"protocol/FROZEN.sha256"
    if not freeze.exists():
        raise SystemExit("Missing protocol/FROZEN.sha256")
    bad=[];checked=0
    for line in freeze.read_text(encoding="utf-8").splitlines():
        if not line.strip():continue
        expected,rel=line.split(maxsplit=1);path=root/rel.strip()
        if not path.exists():
            bad.append((rel,"MISSING",expected));continue
        actual=sha256_file(path);checked+=1
        if actual!=expected:bad.append((rel,actual,expected))
    if bad:
        for rel,actual,expected in bad:
            print(f"FAIL {rel}\n  expected {expected}\n  actual   {actual}")
        raise SystemExit(f"Frozen protocol verification failed for {len(bad)} file(s)")
    print(f"PASS: {checked} frozen files match protocol/FROZEN.sha256")

if __name__=="__main__":main()
