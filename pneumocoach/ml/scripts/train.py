"""Generate the cohort, train, quantise, report.

Run:  python scripts/train.py [--subjects 60] [--seed 0]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path as _P
sys.path.insert(0, str(_P(__file__).resolve().parents[2] / "tools"))
import _consola  # noqa: E402,F401  UTF-8 en Windows
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pneumocoach import dataset, train  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--subjects", type=int, default=60)
    ap.add_argument("--bouts", type=int, default=14)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=Path, default=Path(__file__).resolve().parents[1] / "artifacts")
    args = ap.parse_args()

    t0 = time.time()
    print(f"simulating {args.subjects} subjects ...")
    ds = dataset.generate(n_subjects=args.subjects, seed=args.seed, bouts=args.bouts)
    print(f"  {len(ds)} windows in {time.time() - t0:.1f}s   {ds.class_counts()}")

    train.run(ds, args.out, seed=args.seed)
    print(f"\nartifacts written to {args.out}")


if __name__ == "__main__":
    main()
