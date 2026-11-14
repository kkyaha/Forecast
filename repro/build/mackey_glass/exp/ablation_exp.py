"""Thực nghiệm bổ sung cho báo cáo (ngoài phạm vi bài báo gốc).

Chạy từ repro/build/mackey_glass/:
    PYTHONPATH=. python exp/ablation_exp.py --exp lookback --arch rnn --lookback 10 --seeds 0,1,2,3,4,42
    PYTHONPATH=. python exp/ablation_exp.py --exp teacher  --arch rnn --teacher_epochs 1 --seeds 0,1,2
    PYTHONPATH=. python exp/ablation_exp.py --exp alpha    --arch rnn --alpha 0.25 --seeds 0,1,2
    PYTHONPATH=. python exp/ablation_exp.py --exp diag     --arch rnn --seeds 42

Mỗi lần chạy ghi thêm một dòng vào results/<exp>.csv (một dòng / seed), nên có thể chạy
song song nhiều tiến trình với các cấu hình khác nhau. --exp diag lưu thêm dự đoán trên tập
test (chỉ số bin) vào results/diag_<arch>_seed<s>.npz.

Mọi cấu hình khác giữ đúng kịch bản chuẩn: H=5, 50 bin, alpha=0.5, T=4, 20 epoch, batch 1.
"""
import argparse
import csv
import os
import time
from pathlib import Path

import numpy as np

from base_exp import train_student_model

RESULTS = Path(__file__).resolve().parent.parent / "results"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp", required=True, choices=["lookback", "teacher", "alpha", "diag", "smooth"])
    ap.add_argument("--arch", default="rnn", choices=["rnn", "lstm", "gru"])
    ap.add_argument("--seeds", default="42")
    ap.add_argument("--horizon", type=int, default=5)
    ap.add_argument("--lookback", type=int, default=1)
    ap.add_argument("--alpha", type=float, default=0.5)
    ap.add_argument("--temperature", type=float, default=4)
    ap.add_argument("--teacher_epochs", type=int, default=None)
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--label_smoothing", type=float, default=0.0)
    args = ap.parse_args()

    RESULTS.mkdir(exist_ok=True)
    out_csv = RESULTS / f"{args.exp}.csv"
    fields = ["exp", "arch", "seed", "horizon", "lookback", "alpha", "temperature",
              "epochs", "teacher_epochs", "label_smoothing", "teacher_mse", "baseline_mse", "student_mse",
              "mse_reduction_pct", "seconds"]

    for s in [int(x) for x in args.seeds.split(",")]:
        t0 = time.time()
        r = train_student_model(
            student_horizon=args.horizon, alpha=args.alpha, num_bins=50,
            val_size=0.2, test_size=0.2, epochs=args.epochs,
            temperature=args.temperature, lookback_window=args.lookback,
            batch_size=1, seed=s, arch=args.arch,
            teacher_epochs=args.teacher_epochs, return_preds=(args.exp == "diag"),
            label_smoothing=args.label_smoothing,
        )
        row = {
            "exp": args.exp, "arch": args.arch, "seed": s, "horizon": args.horizon,
            "lookback": args.lookback, "alpha": args.alpha, "temperature": args.temperature,
            "epochs": args.epochs, "label_smoothing": args.label_smoothing,
            "teacher_epochs": args.teacher_epochs if args.teacher_epochs else args.epochs,
            "teacher_mse": r["teacher"], "baseline_mse": r["baseline"],
            "student_mse": r["student"],
            "mse_reduction_pct": (r["baseline"] - r["student"]) / r["baseline"] * 100,
            "seconds": round(time.time() - t0, 1),
        }
        new = not out_csv.exists()
        with open(out_csv, "a", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fields)
            if new:
                w.writeheader()
            w.writerow(row)
            f.flush(); os.fsync(f.fileno())
        if args.exp == "diag":
            p = r["preds"]
            np.savez(RESULTS / f"diag_{args.arch}_seed{s}.npz",
                     **{f"{k}_{n}": v for k, (a, b) in p.items() for n, v in (("pred", a), ("true", b))})
        print("ROW", row, flush=True)


if __name__ == "__main__":
    main()
