"""
Time-series cross-validation (rolling-origin / expanding-window) cho thực nghiệm Mackey-Glass.

Khác với k-fold chuẩn (xáo trộn ngẫu nhiên rồi chia k phần), script này KHÔNG xáo trộn dữ liệu:
mỗi fold dùng một PREFIX (đoạn đầu) ngày càng dài của chuỗi gốc, rồi vẫn áp dụng đúng logic
chia train/val/test hiện có của create_time_series_dataset (luôn lấy đoạn CUỐI của phần dữ
liệu được đưa vào làm test). Vì vậy, các fold có càng nhiều dữ liệu train (expanding window)
và tập test của mỗi fold nằm ở một đoạn càng về sau trên chuỗi thời gian -- đúng kiểu
rolling-origin cross-validation chuẩn cho time series (tương đương sklearn.TimeSeriesSplit),
không làm rò dữ liệu tương lai vào lúc huấn luyện.

Đây là cải tiến tự đề xuất, không có trong code hay paper gốc. Không sửa gì trong utils.py hay
logic huấn luyện của base_exp.py -- chỉ thêm tham số external_data (đã vá) để truyền vào một
prefix của data.pkl thay vì luôn đọc toàn bộ file.
"""
import argparse
import json
import pickle

from base_exp import train_student_model

FRACTIONS = [0.7, 0.85, 1.0]


def run(horizon, alpha, num_bins, epochs, val_size, test_size,
        lookback_window, batch_size, seed, archs):
    with open("data.pkl", "rb") as f:
        full_data = pickle.load(f)
    M = len(full_data)

    results = []
    for fold_id, frac in enumerate(FRACTIONS, start=1):
        cutoff = int(M * frac)
        fold_data = full_data[:cutoff]
        for arch in archs:
            print(f"\n=== Fold {fold_id}/{len(FRACTIONS)} (prefix {frac:.0%}, "
                  f"N={cutoff}) | arch={arch} ===")
            metrics = train_student_model(
                student_horizon=horizon,
                alpha=alpha,
                num_bins=num_bins,
                val_size=val_size,
                test_size=test_size,
                epochs=epochs,
                temperature=4,
                lookback_window=lookback_window,
                batch_size=batch_size,
                seed=seed,
                arch=arch,
                external_data=fold_data,
            )
            teacher, baseline, student = metrics["teacher"], metrics["baseline"], metrics["student"]
            drop_pct = (baseline - student) / baseline * 100.0
            results.append({
                "fold": fold_id,
                "prefix_frac": frac,
                "n_points": cutoff,
                "arch": arch,
                "teacher_mse": teacher,
                "baseline_mse": baseline,
                "student_mse": student,
                "drop_pct": drop_pct,
            })
            print(f"[Fold {fold_id} | {arch}] Giam MSE (Baseline->Student): {drop_pct:.2f}%")

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Rolling-origin time-series cross-validation cho Mackey-Glass (tu de xuat)"
    )
    parser.add_argument("--horizon", type=int, default=5)
    parser.add_argument("--alpha", type=float, default=0.5)
    parser.add_argument("--num_bins", type=int, default=50)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--val_size", type=float, default=0.2)
    parser.add_argument("--test_size", type=float, default=0.2)
    parser.add_argument("--lookback_window", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--archs", type=str, default="rnn,lstm,gru",
                         help="Danh sach kien truc, phan cach bang dau phay")
    parser.add_argument("--out", type=str, default="cv_results.json")
    args = parser.parse_args()

    archs = [a.strip() for a in args.archs.split(",") if a.strip()]
    results = run(
        horizon=args.horizon, alpha=args.alpha, num_bins=args.num_bins,
        epochs=args.epochs, val_size=args.val_size, test_size=args.test_size,
        lookback_window=args.lookback_window, batch_size=args.batch_size,
        seed=args.seed, archs=archs,
    )

    with open(args.out, "w") as f:
        json.dump(results, f, indent=2)

    print("\n\n=== TONG HOP ===")
    print(f"{'Fold':<6}{'Arch':<6}{'Teacher':>10}{'Baseline':>12}{'Student':>10}{'Giam MSE':>12}")
    for r in results:
        print(f"{r['fold']:<6}{r['arch']:<6}{r['teacher_mse']:>10.4f}"
              f"{r['baseline_mse']:>12.4f}{r['student_mse']:>10.4f}{r['drop_pct']:>11.2f}%")

    print(f"\nDa luu ket qua day du vao {args.out}")
