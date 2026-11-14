"""Tổng hợp kết quả exp/ablation_exp.py và các baseline đơn giản (không học sâu).

Chạy từ repro/build/mackey_glass/:  PYTHONPATH=. python exp/analyze_ablation.py
In bảng ra màn hình và lưu hình vào docs/mau-latex-tieu-luan/hinh/ (nếu tìm thấy).
"""
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent.parent
RES = HERE / "results"

# Kết quả đã có trong báo cáo (lookback = 1, alpha = 0.5, T = 4, đủ 20 epoch teacher)
SEED_SWEEP_L1 = {
    "rnn":  {0: (4.8331, 156.1701, 102.7514), 1: (5.4642, 136.5673, 106.0795), 2: (5.7293, 143.5308, 104.4262),
             3: (6.2236, 140.3592, 107.6203), 4: (7.4506, 147.5238, 108.6868), 42: (5.4391, 139.3832, 104.3942)},
    "lstm": {0: (4.1850, 145.4412, 92.5198), 1: (4.6782, 138.9365, 89.1096), 2: (4.6281, 142.7424, 94.6453),
             3: (4.1689, 139.3032, 92.2201), 4: (4.7799, 143.8094, 98.7154), 42: (4.4145, 144.6108, 97.3967)},
    "gru":  {0: (4.4406, 143.1596, 98.4052), 1: (4.5018, 142.7224, 98.0985), 2: (4.2662, 140.9470, 94.1331),
             3: (4.2195, 146.1191, 97.3077), 4: (4.4276, 143.0985, 87.6893), 42: (4.7373, 142.0975, 98.7189)},
}


def find_hinh():
    for a in Path(__file__).resolve().parents:
        c = a / "docs" / "mau-latex-tieu-luan" / "hinh"
        if c.is_dir():
            return c
    return HERE / "results"


def ci95(x):
    x = np.asarray(x, float)
    m, se = x.mean(), x.std(ddof=1) / np.sqrt(len(x))
    h = se * stats.t.ppf(0.975, len(x) - 1)
    return m, m - h, m + h


# ---------- 1. Baseline đơn giản, cùng thang đo (chỉ số bin) ----------
def simple_baselines(H=5, nb=50):
    d = np.asarray(pickle.load(open(HERE / "data.pkl", "rb")))
    x, y = d[:, 0], d[:, 1]
    rows = []
    for L in (1, 10):
        n = len(x) - L - H + 1
        X = np.stack([x[i:i + L] for i in range(n)])
        Y = np.array([y[i + L + H - 1] for i in range(n)])
        nt = int(n * .2); ntr = n - 2 * nt
        e = np.linspace(Y[:ntr].min(), Y[:ntr].max(), nb - 1)
        Xb, Yb = np.digitize(X, e), np.digitize(Y, e)
        Xtr, Xte, Ytr, Yte = Xb[:ntr], Xb[-nt:], Yb[:ntr], Yb[-nt:]
        mse = lambda p: float(np.mean((p - Yte) ** 2))
        rows.append(("Trung bình tập huấn luyện", L, mse(np.full(len(Yte), Ytr.mean()))))
        rows.append(("Persistence (giữ giá trị hiện tại)", L, mse(Xte[:, -1].astype(float))))
        A = np.c_[Xtr, np.ones(len(Xtr))].astype(float)
        w = np.linalg.lstsq(A, Ytr.astype(float), rcond=None)[0]
        rows.append(("Hồi quy tuyến tính (AR)", L, mse(np.c_[Xte, np.ones(len(Xte))].astype(float) @ w)))
        bw = (Y[:ntr].max() - Y[:ntr].min()) / (nb - 2)
    return pd.DataFrame(rows, columns=["method", "lookback", "mse_bin"]), bw, float(y.std())


def main():
    hinh = find_hinh()
    sb, bw, ystd = simple_baselines()
    print(f"Độ rộng 1 bin = {bw:.5f}; độ lệch chuẩn chuỗi = {ystd:.4f} (= {ystd / bw:.2f} bin)")
    print(sb.to_string(index=False))

    # ---------- 2. Chẩn đoán dự đoán ----------
    for f in sorted(RES.glob("diag_*.npz")):
        z = np.load(f)
        print(f"\n[{f.name}]")
        for m in ("teacher", "baseline", "student"):
            p, t = z[f"{m}_pred"], z[f"{m}_true"]
            r = np.corrcoef(p, t)[0, 1] if p.std() > 0 else float("nan")
            vals, cnt = np.unique(p, return_counts=True)
            print(f"  {m:8s}: MSE={np.mean((p - t) ** 2):7.2f}  std(pred)={p.std():5.2f}  "
                  f"std(true)={t.std():5.2f}  corr={r:5.3f}  #bin dự đoán={len(vals):2d}  "
                  f"bin phổ biến nhất chiếm {cnt.max() / len(p):.0%}")

    # ---------- 3. Lookback ----------
    lb = pd.read_csv(RES / "lookback.csv") if (RES / "lookback.csv").exists() else None
    if lb is not None:
        print("\n=== Lookback 1 vs 10 (6 seed) ===")
        for a in ("rnn", "lstm", "gru"):
            d10 = lb[lb.arch == a].sort_values("seed")
            if d10.empty:
                continue
            seeds = d10.seed.tolist()
            l1 = np.array([SEED_SWEEP_L1[a][s] for s in seeds])
            red1 = (l1[:, 1] - l1[:, 2]) / l1[:, 1] * 100
            m, lo, hi = ci95(d10.mse_reduction_pct)
            pt = stats.ttest_rel(d10.baseline_mse, d10.student_mse).pvalue if len(d10) > 1 else np.nan
            print(f"{a:4s} L=1 : T={l1[:,0].mean():6.2f} B={l1[:,1].mean():7.2f} S={l1[:,2].mean():7.2f} giảm={red1.mean():6.2f}%")
            print(f"{a:4s} L=10: T={d10.teacher_mse.mean():6.2f} B={d10.baseline_mse.mean():7.2f} "
                  f"S={d10.student_mse.mean():7.2f} giảm={m:6.2f}% [{lo:.2f}; {hi:.2f}] "
                  f"FGL tốt hơn {int((d10.student_mse < d10.baseline_mse).sum())}/{len(d10)}  p(t ghép cặp)={pt:.2g}  n={len(d10)}")

    # ---------- 4. Chất lượng teacher ----------
    te = pd.read_csv(RES / "teacher.csv") if (RES / "teacher.csv").exists() else None
    if te is not None:
        full = pd.DataFrame([{"seed": s, "teacher_epochs": 20, "teacher_mse": v[0], "baseline_mse": v[1],
                              "student_mse": v[2], "mse_reduction_pct": (v[1] - v[2]) / v[1] * 100}
                             for s, v in SEED_SWEEP_L1["rnn"].items() if s in (0, 1, 2)])
        te = pd.concat([te, full], ignore_index=True)
        g = te.groupby("teacher_epochs")[["teacher_mse", "baseline_mse", "student_mse", "mse_reduction_pct"]].agg(["mean", "std"])
        print("\n=== Làm yếu teacher (RNN, seed 0,1,2) ===")
        print(g.round(2).to_string())
        rho, p = stats.spearmanr(te.teacher_mse, te.mse_reduction_pct)
        print(f"Spearman(teacher MSE, mức giảm MSE) = {rho:.3f}, p = {p:.2g}, n = {len(te)}")

    # ---------- 5. Alpha ----------
    al = pd.read_csv(RES / "alpha.csv") if (RES / "alpha.csv").exists() else None
    if al is not None:
        half = pd.DataFrame([{"seed": s, "alpha": 0.5, "teacher_mse": v[0], "baseline_mse": v[1],
                              "student_mse": v[2], "mse_reduction_pct": (v[1] - v[2]) / v[1] * 100}
                             for s, v in SEED_SWEEP_L1["rnn"].items() if s in (0, 1, 2)])
        al = pd.concat([al, half], ignore_index=True)
        g = al.groupby("alpha")[["student_mse", "mse_reduction_pct"]].agg(["mean", "std"])
        print("\n=== Quét alpha (RNN, seed 0,1,2) ===")
        print(g.round(2).to_string())

    # ---------- 6. Label smoothing cho Baseline (không teacher) ----------
    sm = pd.read_csv(RES / "smooth.csv") if (RES / "smooth.csv").exists() else None
    if sm is not None:
        ref = {s: v for s, v in SEED_SWEEP_L1["rnn"].items() if s in (0, 1, 2)}
        print("\n=== Baseline + label smoothing (RNN, seed 0,1,2) so với Baseline gốc và Student FGL ===")
        for e, d in sm.groupby("label_smoothing"):
            d = d.sort_values("seed")
            b0 = np.array([ref[s][1] for s in d.seed]); fgl = np.array([ref[s][2] for s in d.seed])
            red = (b0 - d.baseline_mse.values) / b0 * 100
            print(f"eps={e}: Baseline+LS MSE={d.baseline_mse.mean():7.2f} (±{d.baseline_mse.std():.2f})  "
                  f"giảm so với Baseline gốc={red.mean():6.2f}%  | Baseline gốc={b0.mean():.2f}  FGL(α=0.5)={fgl.mean():.2f}")

    # ---------- Hình ----------
    if te is not None and al is not None:
        fig, ax = plt.subplots(1, 2, figsize=(11, 4))
        for k, (ep, d) in enumerate(te.groupby("teacher_epochs")):
            ax[0].scatter(d.teacher_mse, d.mse_reduction_pct, s=45, label=f"teacher {ep} epoch",
                          color=plt.cm.viridis(k / 4), zorder=3)
        ax[0].set_xlabel("MSE của Teacher (bin²)"); ax[0].set_ylabel("Giảm MSE Baseline → Student (%)")
        ax[0].set_title("(a) Mức giảm MSE theo chất lượng teacher (RNN, 3 seed)")
        ax[0].grid(alpha=.3); ax[0].legend(fontsize=8)
        g = al.groupby("alpha").mse_reduction_pct
        ax[1].errorbar(g.mean().index, g.mean().values, yerr=g.std().values, fmt="o-", capsize=4)
        ax[1].axhline(0, color="grey", lw=1)
        ax[1].set_xlabel(r"$\alpha$ (trọng số nhãn thật)"); ax[1].set_ylabel("Giảm MSE Baseline → Student (%)")
        ax[1].set_title(r"(b) Ảnh hưởng của $\alpha$ (TB $\pm$ std, 3 seed)"); ax[1].grid(alpha=.3)
        fig.tight_layout()
        for ext in ("pdf", "png"):
            fig.savefig(hinh / f"kq-mackeyglass-ablation.{ext}", dpi=200)
        print(f"\nĐã lưu hình vào {hinh}/kq-mackeyglass-ablation.(pdf|png)")


if __name__ == "__main__":
    main()
