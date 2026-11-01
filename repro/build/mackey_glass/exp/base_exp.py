import argparse
import pickle
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
from tqdm import tqdm

import matplotlib
matplotlib.use("Agg")  # no display needed; script may run headless (Kaggle, CI, ...)
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

from utils.utils import RNN, create_time_series_dataset, KL

# Check for GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# —— Dataviz palette (validated via dataviz skill's validate_palette.js) ——
_BLUE, _ORANGE = "#2a78d6", "#eb6834"
_INK, _INK_SEC, _MUTED = "#0b0b0b", "#52514e", "#898781"
_GRID, _BASELINE_LINE, _SURFACE = "#e1e0d9", "#c3c2b7", "#fcfcfb"


def find_thesis_hinh_dir():
    """Locate docs/mau-latex-tieu-luan/hinh by walking up from this file. Falls back to
    a local 'figures' folder next to the script (e.g. a fresh clone on Kaggle, where the
    thesis directory doesn't exist)."""
    here = Path(__file__).resolve()
    for ancestor in here.parents:
        candidate = ancestor / "docs" / "mau-latex-tieu-luan" / "hinh"
        if candidate.is_dir():
            return candidate
    fallback = here.parent / "figures"
    fallback.mkdir(exist_ok=True)
    return fallback


def plot_seed_sweep(df, out_dir):
    """Bar chart of MSE reduction (%) per seed, with the across-seed average as a
    reference line."""
    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 10, "text.color": _INK,
        "axes.edgecolor": _BASELINE_LINE, "axes.labelcolor": _INK_SEC,
        "xtick.color": _MUTED, "ytick.color": _MUTED,
        "figure.facecolor": _SURFACE, "axes.facecolor": _SURFACE,
        "savefig.facecolor": _SURFACE,
    })

    fig, ax = plt.subplots(figsize=(6.3, 3.6))
    ax.set_facecolor(_SURFACE)
    for spine in ["top", "right", "left"]:
        ax.spines[spine].set_visible(False)
    ax.spines["bottom"].set_color(_BASELINE_LINE)
    ax.tick_params(axis="both", length=0)
    ax.grid(axis="y", color=_GRID, linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)

    x = np.arange(len(df))
    ax.bar(x, df["mse_reduction_pct"], width=0.55, color=_BLUE, zorder=3)
    for xi, v in zip(x, df["mse_reduction_pct"]):
        ax.text(xi, v + 0.8, f"{v:.1f}%", ha="center", va="bottom", fontsize=9,
                color=_INK_SEC)

    avg = df["mse_reduction_pct"].mean()
    ax.axhline(avg, color=_ORANGE, linewidth=1.5, linestyle=(0, (4, 3)), zorder=2)
    ax.text(0.02, 0.94, f"TB {len(df)} seed: {avg:.2f}%", transform=ax.transAxes,
            ha="left", va="top", fontsize=9, color=_ORANGE)

    ax.set_xlabel("Seed")
    ax.set_xticks(x)
    ax.set_xticklabels(df["seed"].astype(str))
    ax.set_ylim(0, df["mse_reduction_pct"].max() + 10)
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%d%%"))
    ax.set_ylabel("Giảm MSE: Baseline → Student (%)")
    ax.set_title("Sweep seed — Mackey-Glass", fontsize=10, color=_INK, pad=8)

    fig.tight_layout()
    out_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_dir / "kq-mg-seed.pdf", bbox_inches="tight")
    fig.savefig(out_dir / "kq-mg-seed.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"[plot] Saved kq-mg-seed.{{pdf,png}} to {out_dir}")

class EarlyStopper:
    def __init__(self, patience=5, min_delta=1e-4):
        self.patience   = patience
        self.min_delta  = min_delta
        self.best_loss  = float('inf')
        self.counter    = 0
        self.best_state = None

    def step(self, current_loss, model):
        # improvement?
        if current_loss + self.min_delta < self.best_loss:
            self.best_loss  = current_loss
            self.counter    = 0
            self.best_state = {k:v.cpu() for k,v in model.state_dict().items()}
            return False    # don’t stop
        else:
            self.counter += 1
            return (self.counter >= self.patience)

    def restore(self, model):
        if self.best_state:
            model.load_state_dict(self.best_state)

def train_student_model(student_horizon, alpha, num_bins,
                        val_size, test_size, epochs,
                        temperature, lookback_window, batch_size,
                        patience=5, seed=42, arch='rnn', external_data=None):
    torch.manual_seed(seed)
    print(f"\nTraining | Horizon={student_horizon} Alpha={alpha} "
          f"Num bins={num_bins} Val size={val_size} Test size={test_size}"
          f"Epochs={epochs} Temp={temperature} "
          f"Lookback={lookback_window} Batch={batch_size} Arch={arch}")

    # hyperparams
    hidden_size   = 128
    output_size   = num_bins
    num_layers    = 2
    lr            = 1e-4

    # load data (or use a caller-provided slice, e.g. for cross-validation folds)
    if external_data is not None:
        data = external_data
    else:
        with open("data.pkl","rb") as f:
            data = pickle.load(f)

    # teacher (1-step, offset=H-1)
    teacher_train, teacher_val, teacher_test, _, _ = create_time_series_dataset(
        data=data, 
        lookback_window=lookback_window, 
        forecasting_horizon=1,
        num_bins=num_bins, 
        val_size=val_size,
        test_size=test_size,
        offset=student_horizon-1, 
        batch_size=batch_size
    )
    # student (H-step, offset=0)
    student_train, student_val, student_test, _, _ = create_time_series_dataset(
        data=data, 
        lookback_window=lookback_window, 
        forecasting_horizon=student_horizon,
        num_bins=num_bins, 
        val_size=val_size,
        test_size=test_size,
        offset=0, 
        batch_size=batch_size
    )

    mse   = nn.MSELoss()
    celoss= nn.CrossEntropyLoss()

    # ---- Teacher loop with early stopping ----
    teacher     = RNN(lookback_window, hidden_size, output_size, num_layers, cell_type=arch).to(device)
    opt_t       = torch.optim.Adam(teacher.parameters(), lr=lr)
    stop_t      = EarlyStopper(patience=patience)
    for epoch in range(epochs):
        teacher.train()
        for _, x, y in teacher_train:
            x = x.float().to(device).view(-1,1,lookback_window)
            y = y.long().to(device) 
            out = teacher(x)
            loss = celoss(out, y)
            opt_t.zero_grad(); loss.backward(); opt_t.step()

        # validate
        teacher.eval()
        with torch.no_grad():
            val_loss = 0.
            for _, x, y in teacher_val:
                x = x.float().to(device).view(-1,1,lookback_window)
                y = y.long().to(device)
                out = teacher(x)
                val_loss += celoss(out, y).item()
            val_loss /= len(teacher_val)

        #print(f"[Teacher] Epoch {epoch+1}: Val MSE={val_loss:.4f}")
        if stop_t.step(val_loss, teacher):
            print(f"[Teacher] Early stopping at epoch {epoch+1}")
            break
    stop_t.restore(teacher)
    print(f"[Teacher] Best Val CE = {stop_t.best_loss:.4f}")

    # ---- Baseline loop with early stopping ----
    baseline   = RNN(lookback_window, hidden_size, output_size, num_layers, cell_type=arch).to(device)
    opt_b      = torch.optim.Adam(baseline.parameters(), lr=lr)
    stop_b     = EarlyStopper(patience=patience)
    for epoch in range(epochs):
        baseline.train()
        for _, x, y in student_train:
            x = x.float().to(device).view(-1,1,lookback_window)
            y = y.long().to(device)
            out = baseline(x)
            loss = celoss(out, y)
            opt_b.zero_grad(); loss.backward(); opt_b.step()

        # validate
        baseline.eval()
        with torch.no_grad():
            val_loss = 0.
            for _, x, y in student_val:
                x = x.float().to(device).view(-1,1,lookback_window)
                y = y.long().to(device)
                out = baseline(x)
                val_loss += celoss(out, y).item()
            val_loss /= len(student_val)

        #print(f"[Baseline] Epoch {epoch+1}: Val MSE={val_loss:.4f}")
        if stop_b.step(val_loss, baseline):
            print(f"[Baseline] Early stopping at epoch {epoch+1}")
            break
    stop_b.restore(baseline)
    print(f"[Baseline] Best Val CE = {stop_b.best_loss:.4f}")

    # ---- Student + distillation loop w/ early stopping ----
    student = RNN(lookback_window, hidden_size, output_size, num_layers, cell_type=arch).to(device)
    opt_s    = torch.optim.Adam(student.parameters(), lr=lr)
    stop_s   = EarlyStopper(patience=patience)

    for epoch in range(epochs):
        student.train()
        for (i_s, x_s, y_s), (i_t, x_t, y_t) in zip(student_train, teacher_train):
            x_s = x_s.float().to(device).view(-1,1,lookback_window)
            targets = y_s.long().to(device)
            outputs = student(x_s)

            x_t = x_t.float().to(device).view(-1,1,lookback_window)
            with torch.no_grad():
                logits = teacher(x_t)

            loss = alpha * celoss(outputs, targets) + KL(outputs, logits, temperature, alpha)
            opt_s.zero_grad(); loss.backward(); opt_s.step()

        # validate student
        student.eval()
        with torch.no_grad():
            val_loss = 0.
            for _, x, y in student_val:
                x = x.float().to(device).view(-1,1,lookback_window)
                y = y.long().to(device)
                out = student(x)
                val_loss += celoss(out, y).item()
            val_loss /= len(student_val)

        #print(f"[Student] Epoch {epoch+1}: Val MSE={val_loss:.4f}")
        if stop_s.step(val_loss, student):
            print(f"[Student] Early stopping at epoch {epoch+1}")
            break
    stop_s.restore(student)
    print(f"[Student] Best Val CE = {stop_s.best_loss:.4f}")

    # final test metrics are in MSE
    def evaluate(model, loader):
        model.eval()
        tot = 0.
        with torch.no_grad():
            for _, x, y in loader:
                x = x.float().to(device).view(-1,1,lookback_window)
                tot += mse(model(x).argmax(dim=1).float(), y.float().to(device).squeeze(-1)).item()
        return tot / len(loader)

    Tmse = evaluate(teacher, teacher_test); print(f"Teacher Test MSE:  {Tmse:.4f}")
    Bmse = evaluate(baseline, student_test); print(f"Baseline Test MSE: {Bmse:.4f}")
    Smse = evaluate(student, student_test); print(f"Student Test MSE:  {Smse:.4f}")

    return {"teacher": Tmse, "baseline": Bmse, "student": Smse}

# … rest of your __main__ unchanged …


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Future-Guided Learning for Time-Series Forecasting"
    )
    parser.add_argument("--horizon", type=int, required=True, 
                        help="Student horizon (N)")
    parser.add_argument("--alpha",   type=float, required=True, 
                        help="Loss weight α")
    parser.add_argument("--num_bins",type=int, default=50,   
                        help="Number of bins")
    parser.add_argument("--epochs",  type=int, default=10,   
                        help="Training epochs")
    parser.add_argument("--temperature", type=float, default=4,
                        help='Controls softness of teacher logits')
    parser.add_argument("--lookback_window", type=int, default=1,
                        help="Length of history fed to RNN")
    parser.add_argument("--batch_size",      type=int, default=1,
                        help="Batch size for all loaders")
    parser.add_argument("--val_size",  type=float, default=0.2,
                        help="Fraction of data for validation")
    parser.add_argument("--test_size", type=float, default=0.2,
                        help="Fraction of data for testing")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed")
    parser.add_argument("--arch", type=str, default="rnn", choices=["rnn", "lstm", "gru"],
                        help="Recurrent cell for Teacher/Baseline/Student (default: rnn, "
                             "matches the author's original architecture)")
    parser.add_argument("--seed_sweep", type=str, default=None,
                        help="Comma-separated seeds (e.g. 0,1,2,3,4,42). When given, runs "
                             "train_student_model once per seed instead of a single run with "
                             "--seed, then saves seed_sweep_results.csv and a summary chart "
                             "(mirrors the stability check in Chuong 5 of the thesis).")

    args = parser.parse_args()

    if args.seed_sweep:
        seeds = [int(s) for s in args.seed_sweep.split(",")]
        rows = []
        for s in seeds:
            r = train_student_model(
                student_horizon   = args.horizon,
                alpha             = args.alpha,
                num_bins          = args.num_bins,
                val_size          = args.val_size,
                test_size         = args.test_size,
                epochs            = args.epochs,
                seed              = s,
                temperature       = args.temperature,
                lookback_window   = args.lookback_window,
                batch_size        = args.batch_size,
                arch              = args.arch
            )
            reduction_pct = (r["baseline"] - r["student"]) / r["baseline"] * 100
            rows.append({
                "seed": s,
                "teacher_mse": r["teacher"],
                "baseline_mse": r["baseline"],
                "student_mse": r["student"],
                "mse_reduction_pct": reduction_pct,
            })

        df_seed = pd.DataFrame(rows)
        df_seed.to_csv("seed_sweep_results.csv", index=False)
        print("\nSaved raw results to seed_sweep_results.csv")
        print(df_seed)

        plot_seed_sweep(df_seed, find_thesis_hinh_dir())
    else:
        results = train_student_model(
            student_horizon   = args.horizon,
            alpha             = args.alpha,
            num_bins          = args.num_bins,
            val_size          = args.val_size,
            test_size         = args.test_size,
            epochs            = args.epochs,
            seed              = args.seed,
            temperature       = args.temperature,
            lookback_window   = args.lookback_window,
            batch_size        = args.batch_size,
            arch              = args.arch
        )
