#!/usr/bin/env python3
"""
Kiểm chứng giả thuyết rò rỉ thông tin trong thực nghiệm Weather của bài báo ConEm
(Knowledge-Based Systems 329 (2025) 114312, Mục 4.1.4 / 4.3.4 / Bảng 10-13).

Bài báo dùng 6 chuỗi khí tượng - p, sh, Tpot, H2OC, rho, SWDR - làm "yếu tố ngoại sinh
đã biết trước" để dự báo 9 chuỗi còn lại. Nhưng vài chuỗi trong số đó liên hệ với target
bằng hằng đẳng thức nhiệt động học, không phải quan hệ thống kê cần học.

Script này không huấn luyện gì. Nó chỉ trả lời: nếu ta BIẾT TRƯỚC 6 biến đó, thì có cần
mô hình dự báo nào không?

  Kiểm tra 1: suy T(degC) từ Tpot và p bằng công thức thế vị (0 tham số).
  Kiểm tra 2: OLS khớp trên train, đánh giá out-of-sample trên test, chỉ dùng 6 biến
              ngoại sinh CÙNG THỜI ĐIỂM, không dùng lịch sử target.

Cách dùng:  python leakage_test.py
"""
import os
from pathlib import Path
import numpy as np
import pandas as pd

os.chdir(Path(__file__).resolve().parent.parent.parent)   # -> repro/

TRAIN = 'data/weather/train_weather.csv'
TEST = 'data/weather/test_weather_df_final.csv'

EXT = ['p (mbar)', 'sh (g/kg)', 'Tpot (K)', 'H2OC (mmol/mol)', 'rho (g/m**3)', 'SWDR (W/m²)']

# MSE/MAE mà bài báo báo cáo cho FEDformer + ConEm, I=96 -> O=96 (Bảng 11-13).
# LƯU Ý: nhãn cột MSE/MAE ở Bảng 10 bị đảo so với Bảng 11-13; giá trị dưới đây theo
# Bảng 11-13, khớp với thứ tự trả về của metric() trong utils/metrics.py (mae, mse, ...).
PAPER = {
    'T (degC)':     (0.0343, 0.1341),
    'rh (%)':       (0.7572, 0.6565),
    'Tdew (degC)':  (0.0142, 0.0831),
    'VPmax (mbar)': (0.0282, 0.1242),
    'VPact (mbar)': (0.0032, 0.0444),
    'VPdef (mbar)': (0.0979, 0.2206),
    'wv (m/s)':     (2.1452, 1.0900),
    'wd (deg)':     (6480.8, 60.390),
    'rain (mm)':    (0.0042, 0.0155),
}


def drop_sentinel(df):
    """Bỏ các hàng còn giá trị khuyết -9999 (vẫn còn trong file train của tác giả)."""
    num = df.drop(columns=['Date Time', 'location'])
    return df[~(num <= -9000).any(axis=1)].reset_index(drop=True)


def ols_predict(Xtr, ytr, Xte):
    """OLS qua lstsq; nhân ma trận bằng vòng lặp để tránh cờ FP giả của BLAS/numpy."""
    beta, *_ = np.linalg.lstsq(Xtr, ytr, rcond=None)
    out = np.zeros(len(Xte))
    for j in range(Xte.shape[1]):
        out = out + Xte[:, j] * beta[j]
    return out


def main():
    tr, te = pd.read_csv(TRAIN), pd.read_csv(TEST)
    n_bad_tr = len(tr) - len(drop_sentinel(tr))
    trc, tec = drop_sentinel(tr), drop_sentinel(te)

    print("=" * 78)
    print("DỮ LIỆU")
    print("=" * 78)
    print(f"  train {len(tr):>6} hàng  ({n_bad_tr} hàng còn sentinel -9999 -> đã lọc)")
    print(f"  test  {len(te):>6} hàng  ({len(te) - len(tec)} hàng còn sentinel -9999)")
    print(f"  yếu tố ngoại sinh: {EXT}")
    print()
    print("  Ghi chú: -9999 vẫn nằm trong file train của tác giả ở 5 cột ngoại sinh")
    print("  (p, sh, Tpot, H2OC, rho). Bài báo nói đã loại các chuỗi này khỏi TARGET,")
    print("  nhưng chúng vẫn được dùng làm ĐẦU VÀO ngoại sinh với giá trị -9999 nguyên vẹn.")

    print()
    print("=" * 78)
    print("KIỂM TRA 1 — suy T(degC) từ Tpot & p bằng công thức nhiệt độ thế vị")
    print("            T = Tpot * (p/1000)^0.286 - 273.15     (0 tham số, 0 huấn luyện)")
    print("=" * 78)
    T = tec['T (degC)'].values
    Th = tec['Tpot (K)'].values * (tec['p (mbar)'].values / 1000.0) ** 0.286 - 273.15
    mse, mae = np.mean((T - Th) ** 2), np.mean(np.abs(T - Th))
    pm, pa = PAPER['T (degC)']
    print(f"  công thức vật lý      MSE {mse:12.6f}   MAE {mae:10.6f}")
    print(f"  FEDformer + ConEm     MSE {pm:12.6f}   MAE {pa:10.6f}")
    print(f"  FEDformer gốc         MSE {1.5919:12.6f}   MAE {0.9804:10.6f}")
    print(f"  -> công thức tốt hơn ConEm {pm / mse:.0f}x (MSE), {pa / mae:.0f}x (MAE)")

    print()
    print("=" * 78)
    print("KIỂM TRA 2 — OLS khớp trên TRAIN, đánh giá OUT-OF-SAMPLE trên TEST")
    print("            chỉ 6 biến ngoại sinh cùng thời điểm; KHÔNG dùng lịch sử target")
    print("=" * 78)
    Xtr = np.c_[np.asarray(trc[EXT], np.float64), np.ones(len(trc))]
    Xte = np.c_[np.asarray(tec[EXT], np.float64), np.ones(len(tec))]
    print(f"  {'target':16s} {'R2':>7s} {'OLS MSE':>11s} {'OLS MAE':>9s} "
          f"{'ConEm MSE':>10s} {'ConEm MAE':>9s}  kết luận")
    for t, (pm, pa) in PAPER.items():
        y = np.asarray(tec[t], np.float64)
        pred = ols_predict(Xtr, np.asarray(trc[t], np.float64), Xte)
        r = y - pred
        m, a = np.mean(r ** 2), np.mean(np.abs(r))
        r2 = 1 - np.sum(r ** 2) / np.sum((y - y.mean()) ** 2)
        if r2 > 0.9:
            verdict = "biến ngoại sinh ĐÃ xác định target"
        elif r2 > 0.5:
            verdict = "phụ thuộc mạnh"
        else:
            verdict = "biến ngoại sinh không mang thông tin"
        print(f"  {t:16s} {r2:7.4f} {m:11.4f} {a:9.4f} {pm:10.4f} {pa:9.4f}  {verdict}")

    print()
    print("=" * 78)
    print("KẾT LUẬN")
    print("=" * 78)
    print("  6 target mà bài báo báo cáo 'cải thiện đáng kể' (T, rh, Tdew, VPmax, VPact,")
    print("  VPdef) đúng là 6 target bị 6 biến ngoại sinh xác định gần như hoàn toàn.")
    print("  3 target mà ConEm hầu như không cải thiện (wv, wd, rain) đúng là 3 target mà")
    print("  các biến đó không mang thông tin. Sự tương ứng này không phải trùng hợp:")
    print("  thực nghiệm Weather đo khả năng NGHỊCH ĐẢO ĐẠI SỐ, không phải khả năng dự báo.")


if __name__ == "__main__":
    main()
