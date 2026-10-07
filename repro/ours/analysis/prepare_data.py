#!/usr/bin/env python3
"""
Dựng file dữ liệu Weather hợp nhất cho loader của tác giả.

Loader Dataset_Train chia train/val/test từ MỘT file, nên ta ghép train_weather.csv và
test_weather_df_final.csv rồi để loader chia theo mốc 2023-01-01 (if_split_by_date).

Hai biến thể, cả hai đều ghi ra để so sánh được:

  weather_all.csv        Y NGUYÊN dữ liệu tác giả, giữ cả giá trị khuyết -9999.
                         Đây là thứ pipeline đã công bố thực sự nhận.

  weather_all_clean.csv  Thay -9999 bằng nội suy theo thời gian trong từng trạm.

VÌ SAO CẦN BẢN CLEAN
--------------------
train_weather.csv còn 261 hàng mang -9999 ở 5 cột ngoại sinh (p, sh, Tpot, H2OC, rho) và
1 hàng ở SWDR — giai đoạn 2022-01-21..01-23 mà Mục 4.2 của bài báo có nhắc. Bài báo loại
các chuỗi này khỏi danh sách TARGET vì lý do đó, nhưng vẫn dùng chúng làm ĐẦU VÀO ngoại
sinh với -9999 nguyên vẹn.

Loader fit StandardScaler trên toàn cột, không lọc gì, nên 261 hàng đó phá thống kê:

    cột            std có -9999     std đúng     sai lệch
    p (mbar)             489,8         10,8          45x
    sh (g/kg)            445,6          2,53        176x
    Tpot (K)             458,0          8,42         54x

Hệ quả: tín hiệu thật của 6 yếu tố ngoại sinh bị nén gần như về 0 phương sai sau chuẩn
hoá, còn -9999 nằm ở z = -22,4. Huấn luyện phân kỳ (quan sát được: epoch 1 train loss
28,8 -> epoch 2 train loss 4,9e7).

Cách dùng:  python prepare_data.py
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent.parent.parent   # -> repro/
WD = HERE / 'data' / 'weather'
TIME = 'Date Time'
SENTINEL = -9000          # mọi giá trị <= ngưỡng này coi là khuyết


def load():
    tr = pd.read_csv(WD / 'train_weather.csv')
    te = pd.read_csv(WD / 'test_weather_df_final.csv')
    if list(tr.columns) != list(te.columns):
        sys.exit('Cột của train và test không khớp.')
    a = pd.concat([tr, te], ignore_index=True)
    a[TIME] = pd.to_datetime(a[TIME])
    return a.sort_values([ 'location', TIME]).reset_index(drop=True), len(tr), len(te)


def clean(df):
    """Thay giá trị khuyết bằng nội suy theo thời gian, riêng từng trạm."""
    num = [c for c in df.columns if c not in (TIME, 'location')]
    out = df.copy()
    n_before = int((out[num] <= SENTINEL).sum().sum())
    out[num] = out[num].mask(out[num] <= SENTINEL)
    parts = []
    for loc, g in out.groupby('location', sort=False):
        g = g.sort_values(TIME).set_index(TIME)
        g[num] = (g[num].interpolate(method='time', limit_direction='both')
                        .ffill().bfill())
        parts.append(g.reset_index())
    out = pd.concat(parts, ignore_index=True)
    n_after = int(out[num].isna().sum().sum())
    return out, n_before, n_after


def report_scaler_damage(raw, cln):
    ext = ['p (mbar)', 'sh (g/kg)', 'Tpot (K)', 'H2OC (mmol/mol)',
           'rho (g/m**3)', 'SWDR (W/m²)']
    print('\nẢnh hưởng lên StandardScaler (loader fit trên toàn cột, không lọc):\n')
    print(f"  {'cột':20s} {'std có -9999':>13s} {'std đúng':>11s} {'sai lệch':>10s} {'z(-9999)':>10s}")
    for c in ext:
        s_raw = raw[c].astype(float)
        sd_bad, mu_bad = s_raw.std(), s_raw.mean()
        sd_ok = cln[c].astype(float).std()
        z = (-9999 - mu_bad) / sd_bad if sd_bad else np.nan
        print(f"  {c:20s} {sd_bad:13.1f} {sd_ok:11.3f} {sd_bad/sd_ok:9.0f}x {z:10.1f}")


def main():
    a, n_tr, n_te = load()
    print(f"ghép: {a.shape}  ({n_tr} train + {n_te} test)")
    print(f"      {a[TIME].min()} -> {a[TIME].max()}")

    nt = a[TIME].nunique()
    ntr = a.loc[a[TIME] < '2023-01-01', TIME].nunique()
    print(f"\nmốc thời gian duy nhất: {nt}  (trước 2023-01-01: {ntr}, test: {nt-ntr})")
    print(f"với train_perc_split_by_date=0.8 -> train {int(ntr*0.8)}, "
          f"val {nt-int(ntr*0.8)-(nt-ntr)}, test {nt-ntr}")

    # bản y nguyên
    raw_path = WD / 'weather_all.csv'
    a.sort_values(TIME).to_csv(raw_path, index=False)
    print(f"\n[1] {raw_path.name:24s} y nguyên dữ liệu tác giả (giữ -9999)")

    # bản làm sạch
    c, n_before, n_after = clean(a)
    clean_path = WD / 'weather_all_clean.csv'
    c.sort_values(TIME).to_csv(clean_path, index=False)
    print(f"[2] {clean_path.name:24s} đã nội suy {n_before} giá trị khuyết "
          f"(còn NaN: {n_after})")

    report_scaler_damage(a, c)
    print("\nDùng --data_path weather_all.csv       -> tái hiện y như pipeline đã công bố")
    print("    --data_path weather_all_clean.csv -> bản đã sửa dữ liệu khuyết")


if __name__ == '__main__':
    main()
