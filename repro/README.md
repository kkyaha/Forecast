# Tái hiện ConEm

Bài báo: **ConEm: A novel framework for integrating external factors with inner and outer
correlations in time series forecasting**, Knowledge-Based Systems 329 (2025) 114312,
[doi:10.1016/j.knosys.2025.114312](https://doi.org/10.1016/j.knosys.2025.114312), CC BY.
Repo tác giả: <https://github.com/anonymous7594/conem>, commit `80f0e01`.

Trình bày: **`ConEm_TaiHien.ipynb`**

---

## Cấu trúc: ba tầng, ranh giới dứt khoát

```
repro/
├── upstream/conem/      100% CỦA TÁC GIẢ. Không bao giờ sửa. Còn .git để kiểm chứng nguồn.
├── ours/                100% CỦA CHÚNG TA.
└── build/conem/         SINH RA = upstream + patches. Không sửa tay. Xoá được, dựng lại được.
```

Nhờ cách này, câu hỏi **"dòng code này của ai"** luôn có câu trả lời dứt khoát, và không
bao giờ cần sửa file của tác giả tại chỗ.

```bash
python ours/build.py --list       # liệt kê các bản vá
python ours/build.py --audit      # dựng build/ + xuất build/audit.diff
```

### Kiểm chứng ranh giới

```bash
git -C upstream/conem status --porcelain    # trống = code tác giả chưa bị sửa
git -C upstream/conem rev-parse HEAD        # 80f0e01c40e2608b9a838771bb1f308b2aa76ae1
cat build/PATCHES.md                        # từng bản vá: vì sao, bằng chứng, thay đổi gì
cat build/audit.diff                        # diff thống nhất so với upstream
```

| | con số |
|---|---|
| Code tác giả | **58 file `.py`, 19.430 dòng** |
| Trong đó giống hệt từng byte sau khi dựng | **47 file** |
| Trong đó được vá | **11 file, 197 dòng** |
| Code của chúng ta (file riêng) | **~2.000 dòng** |
| Số dòng mạng neural chúng ta viết | **0** |

Lõi cơ chế ConEm — `layers/Embed.py` (`Customize_Tokenizer` = Feature Tokenizer Hình 4,
`FastGLU` = Eq. 3, `ShallowMLP` = Eq. 4, `Local_context` = LoEm, `Temporal_embedding` =
PaEm/FuEm) và `Embed_without_{LoEm,FuEm,PaEm}.py` — **không bị chạm một byte nào**.

---

## Cây tệp

```
upstream/conem/              code tác giả, nguyên trạng (clone + .git)

ours/
├── build.py                 dựng build/ từ upstream + patches
├── patches/
│   ├── required/            KHÔNG CÓ THÌ CRASH — không còn lựa chọn nào khác
│   │   ├── a1_a2_fed_based.py       A1+A2  FEDformer+ConEm không chạy
│   │   ├── a3_ablation_imports.py   A3     ablation bỏ FuEm không chạy
│   │   ├── a4_baseline_chdir.py     A4     3 baseline os.chdir máy tác giả
│   │   ├── b1_mps_device.py         B1     thêm nhánh thiết bị MPS
│   │   └── b2_map_location.py       B2     map_location hardcode cuda:0
│   └── choices/             QUYẾT ĐỊNH CỦA CHÚNG TA — code vẫn chạy nếu bỏ
│       ├── c1_dynamic_import.py     C1     chọn backbone không cần sửa file
│       ├── a6fix_rin.py             A6fix  RIN dùng dạng (sum+1)
│       ├── a8_lr_decay.py           A8     bật lại adjust_learning_rate
│       ├── c2_stride.py             C2     stride cửa sổ train/val
│       ├── c3_skip_test_eval.py     C3     bỏ eval test mỗi epoch
│       └── c4_grad_clip.py          C4     THÊM gradient clipping
├── runner/
│   ├── run.py               entry point tác giả thiếu; dựng 68 tham số; 0 dòng mạng neural
│   └── run_all.sh           ba run A/C/B
├── analysis/
│   ├── leakage_test.py      PHÁT HIỆN CHÍNH — không huấn luyện gì
│   ├── prepare_data.py      dựng weather_all.csv và weather_all_clean.csv
│   └── collect_results.py   tổng hợp, đối chiếu bài báo
├── verify/
│   └── verify_pristine.py   chứng minh bản gốc chạy được tới đâu
└── notebook/
    └── build_notebook.py    sinh ConEm_TaiHien.ipynb

build/conem/                 SINH RA — không sửa tay
build/PATCHES.md             báo cáo: áp gì, vì sao, bằng chứng gì
build/audit.diff             diff thống nhất so với upstream

data/weather/                dữ liệu tác giả + 2 bản hợp nhất
logs/ results/ checkpoints/  kết quả chạy
```

---

## Chạy

```bash
# 1. môi trường
python3 -m venv ../.venv
../.venv/bin/python -m pip install -r requirements.txt

# 2. code tác giả — KHÔNG nằm trong repo này, phải tự clone và pin commit
git clone https://github.com/anonymous7594/conem upstream/conem
git -C upstream/conem checkout 80f0e01c40e2608b9a838771bb1f308b2aa76ae1
git -C upstream/conem status --porcelain   # trống = chưa bị sửa

# 3. dựng bản làm việc
python ours/build.py --audit
python ours/verify/verify_pristine.py        # bản gốc chạy được tới đâu

# 4. dữ liệu
curl -sSL "https://drive.usercontent.google.com/download?id=1TGScqj_hV3dRYNRAlwGPpcNNSECIpite&export=download" -o weather.zip
unzip -q weather.zip -d data/weather && rm weather.zip
python ours/analysis/prepare_data.py

# 5. PHÁT HIỆN CHÍNH — nhanh, không huấn luyện
python ours/analysis/leakage_test.py

# 6. huấn luyện (xem cảnh báo về compute ở dưới)
./ours/runner/run_all.sh clean
python ours/analysis/collect_results.py
```

---

## Cái gì tái hiện được

| Thực nghiệm bài báo | Dữ liệu | Trạng thái |
|---|---|---|
| Australian Pharmaceutical Weekly (Bảng 5, 6) | Bảo mật thương mại | **Không thể** |
| Australian Daily POS (Bảng 5, 7) | Bảo mật thương mại | **Không thể** |
| Corporación Favorita (Bảng 8, 9, Hình 11) | Drive `favorita.zip` 108 MB | Được, chưa làm |
| **Weather (Bảng 10–13, 17)** | Drive `weather.zip` 4,3 MB | Phân tích xong; huấn luyện bị giới hạn compute |
| EPF (Bảng 14–16, 18) | [Zenodo 4624805](https://zenodo.org/records/4624805) | Không có loader EPF, không có PatchTST/iTransformer |

[Thư mục Drive](https://drive.google.com/drive/folders/1XFjmbIZ4jrb11NWmhr4QSrLBx-yzqoFv) mà
README repo trỏ tới: `weather.zip` (`1TGScqj_hV3dRYNRAlwGPpcNNSECIpite`),
`favorita.zip` (`1wITO0KWzeHRLKk8Po0oMAJDJ1UPyus-C`).

---

## Chín vấn đề trong code/dữ liệu đã công bố

Chi tiết đầy đủ kèm bằng chứng: `build/PATCHES.md`.

| # | Vấn đề | Loại |
|---|---|---|
| 0 | **Không có entry point nào** — 0 file có `__main__`/`argparse` | thiếu |
| A1, A2 | `model_fed_based.py` lệch 2 tham số + không giải nén decoder | crash |
| A3 | `model_without_FuEm.py` 3 import bị comment nhưng vẫn dùng | crash |
| A4 | **3 file baseline** `os.chdir("/home/ad/20813716/...")` | crash |
| A5 | `model_fedformer.py` embedding khai 11 đầu vào, nạp 5 | crash |
| A6 | RIN: bản an toàn `(sum+1)` bị comment, bản bất ổn `sum` đang dùng | phân kỳ |
| A7 | `StandardScaler` fit trên **cả tập test** | rò rỉ nhẹ |
| A8 | `adjust_learning_rate` bị comment; **không có gradient clipping** | phân kỳ |
| — | `-9999` còn trong `train_weather.csv`, phá `std` 42–177× | dữ liệu |

**A4 + A5 nghiêm trọng nhất về tái hiện**: cả ba file baseline không chạy được ngoài máy
tác giả, nên **so sánh cốt lõi backbone vs backbone+ConEm không tái hiện được từ code đã
công bố**.

**A8 nghiêm trọng nhất về huấn luyện**: không mô hình nào huấn luyện ổn định được từ cấu
hình đã công bố. Loss học bình thường ~230 step rồi nổ đơn điệu (0,97 → 3,8 → 27,6 → 391 →
1.909 → 17.777 → 571.064). Phải **thêm** gradient clipping (C4) — thứ không có trong repo.

Một chỗ chúng ta nghi là lỗi nhưng **code tác giả đúng**: khởi tạo các tensor phức khổng lồ
dùng `scale = 1/(c·k)²` rồi `scale·torch.rand`, khớp FEDformer gốc của thuml.

---

## Phát hiện chính: thực nghiệm Weather bị rò rỉ thông tin

`python ours/analysis/leakage_test.py` — **không huấn luyện gì, chỉ dùng dữ liệu của tác giả**.
Đây là phần kết luận đáng tin nhất, vì nó không phụ thuộc vào bất kỳ thay đổi nào của chúng ta.

Bài báo dùng 6 chuỗi khí tượng (`p`, `sh`, `Tpot`, `H2OC`, `rho`, `SWDR`) làm yếu tố ngoại
sinh **đã biết trước**. Nhưng nhiệt độ thế vị liên hệ với target bằng **hằng đẳng thức nhiệt
động học**:

$$T_{pot} = (T + 273{,}15)\left(\tfrac{1000}{p}\right)^{0{,}286}
\;\Longrightarrow\; T = T_{pot}\left(\tfrac{p}{1000}\right)^{0{,}286} - 273{,}15$$

Áp công thức đó trên **tập test của tác giả**, 0 tham số, 0 huấn luyện:

| | MSE | MAE |
|---|---|---|
| **Công thức vật lý** | **0,000014** | **0,003091** |
| Bài báo, FEDformer + ConEm (Bảng 11) | 0,0343 | 0,1341 |
| Bài báo, FEDformer gốc (Bảng 11) | 1,5919 | 0,9804 |

Một hằng đẳng thức đại số tốt hơn ConEm **2.403×** về MSE.

OLS khớp trên train, đánh giá **out-of-sample** trên test, chỉ dùng 6 biến ngoại sinh
**cùng thời điểm**, không dùng lịch sử target:

| target | R² | OLS MSE | ConEm MSE (bài báo) | |
|---|---|---|---|---|
| T (degC) | 1,0000 | 0,0006 | 0,0343 | biến ngoại sinh **đã xác định** target |
| VPact (mbar) | 0,9998 | 0,0010 | 0,0032 | biến ngoại sinh **đã xác định** target |
| Tdew (degC) | 0,9757 | 0,4498 | 0,0142 | biến ngoại sinh **đã xác định** target |
| VPmax (mbar) | 0,9325 | 0,7958 | 0,0282 | biến ngoại sinh **đã xác định** target |
| rh (%) | 0,8809 | 23,5680 | 0,7572 | phụ thuộc mạnh |
| VPdef (mbar) | 0,8155 | 0,7520 | 0,0979 | phụ thuộc mạnh |
| wv (m/s) | 0,0988 | 2,7733 | 2,1452 | **không mang thông tin** |
| wd (deg) | 0,0129 | 6530,33 | 6480,80 | **không mang thông tin** |
| rain (mm) | 0,0194 | 0,0034 | 0,0042 | **không mang thông tin** |

**Tương ứng hoàn hảo.** 6 phép đo bài báo báo cáo "cải thiện đáng kể" (T, rh, Tdew, VPmax,
VPact, VPdef) đúng là 6 phép đo bị các biến ngoại sinh xác định. 3 phép đo ConEm hầu như
không cải thiện (`wv` 1,738→1,090; `wd` 60,635→60,390; `rain` gần như không đổi) đúng là 3
phép đo các biến đó không mang thông tin.

**Baseline còn bị bỏ đói thông tin.** `data_loader_without_ext.py`:

```python
seq_x_mark = None   # self.data_stamp[index][0:self.seq_len]
```

Các cột backbone ở Bảng 10 là mô hình chỉ huấn luyện trên chuỗi target, trong khi cột
`+ConEm` nhận 6 biến xác định target. Mục 4.2 xác nhận điều này cho các baseline khác:
*"trained and evaluated exclusively on the time series data"*. Bằng chứng nội tại khớp:
**TFT** — baseline duy nhất *có* nhận covariate — là baseline duy nhất cạnh tranh được
(Bảng 12: `VPdef` MSE 0,0200, **tốt hơn** ConEm 0,0979).

> Mức cải thiện trên Weather không đo giá trị của *kiến trúc* ConEm, mà đo giá trị của việc
> **được biết trước biến quyết định target**. Bài báo gọi đây là "simulation" (Mục 4.1.4)
> nhưng vẫn trình bày như bằng chứng về khả năng tổng quát hoá.

---

## Giới hạn: phần huấn luyện chưa hoàn tất

Máy: Apple M5, 16 GB, **không CUDA**. Mô hình FEDformer+ConEm có **116.092.098 tham số**,
trong đó **86,7% là số phức** (6 tensor `complex64` kích thước `1024×1024×16`).

| stride | mẫu train | step/epoch | phút/epoch | 4 epoch × 3 run |
|---|---|---|---|---|
| **1 (như tác giả)** | 83.630 | 5.226 | 192 | **38,4 h** |
| 12 | 6.969 | 435 | 16 | 3,2 h |

Để bằng ngân sách của tác giả chỉ **một** epoch đã cần ~3,2 giờ. Với ngân sách nhỏ hơn thì
mô hình không học đủ — kết quả đo được còn tệ hơn persistence:

| | MSE trên tập test, `T (degC)` |
|---|---|
| dự báo hằng số = mean(test) (= phương sai) | 28,33 |
| tái hiện của chúng ta (`RIN=True`, không clipping) | 14,42 |
| **persistence — lặp giá trị cuối cửa sổ** | **10,54** |
| bài báo, FEDformer gốc | 1,59 |
| bài báo, FEDformer + ConEm | 0,034 |

**Các con số huấn luyện của chúng ta không dùng được làm đối chiếu**, và không nên trình bày
như bằng chứng phản bác bài báo. Hai lý do:

1. Ngân sách compute nhỏ hơn ~12× mỗi epoch.
2. Cấu hình huấn luyện đã có **bổ sung của chúng ta** (nhóm `choices`, đặc biệt C4 gradient
   clipping) nên không còn là tái hiện trung thực.

Và kể cả khớp được 0,0343 cũng **không chứng minh gì**, vì OLS 7 tham số đã đạt 0,0006 trên
cùng tập test. Phép thử có ý nghĩa là so sánh **A (ngoại sinh xác định target)** với
**B (`wv`/`wd`/`rain`, R² < 0,1)** dưới cùng ngân sách — xem `ours/runner/run_all.sh`.

---

## Bước tiếp theo nếu tiếp tục

- **Favorita** (`favorita.zip`, 108 MB): dataset bán lẻ công khai duy nhất, đúng bài toán
  gốc, và promotion là yếu tố ngoại sinh **thật** — không xác định target bằng đại số.
- **Mô hình thu nhỏ** (`d_model` 512→128, `modes` 64→16, ~7M tham số): hết swap, ~1 phút/epoch,
  đủ để chạy so sánh A/B/C. Đổi lại: lệch cấu hình bài báo.
- **Ablation Bảng 17**: `--variant ablation --ablate {LoEm,FuEm,PaEm}` — chạy được sau khi dựng.
- Thêm baseline **LightGBM**: bài báo không có baseline cây nào, dù GBDT thắng chính cuộc thi
  Favorita 2017.
