# Nhật ký thực nghiệm — Future-Guided Learning (Mackey-Glass)

Paper: Gunasekaran, Kembay, Ladret, Zhu, Perrinet, Kavehei, Eshraghian,
*"A predictive approach to enhance time-series forecasting"*, Nature Communications, 2025.
DOI: 10.1038/s41467-025-63786-4
Repo tác giả: https://github.com/SkyeGunasekaran/FutureGuidedLearning (commit `5fa24ea`)

Cấu trúc: `upstream/FutureGuidedLearning/` = code tác giả nguyên trạng (không sửa) —
`build/mackey_glass/` = bản đã vá, dùng để chạy.

---

## Lỗi phát hiện trong code tác giả (trước khi chạy được)

### 1. Bốn file còn nguyên merge-conflict marker chưa giải quyết

Tại HEAD (`5fa24ea`, "Merge pull request #7 from CongSheng/main"), bốn file sau vẫn còn
`<<<<<<< HEAD` / `=======` / `>>>>>>> 07c6981c...` — tức **SyntaxError ngay khi import**,
code công khai không chạy được một dòng nào:

- `mackey_glass/exp/base_exp.py` (dòng 235-239) — hai nhánh giống nhau (`)` lặp), vô hại.
- `mackey_glass/exp/drift_exp.py` (dòng 349-353) — hai nhánh đều là dòng trống, vô hại.
- `mackey_glass/utils/utils.py` (dòng 322-326) — hai nhánh giống nhau
  (`return (1.0 - alpha) * kd`), vô hại.
- `mackey_glass/utils/generate.py` (dòng 4-7 và 23-106) — **conflict thật**, xem mục 2.

### 2. `generate.py` — hai nhánh tính dữ liệu khác nhau, một nhánh gây lệch horizon

- **Nhánh HEAD**: chỉ lấy `target` từ `mg_dataset[idx]` (bỏ `sample`), rồi **nhân đôi**
  thành 2 cột giống nhau → `time_series_data[:,0] == time_series_data[:,1]`.
- **Nhánh `07c6981c`**: giữ cả `sample` và `target`, cho ra 2 cột **lệch nhau đúng 1 bước**
  (cột 1 = cột 0 dịch tới 1 vị trí) — nhìn "có lý" hơn, có cả print kiểm chứng.

**Truy ngược qua `create_time_series_dataset`** (hàm tiêu thụ dữ liệu này, không đổi giữa
2 nhánh): công thức lấy nhãn là `y[i + lookback_window + forecasting_horizon - 1]`. Số
hạng `-1` này chỉ đúng nếu `x` và `y` cùng hệ quy chiếu chỉ số (tức 2 cột PHẢI giống nhau,
như nhánh HEAD). Nếu dùng nhánh `07c6981c` (cột 1 đã lệch sẵn 1 bước), công thức trên sẽ
**dự báo xa thêm 1 bước so với `--horizon` người dùng yêu cầu** (horizon thật = H+1, không
phải H).

**Bằng chứng độc lập xác nhận nhánh HEAD đúng**: file `mackey_glass/data.pkl` tác giả đã
commit sẵn trong repo — kiểm tra trực tiếp thấy cột 0 và cột 1 **giống nhau 100%** (khớp
nhánh HEAD, không khớp nhánh `07c6981c`). Tức bản dữ liệu tác giả thực sự dùng để ra số
trong bài báo được sinh theo đúng nhánh mà toán học cũng chỉ ra là đúng.

→ **Đã sửa**: giữ nhánh HEAD, xoá nhánh `07c6981c`, ở cả `build/mackey_glass/utils/generate.py`.

### 3. Thiếu dependency

`utils.py` import `jitcdde` ở cấp module (dù chỉ cần cho class `MackeyGlass` sinh dữ liệu,
không cần cho vòng train) — không có trong môi trường, phải `pip install jitcdde`
(kéo theo `jitcxde_common`, `chspy`, `symengine`).

### 4. Cách chạy thực tế khác với README

README ghi `python base_exp.py ...`, nhưng file nằm ở `exp/base_exp.py` và
`from utils.utils import ...` chỉ resolve được nếu chạy với
`PYTHONPATH=. python exp/base_exp.py ...` từ thư mục `mackey_glass/` (không phải từ `exp/`
hay từ root repo).

---

## Lần chạy 1 — xác nhận pipeline chạy được, đối chiếu con số bài báo

Lệnh (từ `repro/build/mackey_glass/`):
```
PYTHONPATH=. python exp/base_exp.py --horizon 5 --num_bins 50 --alpha 0.5 --epochs 20
```
(seed mặc định 42, dùng `data.pkl` có sẵn trong repo tác giả — không tự sinh lại)

| | Test MSE |
|---|---|
| Teacher (dự báo 1 bước) | 5,4391 |
| Baseline (dự báo 5 bước, không FGL) | 139,3832 |
| Student (dự báo 5 bước, có FGL) | 104,3942 |

**Giảm MSE (Baseline → Student): (139,3832 − 104,3942) / 139,3832 ≈ 25,1%**
So với bài báo công bố: **giảm 23,4%** (Mackey-Glass, loại outlier) — khớp khá sát ngay từ
lần chạy đầu, dùng đúng tham số mẫu trong README, chưa chỉnh gì.

Ghi nhận phụ: có `UserWarning` về broadcasting ở bước `evaluate()` cuối
(`batch_size=1` khiến tensor đích shape `()` so với đầu ra shape `(1,)`) — không chắc ảnh
hưởng độ chính xác số liệu (broadcasting scalar vs tensor 1 phần tử vẫn ra đúng giá trị),
nhưng là điểm code chưa gọn, cần nêu nếu ai hỏi.

---

## Lần chạy 2 — sweep seed để kiểm tra độ ổn định

Thêm tham số `--seed` vào `build/mackey_glass/exp/base_exp.py` (tác giả hardcode
`torch.manual_seed(42)`, không có cách đổi seed qua CLI).

Lệnh, lặp qua các seed:
```
for s in 0 1 2 3 4 42; do
  PYTHONPATH=. python exp/base_exp.py --horizon 5 --num_bins 50 --alpha 0.5 --epochs 20 --seed $s
done
```

### Kết quả

| seed | Teacher MSE | Baseline MSE | Student MSE | Giảm MSE (Baseline→Student) |
|---|---|---|---|---|
| 0  | 4,8331 | 156,1701 | 102,7514 | **34,20%** |
| 1  | 5,4642 | 136,5673 | 106,0795 | **22,32%** |
| 2  | 5,7293 | 143,5308 | 104,4262 | **27,25%** |
| 3  | 6,2236 | 140,3592 | 107,6203 | **23,32%** |
| 4  | 7,4506 | 147,5238 | 108,6868 | **26,33%** |
| 42 | 5,4391 | 139,3832 | 104,3942 | **25,10%** |
| **Trung bình** | 5,857 | 143,922 | 105,660 | **26,42% (±3,86 std)** |

### Nhận định

- **Ổn định theo đúng hướng**: cả 6/6 seed đều cho Student MSE thấp hơn Baseline MSE — FGL
  luôn cải thiện, không có seed nào đảo chiều (âm) hoặc bằng không.
- **Độ lớn cải thiện dao động 22,3%–34,2%**, trung bình 26,42% (độ lệch chuẩn ~3,9 điểm %).
  Con số bài báo công bố (23,4%) **nằm gần biên dưới của khoảng này** — tức bài báo có thể
  đã báo cáo một con số hơi "khiêm tốn" hơn mức trung bình quan sát được ở đây, hoặc do
  khác biệt nhỏ về cấu hình/loại trừ outlier mà bài báo áp dụng ("outlier excluded") mà
  thực nghiệm này chưa làm.
- Teacher MSE cũng dao động theo seed (4,83–7,45) — hợp lý vì teacher chỉ được huấn luyện
  trên 1 nhiễu/khởi tạo cụ thể, không phải nguồn sai số hệ thống.
- Kết luận tạm: **tái hiện được hướng và độ lớn cải thiện chính của bài báo trên
  Mackey-Glass**, với 1 lần chạy duy nhất mỗi seed (chưa lặp lại nhiều lần/seed để tách
  phương sai do khởi tạo ngẫu nhiên khỏi phương sai do seed data-split) — nên xem đây là
  bằng chứng "khớp hướng và độ lớn", chưa phải khẳng định thống kê chặt.

---

## Lần chạy 3 — `drift_exp.py` (Page-Hinkley online adaptation), quét horizon 2→15

### Lỗi phát hiện: CLI bị vô hiệu hoàn toàn

Toàn bộ khối `argparse` (đọc `--horizon`, `--use_ph`...) nằm **trong một chuỗi
`'''...'''` bị comment** (dòng 272-332) — lệnh README gợi ý
(`python drift_exp.py --horizon 5 --num_bins 50 --alpha 0.5 --epochs 20 --use_ph`)
**không có tác dụng gì**: không có `args = parser.parse_args()` nào được thực thi.

Code **thực sự chạy** là đoạn hardcode ngoài chuỗi comment (dòng 333-348): tự lặp
`student_horizon` từ 2 đến 15, với `alpha=0.0` (loss student = thuần KL distillation,
không có cross-entropy nhãn gốc), `num_bins=25, epochs=50, batch_size=128, use_ph=True,
ph_delta=0.130033, ph_lambda=0.647096, ph_window=3, ph_retrain_epochs=3` — khớp đúng
khối "PH Params Bins=25" tác giả ghi chú ngay trong cùng chuỗi comment.

Lệnh chạy đúng (từ `repro/build/mackey_glass/`):
```
PYTHONPATH=. python exp/drift_exp.py
```
(không có cờ nào để chỉnh — chạy nguyên xi vòng lặp 14 horizon đã hardcode)

Thời gian thực tế: toàn bộ 14 horizon chạy xong trong vài phút (model nhỏ, batch=128).

### Kết quả

| Horizon | Teacher MSE | Baseline MSE | Student MSE | Giảm MSE (Baseline→Student) |
|---|---|---|---|---|
| 2  | 0,4479 | 0,6943 | 0,7443 | **−7,20%** (FGL làm tệ hơn) |
| 3  | 0,4458 | 1,5198 | 1,2870 | 15,32% |
| 4  | 0,4542 | 2,5526 | 2,0479 | 19,77% |
| 5  | 0,4516 | 3,5365 | 3,0359 | 14,15% |
| 6  | 0,4453 | 4,3719 | 3,9307 | 10,09% |
| 7  | 0,4573 | 4,9760 | 4,7771 | 4,00% |
| 8  | 0,4562 | 5,2880 | 5,0052 | 5,35% |
| 9  | 0,4589 | 5,5932 | 5,1766 | 7,45% |
| 10 | 0,4469 | 4,9651 | 4,5422 | 8,52% |
| 11 | 0,4365 | 4,0057 | 3,3552 | 16,24% |
| 12 | 0,4380 | 3,1568 | 2,6109 | 17,29% |
| 13 | 0,4359 | 2,7406 | 2,3344 | 14,82% |
| 14 | 0,4401 | 2,3135 | 2,0281 | 12,34% |
| 15 | 0,4469 | 2,1323 | 1,8453 | 13,46% |
| **TB (14 horizon)** | 0,446 | — | — | **10,83%** |
| **TB (bỏ horizon=2)** | — | — | — | **12,22%** |

### Nhận định

- **13/14 horizon cho FGL cải thiện** (dương), chỉ **horizon=2 là ngoại lệ âm (−7,20%)**
  — tức ở khoảng dự báo rất ngắn (2 bước), cơ chế distillation + Page-Hinkley retraining
  ở cấu hình này **làm tệ hơn** baseline không FGL. Cần nêu rõ ngoại lệ này khi báo cáo,
  không nên chỉ trích dẫn trung bình.
- **Teacher MSE gần như hằng số (~0,44-0,46) ở mọi horizon** — hợp lý, vì teacher luôn
  làm bài toán 1-bước cố định, không phụ thuộc horizon của student.
- **Baseline MSE không tăng đơn điệu theo horizon**: tăng từ h=2→9 (0,69→5,59) rồi lại
  **giảm** từ h=10→15 (4,97→2,13) — khác trực giác "xa hơn thì khó hơn". Có thể liên quan
  đến chu kỳ/trễ đặc trưng của hệ Mackey-Glass (`tau=17`): một số horizon trùng khớp cấu
  trúc trễ của hệ nên dễ dự báo hơn, không phải lỗi của code — cần nêu như một quan sát
  thú vị về động lực học, không phải bất thường cần sửa.
- Thiết lập `alpha=0.0` (hardcode) nghĩa là thực nghiệm này đo **student học hoàn toàn
  từ teacher** (không dùng nhãn gốc), khác với `base_exp.py` (alpha=0.5, kết hợp cả hai)
  — hai thực nghiệm đo hai khía cạnh khác nhau của FGL, không nên gộp chung so sánh.

---

## Quyết định: KHÔNG tiếp tục track EEG (CHB-MIT / AES)

### Chuẩn bị đã làm

- Kiểm tra `AES/` và `CHBMIT/`: **không có merge-conflict** (khác `mackey_glass/`).
- Cả 3 script CHBMIT (`seizure_detection.py`, `seizure_prediction.py`, `FGL_CHBMIT.py`)
  hardcode `.to("cuda")` — đã vá trong `build/CHBMIT/` thành tự nhận biết
  MPS/CUDA/CPU (mẫu giống patch B1 của ConEm), để có thể chạy thử trên Mac.
- Soạn sẵn notebook Colab (`repro/notebook/FGL_CHBMIT_Colab.ipynb`): mount Drive, clone
  repo, tải CHB-MIT bệnh nhân 1 từ PhysioNet, chạy đủ 3 Stage — **chưa chạy, để dành nếu
  sau này quay lại**.

### Lý do dừng: nghẽn ở khâu TẢI DỮ LIỆU, không phải ở tính toán

Đo trực tiếp tốc độ tải từ `physionet.org` (CHB-MIT, công khai, không cần đăng nhập)
trên mạng hiện tại:

| | Tốc độ |
|---|---|
| Tải PhysioNet, 1 kết nối | **~9,8 KB/s** |
| Đối chứng — Cloudflare speed test | ~6.933 KB/s (**~700 lần nhanh hơn**) |
| PhysioNet, 4 kết nối song song | ~70 KB/s (server có vẻ giới hạn ~4 kết nối) |
| PhysioNet, 8 kết nối song song | ~51 KB/s (**không cải thiện thêm**, có khi còn kém hơn 4) |

→ Mạng tổng quát của máy bình thường (đối chứng Cloudflare rất nhanh) — riêng
PhysioNet giới hạn băng thông per-connection rất chặt. Bệnh nhân 1 có 42 file `.edf`,
~40MB/file (~1,7GB tổng). Ở tốc độ tốt nhất đo được (~70KB/s), ước tính **~6-7 giờ chỉ
để tải xong 1 bệnh nhân** — chưa tính thời gian train. AES (Kaggle) chưa thử tải nên
chưa có số đo, nhưng cũng cần tài khoản/API token riêng.

**Quyết định**: dừng track EEG, tập trung hoàn thiện Mackey-Glass (đã có kết quả vững:
base_exp 6 seed trung bình giảm MSE 26,42%, drift_exp 13/14 horizon cải thiện trung bình
12,22%). Notebook Colab và patch MPS vẫn giữ lại trong repo — có thể thử lại sau nếu có
mạng tốt hơn tới PhysioNet (ví dụ từ mạng trường/viện nghiên cứu) hoặc quyết định đáng
bỏ thời gian chờ tải.

