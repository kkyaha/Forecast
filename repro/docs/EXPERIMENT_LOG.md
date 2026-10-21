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

## Lần chạy 4 — So sánh kiến trúc RNN vs LSTM vs GRU (tự đề xuất cải tiến)

### Động cơ

Tác giả dùng RNN vanilla (`nn.RNN`) cho cả Teacher/Baseline/Student trên Mackey-Glass —
kiến trúc đơn giản nhất trong họ hồi quy, dễ gặp vanishing/exploding gradient hơn LSTM/GRU.
Đây là phần **tự đề xuất cải tiến** (không có trong code hay paper gốc), nhằm kiểm tra: kiến
trúc mạnh hơn có giúp FGL cải thiện nhiều hơn không, hay mức cải thiện tương đối giữa
Baseline/Student là bất biến theo kiến trúc.

### Thay đổi code

Tham số hoá `class RNN` trong `build/mackey_glass/utils/utils.py` bằng `cell_type ∈
{rnn, lstm, gru}` (mặc định `'rnn'`, không đổi hành vi cũ), và thêm cờ `--arch` vào
`exp/base_exp.py`, áp dụng **cùng kiến trúc cho cả Teacher/Baseline/Student** (giữ đúng
nguyên tắc "1 họ kiến trúc dùng chung" của paper, chỉ đổi loại cell). Đã xác nhận
`--arch rnn` cho ra đúng số liệu y hệt Lần chạy 1 (Teacher 5.4391, Baseline 139.3832,
Student 104.3942) trước khi chạy thử LSTM/GRU — đảm bảo refactor không làm lệch kết quả gốc.

### Lệnh chạy (từ `repro/build/mackey_glass/`)

```
for arch in lstm gru; do
  for s in 0 1 2 3 4 42; do
    PYTHONPATH=. python exp/base_exp.py --horizon 5 --num_bins 50 --alpha 0.5 --epochs 20 --arch $arch --seed $s
  done
done
```

### Kết quả — LSTM (6 seed)

| seed | Teacher MSE | Baseline MSE | Student MSE | Giảm MSE (Baseline→Student) |
|---|---|---|---|---|
| 0  | 4,1850 | 145,4412 | 92,5198 | **36,39%** |
| 1  | 4,6782 | 138,9365 | 89,1096 | **35,86%** |
| 2  | 4,6281 | 142,7424 | 94,6453 | **33,70%** |
| 3  | 4,1689 | 139,3032 | 92,2201 | **33,80%** |
| 4  | 4,7799 | 143,8094 | 98,7154 | **31,36%** |
| 42 | 4,4145 | 144,6108 | 97,3967 | **32,65%** |
| **Trung bình** | 4,476 | 142,477 | 94,101 | **33,96% (±1,74 std)** |

### Kết quả — GRU (6 seed)

| seed | Teacher MSE | Baseline MSE | Student MSE | Giảm MSE (Baseline→Student) |
|---|---|---|---|---|
| 0  | 4,4406 | 143,1596 | 98,4052 | **31,26%** |
| 1  | 4,5018 | 142,7224 | 98,0985 | **31,27%** |
| 2  | 4,2662 | 140,9470 | 94,1331 | **33,21%** |
| 3  | 4,2195 | 146,1191 | 97,3077 | **33,41%** |
| 4  | 4,4276 | 143,0985 | 87,6893 | **38,72%** |
| 42 | 4,7373 | 142,0975 | 98,7189 | **30,53%** |
| **Trung bình** | 4,432 | 142,197 | 95,726 | **33,07% (±2,74 std)** |

### So sánh 3 kiến trúc (trung bình 6 seed)

| Kiến trúc | Giảm MSE trung bình | So với RNN |
|---|---|---|
| RNN (gốc tác giả) | 26,42% (±3,86) | — |
| **LSTM** | **33,96% (±1,74)** | **+7,54 điểm %** |
| **GRU** | **33,07% (±2,74)** | **+6,65 điểm %** |

### Nhận định

- **Cả LSTM và GRU đều cải thiện trên 6/6 seed**, không có ngoại lệ — kết quả ổn định, không
  phải may rủi của 1 seed cụ thể.
- **LSTM nhỉnh hơn GRU** cả về mức cải thiện trung bình (33,96% vs 33,07%) lẫn độ ổn định
  (std 1,74 vs 2,74) — hợp lý vì LSTM có thêm cổng quên/nhớ (forget/cell gate) kiểm soát
  dòng thông tin tốt hơn GRU trên chuỗi có tính hỗn loạn cao như Mackey-Glass.
- **MSE của Teacher cũng giảm rõ rệt** khi đổi kiến trúc (RNN ~5,86 trung bình ở Lần chạy 2
  → LSTM ~4,48 → GRU ~4,43) — tức bản thân nhiệm vụ dự báo 1-bước (của Teacher) cũng dễ hơn
  với kiến trúc mạnh hơn, không chỉ nhiệm vụ H-bước (của Student).
- **Kết luận cải tiến**: đổi kiến trúc hồi quy từ RNN vanilla sang LSTM là một cải tiến đơn
  giản (chỉ đổi 1 tham số dòng lệnh, không đổi logic FGL/KD) nhưng mang lại cải thiện đáng
  kể và nhất quán — phù hợp để đưa vào phần "đóng góp thêm" của báo cáo, đi kèm lưu ý: đây
  là thử nghiệm tự đề xuất, không có trong paper gốc, nên trình bày rõ ràng là phần mở rộng
  của người làm tiểu luận, không phải kết quả tác giả công bố.

---

## Lần chạy 5 — Time-series cross-validation (rolling-origin) trên 3 kiến trúc

### Động cơ

Lần chạy 4 (sweep 6 seed) chỉ đổi khởi tạo trọng số ngẫu nhiên, trong khi điểm chia train/
val/test trên chuỗi Mackey-Glass luôn cố định (60/20/20 đầu chuỗi). Để kiểm tra riêng ảnh hưởng
của **vị trí chia dữ liệu** tới mức cải thiện FGL, thêm thực nghiệm time-series cross-
validation kiểu rolling-origin (mở rộng cửa sổ, tương đương `sklearn.TimeSeriesSplit`). Lưu ý:
đây **không phải** k-fold chuẩn (xáo trộn ngẫu nhiên) — xáo trộn sẽ làm rò dữ liệu tương lai
vào lúc huấn luyện, không hợp lệ với time series.

### Thiết kế

- Thêm tham số `external_data` (mặc định `None`) vào `train_student_model` trong
  `build/mackey_glass/exp/base_exp.py`, để truyền vào một prefix của `data.pkl` thay vì luôn
  đọc toàn bộ file — không đổi hành vi cũ, đã xác nhận `--horizon 5 --num_bins 50 --alpha 0.5
  --epochs 20` (không truyền `external_data`) vẫn ra đúng số liệu Lần chạy 1
  (Teacher 5.4391, Baseline 139.3832, Student 104.3942).
- File mới `build/mackey_glass/exp/cv_exp.py`: dùng 3 prefix ngày càng dài của `data.pkl`
  (70%, 85%, 100% của 10000 điểm) làm 3 "fold"; trong mỗi fold, logic chia 60/20/20 cũ của
  `create_time_series_dataset` vẫn áp dụng như bình thường (không sửa file này) — vì vậy tập
  test của mỗi fold nằm ở một đoạn càng về sau trên chuỗi gốc (expanding window đúng kiểu
  rolling-origin CV).

### Lệnh chạy (từ `repro/build/mackey_glass/`)

```
PYTHONPATH=. python exp/cv_exp.py --horizon 5 --num_bins 50 --alpha 0.5 --epochs 20 --seed 42 --archs rnn,lstm,gru
```

### Kết quả

| Fold (% dữ liệu) | Kiến trúc | Teacher MSE | Baseline MSE | Student MSE | Giảm MSE |
|---|---|---|---|---|---|
| 1 (70%, N=7000)  | RNN  | 6,9168 | 148,8392 | 112,5397 | 24,39% |
| 1 (70%, N=7000)  | LSTM | 5,8645 | 145,9056 | 98,0264  | 32,82% |
| 1 (70%, N=7000)  | GRU  | 5,8000 | 140,2981 | 101,9886 | 27,31% |
| 2 (85%, N=8500)  | RNN  | 5,3982 | 139,8405 | 98,6315  | 29,47% |
| 2 (85%, N=8500)  | LSTM | 4,5499 | 136,1424 | 91,5927  | 32,72% |
| 2 (85%, N=8500)  | GRU  | 4,6071 | 139,0277 | 96,8740  | 30,32% |
| 3 (100%, N=10000)| RNN  | 5,4391 | 139,3832 | 104,3942 | 25,10% |
| 3 (100%, N=10000)| LSTM | 4,4145 | 144,6108 | 97,3967  | 32,65% |
| 3 (100%, N=10000)| GRU  | 4,7373 | 142,0975 | 98,7189  | 30,53% |

| Kiến trúc | Giảm MSE trung bình (3 fold) |
|---|---|
| RNN | 26,32% (±2,25) |
| LSTM | **32,73% (±0,07)** |
| GRU | 29,38% (±1,47) |

### Nhận định

- **Fold 3 (100% dữ liệu) khớp chính xác từng số lẻ** với số liệu seed=42 của Lần chạy 1/2/4
  (RNN 25,10%, LSTM 32,65%, GRU 30,53%) — vì fold 3 dùng đúng cấu hình cũ (toàn bộ chuỗi,
  60/20/20) — xác nhận script `cv_exp.py` cài đặt đúng, không có lỗi âm thầm làm lệch số liệu.
- **Cả 3 kiến trúc cải thiện dương trên cả 3 fold**, không có trường hợp âm — khác với Lần
  chạy 3 (quét horizon), nơi horizon=2 là ngoại lệ âm.
- **LSTM có độ lệch chuẩn giữa fold rất nhỏ (0,07 điểm %)**, nhỏ hơn hẳn RNN (2,25) và GRU
  (1,47) — mức cải thiện của LSTM ổn định không chỉ qua seed khởi tạo (Lần chạy 4) mà còn qua
  vị trí chia dữ liệu khác nhau trên chuỗi — bằng chứng bổ sung ủng hộ chọn LSTM.
- Hạn chế: chỉ 3 fold, 1 seed mỗi fold (không lặp seed × fold đầy đủ do giới hạn thời gian máy
  cá nhân, batch_size=1 mặc định của tác giả khiến mỗi lần train khá chậm trên CPU) — nên kết
  quả là "bằng chứng bổ sung", chưa phải kiểm định thống kê đầy đủ (seed × fold đầy đủ sẽ cần
  nhiều lần chạy hơn).

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

---

## Lần chạy 6 — CHB-MIT trên Kaggle (patient 1), bản gốc tác giả (đã tái hiện đầy đủ 3 Stage)

**Về nguồn code**: chạy trực tiếp từ clone sạch của repo gốc tác giả
(`SkyeGunasekaran/FutureGuidedLearning`, thư mục Kaggle
`/kaggle/working/FutureGuidedLearning/CHBMIT`) — **không** phải từ kho cá nhân
`kkyaha/Forecast` và **không** dùng cờ `--bidirectional` (cờ này không tồn tại trong repo gốc,
truyền vào sẽ báo lỗi `unrecognized arguments` ngay lập tức — tức nếu lệnh chạy thành công và
ra kết quả, chắc chắn không dùng cờ đó). Đã xác nhận lại với người chạy thực nghiệm và đối
chiếu đúng dòng lệnh gốc. Đây là số liệu tái hiện thuần túy, dùng cho báo cáo giữa kỳ.

### Lệnh đã chạy (patient 1; patient 3, 5 không chạy được do chưa tải dữ liệu)

```
PYTHONPATH=. python exp/seizure_detection.py --patient 1 --epochs 50 --patience 5     # x3 lần
python -m exp.seizure_prediction --patient 1 --model CNN_LSTM --trials 3 --epochs 25  # x3 lần (x3 trial/lần)
python -m exp.FGL_CHBMIT --patient 1 --epochs 30 --trials 3 --optimizer Adam --alpha 0.5 --temperature 4   # x3 lần (x3 trial/lần)
```

### Kết quả

**Stage 1 — Teacher/Detection, AUC-ROC, 3 lần chạy:**

| Lần | AUC-ROC |
|---|---|
| 1 | 0,6252 |
| 2 | 0,6431 |
| 3 | 0,6669 |
| **Trung bình** | **0,6451 (±0,0171)** |

So với lần chạy local trước đó (AUC=0,28, thấp hơn ngẫu nhiên, mục "Trạng thái tái hiện trên
máy này" ở `PHAN_TICH_LY_THUYET_VA_CODE.md`): cả 3 lần chạy trên Kaggle đều cho AUC hợp lý
(0,63–0,67, cao hơn ngẫu nhiên 0,5 rõ rệt) — ngược hẳn với kết quả local. Vì script
`seizure_detection.py` không cố định seed (`torch.manual_seed`), nhiều khả năng lần chạy local
0,28 là một lần khởi tạo ngẫu nhiên không may, không phải lỗi hệ thống do `shuffle=False` như
nghi vấn trước đó — nhưng nghi vấn `shuffle=False` vẫn là một quan sát hợp lệ về thiết kế code,
chưa bị bác bỏ hoàn toàn, chỉ là chưa đủ bằng chứng khẳng định đó là *nguyên nhân chính*.

**Stage 2 — Baseline (Seizure Prediction, không FGL), [FPR, Sensitivity, AUC-ROC] × 3 trial × 3 lần chạy:**

| Lần chạy | Trial 1 | Trial 2 | Trial 3 |
|---|---|---|---|
| 1 | [0,052; 1,000; 0,9972] | [0,082; 0,967; 0,9872] | [0,048; 0,725; 0,8610] |
| 2 | [0,015; 1,000; 0,9991] | [0,024; 0,708; 0,9217] | [0,022; 0,892; 0,9714] |
| 3 | [0,115; 0,642; 0,7815] | [0,031; 0,875; 0,9733] | [0,491; 0,992; 0,7628] |

AUC-ROC trung bình (9 trial): **0,9173 (±0,0879)**, dao động 0,7628–0,9991.

**Stage 3 — Student/FGL (bản gốc, alpha=0.5), AUC-ROC × 3 trial × 3 lần chạy:**

| Lần chạy | Trial 1 | Trial 2 | Trial 3 |
|---|---|---|---|
| 1 | 0,9881 | 0,9285 | 0,9615 |
| 2 | 0,4376 | 0,9413 | 0,9310 |
| 3 | 0,9881 | 0,9905 | 0,9948 |

AUC-ROC trung bình (9 trial): **0,9068 (±0,1678)**, dao động 0,4376–0,9948.
Loại 1 trial bất thường (lần chạy 2, trial 1 = 0,4376, gần ngẫu nhiên): 8 trial còn lại trung
bình **0,9655 (±0,0266)**.

### Nhận định

- **Teacher/Detection hợp lý trên cả 3 lần chạy** (0,63–0,67, cao hơn ngẫu nhiên 0,5 rõ rệt) —
  ngược hẳn với lần chạy local trước đó (AUC=0,28, thấp hơn ngẫu nhiên, mục "Trạng thái tái
  hiện trên máy này" ở `PHAN_TICH_LY_THUYET_VA_CODE.md`). Vì `seizure_detection.py` không cố
  định seed (`torch.manual_seed`), nhiều khả năng lần chạy local 0,28 là một lần khởi tạo ngẫu
  nhiên không may, không phải lỗi hệ thống do `shuffle=False` như nghi vấn trước đó — nghi vấn
  `shuffle=False` chưa bị bác bỏ hoàn toàn nhưng không còn đủ bằng chứng khẳng định là nguyên
  nhân chính.
- **So Baseline vs Student (FGL) — không ủng hộ rõ claim 44,8% của bài báo**: tính trên toàn bộ
  9 trial, AUC trung bình giảm nhẹ (0,9173 → 0,9068, **−1,14%**), chủ yếu do 1 trial sụp (AUC
  0,4376). Nếu loại trial này, Student trung bình **0,9655**, cao hơn Baseline (0,9173) khoảng
  **+5,25%** — vẫn thấp hơn rất nhiều so với 44,8% bài báo công bố, và việc loại bỏ 1/9 trial
  (hơn 11% số liệu) để có con số dương cần được nêu rõ là một lựa chọn diễn giải, không phải
  kết quả "sạch".
- **Độc lập ngẫu nhiên, không cố định seed**: cả `seizure_prediction.py` và `FGL_CHBMIT.py`
  không gọi `torch.manual_seed`, nên 9 trial của mỗi mô hình là 9 lần khởi tạo ngẫu nhiên hoàn
  toàn độc lập — độ lệch chuẩn lớn (đặc biệt Student, ±0,168, chủ yếu do 1 trial sụp) phản ánh
  đúng mức bất ổn định thực sự của việc không cố định seed trong mã nguồn tác giả, không phải
  lỗi đo đạc của báo cáo.
- **Khác biệt với Mackey-Glass**: trên Mackey-Glass, FGL cải thiện nhất quán trên toàn bộ 6/6
  seed (Lần chạy 2). Trên CHB-MIT, FGL **không** cho thấy cải thiện nhất quán tương tự — đây là
  một phát hiện quan trọng cần nêu trung thực trong báo cáo, không nên chỉ trích dẫn mức tăng
  44,8% của bài báo mà bỏ qua việc tái hiện không khớp.


---

## Lần chạy 7 — CHB-MIT trên Kaggle, mở rộng sang bệnh nhân 3 và 5

Sau Lần chạy 6 (chỉ bệnh nhân 1, vì ban đầu chỉ tải được dữ liệu bệnh nhân này), tải thêm được
dữ liệu bệnh nhân 3 và 5 (cũng thuộc danh sách 12 bệnh nhân mặc định của tác giả), chạy đủ ba
Stage cho mỗi bệnh nhân (một lần chạy, ba trial mỗi lần cho Stage 2–3), cùng bản gốc tác giả,
không bidirectional.

### Kết quả thô

**Stage 1 — Teacher/Detection, AUC-ROC:**

| Bệnh nhân | AUC-ROC |
|---|---|
| 3 | 0,9067 |
| 5 | 0,9911 |

**Stage 2 — Baseline, [FPR, Sensitivity, AUC-ROC] × 3 trial:**

| Bệnh nhân | Trial 1 | Trial 2 | Trial 3 |
|---|---|---|---|
| 3 | [0,416; 0,617; 0,5991] | [0,999; 1,000; 0,2956] | [0,726; 0,942; 0,3882] |
| 5 | [0,979; 1,000; 0,1202] | [0,964; 1,000; 0,3890] | [0,000; 0,000; 0,0797] |

**Stage 3 — Student/FGL, AUC-ROC × 3 trial:**

| Bệnh nhân | Trial 1 | Trial 2 | Trial 3 |
|---|---|---|---|
| 3 | 0,7989 | 0,9537 | 0,4234 |
| 5 | 0,8255 | 0,1313 | 0,2135 |

### Tổng hợp cả ba bệnh nhân (1, 3, 5)

| Bệnh nhân | Baseline AUC (TB) | Student AUC (TB) | Tăng |
|---|---|---|---|
| 1 (9 trial) | 0,9173 | 0,9068 | −1,1% |
| 3 (3 trial) | 0,4276 | 0,7253 | +69,6% |
| 5 (3 trial) | 0,1963 | 0,3901 | +98,7% |

Hai cách gộp:
- **Gộp theo trial** (15 Baseline + 15 Student, không phân biệt bệnh nhân): Baseline TB
  **0,6751**, Student TB **0,7672** → tăng **+13,63%**. Bị chi phối bởi bệnh nhân 1 (9/15 =
  60% trial).
- **Trung bình của trung bình từng bệnh nhân** (mỗi bệnh nhân góp 1 điểm, không phân biệt số
  trial): Baseline TB **0,5137**, Student TB **0,6741** → tăng **+31,21%**.

### Nhận định

- **Baseline hoạt động rất khác nhau giữa bệnh nhân**: AUC từ 0,1963 (bệnh nhân 5, có 1 trial
  suy biến thành bộ phân loại toàn-lớp-âm, FPR=Sensitivity=0) đến 0,9173 (bệnh nhân 1). Đây là
  dấu hiệu `CNN_LSTM_Model` + kịch bản huấn luyện của tác giả không ổn định giữa các bệnh nhân
  có đặc điểm dữ liệu khác nhau (lượng ictal, tỉ lệ mất cân bằng lớp...).
- **Mức tăng tương đối lớn nhất rơi vào bệnh nhân có Baseline kém nhất** (3, 5) — gợi ý FGL có
  thể "cứu" được các trường hợp Baseline thất bại nặng, nhưng mức tăng % lớn một phần do mẫu số
  (AUC Baseline) quá nhỏ, không hẳn vì Student đạt AUC tuyệt đối cao (Student bệnh nhân 5 vẫn
  chỉ 0,39 — còn khá kém).
- **Cả 2 cách gộp đều thấp hơn 44,8% công bố**, nhưng cách "trung bình của trung bình theo bệnh
  nhân" (+31,21%) gần hơn nhiều so với cách gộp theo trial (+13,63%) — gợi ý nếu bài báo tính
  trung bình theo đơn vị bệnh nhân (không theo trial) trên nhiều bệnh nhân hơn, con số có thể
  tiến gần 44,8% hơn những gì quan sát được ở đây với chỉ 3 bệnh nhân, 1-3 lần chạy mỗi bệnh
  nhân.
- **Cỡ mẫu còn rất nhỏ** (bệnh nhân 3, 5 mới có 1 lần chạy = 3 trial mỗi bệnh nhân) để kết luận
  chắc chắn — cần chạy lại nhiều lần hơn cho từng bệnh nhân, và mở rộng sang các bệnh nhân khác
  trong danh sách 12 bệnh nhân mặc định của tác giả, mới có thể so sánh công bằng với con số
  44,8% (có thể được tính trên toàn bộ hoặc phần lớn 12 bệnh nhân này).
