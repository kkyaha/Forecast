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

<!-- KẾT QUẢ SẼ ĐƯỢC ĐIỀN SAU KHI CHẠY XONG -->

