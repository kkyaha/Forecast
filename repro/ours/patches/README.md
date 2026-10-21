# Patches — cải tiến tự đề xuất (không phải code tác giả)

Thư mục này lưu các thay đổi **do người làm tiểu luận tự đề xuất** trên nền code đã tái hiện
(`repro/build/`), tách biệt khỏi:
- `repro/upstream/` — code gốc tác giả, không sửa.
- `repro/build/` — bản đã vá để **chạy được** (sửa lỗi, tương thích môi trường — xem mục
  "Lỗi phát hiện trong code tác giả" ở `repro/docs/EXPERIMENT_LOG.md`), vẫn là tái hiện
  đúng phương pháp của paper, không phải cải tiến.

Mỗi file `.patch` ở đây là một `git diff` áp được trực tiếp lên `repro/build/`, đại diện cho
**một cải tiến cụ thể, ngoài phạm vi paper gốc**.

## `arch_lstm_gru.patch`

**Cải tiến**: cho phép chọn kiến trúc hồi quy (`RNN` vanilla / `LSTM` / `GRU`) cho cả ba vai
trò Teacher/Baseline/Student trên Mackey-Glass, thay vì chỉ RNN vanilla như code gốc.

**Thay đổi**:
- `repro/build/mackey_glass/utils/utils.py`: tham số hoá `class RNN` bằng `cell_type ∈
  {rnn, lstm, gru}` (mặc định `'rnn'` — hành vi cũ giữ nguyên 100%, đã xác nhận bằng cách
  chạy lại `--arch rnn` ra đúng số liệu Lần chạy 1 trong `EXPERIMENT_LOG.md`).
- `repro/build/mackey_glass/exp/base_exp.py`: thêm cờ `--arch rnn|lstm|gru`.

**Kết quả** (sweep 6 seed, xem đầy đủ ở `repro/docs/EXPERIMENT_LOG.md`, mục "Lần chạy 4"):

| Kiến trúc | Giảm MSE trung bình |
|---|---|
| RNN (gốc tác giả) | 26,42% |
| LSTM | **33,96%** |
| GRU | **33,07%** |

**Áp patch** (nếu cần tái tạo từ `repro/build/` sạch, ví dụ sau khi clone lại từ
`repro/upstream/` và vá theo `EXPERIMENT_LOG.md`):
```
cd "repro/build/mackey_glass"   # patch dùng đường dẫn tương đối từ root repo, áp từ root:
cd ../../..
git apply repro/ours/patches/arch_lstm_gru.patch
```

**Notebook chạy thử trên Kaggle**: bản độc lập (không cần clone repo, tự sinh dữ liệu
Mackey-Glass) nằm tại `repro/notebook/FGL_MackeyGlass_Arch_Kaggle.ipynb`.

## `chbmit_bidirectional_lstm.patch`

**Cải tiến**: thêm lựa chọn LSTM 2 chiều (Bidirectional) cho `CNN_LSTM_Model` dùng ở cả
Teacher (Stage 1 — seizure detection) và Student/FGL (Stage 3 — seizure prediction) trên
CHB-MIT, thay vì chỉ LSTM 1 chiều như code gốc. Cùng tinh thần với `arch_lstm_gru.patch`
trên Mackey-Glass: LSTM 2 chiều cho mỗi bước thời gian "thấy" được cả ngữ cảnh trước lẫn sau
trong cùng cửa sổ 30 giây, thường cải thiện phân loại chuỗi EEG.

**Thay đổi**:
- `repro/build/CHBMIT/models/model.py`: tham số hoá `CNN_LSTM_Model` bằng `bidirectional:
  bool` (mặc định `False` — hành vi cũ giữ nguyên 100%; đã kiểm chứng cơ chế LSTM+fc1 với
  input giả lập cho cả 2 chế độ, ra đúng shape logits `(batch, 2)`).
- `repro/build/CHBMIT/exp/seizure_detection.py`, `exp/FGL_CHBMIT.py`: thêm cờ
  `--bidirectional`.

**Kết quả**: chưa có số liệu BiLSTM thật. Các lần chạy trên Kaggle tưởng là bản BiLSTM (bệnh
nhân 1, 3, 5) hoá ra là chạy trực tiếp từ clone sạch repo gốc tác giả (không qua kho
`kkyaha/Forecast`, không dùng cờ `--bidirectional`) — số liệu đó là **bản gốc**, đã chuyển sang
dùng cho báo cáo giữa kỳ (phần tái hiện), xem `EXPERIMENT_LOG.md` mục "Lần chạy 6" (bệnh nhân
1) và "Lần chạy 7" (bệnh nhân 3, 5). Dự định chạy lại đúng bản BiLSTM (qua `kkyaha/Forecast`,
có cờ `--bidirectional`) cho báo cáo cuối kỳ.

**Áp patch** (lên `repro/build/CHBMIT/` sạch, ví dụ sau khi vá lại theo `EXPERIMENT_LOG.md`):
```
git apply repro/ours/patches/chbmit_bidirectional_lstm.patch   # chạy từ root repo
```

**Notebook chạy trên Kaggle**: `repro/notebook/FGL_CHBMIT_Kaggle.ipynb` — Stage 1 và Stage 3
mặc định chạy bản gốc (không `--bidirectional`), dùng cho báo cáo giữa kỳ. Biến thể BiLSTM nằm
ở cell riêng ngay dưới mỗi Stage (lệnh bị comment `#`, bỏ dấu `#` để chạy cho báo cáo cuối kỳ).

## `mg_timeseries_cv.patch`

**Cải tiến**: thêm kịch bản time-series cross-validation (rolling-origin / expanding-window,
tương đương `sklearn.TimeSeriesSplit`) cho Mackey-Glass, để kiểm tra độ bền của mức cải thiện
FGL khi đổi **vị trí chia** train/test trên chuỗi — bổ sung cho sweep 6 seed ở
`arch_lstm_gru.patch` (chỉ đổi khởi tạo ngẫu nhiên, không đổi điểm chia dữ liệu). **Không** dùng
k-fold chuẩn (xáo trộn ngẫu nhiên) vì sẽ làm rò dữ liệu tương lai vào lúc huấn luyện.

**Thay đổi**:
- `repro/build/mackey_glass/exp/base_exp.py`: thêm tham số tùy chọn `external_data` cho
  `train_student_model` (mặc định `None` — hành vi cũ giữ nguyên 100%, đã xác nhận chạy lại
  `--horizon 5 --num_bins 50 --alpha 0.5 --epochs 20` ra đúng số liệu Lần chạy 1); đổi
  `return None` thành `return {"teacher", "baseline", "student"}` MSE (không phá vỡ gì vì
  `__main__` cũ không dùng giá trị trả về).
- `repro/build/mackey_glass/exp/cv_exp.py` (file mới): dùng 3 đoạn đầu chuỗi ngày càng dài
  (70%, 85%, 100%) làm 3 fold, chạy cả RNN/LSTM/GRU trên mỗi fold.

**Kết quả** (3 fold, seed 42, xem đầy đủ ở `repro/docs/EXPERIMENT_LOG.md`):

| Fold (% dữ liệu) | RNN | LSTM | GRU |
|---|---|---|---|
| 1 (70%) | 24,39% | 32,82% | 27,31% |
| 2 (85%) | 29,47% | 32,72% | 30,32% |
| 3 (100%) | 25,10% | 32,65% | 30,53% |
| **Trung bình** | 26,32% (±2,25) | **32,73% (±0,07)** | 29,38% (±1,47) |

Fold 3 (100% dữ liệu) khớp *chính xác* với số liệu seed=42 đã có ở `arch_lstm_gru.patch` —
xác nhận script chạy đúng. Cả 3 kiến trúc cải thiện dương trên cả 3 fold; LSTM có độ lệch
chuẩn nhỏ nhất (0,07), ổn định nhất qua các vị trí chia dữ liệu khác nhau.

**Áp patch**:
```
git apply repro/ours/patches/mg_timeseries_cv.patch   # chạy từ root repo
```

**Chạy**: từ `repro/build/mackey_glass/`,
`PYTHONPATH=. python exp/cv_exp.py --horizon 5 --num_bins 50 --alpha 0.5 --epochs 20 --seed 42`.
