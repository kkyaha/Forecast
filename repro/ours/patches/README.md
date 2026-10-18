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

**Kết quả**: chưa có số liệu tái hiện trên CHB-MIT patient 1 trên máy local (Stage 2-3 từng
crash do OOM — xem `EXPERIMENT_LOG.md`). Dự định chạy thử trên Kaggle (dữ liệu đã upload
sẵn), xem notebook bên dưới.

**Áp patch** (lên `repro/build/CHBMIT/` sạch, ví dụ sau khi vá lại theo `EXPERIMENT_LOG.md`):
```
git apply repro/ours/patches/chbmit_bidirectional_lstm.patch   # chạy từ root repo
```

**Notebook chạy trên Kaggle**: `repro/notebook/FGL_CHBMIT_Kaggle.ipynb` — clone thẳng từ
`https://github.com/kkyaha/Forecast.git` (đã có sẵn patch này trong code, không cần vá lúc
chạy), nạp dữ liệu CHB-MIT từ Kaggle Dataset có sẵn, chạy Stage 1 + Stage 3 với
`--bidirectional`.
