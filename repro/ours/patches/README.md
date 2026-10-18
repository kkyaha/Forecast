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
