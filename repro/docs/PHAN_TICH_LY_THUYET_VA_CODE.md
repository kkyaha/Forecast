# Phân tích lý thuyết & code — Future-Guided Learning (FGL)

Tài liệu tham khảo để viết báo cáo tiểu luận / làm slide. Mọi khẳng định về **code** trong
tài liệu này đều được kiểm chứng trực tiếp bằng cách đọc source trong
`repro/build/{mackey_glass,CHBMIT}/` và `repro/upstream/FutureGuidedLearning/` (không suy
diễn). Khẳng định về **paper** lấy từ README chính thức của tác giả (không lấy được bản PDF
đầy đủ do Nature Communications chặn fetch tự động — nếu cần trích dẫn abstract/số liệu
chính xác 100%, nên mở trực tiếp DOI bên dưới và đọc bản gốc).

> **Paper:** Gunasekaran, Kembay, Ladret, Zhu, Perrinet, Kavehei, Eshraghian (2025),
> *"A predictive approach to enhance time-series forecasting"*, Nature Communications 16,
> 8645. DOI: [10.1038/s41467-025-63786-4](https://doi.org/10.1038/s41467-025-63786-4)
> **Code gốc:** https://github.com/SkyeGunasekaran/FutureGuidedLearning (commit `5fa24ea`)

---

## 1. Bài toán & ý tưởng cốt lõi

### 1.1 Vấn đề

Dự báo thời gian thực (time-series event forecasting) có một đánh đổi cố hữu: dự báo
**sớm** (long-horizon — ví dụ "cơn động kinh sẽ xảy ra trong 30 phút nữa") thì **hữu dụng**
nhưng **khó/kém chính xác** (nhiễu, bất định tích lũy theo thời gian); dự báo **muộn**
(short-horizon — "cơn động kinh đang xảy ra ngay bây giờ") thì **dễ/chính xác** nhưng
**vô dụng** về mặt ứng dụng (biết để làm gì khi chuyện đã xảy ra?).

### 1.2 Ý tưởng: Future-Guided Learning (FGL)

Thay vì bắt một model duy nhất tự học đánh đổi đó, FGL **tách thành 2 model**:

- **Teacher**: được cho "nhìn" vào dữ liệu ở *tương lai gần hơn so với student* (gần/ngay
  tại thời điểm biến cố xảy ra) → bài toán của teacher **dễ**, học nhanh, chính xác cao.
- **Student**: chỉ được nhìn dữ liệu ở *quá khứ tương đối* (trước khi biến cố xảy ra) →
  bài toán **khó** (đây mới là bài toán thực sự cần dùng).
- **Cầu nối**: trong lúc train student, cho cùng một batch input đi qua **cả teacher và
  student**, rồi ép phân phối xác suất đầu ra của student **bắt chước** phân phối của
  teacher (chưng cất tri thức — *knowledge distillation*, KD) bằng Kullback–Leibler (KL)
  divergence, **cộng thêm** cross-entropy loss chuẩn so với nhãn thật.

Tác giả nói ý tưởng này lấy cảm hứng từ **predictive coding** (lý thuyết não bộ: não luôn
dự báo trước tín hiệu cảm giác sẽ đến, rồi sửa sai bằng chênh lệch dự báo-thực tế) — ở đây
vai trò "tín hiệu sửa sai" được đóng bởi KL-loss giữa teacher và student.

### 1.3 Công thức loss (tương đương Eq. 1 trong paper)

Đây là công thức **Knowledge Distillation chuẩn** (Hinton et al. 2015), xác nhận trực tiếp
trong code (`repro/build/mackey_glass/exp/base_exp.py:160`,
`repro/build/CHBMIT/exp/FGL_CHBMIT.py` hàm `distill_student_model`):

```
L_student = α · CE(y_true, p_student)  +  (1 − α) · T² · KL( softmax(z_teacher/T) ‖ log_softmax(z_student/T) )
```

- `z_teacher, z_student`: logits (đầu ra trước softmax) của teacher/student trên **cùng
  input**.
- `T` (**temperature**): làm "mềm" phân phối xác suất trước khi so KL — T lớn → phân phối
  dàn đều hơn → student học được các quan hệ tương đối *giữa các lớp* (dark knowledge) chứ
  không chỉ lớp có xác suất cao nhất. Nhân `T²` để giữ thang gradient ổn định khi đổi T
  (chuẩn kỹ thuật trong KD, không phải phát minh riêng của paper này).
- `α` (**alpha**): trọng số giữa "tin nhãn thật" (CE) và "tin thầy" (KL). `α=1` ⇒ bỏ hoàn
  toàn teacher (= baseline không FGL). `α=0` ⇒ học thuần từ teacher, không dùng nhãn thật.
- **Lưu ý khác biệt nhỏ giữa 2 bản code**: `FGL_CHBMIT.py` nhân `β=(1−α)` ở ngoài vào
  *distillation_loss = KL·T²*, còn `mackey_glass/utils/utils.py::KL()` cũng trả về đã nhân
  sẵn `(1-α)` — hai cách viết khác nhau nhưng **toán học tương đương**.

### 1.4 "Teacher nhìn tương lai" được *thực thi* cụ thể ra sao?

Đây là phần hay nhất để giải thích trong báo cáo — **cơ chế khác nhau tùy bài toán**, nhưng
cùng một triết lý. Xem mục 2 và 3 dưới đây — mỗi bài toán hiện thực hóa "tương lai" theo
một cách khác nhau.

---

## 2. Ứng dụng 1 — Mackey-Glass (dự báo chuỗi thời gian hỗn loạn)

### 2.1 Dữ liệu: hệ Mackey–Glass

Phương trình vi phân có trễ (delay differential equation), sinh chuỗi thời gian **hỗn loạn
(chaotic)** — thường dùng làm benchmark chuẩn cho bài toán dự báo phi tuyến vì khó dự báo
dài hạn một cách căn bản (nhạy cảm với điều kiện ban đầu — số mũ Lyapunov dương):

```
dy/dt = β·y(t−τ) / (1 + y(t−τ)^n) − γ·y(t)
```

Code sinh dữ liệu: `repro/build/mackey_glass/utils/utils.py` (class `MackeyGlass`, dùng thư
viện `jitcdde` để giải số phương trình trễ).

### 2.2 Trick biến hồi quy → phân loại (binning)

KD cổ điển định nghĩa trên **phân phối xác suất rời rạc** (softmax trên các lớp), nhưng dự
báo chuỗi thời gian vốn là bài toán **hồi quy** (giá trị liên tục). Code giải quyết bằng
cách **rời rạc hóa**: chia khoảng giá trị quan sát được thành `num_bins` "lớp" bằng
`np.digitize` (`utils.py::create_time_series_dataset`, dòng ~216-223), biến bài toán thành
**phân loại nhiều lớp** → có thể áp KD/KL bình thường. Đây là một lựa chọn thiết kế thông
minh nhưng cũng là **giới hạn**: độ phân giải dự báo bị chặn bởi `num_bins`.

### 2.3 Cơ chế "teacher thấy tương lai" ở đây: lệch cửa sổ thời gian (offset)

Cùng hàm `create_time_series_dataset` được gọi **2 lần** với tham số khác nhau
(`base_exp.py:57-77`):

| | lookback_window | forecasting_horizon | offset |
|---|---|---|---|
| **Teacher** | giống student | **1** (chỉ 1 bước) | `student_horizon − 1` |
| **Student** | L | **H** (horizon thật, vd 5) | `0` |

Nhãn được lấy tại `y[i + lookback_window + forecasting_horizon − 1]` — thay số vào sẽ thấy
**cả hai đều nhắm tới đúng cùng một mốc thời gian tuyệt đối `t = i + L + H − 1`**:
- Student: đứng ở mốc `i`, nhìn `L` bước lùi, phải dự báo **xa** `H` bước tới.
- Teacher: đứng ở mốc `i + (H−1)` (nhờ `offset`), nhìn `L` bước lùi **tính từ đó**, chỉ cần
  dự báo **1** bước tới — vẫn ra đúng mốc `t` như trên, nhưng bài toán *dễ hơn nhiều* vì chỉ
  phải ngoại suy 1 bước thay vì H bước.

→ Đây chính là "teacher sống ở tương lai gần hơn student": **cùng đích, khác xuất phát
điểm** (teacher xuất phát gần đích hơn).

### 2.4 Kiến trúc & pipeline huấn luyện

- Model: `RNN` 2 lớp (vanilla `nn.RNN`, không phải LSTM/GRU) + 2 tầng fully-connected
  (`utils.py::RNN`, 128 hidden units).
- 3 model được train **độc lập, tuần tự**, cùng early-stopping (patience mặc định 5):
  1. **Teacher** — train trên nhiệm vụ dễ (1-step, đã lệch offset).
  2. **Baseline** — train trực tiếp H-step, **không** có teacher (đối chứng).
  3. **Student (FGL)** — train H-step **có** teacher chỉ dẫn qua loss ở mục 1.3.
- Đánh giá cuối: MSE trên tập test, so sánh `Baseline` (không FGL) vs `Student` (có FGL).

### 2.5 Kết quả đã tái hiện trên máy này (đối chiếu paper)

Chi tiết đầy đủ, gồm lệnh chạy và log, nằm ở [`repro/docs/EXPERIMENT_LOG.md`](EXPERIMENT_LOG.md)
— dưới đây là bảng tóm tắt dùng ngay cho slide:

**Lần chạy chuẩn** (`--horizon 5 --num_bins 50 --alpha 0.5 --epochs 20`, seed mặc định 42):

| | Test MSE |
|---|---|
| Teacher (1 bước) | 5.44 |
| Baseline (5 bước, không FGL) | 139.38 |
| Student (5 bước, có FGL) | 104.39 |
| **Giảm MSE (Baseline → Student)** | **25.1%** |

Paper công bố: **giảm 23.4%** (loại outlier) — khớp khá sát.

**Sweep 6 seed** (để kiểm tra độ ổn định, không có trong paper — tự làm thêm):

| seed | Teacher | Baseline | Student | Giảm MSE |
|---|---|---|---|---|
| 0 | 4.83 | 156.17 | 102.75 | 34.2% |
| 1 | 5.46 | 136.57 | 106.08 | 22.3% |
| 2 | 5.73 | 143.53 | 104.43 | 27.3% |
| 3 | 6.22 | 140.36 | 107.62 | 23.3% |
| 4 | 7.45 | 147.52 | 108.69 | 26.3% |
| 42 | 5.44 | 139.38 | 104.39 | 25.1% |
| **TB** | 5.86 | 143.92 | 105.66 | **26.42% (±3.86)** |

**Nhận định dùng cho phần thảo luận/kết luận của báo cáo**: cả 6/6 seed đều cho FGL cải
thiện (không seed nào đảo chiều) — tái hiện được **đúng hướng và đúng độ lớn** so với paper,
dù chỉ 1 lần chạy/seed (chưa tách được phương sai do khởi tạo ngẫu nhiên khỏi phương sai do
seed chia dữ liệu — nên nói "khớp hướng và độ lớn", tránh nói "khẳng định thống kê chặt").

### 2.6 Stage bổ sung CỦA TÁC GIẢ: thích ứng trôi dạt (drift) với Page-Hinkley — `drift_exp.py`

**Quan trọng về nguồn**: file `drift_exp.py`, cơ chế Page-Hinkley, và cả vòng lặp quét
`horizon` 2→15 **đều là code gốc của tác giả** (một trong 3 giai đoạn thí nghiệm
Mackey-Glass mà README chính thức liệt kê — "FGL with Page-Hinkley Drift Detection"), **không
phải do mình thêm vào**. Việc của mình ở đây chỉ là **sửa lỗi** (khối `argparse` bị comment
khiến CLI vô hiệu — mục 4, điểm 3) để script chạy được, rồi **thực thi** nó. Ngược lại,
**bảng số liệu kết quả** ngay dưới đây là do mình tự chạy ra trên máy này — chưa đối chiếu
được với số liệu paper công bố (nếu có) cho phần drift này.

**Lý thuyết Page-Hinkley test**: một phương pháp phát hiện *concept drift* (phân phối dữ
liệu thay đổi theo thời gian) trực tuyến (online), dựa trên tích lũy độ lệch giữa giá trị
quan sát và trung bình trượt của nó; khi tích lũy vượt ngưỡng `λ` → báo "có drift" → trigger
retrain. Ở đây dùng để: khi phát hiện drift trong luồng dữ liệu test, **retrain lại student
vài epoch** (`ph_retrain_epochs`) ngay trong lúc inference, mô phỏng kịch bản triển khai
thực tế (dữ liệu luôn thay đổi theo thời gian).

Kết quả quét `horizon` 2→15, tự chạy trên máy này (bảng đầy đủ ở EXPERIMENT_LOG.md mục "Lần chạy 3"):
- **13/14 horizon** cho FGL cải thiện so với baseline, **chỉ horizon=2 bị âm (−7.2%)**.
- Trung bình cải thiện **10.83%** (toàn bộ) / **12.22%** (loại horizon=2).
- Điểm hay để nêu trong slide: baseline MSE **không tăng đơn điệu** theo horizon (tăng rồi
  giảm) — gợi ý liên quan tới chu kỳ trễ đặc trưng `τ=17` của hệ Mackey-Glass, không phải
  lỗi code.

---

## 3. Ứng dụng 2 — CHB-MIT EEG (dự báo cơn động kinh)

### 3.1 Dữ liệu

CHB-MIT (PhysioNet): EEG da đầu (scalp EEG) của bệnh nhân động kinh nhi khoa, 23 kênh
(montage 10-20), lấy mẫu 256Hz. Trên máy này đã tải xong **bệnh nhân 1** (42 file `.edf`,
~1.6GB) — xem `repro/build/CHBMIT/download_chb01.sh`.

### 3.2 Tiền xử lý (feature engineering)

`utils/load_signals_{teacher,student}.py` + `PrepData*.preprocess()`:

1. Đọc `.edf` bằng `mne.io.read_raw_edf`, chọn đúng 18-23 kênh theo montage chuẩn (một số
   bệnh nhân — 4, 13, 16 — có bộ kênh khác, code có nhánh riêng).
2. Cắt cửa sổ **30 giây** (`numts=30`, 256Hz → 7680 mẫu/cửa sổ).
3. Biến đổi **STFT** (short-time Fourier transform) mỗi cửa sổ → **spectrogram** (biểu diễn
   năng lượng theo tần số-thời gian), lấy log, cắt bỏ một số dải tần (dòng
   `stft_data[:,:,1:57] + [64:117] + [124:]` — loại dải quanh 57-64Hz và 117-124Hz, nhiều
   khả năng để loại nhiễu điện lưới 60Hz và hài của nó).
4. **Oversampling lớp ictal**: với cửa sổ ictal, trượt thêm cửa sổ chồng lắp (overlap) để
   tăng số mẫu dương (lớp ictal luôn hiếm hơn interictal rất nhiều) — các mẫu oversample gán
   nhãn tạm `y=2`, **chỉ dùng khi train**, bị loại khỏi tập test (tránh data leakage giữa
   các cửa sổ chồng lắp nhau).
5. Cache kết quả ra `cachedir-teacher/` hoặc `cachedir-student/` bằng `hickle` (để không
   phải xử lý lại EDF/STFT mỗi lần chạy) — **đây chính là bước làm máy 16GB RAM hết bộ nhớ
   khi chạy student** (xem mục 3.5).

### 3.3 Hai nhãn khác nhau cho teacher và student — đây là chỗ FGL "tương lai" lộ rõ nhất

Khác với Mackey-Glass (lệch *cửa sổ thời gian đầu vào*), ở CHB-MIT, FGL hiện thực hóa "nhìn
tương lai" bằng cách cho **teacher và student học hai NHÃN khác nhau trên cùng loại tín
hiệu**:

| | Teacher (`load_signals_teacher.py`) | Student (`load_signals_student.py`) |
|---|---|---|
| **Tên bài toán** | *Seizure Detection* (phát hiện cơn **đang** xảy ra) | *Seizure Prediction* (dự báo cơn **sẽ** xảy ra) |
| **Nhãn "ictal" (1) lấy từ** | `data[seizure_start : seizure_stop]` — đúng đoạn tín hiệu **trong lúc** co giật | `data[st − SOP : st]` với `st = seizure_start − 5phút`, `SOP = 30 phút` — đoạn **30 phút TRƯỚC** cơn, cách onset 5 phút (SPH = seizure prediction horizon) |
| **Thông tin có sẵn** | Thấy trực tiếp biến cố → bài toán *dễ*, AUC cao tự nhiên | Chỉ thấy tín hiệu **trước** biến cố, chưa có dấu hiệu rõ ràng → bài toán *khó* |

→ Teacher = "người đứng ở tương lai" (tại thời điểm cơn xảy ra) nhìn thấy sự thật trần trụi;
student = "người đứng ở quá khứ" (30 phút trước) phải *đoán trước*. Khi distill, script
`FGL_CHBMIT.py` cho **input preictal/interictal của student** chạy qua **teacher** (dù
teacher chưa từng được train trên preictal — nó chỉ được hỏi ý kiến, không được train lại)
để lấy "cảm nhận mềm" (soft logits) của teacher về việc tín hiệu này "giống cơn động kinh"
tới mức nào, rồi ép student học theo cảm nhận đó cộng với nhãn thật. Trực giác: dù teacher
chưa thấy preictal bao giờ, các đặc trưng tần số nó học được để nhận ra cơn động kinh vẫn
mang thông tin hữu ích cho việc đoán trước.

### 3.4 Kiến trúc model

Hai lựa chọn, định nghĩa ở `repro/build/CHBMIT/models/model.py`:

- **`CNN_LSTM_Model`**: Conv3D (xử lý spectrogram đa kênh như "ảnh 3D") → MaxPool/AvgPool →
  flatten → LSTM 3 lớp (512 hidden) → 2 tầng FC → 2 lớp đầu ra (ictal/interictal). Dùng làm
  model cho **cả teacher và student/baseline** mặc định.
- **`MViT`** ("multi-channel ViT"): mỗi kênh EEG được patch-embed **riêng** thành 1 chuỗi
  token, qua **Transformer Encoder riêng cho từng kênh** (không chia sẻ trọng số giữa
  kênh), lấy token `[CLS]` của mỗi kênh, nối lại → 1 tầng Linear phân loại. Ý tưởng: coi mỗi
  kênh EEG như một "modal" riêng, học đặc trưng không gian-thời gian độc lập trước khi fuse.
  Dùng cho baseline/student ở Stage 2-3 (README gợi ý `--model MViT`).

### 3.5 Trạng thái tái hiện trên máy này (quan trọng — nêu trong phần "Hạn chế" của báo cáo)

- ✅ **Dữ liệu**: patient 1 đã tải xong (1.6GB `.edf`).
- ✅ **Stage 1 (Teacher — seizure detection)**: đã train xong, checkpoint lưu tại
  `pytorch_models/Patient_1_detection`. **Nhưng AUC-ROC đo được chỉ 0.28** (`Detection_results.txt`)
  — **thấp hơn random (0.5)**, bất thường. Nghi vấn kỹ thuật: `DataLoader` huấn luyện dùng
  `shuffle=False` (`seizure_detection.py`) trong khi dữ liệu train được nối theo thứ tự
  **[toàn bộ interictal, rồi toàn bộ ictal]** (`train_val_test_split_continual_t`) — nghĩa
  là mỗi epoch, model thấy hàng loạt batch **toàn nhãn 0** trước, rồi mới tới batch toàn
  nhãn 1 — đây là kiểu sắp xếp dữ liệu dễ gây học lệch/không ổn định (catastrophic
  forgetting giữa batch), **rất có thể là nguyên nhân** AUC thấp bất thường — *nhưng chưa
  chạy lại để xác nhận giả thuyết này*, cần ghi rõ "nghi vấn, chưa kiểm chứng" nếu đưa vào
  báo cáo.
- ❌ **Stage 2 (Baseline) / Stage 3 (Student FGL)**: **chưa hoàn thành** — crash giữa lúc
  build cache `cachedir-student/` (xử lý STFT cho toàn bộ EEG patient 1). Bằng chứng: không
  tiến trình Python nào còn chạy, không crash report hệ thống, nhưng **swap đã dùng
  11.8/13.3GB** trên máy chỉ có **16GB RAM** ngay sau đó — dấu hiệu mạnh của hết bộ nhớ
  (OOM), vì code gom **toàn bộ** windows STFT của một loại nhãn (ictal hoặc interictal) vào
  1 mảng NumPy duy nhất trong RAM rồi mới ghi cache 1 lần (không ghi tăng dần theo file).
- Hướng xử lý đang xem xét: (a) thử Colab free (RAM ảo ~12-13GB, **ít hơn** máy hiện tại,
  không chắc cứu được nếu đúng là thuật toán cần >13GB đỉnh), hoặc (b) sửa code để ghi cache
  tăng dần theo từng file EDF, giảm đỉnh RAM thực sự — chưa quyết định/thực hiện tại thời
  điểm viết tài liệu này.

---

## 4. Ứng dụng 3 — AES (Kaggle Seizure Prediction Challenge)

**Chưa tái hiện trên máy này** (`repro/build/AES/` hiện trống, chưa tải dữ liệu) — mục này
chỉ phân tích **lý thuyết + code gốc của tác giả** (`repro/upstream/FutureGuidedLearning/AES/`),
dùng được cho phần lý thuyết của báo cáo, nhưng **không có số liệu tái hiện** để đối chiếu.

### 4.1 Dữ liệu

iEEG (intracranial EEG — điện não trong sọ, khác CHB-MIT là scalp EEG ngoài da đầu) từ cuộc
thi Kaggle "Seizure Prediction Challenge": **5 chó** (`Dog_1`–`Dog_5`) và **2 người**
(`Patient_1`, `Patient_2`). Cùng bài toán *seizure prediction* như CHB-MIT, nên về bản chất
đây **cũng là biến thể (B) — event prediction** ở Hình 3 của paper, không phải biến thể
riêng — nhưng cơ chế xây **teacher** khác CHB-MIT, xem mục 4.2.

### 4.2 Khác biệt cốt lõi so với CHB-MIT: "universal teacher" thay vì teacher riêng từng bệnh nhân

README tác giả ghi rõ lý do: *"The AES dataset lacks seizure labels for training, so we
create two 'universal' teacher models... one for dogs and one for humans."* — xác nhận
trong code (`AES/exp/create_teacher.py::train_teacher_model`): chỉ train **2 teacher duy
nhất** — `teacher_dog.pth` (mode `"Dog"`, học gộp dữ liệu ictal/interictal nhóm "Dog") và một
teacher cho nhóm người — rồi **tái sử dụng cùng 1 teacher đó cho mọi subject cùng loài** khi
chạy FGL (`AES/exp/FGL_AES.py`, dòng ~59-64: `if 'Dog_' in target: model_path = 'teacher_dog.pth'`
cho cả `Dog_1` → `Dog_5`).

→ Khác hẳn CHB-MIT (teacher **riêng cho từng bệnh nhân**, train trên chính cơn co giật của
bệnh nhân đó): ở AES, "tương lai" mà teacher nắm được là tri thức **gộp chung của cả loài**
(chó nói chung / người nói chung) về "co giật trông như thế nào", không đặc thù theo cá thể
— một cách khác để hiện thực hóa biến thể (B) khi dữ liệu nhãn ictal của riêng từng subject
không đủ để tự train teacher riêng.

### 4.3 Kiến trúc & pipeline

- Model: **cùng `CNN_LSTM_Model`** như CHB-MIT (`AES/models/models.py` — kiến trúc giống
  100%, chỉ thay dữ liệu đầu vào).
- Tiền xử lý: cùng công thức STFT/log-spectrogram như CHB-MIT (`AES/utils/load_signals_*.py`
  import `stft`, cùng logic windowing/oversampling ictal).
- Lệnh chạy theo README: `create_teacher.py --epochs 50` (dựng 2 universal teacher) →
  `seizure_prediction.py --subject Dog_1 --model CNN_LSTM --trials 3` (baseline) →
  `FGL_AES.py --subject Dog_1 --epochs 25 --trials 3 --optimizer Adam --alpha 0.7
  --temperature 4` (student FGL — lưu ý `α=0.7`, khác CHB-MIT dùng `α=0.5`).

### 4.4 Lý do chưa tái hiện

Theo `EXPERIMENT_LOG.md`: quyết định dừng ưu tiên track dữ liệu EEG (cả CHB-MIT và AES) sau
khi đo tốc độ tải PhysioNet quá chậm; CHB-MIT patient 1 sau đó vẫn tải được (ở phiên làm việc
sau) nhưng AES — cần tài khoản/API token Kaggle riêng, chưa thử — **vẫn chưa được bắt đầu**.
Nếu đưa vào báo cáo: nên trình bày đây là phần **hiểu biết lý thuyết + đọc code**, không
nhận là "đã tái hiện".

---

## 5. Tổng hợp các vấn đề phát hiện trong code gốc của tác giả

Hữu ích cho phần "Thảo luận / Reproducibility" của tiểu luận — cho thấy code công khai đi
kèm paper **không chạy được ngay** mà cần vá, một quan sát đáng đưa vào báo cáo khi đánh giá
tính tái lập (reproducibility) của nghiên cứu:

1. **4 file còn merge-conflict marker chưa giải quyết** tại HEAD của repo tác giả
   (`mackey_glass/exp/base_exp.py`, `drift_exp.py`, `utils/utils.py`, `utils/generate.py`)
   → `SyntaxError` ngay khi import, code **không chạy được một dòng nào** ở trạng thái gốc.
2. **`generate.py`**: 2 nhánh conflict tính dữ liệu khác nhau — một nhánh làm lệch
   `forecasting_horizon` đi 1 bước so với giá trị `--horizon` người dùng truyền vào (xác
   minh bằng cách đối chiếu `data.pkl` tác giả tự commit sẵn — khớp nhánh HEAD).
3. **`drift_exp.py`**: toàn bộ khối `argparse` nằm trong một chuỗi bị **comment** → cờ CLI
   (`--horizon`, `--use_ph`...) **vô hiệu hoàn toàn**; code thật sự chạy là đoạn hardcode
   (quét horizon 2-15, `alpha=0.0`) nằm ngoài vùng comment.
4. **Thiếu dependency** không ghi trong `requirements.txt` gốc (`jitcdde` và các gói phụ
   thuộc của nó).
5. **Đường dẫn chạy thực tế khác README**: README ghi `python base_exp.py`, nhưng file nằm
   ở `exp/base_exp.py` và import `from utils.utils import ...` chỉ resolve nếu chạy bằng
   `PYTHONPATH=. python exp/base_exp.py` từ thư mục gốc của từng bài toán.
6. **Hardcode `.to("cuda")`** ở 3 script CHBMIT — không chạy được trên máy không có NVIDIA
   GPU (đã vá thành tự nhận biết MPS/CUDA/CPU trong `repro/build/CHBMIT/`).
7. **`shuffle=False` trên dữ liệu được nối theo thứ tự lớp** (mục 3.5) — nghi vấn gây AUC
   bất thường ở teacher CHBMIT, chưa kiểm chứng.
8. **Thiết kế gom toàn bộ dữ liệu một loại nhãn vào RAM trước khi cache** — không scale với
   máy RAM nhỏ khi xử lý EEG nhiều giờ liên tục (mục 3.5).

---

## 6. Gợi ý cấu trúc báo cáo / slide

1. **Giới thiệu**: vấn đề đánh đổi sớm/muộn trong dự báo biến cố; paper + repo tác giả.
2. **Phương pháp**: ý tưởng FGL (mục 1.2), công thức loss KD (mục 1.3) — 1 slide công thức
   là đủ, highlight vai trò α/T.
3. **Cơ chế "nhìn tương lai" cụ thể hóa** (mục 1.4): 1 slide so sánh 2 cách hiện thực hóa —
   lệch *thời gian* (Mackey-Glass) vs lệch *định nghĩa nhãn* (CHB-MIT **và** AES) — đây là
   insight hay nhất để trình bày, cho thấy cùng 1 ý tưởng áp dụng linh hoạt ra sao.
4. **Thực nghiệm & kết quả** (nên nói lướt qua, không cần đi quá sâu từng con số): bảng mục
   2.5 (Mackey-Glass, đã có số liệu đầy đủ) + mục 2.6 (drift, bonus).
5. **CHB-MIT & AES**: trình bày tiền xử lý + kiến trúc (mục 3.2-3.4, 4.2-4.3) như hiểu biết
   lý thuyết; nói rõ **trạng thái thật** — CHB-MIT dừng ở teacher (mục 3.5), AES chưa bắt
   đầu tái hiện (mục 4.4) — biến hạn chế thành điểm phân tích kỹ thuật, không chỉ "chưa chạy
   được".
6. **Thảo luận / Hạn chế**: mục 5 — thể hiện đã đọc kỹ code, không chỉ chạy lệnh có sẵn.
7. **Kết luận**: tái hiện được đúng hướng & độ lớn cải thiện trên Mackey-Glass (26.4% vs
   23.4% công bố); CHB-MIT & AES trình bày được cơ chế/kiến trúc nhưng dừng ở lý thuyết do
   giới hạn tài nguyên/dữ liệu.

---

## 7. Nguồn

- Code đã đọc trực tiếp: `repro/build/mackey_glass/{exp,utils}/*.py`,
  `repro/build/CHBMIT/{exp,utils,models}/*.py`,
  `repro/upstream/FutureGuidedLearning/{README.md,AES/**/*.py}`.
- Số liệu thực nghiệm Mackey-Glass: [`repro/docs/EXPERIMENT_LOG.md`](EXPERIMENT_LOG.md)
  (đã chạy và ghi log trực tiếp trên máy này).
- Thông tin paper (abstract, headline numbers) lấy từ README tác giả — **chưa đối chiếu
  được với bản PDF gốc** (Nature Communications chặn fetch tự động lúc viết tài liệu này,
  2026-10-14) — nên trích dẫn số liệu "theo README tác giả công bố" thay vì "theo paper",
  trừ khi tự mở DOI kiểm tra lại.
