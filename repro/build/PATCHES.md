# build/conem — những gì đã được áp

> **Sinh tự động bởi `ours/build.py`. Không sửa tay thư mục `build/`.**
> Code tác giả nguyên trạng ở `upstream/conem/`; các bản vá ở `ours/patches/`.

Chế độ dựng: **required + choices**

## Phân loại

| nhóm | nghĩa |
|---|---|
| **required** | Không có thì code **crash** — không chạy được gì. Không có lựa chọn khác. |
| **choice** | **Quyết định của chúng ta.** Code vẫn chạy nếu bỏ. Tức KHÔNG phải tái hiện trung thực. |

## BẮT BUỘC — không có thì crash

### `A1+A2` — model_fed_based.py: lời gọi DataEmbedding lệch 2 tham số + không giải nén decoder

Module: `ours/patches/required/a1_a2_fed_based.py`

File của tác giả bị chạm:

- `model/model/model_fed_based.py`

**Vì sao**

```
Đây là biến thể FEDformer+ConEm — chính biến thể bài báo báo cáo tốt nhất trên Weather
(Bảng 11-13) và trên Pharma (Bảng 6). Không vá thì không chạy được một dòng nào.

A1. layers/Embed.py có chữ ký MỚI:
      DataEmbedding_encoder(c_in, d_model, static_d_token, contextual_encoder_d_token,
                            embed_type, freq, dropout, num_cat, num_num, num_static)
    nhưng model_fed_based.py:107 còn gọi theo bản CŨ, thiếu 2 tham số giữa:
      DataEmbedding_encoder(enc_in, d_model, embed, freq, dropout, ...)
    -> mọi tham số lệch 2 vị trí; freq nhận len(cat_vab) = 0.

A2. DataEmbedding_decoder.forward trả về `self.dropout(x), x_combined` (2 giá trị)
    nhưng model_fed_based.py:356 gán 1 biến -> dec_out là tuple.
```

**Bằng chứng**

```
A1: KeyError: 0            tại layers/Embed.py:293  (self.freq_map[freq] với freq=0)
A2: AttributeError: 'tuple' object has no attribute 'shape'
                           tại layers/Correlation.py:158

Mẫu để sửa KHÔNG do chúng ta nghĩ ra: model_info_based.py và model_auto_based.py
trong cùng repo gọi ĐÚNG cả hai chỗ. Vá này copy y nguyên cách gọi của chúng.
```

**Thay đổi thực tế khi dựng**

- A1 encoder: thêm static_d_token, contextual_encoder_d_token
- A1 decoder: thêm static_d_token, contextual_decoder_d_token
- A2: giải nén 2 giá trị trả về của dec_embedding

### `A3` — model_without_FuEm.py: cả ba import của khối 'Custom Embedding' bị comment

Module: `ours/patches/required/a3_ablation_imports.py`

File của tác giả bị chạm:

- `model/model/model_without_FuEm.py`

**Vì sao**

```
Biến thể ablation "bỏ FuEm" của Bảng 17. Ba dòng import ở dòng 21-23 bị comment:
    #from model.layers.Embed import Customize_Tokenizer
    #import torch.nn.init as nn_init
    #import math
nhưng nn_init và math vẫn được dùng ở dòng 96.

Mẫu để sửa có sẵn trong repo: model_without_LoEm.py có đúng ba dòng này KHÔNG bị comment.
```

**Bằng chứng**

```
NameError: name 'nn_init' is not defined   tại model/model/model_without_FuEm.py:96
rồi sau khi sửa dòng đó: NameError: name 'math' is not defined

LƯU Ý: model_without_LoEm.py và model_without_PaEm.py KHÔNG cần vá — chúng dùng
layers/Embed_without_*.py có chữ ký CŨ (8 tham số, không static_d_token) và decoder
trả về 1 giá trị, nên lời gọi cũ trong chúng là ĐÚNG. Một bản vá trước đây của chúng ta
đã áp A1/A2 lên chúng và LÀM HỎNG hai file đang chạy được:
    TypeError: __init__() takes from 3 to 9 positional arguments but 11 were given
```

**Thay đổi thực tế khi dựng**

- bỏ comment: from model.layers.Embed import Customize_Tokenizer
- bỏ comment: import torch.nn.init as nn_init
- bỏ comment: import math

### `A4` — 3 file baseline: os.chdir sang đường dẫn máy tác giả + import tương đối

Module: `ours/patches/required/a4_baseline_chdir.py`

File của tác giả bị chạm:

- `model/model/model_fedformer.py`
- `model/model/model_autoformer.py`
- `model/model/model_informer.py`

**Vì sao**

```
Đây là CÁC CỘT BASELINE của Bảng 10 — tức nửa còn lại của so sánh cốt lõi
"backbone vs backbone + ConEm". Cả ba file có ở dòng 4:

    os.chdir("/home/ad/20813716/transformer_sales_forecast/model")

là đường dẫn tuyệt đối trên máy riêng của tác giả, và ngay sau đó import kiểu
`from layers.Embed import ...` chỉ hoạt động SAU cú chdir đó.

HỆ QUẢ: không máy nào khác chạy được baseline, nên KHÔNG THỂ tái hiện so sánh cốt lõi
của bài báo từ code đã công bố.

Mẫu để sửa có sẵn: các file *_based.py trong cùng repo đã comment chdir và dùng
`from model.layers.` — vá này áp y nguyên cách đó.
```

**Bằng chứng**

```
FileNotFoundError: [Errno 2] No such file or directory:
    '/home/ad/20813716/transformer_sales_forecast/model'
    tại model/model/model_fedformer.py:4

grep os.chdir trên bản gốc:
    model_fedformer.py:4     os.chdir("/home/ad/20813716/...")
    model_autoformer.py:4    os.chdir("/home/ad/20813716/...")
    model_informer.py:4      os.chdir("/home/ad/20813716/...")
    model_auto_based.py:16   #os.chdir(...)   <- đã comment
    model_info_based.py:17   #os.chdir(...)   <- đã comment

A5 (liên quan, KHÔNG vá ở đây): cùng ba file này khai báo embedding cho
freq + cat + num = 11 đầu vào nhưng forward chỉ nạp 5 (x_mark_enc = x_temporal_enc,
dòng ghép ngoại sinh đã bị comment) ->
    RuntimeError: linear(): input and weight.T shapes cannot be multiplied (96x5 and 11x512)
A5 được xử lý bằng CẤU HÌNH trong ours/runner/run.py (num_vab=[target] cho biến thể
backbone) để không phải chạm thêm vào code tác giả.
```

**Thay đổi thực tế khi dựng**

- model_fedformer.py: vô hiệu os.chdir + import tuyệt đối theo package
- model_autoformer.py: vô hiệu os.chdir + import tuyệt đối theo package
- model_informer.py: vô hiệu os.chdir + import tuyệt đối theo package

### `B1` — exp_basic.py: thêm nhánh thiết bị MPS (Apple Silicon)

Module: `ours/patches/required/b1_mps_device.py`

File của tác giả bị chạm:

- `model/exp/exp_basic.py`

**Vì sao**

```
_acquire_device() chỉ có hai nhánh: cuda nếu use_gpu, ngược lại cpu. Máy chạy là
Apple M5 không có CUDA, nên nếu không vá thì toàn bộ huấn luyện rơi về CPU —
đo được chậm hơn MPS 3x (4,970 vs 1,686 s/iter, batch 8).

Đây là 'required' theo nghĩa thực tế: trên CPU một epoch ở stride 1 mất ~866 phút,
tức không thể chạy.
```

**Bằng chứng**

```
Đo trên máy này, forward+backward, batch 8, mô hình FEDformer+ConEm 116M tham số:
    CPU  4,970 s/iter  -> ~866 phút/epoch @stride 1
    MPS  1,686 s/iter  -> ~294 phút/epoch @stride 1
torch 2.8.0, torch.backends.mps.is_available() = True
```

**Thay đổi thực tế khi dựng**

- exp_basic.py: thêm nhánh MPS

### `B2` — exp_main_*.py: torch.load(map_location='cuda:0') -> self.device

Module: `ours/patches/required/b2_map_location.py`

File của tác giả bị chạm:

- `model/exp/exp_main_compare.py`
- `model/exp/exp_main_without_ext.py`
- `model/exp/exp_main_ablation.py`

**Vì sao**

```
Sau early stopping, train() nạp lại checkpoint tốt nhất bằng
    torch.load(best_model_path, map_location='cuda:0')
Trên máy không có CUDA, lệnh này crash — tức mọi run đều chết ở bước nạp checkpoint,
ngay trước khi đánh giá test.
```

**Bằng chứng**

```
RuntimeError: Attempting to deserialize object on a CUDA device but
torch.cuda.is_available() is False.
3 chỗ mỗi file (train(), predict(), và một chỗ đã bị comment).
```

**Thay đổi thực tế khi dựng**

- exp_main_compare.py: 3 chỗ map_location
- exp_main_without_ext.py: 3 chỗ map_location
- exp_main_ablation.py: 3 chỗ map_location

## LỰA CHỌN CỦA CHÚNG TA — code vẫn chạy nếu bỏ

### `C1` — exp_main_*.py: import Model ở cấp module -> import động theo args.model_file

Module: `ours/patches/choice/c1_dynamic_import.py`

File của tác giả bị chạm:

- `model/exp/exp_main_compare.py`
- `model/exp/exp_main_without_ext.py`
- `model/exp/exp_main_ablation.py`

**Vì sao**

```
Mỗi file exp hardcode một backbone ở đầu file, ví dụ:
    from model.model.model_info_based import Model
Muốn đổi backbone, tác giả phải sửa dòng này rồi chạy lại. Vá này chuyển sang import
động theo args.model_file để một lệnh chạy chọn được backbone.

THUẦN TIỆN LỢI. Không có nó code vẫn chạy — chỉ là phải sửa file mỗi lần đổi backbone.
Nếu muốn tối giản tuyệt đối, bỏ vá này và sửa tay dòng import.
```

**Bằng chứng**

```
exp_main_compare.py:10     from model.model.model_info_based import Model
exp_main_without_ext.py:10 from model.model.model_informer import Model
exp_main_ablation.py:10    from model.model.model_without_LoEm import Model
```

**Thay đổi thực tế khi dựng**

- exp_main_compare.py: import Model động
- exp_main_without_ext.py: import Model động
- exp_main_ablation.py: import Model động

### `A6fix` — model_fed_based.py: RIN dùng dạng (sum+1) thay vì sum ở hệ số affine

Module: `ours/patches/choice/a6fix_rin.py`

File của tác giả bị chạm:

- `model/model/model_fed_based.py`

**Vì sao**

```
QUYẾT ĐỊNH CỦA CHÚNG TA. Code vẫn chạy nếu bỏ vá này — nhưng phân kỳ.

model_fed_based.py có HAI dòng cạnh nhau, dòng an toàn bị comment:
  275  #x_enc = x_enc*(torch.add(torch.sum(w*x_static_enc,dim=2,keepdim=True),1)) + ...
  276   x_enc = x_enc* torch.sum(w*x_static_enc,dim=2,keepdim=True)              + ...

Dòng 275 (comment) cho hệ số = sum(w*s) + 1 — dạng "1 cộng phần học được", đúng RevIN
chuẩn, mẫu số luôn quanh 1.
Dòng 276 (đang dùng) cho hệ số = sum(w*s). Weather chỉ có MỘT đặc trưng tĩnh (location)
nên đó là MỘT scalar học được duy nhất, và dòng 373 chia cho nó:
    dec_out = dec_out / torch.sum(w*x_static_dec[...], dim=2, keepdim=True)
không epsilon, không clamp. Khởi tạo kaiming_uniform_ trên tensor 1 phần tử cho
uniform(-1.73, +1.73) nên có thể gần 0 ngay từ đầu.

Vá này dùng dạng "+1" ở CẢ hai chiều cho nhất quán.

VÌ SAO KHÔNG DÙNG --RIN 0 THAY THẾ: RIN=0 bỏ hẳn instance normalization. Dữ liệu Weather
có dịch chuyển mức theo mùa rất mạnh (train Jan-Oct 2022, val Oct-Dec 2022, test
Jan-Apr 2023). Đo được với RIN=0: epoch 1 train loss 1,29 nhưng val loss 15,40, và MSE
test cuối 1057,58 — tệ hơn cả dự báo hằng số (28,33). Nên phải GIỮ RIN và sửa mẫu số.

LƯU Ý TRUNG THỰC: dòng là của tác giả, nhưng QUYẾT ĐỊNH dùng dòng nào là của chúng ta.
```

**Bằng chứng**

```
Với dòng 276 (bản gốc), phân kỳ ở epoch 2 bất kể dữ liệu raw hay clean:
    Epoch 1: train 28,80    val 17,30
    Epoch 2: train 6,12e7   val 6,72e8
    Epoch 3: train 8,14e14  -> early stopping
MSE test cuối: 14,4151 (raw) / 14,4297 (clean) — tệ hơn persistence (10,54).
```

**Thay đổi thực tế khi dựng**

- chiều chuẩn hoá: hệ số (sum+1)
- chiều khôi phục: mẫu số (sum+1)

### `A8` — exp_main_*.py: bật lại adjust_learning_rate (bị comment)

Module: `ours/patches/choice/a8_lr_decay.py`

File của tác giả bị chạm:

- `model/exp/exp_main_compare.py`
- `model/exp/exp_main_without_ext.py`
- `model/exp/exp_main_ablation.py`

**Vì sao**

```
QUYẾT ĐỊNH CỦA CHÚNG TA, nhưng là khôi phục bản trước đó của chính tác giả.

Dòng 219 mỗi file:
    #adjust_learning_rate(model_optim, epoch + 1, self.args)
nên learning rate đứng yên ở 1e-5 suốt. Với lradj='type1' hàm này giảm một nửa LR mỗi
epoch — đúng hành vi codebase gốc Informer/Autoformer/FEDformer mà repo này dẫn xuất từ.

BẰNG CHỨNG ĐÂY LÀ THỨ BỊ TẮT VỀ SAU, không phải chủ ý: hai file backup cũ của chính
tác giả vẫn gọi nó KHÔNG comment —
    exp_main - Copy.py:174   adjust_learning_rate(model_optim, epoch + 1, self.args)
    exp_main - dev.py:174    adjust_learning_rate(model_optim, epoch + 1, self.args)
```

**Bằng chứng**

```
grep adjust_learning_rate trên bản gốc:
    exp_main - Copy.py:174      adjust_learning_rate(...)     <- KHÔNG comment
    exp_main - dev.py:174       adjust_learning_rate(...)     <- KHÔNG comment
    exp_main.py:219            #adjust_learning_rate(...)
    exp_main_compare.py:219    #adjust_learning_rate(...)
    exp_main_without_ext.py:219 #adjust_learning_rate(...)
    exp_main_ablation.py:219   #adjust_learning_rate(...)
    exp_main_timesnet.py:245   #adjust_learning_rate(...)

Lưu ý: A8 MỘT MÌNH không đủ. Phân kỳ phụ thuộc SỐ STEP, không phụ thuộc epoch, nên
decay ở cuối epoch không chặn được nổ gradient trong epoch 1 (xem C4).
```

**Thay đổi thực tế khi dựng**

- exp_main_compare.py: bật lại adjust_learning_rate
- exp_main_without_ext.py: bật lại adjust_learning_rate
- exp_main_ablation.py: bật lại adjust_learning_rate

### `C2` — data_loader.py + data_factory.py: thêm stride khi cắt cửa sổ train/val

Module: `ours/patches/choice/c2_stride.py`

File của tác giả bị chạm:

- `model/data/data_loader.py`
- `model/data/data_factory.py`

**Vì sao**

```
QUYẾT ĐỊNH CỦA CHÚNG TA, thuần về chi phí tính toán.

Loader gốc cắt cửa sổ trượt với stride = 1, sinh 83.630 mẫu train từ ~42.000 mốc thời
gian — hai cửa sổ liền nhau trùng 95/96 giá trị. Trên Apple M5 (không CUDA) đó là
5.226 step/epoch, tức ~3,2 giờ/epoch cho mô hình 116M tham số.

stride > 1 CHỈ áp cho train/val. Tập TEST luôn giữ stride = 1 để định nghĩa metric
không đổi.

ĐÂY LÀ ĐỘ LỆCH SO VỚI TÁC GIẢ. Hệ quả: ngân sách gradient step nhỏ hơn, nên các con số
tuyệt đối KHÔNG so sánh trực tiếp được với bài báo.
```

**Bằng chứng**

```
Ngân sách đo trên máy này (2,2 s/step, batch 16):
    stride   mẫu train   step/epoch   phút/epoch   4 epoch x 3 run
         1      83.630        5.226          192           38,4 h
         4      20.907        1.306           48            9,6 h
        12       6.969          435           16            3,2 h
        24       3.484          217            8            1,6 h
```

**Thay đổi thực tế khi dựng**

- data_loader.py: Dataset_Train nhận tham số stride
- data_factory.py: truyền stride (test luôn = 1)

### `C3` — exp_main_*.py: bỏ đánh giá tập test sau mỗi epoch (chỉ dùng để in)

Module: `ours/patches/choice/c3_skip_test_eval.py`

File của tác giả bị chạm:

- `model/exp/exp_main_compare.py`
- `model/exp/exp_main_without_ext.py`
- `model/exp/exp_main_ablation.py`

**Vì sao**

```
QUYẾT ĐỊNH CỦA CHÚNG TA, thuần về chi phí.

Vòng train gọi self.vali(test_data, test_loader, criterion) sau MỖI epoch, nhưng
test_loss chỉ được IN RA — early stopping dùng vali_loss. Tập test chạy ở stride 1
(25.900 mẫu, ~1.619 batch) nên bước này chiếm ~75% thời gian mỗi epoch mà không ảnh
hưởng gì tới việc học.

Sau vá, hành vi do args.eval_test_each_epoch điều khiển (mặc định False).
exp.test() cuối cùng VẪN chạy đủ tập test ở stride 1.
```

**Bằng chứng**

```
Đo được: epoch mất 2.375s, trong đó train chỉ ~430s.
    217 step train  x 2,0s  = ~430s
    val   ~54 batch         = ~100s
    test  ~1.619 batch      = ~1.800s   <- 75%
```

**Thay đổi thực tế khi dựng**

- exp_main_compare.py: bỏ eval test mỗi epoch
- exp_main_without_ext.py: bỏ eval test mỗi epoch
- exp_main_ablation.py: bỏ eval test mỗi epoch

### `C4` — exp_main_*.py: THÊM gradient clipping (không có trong repo tác giả)

Module: `ours/patches/choice/c4_grad_clip.py`

File của tác giả bị chạm:

- `model/exp/exp_main_compare.py`
- `model/exp/exp_main_without_ext.py`
- `model/exp/exp_main_ablation.py`

**Vì sao**

```
ĐÂY LÀ THỨ CHÚNG TA THÊM VÀO, KHÔNG PHẢI CỦA TÁC GIẢ.
`grep -rn clip_grad` trên bản gốc: 0 kết quả.

Lý do: cấu hình đã công bố (lr=1e-5, Adam, không clipping, adjust_learning_rate bị
comment) không huấn luyện ổn định được. Mô hình học bình thường ~230 step rồi nổ
gradient đơn điệu. Phân kỳ phụ thuộc SỐ STEP nên A8 (decay cuối epoch) không cứu được.

KHÔNG dùng được torch.nn.utils.clip_grad_norm_: 86,7% tham số mô hình là SỐ PHỨC
(6 tensor complex64 kích thước 1024x1024x16 trong khối MultiWavelet, 100,6M/116M phần
tử) và nó báo "norm ops are not supported for complex yet". Phải tự tính global norm.

CẨN TRỌNG VỀ BỘ NHỚ: cách viết ngây thơ `((view_as_real(g))**2).sum()` tạo tensor tạm
200 triệu phần tử (800 MB) cho MỖI tensor phức — sáu tensor là ~4,8 GB tạm mỗi step,
làm máy 16 GB swap và tốc độ tụt từ 2,7 s/step xuống 163-660 s/step (đã mắc lỗi này).
Bản dưới đây dùng torch.linalg.vector_norm trên view .real/.imag — là reduction hợp nhất,
không tạo tensor tạm cỡ lớn.

MẶC ĐỊNH TẮT (args.clip_grad = 0) để đường tái hiện trung thực vẫn là mặc định.
Bật bằng --clip_grad 1.0. Khi tắt, code chạy y như bản gốc.

Khởi tạo các tensor phức này ĐÚNG (scale = 1/(c*k)^2 rồi scale*torch.rand, khớp
FEDformer gốc của thuml) — nên nổ gradient KHÔNG đến từ khởi tạo.
```

**Bằng chứng**

```
Quỹ đạo loss đo được trong epoch 1 (stride 12, RIN=0, clean, A8 đã bật, KHÔNG clipping):
    step  10   0.584        step 250      3.77
    step  70   0.154        step 270     27.6
    step 130   0.525        step 290    391
    step 190   0.262        step 310   1909
    step 230   0.971        step 330  17777
                            step 370 571064

Với clipping bật: vượt qua được step 250 (loss 0,16-1,0 ở step 220-300). Xác nhận
clipping là thứ chặn được nổ.

Chi phí clipping đo riêng (model mới khởi tạo, chưa swap): 2,455 -> 2,665 s/iter (+8,5%).
```

**Thay đổi thực tế khi dựng**

- exp_main_compare.py: thêm gradient clipping (mặc định TẮT)
- exp_main_without_ext.py: thêm gradient clipping (mặc định TẮT)
- exp_main_ablation.py: thêm gradient clipping (mặc định TẮT)

### `C5` — exp_main_*.py: THÊM resume giữa các epoch (không có trong repo tác giả)

Module: `ours/patches/choice/c5_resume.py`

File của tác giả bị chạm:

- `model/exp/exp_main_compare.py`
- `model/exp/exp_main_without_ext.py`
- `model/exp/exp_main_ablation.py`

**Vì sao**

```
ĐÂY LÀ THỨ CHÚNG TA THÊM VÀO, KHÔNG PHẢI CỦA TÁC GIẢ.
`grep -rn resume` trên bản gốc: 0 kết quả.

Lý do: chạy trên Colab free. Session bị cắt bất kỳ lúc nào (idle ~90 phút, usage
limit động, có lúc không được cấp GPU). Không có resume thì mỗi lần cắt là mất cả run.

Vòng train của tác giả KHÔNG resume được, vì EarlyStopping.save_checkpoint chỉ lưu
    torch.save(model.state_dict(), path + '/checkpoint.pth')
tức CHỈ trọng số, và chỉ lưu khi vali_loss giảm. Thiếu hết:
  - optimizer state (Adam moment m, v) -> tiếp tục từ weights không có m,v là một
    quỹ đạo huấn luyện KHÁC, không phải tiếp tục
  - số epoch đã xong    -> không biết chạy tiếp từ đâu
  - EarlyStopping state -> counter, best_score, val_loss_min reset về 0/None/Inf
  - RNG state           -> thứ tự batch và dropout mask lệch

Nên bản vá này lưu thêm `resume.pt` BÊN CẠNH `checkpoint.pth` của tác giả, sau MỖI
epoch (không chỉ khi vali_loss giảm). `checkpoint.pth` giữ nguyên ý nghĩa gốc là
"trọng số tốt nhất"; `resume.pt` là trạng thái huấn luyện hiện hành.

MẶC ĐỊNH TẮT (args.resume = 0). Khi tắt, code chạy y như bản gốc: không đọc, không
ghi, không một lệnh nào thêm trong vòng lặp nóng. Bật bằng --resume 1.

GHI CHÚ torch >= 2.6: torch.load mặc định weights_only=True, sẽ từ chối optimizer
state và numpy RNG state. Bản dưới truyền weights_only=False tường minh.
```

**Bằng chứng**

```
Bản gốc, model/utils/tools.py:67-71 — toàn bộ phần lưu checkpoint của tác giả:
    def save_checkpoint(self, val_loss, model, path):
        if self.verbose: print(f'Validation loss decreased ...')
        torch.save(model.state_dict(), path + '/' + 'checkpoint.pth')
        self.val_loss_min = val_loss
-> không optimizer, không epoch, không counter. Không đủ để resume.

EarlyStopping (tools.py:42-50) có 4 trường trạng thái cần giữ:
    counter, best_score, val_loss_min, early_stop

Anchor áp vá, giống hệt trên cả 3 file exp:
    dòng 124  for epoch in range(self.args.train_epochs):
    dòng 238  early_stopping(vali_loss, self.model, path)
```

**Thay đổi thực tế khi dựng**

- exp_main_compare.py: thêm resume giữa các epoch (mặc định TẮT)
- exp_main_without_ext.py: thêm resume giữa các epoch (mặc định TẮT)
- exp_main_ablation.py: thêm resume giữa các epoch (mặc định TẮT)
