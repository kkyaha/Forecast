# -*- coding: utf-8 -*-
"""A6fix — RIN: khôi phục dạng (sum+1) mà tác giả đã comment."""
ID = "A6fix"
TITLE = "model_fed_based.py: RIN dùng dạng (sum+1) thay vì sum ở hệ số affine"
CATEGORY = "choice"
FILES = ["model/model/model_fed_based.py"]

WHY = """
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
"""

EVIDENCE = """
Với dòng 276 (bản gốc), phân kỳ ở epoch 2 bất kể dữ liệu raw hay clean:
    Epoch 1: train 28,80    val 17,30
    Epoch 2: train 6,12e7   val 6,72e8
    Epoch 3: train 8,14e14  -> early stopping
MSE test cuối: 14,4151 (raw) / 14,4297 (clean) — tệ hơn persistence (10,54).
"""

FWD_OLD = ("            x_enc = x_enc*torch.sum(self.affine_weight_input*x_static_enc,"
           "dim=2,keepdim=True) + torch.sum(self.affine_bias_input*x_static_enc,dim=2,keepdim=True)")
FWD_NEW = ('            # [vá A6fix] dạng "1 + phần học được", chính là dòng tác giả đã comment ở trên\n'
           "            x_enc = x_enc*(torch.add(torch.sum(self.affine_weight_input*x_static_enc,"
           "dim=2,keepdim=True),1)) + torch.sum(self.affine_bias_input*x_static_enc,dim=2,keepdim=True)")
INV_OLD = ("            dec_out = dec_out/torch.sum(self.affine_weight_input*"
           "x_static_dec[:,-self.pred_len:,:],dim=2,keepdim=True)")
INV_NEW = ("            # [vá A6fix] mẫu số khớp chiều chuẩn hoá: (sum + 1), luôn quanh 1\n"
           "            dec_out = dec_out/(torch.sum(self.affine_weight_input*"
           "x_static_dec[:,-self.pred_len:,:],dim=2,keepdim=True) + 1)")


def apply(root):
    p = root / FILES[0]
    s = p.read_text()
    if "[vá A6fix]" in s:
        return []
    notes = []
    if FWD_OLD in s:
        s = s.replace(FWD_OLD, FWD_NEW, 1); notes.append("chiều chuẩn hoá: hệ số (sum+1)")
    if INV_OLD in s:
        s = s.replace(INV_OLD, INV_NEW, 1); notes.append("chiều khôi phục: mẫu số (sum+1)")
    if len(notes) != 2:
        raise RuntimeError(f"A6fix chỉ khớp {len(notes)}/2 dòng — kiểm tra thủ công")
    p.write_text(s)
    return notes
