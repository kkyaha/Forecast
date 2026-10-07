# -*- coding: utf-8 -*-
"""C4 — THÊM gradient clipping. Không có trong repo tác giả."""
ID = "C4"
TITLE = "exp_main_*.py: THÊM gradient clipping (không có trong repo tác giả)"
CATEGORY = "choice"
FILES = ["model/exp/exp_main_compare.py",
         "model/exp/exp_main_without_ext.py",
         "model/exp/exp_main_ablation.py"]

WHY = """
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
"""

EVIDENCE = """
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
"""

OLD = ("                    loss.backward()\n"
       "                    model_optim.step()")

NEW = """                    loss.backward()
                    # [vá C4 — THÊM BỞI BẢN TÁI HIỆN, không có trong repo tác giả]
                    # 86,7% tham số là số phức nên clip_grad_norm_ không dùng được;
                    # vector_norm trên view .real/.imag tránh tensor tạm cỡ lớn.
                    _cg = getattr(self.args, 'clip_grad', 0)
                    if _cg and _cg > 0:
                        _sq = None
                        for _q in self.model.parameters():
                            if _q.grad is None:
                                continue
                            _g = _q.grad
                            if _g.is_complex():
                                _n2 = (torch.linalg.vector_norm(_g.real) ** 2
                                       + torch.linalg.vector_norm(_g.imag) ** 2)
                            else:
                                _n2 = torch.linalg.vector_norm(_g) ** 2
                            _sq = _n2 if _sq is None else _sq + _n2
                        if _sq is not None:
                            _co = _cg / (torch.sqrt(_sq) + 1e-6)
                            if _co < 1:
                                for _q in self.model.parameters():
                                    if _q.grad is not None:
                                        _q.grad.mul_(_co)
                    model_optim.step()"""


def apply(root):
    notes = []
    for rel in FILES:
        p = root / rel
        s = p.read_text()
        if "vá C4" in s:
            continue
        s2 = s.replace(OLD, NEW, 1)
        if s2 != s:
            p.write_text(s2)
            notes.append(f"{p.name}: thêm gradient clipping (mặc định TẮT)")
    return notes
