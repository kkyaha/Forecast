# -*- coding: utf-8 -*-
"""A8 — bật lại adjust_learning_rate."""
import re

ID = "A8"
TITLE = "exp_main_*.py: bật lại adjust_learning_rate (bị comment)"
CATEGORY = "choice"
FILES = ["model/exp/exp_main_compare.py",
         "model/exp/exp_main_without_ext.py",
         "model/exp/exp_main_ablation.py"]

WHY = """
QUYẾT ĐỊNH CỦA CHÚNG TA, nhưng là khôi phục bản trước đó của chính tác giả.

Dòng 219 mỗi file:
    #adjust_learning_rate(model_optim, epoch + 1, self.args)
nên learning rate đứng yên ở 1e-5 suốt. Với lradj='type1' hàm này giảm một nửa LR mỗi
epoch — đúng hành vi codebase gốc Informer/Autoformer/FEDformer mà repo này dẫn xuất từ.

BẰNG CHỨNG ĐÂY LÀ THỨ BỊ TẮT VỀ SAU, không phải chủ ý: hai file backup cũ của chính
tác giả vẫn gọi nó KHÔNG comment —
    exp_main - Copy.py:174   adjust_learning_rate(model_optim, epoch + 1, self.args)
    exp_main - dev.py:174    adjust_learning_rate(model_optim, epoch + 1, self.args)
"""

EVIDENCE = """
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
"""


def apply(root):
    notes = []
    for rel in FILES:
        p = root / rel
        s = p.read_text()
        if re.search(r"^\s+adjust_learning_rate\(model_optim", s, re.M):
            continue
        s2 = re.sub(r"^(\s*)#\s*(adjust_learning_rate\(model_optim[^\n]*)$",
                    r"\1\2", s, count=1, flags=re.M)
        if s2 != s:
            p.write_text(s2)
            notes.append(f"{p.name}: bật lại adjust_learning_rate")
    return notes
