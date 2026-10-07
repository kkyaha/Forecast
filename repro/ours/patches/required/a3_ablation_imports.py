# -*- coding: utf-8 -*-
"""A3 — model_without_FuEm.py thiếu import."""
from .._util import uncomment

ID = "A3"
TITLE = "model_without_FuEm.py: cả ba import của khối 'Custom Embedding' bị comment"
CATEGORY = "required"
FILES = ["model/model/model_without_FuEm.py"]

WHY = """
Biến thể ablation "bỏ FuEm" của Bảng 17. Ba dòng import ở dòng 21-23 bị comment:
    #from model.layers.Embed import Customize_Tokenizer
    #import torch.nn.init as nn_init
    #import math
nhưng nn_init và math vẫn được dùng ở dòng 96.

Mẫu để sửa có sẵn trong repo: model_without_LoEm.py có đúng ba dòng này KHÔNG bị comment.
"""

EVIDENCE = """
NameError: name 'nn_init' is not defined   tại model/model/model_without_FuEm.py:96
rồi sau khi sửa dòng đó: NameError: name 'math' is not defined

LƯU Ý: model_without_LoEm.py và model_without_PaEm.py KHÔNG cần vá — chúng dùng
layers/Embed_without_*.py có chữ ký CŨ (8 tham số, không static_d_token) và decoder
trả về 1 giá trị, nên lời gọi cũ trong chúng là ĐÚNG. Một bản vá trước đây của chúng ta
đã áp A1/A2 lên chúng và LÀM HỎNG hai file đang chạy được:
    TypeError: __init__() takes from 3 to 9 positional arguments but 11 were given
"""

WANT = [r"from model\.layers\.Embed import Customize_Tokenizer",
        r"import torch\.nn\.init as nn_init",
        r"import math"]


def apply(root):
    p = root / FILES[0]
    s = p.read_text()
    notes = []
    for pat in WANT:
        s, ok = uncomment(s, pat)
        if ok:
            notes.append("bỏ comment: " + pat.replace('\\', ''))
    if notes:
        p.write_text(s)
    return notes
