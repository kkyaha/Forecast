# -*- coding: utf-8 -*-
"""A1 + A2 — model_fed_based.py không chạy được."""
from .._util import sub_once

ID = "A1+A2"
TITLE = "model_fed_based.py: lời gọi DataEmbedding lệch 2 tham số + không giải nén decoder"
CATEGORY = "required"
FILES = ["model/model/model_fed_based.py"]

WHY = """
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
"""

EVIDENCE = """
A1: KeyError: 0            tại layers/Embed.py:293  (self.freq_map[freq] với freq=0)
A2: AttributeError: 'tuple' object has no attribute 'shape'
                           tại layers/Correlation.py:158

Mẫu để sửa KHÔNG do chúng ta nghĩ ra: model_info_based.py và model_auto_based.py
trong cùng repo gọi ĐÚNG cả hai chỗ. Vá này copy y nguyên cách gọi của chúng.
"""


def apply(root):
    p = root / FILES[0]
    s = p.read_text()
    notes = []
    for old, new, label in [
        ("DataEmbedding_encoder(enc_in, d_model, embed, freq,",
         "DataEmbedding_encoder(enc_in, d_model, static_d_token, contextual_encoder_d_token, embed, freq,",
         "A1 encoder: thêm static_d_token, contextual_encoder_d_token"),
        ("DataEmbedding_decoder(dec_in, d_model, embed, freq,",
         "DataEmbedding_decoder(dec_in, d_model, static_d_token, contextual_decoder_d_token, embed, freq,",
         "A1 decoder: thêm static_d_token, contextual_decoder_d_token"),
        ("        dec_out = self.dec_embedding(seasonal_init,",
         "        dec_out, x_future_context = self.dec_embedding(seasonal_init,",
         "A2: giải nén 2 giá trị trả về của dec_embedding"),
    ]:
        s, ok = sub_once(s, old, new)
        if ok:
            notes.append(label)
    if notes:
        p.write_text(s)
    return notes
