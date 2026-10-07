# -*- coding: utf-8 -*-
"""A4 + A5 — ba file baseline không chạy được ngoài máy tác giả."""
import re

ID = "A4"
TITLE = "3 file baseline: os.chdir sang đường dẫn máy tác giả + import tương đối"
CATEGORY = "required"
FILES = ["model/model/model_fedformer.py",
         "model/model/model_autoformer.py",
         "model/model/model_informer.py"]

WHY = """
Đây là CÁC CỘT BASELINE của Bảng 10 — tức nửa còn lại của so sánh cốt lõi
"backbone vs backbone + ConEm". Cả ba file có ở dòng 4:

    os.chdir("/home/ad/20813716/transformer_sales_forecast/model")

là đường dẫn tuyệt đối trên máy riêng của tác giả, và ngay sau đó import kiểu
`from layers.Embed import ...` chỉ hoạt động SAU cú chdir đó.

HỆ QUẢ: không máy nào khác chạy được baseline, nên KHÔNG THỂ tái hiện so sánh cốt lõi
của bài báo từ code đã công bố.

Mẫu để sửa có sẵn: các file *_based.py trong cùng repo đã comment chdir và dùng
`from model.layers.` — vá này áp y nguyên cách đó.
"""

EVIDENCE = """
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
"""


def apply(root):
    notes = []
    for rel in FILES:
        p = root / rel
        s = orig = p.read_text()
        s = re.sub(r'^os\.chdir\("/home/ad/[^"]*"\)',
                   '# [vá A4] os.chdir sang máy tác giả — đã vô hiệu',
                   s, count=1, flags=re.M)
        s = re.sub(r'^print\(os\.getcwd\(\)\)', '', s, count=1, flags=re.M)
        for mod in ("layers", "utils", "data"):
            s = re.sub(rf'^from {mod}\.', f'from model.{mod}.', s, flags=re.M)
        if s != orig:
            p.write_text(s)
            notes.append(f"{p.name}: vô hiệu os.chdir + import tuyệt đối theo package")
    return notes
