# -*- coding: utf-8 -*-
"""C2 — thêm stride cho train/val."""
ID = "C2"
TITLE = "data_loader.py + data_factory.py: thêm stride khi cắt cửa sổ train/val"
CATEGORY = "choice"
FILES = ["model/data/data_loader.py", "model/data/data_factory.py"]

WHY = """
QUYẾT ĐỊNH CỦA CHÚNG TA, thuần về chi phí tính toán.

Loader gốc cắt cửa sổ trượt với stride = 1, sinh 83.630 mẫu train từ ~42.000 mốc thời
gian — hai cửa sổ liền nhau trùng 95/96 giá trị. Trên Apple M5 (không CUDA) đó là
5.226 step/epoch, tức ~3,2 giờ/epoch cho mô hình 116M tham số.

stride > 1 CHỈ áp cho train/val. Tập TEST luôn giữ stride = 1 để định nghĩa metric
không đổi.

ĐÂY LÀ ĐỘ LỆCH SO VỚI TÁC GIẢ. Hệ quả: ngân sách gradient step nhỏ hơn, nên các con số
tuyệt đối KHÔNG so sánh trực tiếp được với bài báo.
"""

EVIDENCE = """
Ngân sách đo trên máy này (2,2 s/step, batch 16):
    stride   mẫu train   step/epoch   phút/epoch   4 epoch x 3 run
         1      83.630        5.226          192           38,4 h
         4      20.907        1.306           48            9,6 h
        12       6.969          435           16            3,2 h
        24       3.484          217            8            1,6 h
"""


def apply(root):
    notes = []

    p = root / FILES[0]
    s = p.read_text()
    if "self.stride = stride" not in s:
        s2 = s.replace(
            "                 RIN=True\n                 ):\n        self.seq_len = size[0]",
            "                 RIN=True,\n                 stride=1\n                 ):\n"
            "        self.stride = stride\n        self.seq_len = size[0]", 1)
        # chỉ lần xuất hiện ĐẦU TIÊN = vòng cắt cửa sổ của Dataset_Train
        s2 = s2.replace("                split_start += 1",
                        "                split_start += self.stride", 1)
        if s2 == s:
            raise RuntimeError("C2: không khớp mẫu trong data_loader.py")
        p.write_text(s2)
        notes.append("data_loader.py: Dataset_Train nhận tham số stride")

    p = root / FILES[1]
    s = p.read_text()
    if "stride =" not in s:
        s2 = s.replace(
            "            static_cat_vab = args.static_cat_vab,\n            RIN = args.RIN",
            "            static_cat_vab = args.static_cat_vab,\n            RIN = args.RIN,\n"
            "            # test LUÔN stride=1 để định nghĩa metric không đổi\n"
            "            stride = 1 if flag == 'test' else getattr(args, 'stride_train', 1)", 1)
        if s2 == s:
            raise RuntimeError("C2: không khớp mẫu trong data_factory.py")
        p.write_text(s2)
        notes.append("data_factory.py: truyền stride (test luôn = 1)")

    return notes
