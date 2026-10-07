# -*- coding: utf-8 -*-
"""B2 — map_location hardcode CUDA."""
ID = "B2"
TITLE = "exp_main_*.py: torch.load(map_location='cuda:0') -> self.device"
CATEGORY = "required"
FILES = ["model/exp/exp_main_compare.py",
         "model/exp/exp_main_without_ext.py",
         "model/exp/exp_main_ablation.py"]

WHY = """
Sau early stopping, train() nạp lại checkpoint tốt nhất bằng
    torch.load(best_model_path, map_location='cuda:0')
Trên máy không có CUDA, lệnh này crash — tức mọi run đều chết ở bước nạp checkpoint,
ngay trước khi đánh giá test.
"""

EVIDENCE = """
RuntimeError: Attempting to deserialize object on a CUDA device but
torch.cuda.is_available() is False.
3 chỗ mỗi file (train(), predict(), và một chỗ đã bị comment).
"""


def apply(root):
    notes = []
    for rel in FILES:
        p = root / rel
        s = p.read_text()
        n = s.count("map_location='cuda:0'")
        if n:
            p.write_text(s.replace("map_location='cuda:0'", "map_location=self.device"))
            notes.append(f"{p.name}: {n} chỗ map_location")
    return notes
