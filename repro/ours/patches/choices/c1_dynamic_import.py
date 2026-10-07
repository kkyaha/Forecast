# -*- coding: utf-8 -*-
"""C1 — import Model động."""
import re

ID = "C1"
TITLE = "exp_main_*.py: import Model ở cấp module -> import động theo args.model_file"
CATEGORY = "choice"
FILES = ["model/exp/exp_main_compare.py",
         "model/exp/exp_main_without_ext.py",
         "model/exp/exp_main_ablation.py"]

WHY = """
Mỗi file exp hardcode một backbone ở đầu file, ví dụ:
    from model.model.model_info_based import Model
Muốn đổi backbone, tác giả phải sửa dòng này rồi chạy lại. Vá này chuyển sang import
động theo args.model_file để một lệnh chạy chọn được backbone.

THUẦN TIỆN LỢI. Không có nó code vẫn chạy — chỉ là phải sửa file mỗi lần đổi backbone.
Nếu muốn tối giản tuyệt đối, bỏ vá này và sửa tay dòng import.
"""

EVIDENCE = """
exp_main_compare.py:10     from model.model.model_info_based import Model
exp_main_without_ext.py:10 from model.model.model_informer import Model
exp_main_ablation.py:10    from model.model.model_without_LoEm import Model
"""

NEW = '''    def _build_model(self):
        from importlib import import_module
        Model = import_module('model.model.' + self.args.model_file).Model
        model = Model('''


def apply(root):
    notes = []
    for rel in FILES:
        p = root / rel
        s = orig = p.read_text()
        if "import_module('model.model.'" in s:
            continue
        s = re.sub(r"^from model\.model\.[A-Za-z_0-9]+ import Model.*$",
                   "# [vá C1] Model được import động trong _build_model theo args.model_file",
                   s, count=1, flags=re.M)
        s = s.replace("    def _build_model(self):\n        model = Model(", NEW, 1)
        if s != orig:
            p.write_text(s)
            notes.append(f"{p.name}: import Model động")
    return notes
