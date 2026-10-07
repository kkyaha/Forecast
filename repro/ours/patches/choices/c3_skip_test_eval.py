# -*- coding: utf-8 -*-
"""C3 — bỏ đánh giá tập test sau mỗi epoch."""
ID = "C3"
TITLE = "exp_main_*.py: bỏ đánh giá tập test sau mỗi epoch (chỉ dùng để in)"
CATEGORY = "choice"
FILES = ["model/exp/exp_main_compare.py",
         "model/exp/exp_main_without_ext.py",
         "model/exp/exp_main_ablation.py"]

WHY = """
QUYẾT ĐỊNH CỦA CHÚNG TA, thuần về chi phí.

Vòng train gọi self.vali(test_data, test_loader, criterion) sau MỖI epoch, nhưng
test_loss chỉ được IN RA — early stopping dùng vali_loss. Tập test chạy ở stride 1
(25.900 mẫu, ~1.619 batch) nên bước này chiếm ~75% thời gian mỗi epoch mà không ảnh
hưởng gì tới việc học.

Sau vá, hành vi do args.eval_test_each_epoch điều khiển (mặc định False).
exp.test() cuối cùng VẪN chạy đủ tập test ở stride 1.
"""

EVIDENCE = """
Đo được: epoch mất 2.375s, trong đó train chỉ ~430s.
    217 step train  x 2,0s  = ~430s
    val   ~54 batch         = ~100s
    test  ~1.619 batch      = ~1.800s   <- 75%
"""

OLD = ("            if self.args.if_test:\n"
       "                test_loss = self.vali(test_data, test_loader, criterion)")
NEW = ("            if self.args.if_test and getattr(self.args, 'eval_test_each_epoch', False):\n"
       "                test_loss = self.vali(test_data, test_loader, criterion)")


def apply(root):
    notes = []
    for rel in FILES:
        p = root / rel
        s = p.read_text()
        if "eval_test_each_epoch" in s:
            continue
        s2 = s.replace(OLD, NEW, 1)
        if s2 != s:
            p.write_text(s2)
            notes.append(f"{p.name}: bỏ eval test mỗi epoch")
    return notes
