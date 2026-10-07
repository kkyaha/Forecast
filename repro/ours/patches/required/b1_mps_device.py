# -*- coding: utf-8 -*-
"""B1 — thêm nhánh thiết bị MPS."""
ID = "B1"
TITLE = "exp_basic.py: thêm nhánh thiết bị MPS (Apple Silicon)"
CATEGORY = "required"
FILES = ["model/exp/exp_basic.py"]

WHY = """
_acquire_device() chỉ có hai nhánh: cuda nếu use_gpu, ngược lại cpu. Máy chạy là
Apple M5 không có CUDA, nên nếu không vá thì toàn bộ huấn luyện rơi về CPU —
đo được chậm hơn MPS 3x (4,970 vs 1,686 s/iter, batch 8).

Đây là 'required' theo nghĩa thực tế: trên CPU một epoch ở stride 1 mất ~866 phút,
tức không thể chạy.
"""

EVIDENCE = """
Đo trên máy này, forward+backward, batch 8, mô hình FEDformer+ConEm 116M tham số:
    CPU  4,970 s/iter  -> ~866 phút/epoch @stride 1
    MPS  1,686 s/iter  -> ~294 phút/epoch @stride 1
torch 2.8.0, torch.backends.mps.is_available() = True
"""

NEW = '''    def _acquire_device(self):
        if self.args.use_gpu:
            os.environ["CUDA_VISIBLE_DEVICES"] = str(
                self.args.gpu) if not self.args.use_multi_gpu else self.args.devices
            device = torch.device('cuda:{}'.format(self.args.gpu))
            print('Use GPU: cuda:{}'.format(self.args.gpu))
        elif getattr(self.args, 'use_mps', False) and torch.backends.mps.is_available():
            device = torch.device('mps')
            print('Use MPS (Apple Silicon)')
        else:
            device = torch.device('cpu')
            print('Use CPU')
        return device
'''


def apply(root):
    p = root / FILES[0]
    s = p.read_text()
    if "Use MPS (Apple Silicon)" in s:
        return []
    a = s.index("    def _acquire_device(self):")
    b = s.index("    def _get_data(self):")
    p.write_text(s[:a] + NEW + "\n" + s[b:])
    return ["exp_basic.py: thêm nhánh MPS"]
