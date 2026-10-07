"""
Danh mục các bản vá áp lên code tác giả.

Mỗi bản vá là một module riêng, khai báo:

    ID        mã ngắn, dùng trong báo cáo (A1, A4, C4, ...)
    TITLE     một dòng mô tả
    CATEGORY  'required' = không có thì code CRASH, không chạy được gì
              'choice'   = QUYẾT ĐỊNH CỦA CHÚNG TA, code vẫn chạy nếu bỏ
    FILES     các file của tác giả bị chạm tới
    WHY       vì sao cần
    EVIDENCE  bằng chứng kiểm tra được (thông báo lỗi, số dòng, số đo)
    apply(root) -> list[str]   áp vá, trả về mô tả từng thay đổi; idempotent

Phân biệt 'required' và 'choice' là điểm quan trọng nhất của thư mục này:
  - required: nếu bỏ, không thể chạy -> không có lựa chọn nào khác
  - choice  : chúng ta chủ động thay đổi hành vi -> KHÔNG phải tái hiện trung thực

Dựng bằng:  python ours/build.py              (tất cả)
            python ours/build.py --only-required   (chỉ nhóm required, để thấy nó phân kỳ)
"""
from importlib import import_module

# Thứ tự áp. required trước, choices sau.
REQUIRED = [
    'a1_a2_fed_based',
    'a3_ablation_imports',
    'a4_baseline_chdir',
    'b1_mps_device',
    'b2_map_location',
]

CHOICES = [
    'c1_dynamic_import',
    'a6fix_rin',
    'a8_lr_decay',
    'c2_stride',
    'c3_skip_test_eval',
    'c4_grad_clip',
    'c5_resume',
]


def load(name, category):
    return import_module(f'patches.{category}.{name}')


def all_patches(only_required=False):
    out = [(n, load(n, 'required')) for n in REQUIRED]
    if not only_required:
        out += [(n, load(n, 'choices')) for n in CHOICES]
    return out
