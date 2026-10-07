#!/usr/bin/env python3
"""
Dựng build/conem = upstream/conem (code tác giả, nguyên trạng) + các bản vá của chúng ta.

NGUYÊN TẮC CẤU TRÚC
    upstream/conem   100% TÁC GIẢ. Không bao giờ sửa. Còn .git để kiểm chứng nguồn.
    ours/            100% CỦA CHÚNG TA.
    build/conem      SINH RA. Không sửa tay. Xoá được, dựng lại được.

Nhờ vậy, câu hỏi "dòng này của ai" luôn có câu trả lời dứt khoát.

Cách dùng
    python ours/build.py                  dựng đủ (required + choices)
    python ours/build.py --only-required  chỉ nhóm required — để thấy nó phân kỳ
    python ours/build.py --list           chỉ liệt kê các bản vá, không dựng
    python ours/build.py --audit          dựng rồi xuất build/audit.diff

Xuất ra
    build/conem/            cây code đã vá
    build/PATCHES.md        báo cáo đã áp những gì, vì sao, bằng chứng gì
    build/audit.diff        diff thống nhất so với upstream (với --audit)
"""
import argparse
import shutil
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent          # ours/
REPRO = HERE.parent                             # repro/
UPSTREAM = REPRO / 'upstream' / 'conem'
BUILD = REPRO / 'build' / 'conem'

sys.path.insert(0, str(HERE))
import patches  # noqa: E402


def copy_upstream():
    if BUILD.exists():
        shutil.rmtree(BUILD)
    BUILD.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(UPSTREAM, BUILD,
                    ignore=shutil.ignore_patterns('.git', '__pycache__', '*.pyc'))
    n = len(list(BUILD.rglob('*.py')))
    return n


def write_manifest(applied, only_required):
    lines = [
        "# build/conem — những gì đã được áp",
        "",
        "> **Sinh tự động bởi `ours/build.py`. Không sửa tay thư mục `build/`.**",
        "> Code tác giả nguyên trạng ở `upstream/conem/`; các bản vá ở `ours/patches/`.",
        "",
        f"Chế độ dựng: **{'chỉ required' if only_required else 'required + choices'}**",
        "",
        "## Phân loại",
        "",
        "| nhóm | nghĩa |",
        "|---|---|",
        "| **required** | Không có thì code **crash** — không chạy được gì. Không có lựa chọn khác. |",
        "| **choice** | **Quyết định của chúng ta.** Code vẫn chạy nếu bỏ. Tức KHÔNG phải tái hiện trung thực. |",
        "",
    ]
    for cat, label in (('required', 'BẮT BUỘC — không có thì crash'),
                       ('choice', 'LỰA CHỌN CỦA CHÚNG TA — code vẫn chạy nếu bỏ')):
        group = [(n, m, notes) for n, m, notes in applied if m.CATEGORY == cat]
        if not group:
            continue
        lines += [f"## {label}", ""]
        for name, mod, notes in group:
            lines += [
                f"### `{mod.ID}` — {mod.TITLE}",
                "",
                f"Module: `ours/patches/{cat}/{name}.py`",
                "",
                "File của tác giả bị chạm:",
                "",
            ]
            lines += [f"- `{f}`" for f in mod.FILES]
            lines += ["", "**Vì sao**", "", "```", mod.WHY.strip(), "```", ""]
            lines += ["**Bằng chứng**", "", "```", mod.EVIDENCE.strip(), "```", ""]
            lines += ["**Thay đổi thực tế khi dựng**", ""]
            lines += ([f"- {x}" for x in notes] if notes
                      else ["- (không thay đổi gì — đã ở trạng thái đích)"])
            lines += [""]
    (BUILD.parent / 'PATCHES.md').write_text("\n".join(lines), encoding='utf-8')


def audit_diff():
    out = BUILD.parent / 'audit.diff'
    r = subprocess.run(
        ['diff', '-ru', '--exclude=__pycache__', '--exclude=.git',
         str(UPSTREAM), str(BUILD)],
        capture_output=True, text=True)
    out.write_text(r.stdout, encoding='utf-8')
    changed = sum(1 for l in r.stdout.split('\n') if l[:1] in '+-' and l[:3] not in ('---', '+++'))
    return out, changed


def summarise_footprint():
    import hashlib
    h = lambda p: hashlib.md5(p.read_bytes()).hexdigest()
    same = diff = 0
    for f in UPSTREAM.rglob('*.py'):
        if '.git' in f.parts:
            continue
        t = BUILD / f.relative_to(UPSTREAM)
        if not t.exists():
            continue
        if h(f) == h(t):
            same += 1
        else:
            diff += 1
    return same, diff


def main():
    ap = argparse.ArgumentParser(description="Dựng build/conem từ upstream + patches")
    ap.add_argument('--only-required', action='store_true',
                    help="chỉ áp nhóm required (để thấy code tác giả phân kỳ)")
    ap.add_argument('--list', action='store_true', help="chỉ liệt kê các bản vá")
    ap.add_argument('--audit', action='store_true', help="xuất build/audit.diff")
    a = ap.parse_args()

    plan = patches.all_patches(only_required=a.only_required)

    if a.list:
        print(f"{'mã':8s} {'nhóm':10s} mô tả")
        print("-" * 92)
        for name, mod in plan:
            print(f"{mod.ID:8s} {mod.CATEGORY:10s} {mod.TITLE}")
        return

    if not UPSTREAM.exists():
        sys.exit(f"Thiếu {UPSTREAM}\n"
                 f"  git clone https://github.com/anonymous7594/conem {UPSTREAM}")

    print(f"upstream : {UPSTREAM}")
    r = subprocess.run(['git', '-C', str(UPSTREAM), 'rev-parse', 'HEAD'],
                       capture_output=True, text=True)
    if r.returncode == 0:
        print(f"commit   : {r.stdout.strip()[:12]}")
    dirty = subprocess.run(['git', '-C', str(UPSTREAM), 'status', '--porcelain'],
                           capture_output=True, text=True).stdout.strip()
    print(f"nguyên trạng: {'KHÔNG — upstream đã bị sửa!' if dirty else 'có (git status sạch)'}")

    n = copy_upstream()
    print(f"\nđã copy {n} file .py -> {BUILD}\n")

    applied = []
    for name, mod in plan:
        try:
            notes = mod.apply(BUILD)
        except Exception as e:
            sys.exit(f"  LỖI  {mod.ID}: {e}")
        applied.append((name, mod, notes))
        mark = '~' if notes else '='
        print(f"  {mark} {mod.ID:8s} [{mod.CATEGORY:8s}] {mod.TITLE}")
        for x in notes:
            print(f"        - {x}")

    write_manifest(applied, a.only_required)
    same, diff = summarise_footprint()
    print(f"\nDấu chân: {same} file giống hệt upstream, {diff} file đã vá")
    print(f"Báo cáo : {BUILD.parent / 'PATCHES.md'}")

    if a.audit:
        out, changed = audit_diff()
        print(f"Kiểm toán: {out}  ({changed} dòng khác upstream)")

    print("\nLõi cơ chế ConEm (layers/Embed*.py) KHÔNG nằm trong danh sách vá.")


if __name__ == '__main__':
    main()
