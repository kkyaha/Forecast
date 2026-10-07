#!/usr/bin/env python3
"""
Kiểm chứng: bản gốc của tác giả chạy được tới đâu?

Chạy thử dựng từng Model trực tiếp từ upstream/conem/ (y nguyên code tác giả, 0 thay đổi)
để xác định vá nào THỰC SỰ cần thiết và vá nào là dư thừa.

Cách dùng:  python verify_pristine.py
"""
import sys
import traceback
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPRO = HERE.parent.parent
PRISTINE = REPRO / 'upstream' / 'conem'     # 100% tác giả
PATCHED  = REPRO / 'build' / 'conem'        # sinh ra: upstream + patches

KW = dict(seq_len=96, label_len=48, pred_len=96, freq='h', moving_avg=25,
          cat_vab=[], num_vab=list('abcdef') + ['T'], static_vab=['location'],
          e_layers=2, d_layers=1, modes=64, d_model=512, d_ff=1024,
          enc_in=1, dec_in=1, c_out=1, dropout=1e-5, L=3, base='legendre',
          cross_activation='tanh', n_heads=8, activation='gelu', embed='timeF',
          RIN=True, init_weights=1, static_d_token=30,
          contextual_encoder_d_token=30, contextual_decoder_d_token=30)

MODELS = [
    ('model_info_based',   'Informer + ConEm'),
    ('model_auto_based',   'Autoformer + ConEm'),
    ('model_fed_based',    'FEDformer + ConEm   <- bài báo báo cáo tốt nhất'),
    ('model_without_LoEm', 'ablation: bỏ LoEm'),
    ('model_without_FuEm', 'ablation: bỏ FuEm'),
    ('model_without_PaEm', 'ablation: bỏ PaEm'),
    ('model_fedformer',    'FEDformer gốc (baseline)'),
]


def try_build(tree, name):
    """Dựng Model trong tiến trình con để tránh xung đột sys.modules giữa 2 cây code."""
    import subprocess
    code = (
        "import sys, warnings; warnings.filterwarnings('ignore')\n"
        f"sys.path.insert(0, {str(tree)!r})\n"
        "import io, contextlib\n"
        "from importlib import import_module\n"
        f"kw = {KW!r}\n"
        "buf = io.StringIO()\n"
        "try:\n"
        "    with contextlib.redirect_stdout(buf):\n"
        f"        import_module('model.model.{name}').Model(**kw)\n"
        "    print('OK')\n"
        "except Exception as e:\n"
        "    import traceback\n"
        "    tb = traceback.extract_tb(sys.exc_info()[2])[-1]\n"
        "    loc = tb.filename.split('upstream/conem/')[-1].split('conem/')[-1]\n"
        "    print('FAIL|%s: %s|%s:%d' % (type(e).__name__, e, loc, tb.lineno))\n"
    )
    r = subprocess.run([sys.executable, '-c', code], capture_output=True, text=True)
    out = (r.stdout or '').strip().split('\n')[-1]
    if out == 'OK':
        return True, '', ''
    if out.startswith('FAIL|'):
        _, msg, loc = out.split('|', 2)
        return False, msg, loc
    return False, (r.stderr or out).strip()[:90], ''


def main():
    if not PRISTINE.exists():
        sys.exit("Thiếu upstream/conem/ — clone: "
                 "git clone https://github.com/anonymous7594/conem conem_pristine")

    print("=" * 84)
    print("BẢN GỐC TÁC GIẢ (upstream/conem/, commit 80f0e01, 0 thay đổi)")
    print("=" * 84)
    print("\n1) Có entry point nào chạy trực tiếp được không?")
    hits = [p for p in PRISTINE.rglob('*.py')
            if '.git' not in p.parts and
            any(k in p.read_text(errors='ignore') for k in ('__main__', 'ArgumentParser'))]
    print(f"   file có __main__ / ArgumentParser: {len(hits)}")
    print("   -> KHÔNG có cách nào gọi code; README ghi 'Code will be released soon'.")
    print("      Vì vậy run.py (entry point dựng lại 66 tham số) là BẮT BUỘC.\n")

    print("2) Nếu được cấp entry point, từng Model có dựng được không?\n")
    fails = []
    for name, desc in MODELS:
        ok, msg, loc = try_build(PRISTINE, name)
        if ok:
            print(f"   OK     {name:20s}  {desc}")
        else:
            print(f"   CRASH  {name:20s}  {desc}")
            print(f"          {msg}")
            if loc:
                print(f"          tại {loc}")
            fails.append(name)

    print("\n" + "=" * 84)
    print("BẢN ĐÃ VÁ (build/conem/ — sinh ra bởi ours/build.py)")
    print("=" * 84 + "\n")
    for name, desc in MODELS:
        ok, msg, loc = try_build(PATCHED, name)
        print(f"   {'OK    ' if ok else 'CRASH '} {name:20s}  {desc}"
              + ('' if ok else f"\n          {msg}"))

    print("\n" + "=" * 84)
    print("KẾT LUẬN")
    print("=" * 84)
    print(f"   Bản gốc crash ở: {', '.join(fails) if fails else '(không file nào)'}")
    print("   Đây chính xác là những file trong nhóm REQUIRED của ours/patches/.")
    print("   Các file khác KHÔNG bị chạm tới.")


if __name__ == "__main__":
    main()
