#!/usr/bin/env python3
"""
Chạy ma trận thực nghiệm tuần tự, chịu được session bị cắt giữa đường.

PHẦN NÀY LÀ CỦA CHÚNG TA, không phải của tác giả.

Viết cho Colab free, nơi session chết bất kỳ lúc nào (idle ~90 phút, usage limit động,
có lúc không được cấp GPU). Nguyên tắc:

  - MỘT run một lúc. Không song song: VRAM 16GB và dynamic limit sẽ phản tác dụng.
  - Mỗi run gọi run.py như subprocess, với --resume 1 (vá C5).
  - Run nào đã có results/<setting>/metrics.npy thì BỎ QUA. Nên chạy lại cell sau khi
    bị cắt là tiếp đúng chỗ dở, không mất gì.
  - Checkpoint + log ghi thẳng vào Drive, không vào /content (mất khi session chết).
  - Hết ngân sách thời gian thì dừng SẠCH giữa hai run, không cắt giữa một run.

Cách dùng:
  python ours/runner/run_matrix.py --list
  python ours/runner/run_matrix.py \
      --checkpoints /content/drive/MyDrive/conem/checkpoints \
      --logs /content/drive/MyDrive/conem/logs \
      --budget_min 150
"""
import argparse
import os
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
REPRO = os.path.dirname(os.path.dirname(HERE))
RUN_PY = os.path.join(HERE, "run.py")

SHORT = {'T (degC)': 'T', 'rh (%)': 'rh', 'Tdew (degC)': 'Tdew', 'VPmax (mbar)': 'VPmax',
         'VPact (mbar)': 'VPact', 'VPdef (mbar)': 'VPdef', 'wv (m/s)': 'wv',
         'wd (deg)': 'wd', 'rain (mm)': 'rain'}

# Thiết kế A/B/C giữ nguyên từ run_all.sh:
#   A  conem    / ext=paper    tái hiện số liệu chủ đạo
#   C  backbone / không ext    baseline bài báo đối chiếu
#   B  conem    / ext=noninfo  KHỬ RÒ RỈ — cùng kiến trúc, nhưng yếu tố ngoại sinh là
#                              wv/wd/rain (R2 < 0,1 với target, không xác định target)
#
# Nếu A << C mà B ~ C thì mức cải thiện đến từ việc ĐƯỢC BIẾT TRƯỚC biến quyết định
# target, không từ kiến trúc ConEm. Đây là trục lập luận chính của báo cáo.
CONFIGS = [
    ('A', dict(variant='conem',    ext='paper')),
    ('B', dict(variant='conem',    ext='noninfo')),
    ('C', dict(variant='backbone', ext='paper')),
]

# Hai target, chọn theo kết quả leakage_test.py:
#   T (degC)  R2 = 1,0000 từ 6 biến ngoại sinh -> để ĐỐI CHIẾU được với Bảng 11
#   rh (%)    R2 = 0,8809                      -> để KIỂM TRA cơ chế thật
# Chạy trên T một mình thì mọi phương pháp đều rất tốt và so sánh vô nghĩa.
TARGETS = ['T (degC)', 'rh (%)']


def setting_name(a):
    """Dựng lại đúng chuỗi setting mà run.py sinh ra. Phải khớp từng ký tự."""
    short = SHORT.get(a['target'], a['target'])
    return (f"weather_{short}_{a['variant']}"
            f"{'-' + a['ablate'] if a.get('ablate') else ''}_{a['backbone']}"
            f"_ext-{a['ext']}_I{a['seq_len']}_O{a['pred_len']}"
            f"_el{a['e_layers']}_dl{a['d_layers']}"
            f"_lr{a['learning_rate']}_st{a['stride_train']}{a['tag']}")


def build_matrix(args):
    runs = []
    for target in TARGETS:
        for label, cfg in CONFIGS:
            a = dict(backbone=args.backbone, target=target, tag=args.tag,
                     seq_len=96, pred_len=96, e_layers=2, d_layers=1,
                     learning_rate=args.learning_rate,
                     stride_train=args.stride_train, ablate=None, **cfg)
            runs.append((label, a))
    return runs


def is_done(a, results_dir):
    """Run đã xong = có metrics.npy. Đây là thứ exp.test() ghi ra cuối cùng."""
    return os.path.exists(os.path.join(results_dir, setting_name(a), 'metrics.npy'))


def main():
    p = argparse.ArgumentParser(description="Ma tran thuc nghiem ConEm, tuan tu + resume")
    p.add_argument('--checkpoints', default=os.path.join(REPRO, 'checkpoints'),
                   help="NEN tro vao Drive khi chay Colab")
    p.add_argument('--logs', default=os.path.join(REPRO, 'logs'))
    p.add_argument('--results', default=os.path.join(REPRO, 'results'),
                   help="exp.test() cua tac gia hardcode './results/' -> repro/results")
    p.add_argument('--data_path', default='weather_all_clean.csv',
                   help="clean = da xu ly -9999; so sanh cong bang phai dung ban nay")
    p.add_argument('--backbone', default='fedformer')
    p.add_argument('--stride_train', type=int, default=24)
    p.add_argument('--learning_rate', type=float, default=1e-5)
    p.add_argument('--train_epochs', type=int, default=10)
    p.add_argument('--clip_grad', type=float, default=1.0,
                   help="0 = y nhu tac gia (no gradient ~step 250). 1.0 = va C4 bat.")
    p.add_argument('--tag', default='')
    p.add_argument('--budget_min', type=float, default=0,
                   help="0 = khong gioi han. >0 = dung sach giua hai run khi vuot.")
    p.add_argument('--keep_resume', action='store_true',
                   help="giu resume.pt sau khi run xong (~2,58 GB/run). Mac dinh xoa: "
                        "Drive free chi 15 GB.")
    p.add_argument('--list', action='store_true', help="chi liet ke, khong chay")
    args = p.parse_args()

    for d in (args.checkpoints, args.logs, args.results):
        os.makedirs(d, exist_ok=True)

    runs = build_matrix(args)
    t0 = time.time()

    print("=" * 78)
    print(f"  ma tran  : {len(runs)} run  ({len(CONFIGS)} cau hinh x {len(TARGETS)} target)")
    print(f"  backbone : {args.backbone}   stride: {args.stride_train}   clip_grad: {args.clip_grad}")
    print(f"  du lieu  : {args.data_path}")
    print(f"  ckpt     : {args.checkpoints}")
    print(f"  results  : {args.results}")
    print(f"  ngan sach: {args.budget_min or 'khong gioi han'} phut")
    # Canh bao dung luong: do thuc te tu smoke test (mo hinh 116M tham so, 86,7% complex64)
    n_todo = sum(1 for _, a in runs if not is_done(a, args.results))
    print(f"  dung luong du kien: ~{n_todo * 0.89:.1f} GB checkpoint ({n_todo} run x 0,89 GB)"
          f" + 2,6 GB resume.pt (1 ban, xoa sau moi run)")
    if n_todo * 0.89 + 2.6 > 14:
        print("  CANH BAO: vuot ~15 GB cua Drive free. Giam so run hoac don Drive truoc.")
    print("=" * 78)
    for label, a in runs:
        mark = "DA XONG" if is_done(a, args.results) else "cho"
        print(f"  [{label}] {mark:8} {setting_name(a)}")
    print("=" * 78)
    if args.list:
        return 0

    done, skipped, failed = [], [], []
    for i, (label, a) in enumerate(runs, 1):
        name = setting_name(a)

        if is_done(a, args.results):
            print(f"\n[{i}/{len(runs)}] BO QUA (da co metrics.npy): {name}")
            skipped.append(name)
            continue

        if args.budget_min and (time.time() - t0) / 60 > args.budget_min:
            print(f"\n[{i}/{len(runs)}] HET NGAN SACH ({args.budget_min} phut). "
                  f"Dung sach. Chay lai cell nay de tiep tuc.")
            break

        cmd = [sys.executable, RUN_PY,
               '--variant', a['variant'], '--backbone', a['backbone'],
               '--ext', a['ext'], '--target', a['target'],
               '--data_path', args.data_path,
               '--stride_train', str(a['stride_train']),
               '--learning_rate', str(a['learning_rate']),
               '--train_epochs', str(args.train_epochs),
               '--clip_grad', str(args.clip_grad),
               '--checkpoints', args.checkpoints + os.sep,
               '--resume', '1']
        if a['tag']:
            cmd += ['--tag', a['tag']]

        log_path = os.path.join(args.logs, name + '.log')
        print(f"\n[{i}/{len(runs)}] CHAY [{label}] {name}")
        print(f"          log -> {log_path}")
        t1 = time.time()

        # tee: vua hien tren notebook vua ghi Drive. Append de resume khong mat log cu.
        with open(log_path, 'a', buffering=1) as fh:
            fh.write(f"\n{'=' * 78}\n# bat dau {time.strftime('%Y-%m-%d %H:%M:%S')}\n"
                     f"# {' '.join(cmd)}\n{'=' * 78}\n")
            proc = subprocess.Popen(cmd, cwd=REPRO, stdout=subprocess.PIPE,
                                    stderr=subprocess.STDOUT, text=True, bufsize=1)
            for line in proc.stdout:
                sys.stdout.write(line)
                fh.write(line)
            rc = proc.wait()

        mins = (time.time() - t1) / 60
        if rc == 0 and is_done(a, args.results):
            print(f"          XONG sau {mins:.1f} phut")
            done.append(name)
            # resume.pt CHI can trong luc run dang chay. Giu lai se pha Drive:
            # mo hinh 116M tham so, 86,7% complex64 -> checkpoint.pth ~888 MB,
            # resume.pt ~2,58 GB (model + 2 moment cua Adam). 6 run x 3,5 GB = 21 GB,
            # vuot 15 GB cua Drive free. Xoa sau khi run xong -> chi 1 ban ton tai.
            if not args.keep_resume:
                rp = os.path.join(args.checkpoints, name, 'resume.pt')
                if os.path.exists(rp):
                    gb = os.path.getsize(rp) / 1e9
                    os.remove(rp)
                    print(f"          da xoa resume.pt ({gb:.2f} GB) -- run da xong")
        else:
            print(f"          THAT BAI (rc={rc}) sau {mins:.1f} phut -- xem {log_path}")
            failed.append((name, rc))

    print("\n" + "=" * 78)
    print(f"  xong moi : {len(done)}")
    print(f"  bo qua   : {len(skipped)}")
    print(f"  that bai : {len(failed)}")
    for n, rc in failed:
        print(f"      rc={rc}  {n}")
    remaining = [setting_name(a) for _, a in runs if not is_done(a, args.results)]
    print(f"  con lai  : {len(remaining)}")
    for n in remaining:
        print(f"      {n}")
    print("=" * 78)
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
