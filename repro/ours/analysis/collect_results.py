#!/usr/bin/env python3
"""
Thu thập kết quả các run và đối chiếu với số liệu bài báo.

Đọc dòng `mse:... mae:...` cuối cùng trong logs/*.log (do exp.test() in ra), nên không
phụ thuộc vào tên thư mục results/.

Lưu ý thứ tự metric: utils/metrics.py::metric() trả về (mae, mse, rmse, ...) và exp_main
lưu nguyên mảng đó vào metrics.npy. Đây cũng là nguồn gốc việc Bảng 10 của bài báo dán
nhãn MSE/MAE bị đảo. Ở đây ta đọc từ log nên nhãn đã rõ ràng.

Cách dùng:  python collect_results.py
"""
import re
from pathlib import Path

HERE = Path(__file__).resolve().parent.parent.parent   # -> repro/
LOGS = HERE / 'logs'

# Bài báo, Bảng 11 (nhãn MAE, MSE — bản đúng), target T (degC), I=96 -> O=96
PAPER = {
    'conem':    dict(mse=0.0343, mae=0.1341, label='FEDformer + ConEm'),
    'backbone': dict(mse=1.5919, mae=0.9804, label='FEDformer gốc'),
}

RUNS = [
    ('A_conem_paper_raw',     'A-raw',   'conem',    'paper',   'raw',
     'ConEm + 6 biến bài báo, dữ liệu Y NGUYÊN (giữ -9999)'),
    ('A_conem_paper_clean',   'A',       'conem',    'paper',   'clean',
     'ConEm + 6 biến bài báo, dữ liệu đã làm sạch'),
    ('C_backbone_clean',      'C',       'backbone', '—',       'clean',
     'FEDformer gốc, KHÔNG yếu tố ngoại sinh'),
    ('B_conem_noninfo_clean', 'B',       'conem',    'noninfo', 'clean',
     'ConEm + wv/wd/rain (không xác định target)'),
]


def read(stem):
    p = LOGS / f'{stem}.log'
    if not p.exists():
        return None
    s = p.read_text(errors='ignore')
    m = re.findall(r'^mse:([\d.eE+-]+), rmse:[\d.eE+-]+, mae:([\d.eE+-]+)', s, re.M)
    if not m:
        ep = re.findall(r'Epoch: (\d+), Steps', s)
        return dict(status=f'đang chạy ({len(ep)} epoch)')
    diverged = bool(re.search(r'Train Loss: \d{6,}|Train Loss: [\d.]+e\+', s))
    return dict(mse=float(m[-1][0]), mae=float(m[-1][1]), status='xong',
                diverged=diverged)


def main():
    rows = [(lbl, var, ext, mode, desc, read(stem)) for stem, lbl, var, ext, mode, desc in RUNS]

    print('=' * 100)
    print('KẾT QUẢ TÁI HIỆN — Weather, target T (degC), I=96 -> O=96')
    print('MSE/MAE trên thang GỐC (°C) vì RIN=True nên target không bị chuẩn hoá')
    print('=' * 100)
    print(f"{'run':7}{'dữ liệu':9}{'ngoại sinh':12}{'MSE':>11}{'MAE':>9}"
          f"{'MSE bài báo':>13}{'MAE bài báo':>12}")
    print('-' * 100)
    for lbl, var, ext, mode, desc, r in rows:
        ref = PAPER.get(var) if ext in ('paper', '—') else None
        rm = f"{ref['mse']:.4f}" if ref else '—'
        ra = f"{ref['mae']:.4f}" if ref else '—'
        if r is None:
            print(f"{lbl:7}{mode:9}{ext:12}{'(chưa chạy)':>11}{'':>9}{rm:>13}{ra:>12}")
        elif r['status'] != 'xong':
            print(f"{lbl:7}{mode:9}{ext:12}{r['status']:>11}{'':>9}{rm:>13}{ra:>12}")
        else:
            flag = '  PHÂN KỲ' if r.get('diverged') else ''
            print(f"{lbl:7}{mode:9}{ext:12}{r['mse']:>11.4f}{r['mae']:>9.4f}"
                  f"{rm:>13}{ra:>12}{flag}")
    print()
    for lbl, var, ext, mode, desc, r in rows:
        print(f"  {lbl:7} {desc}")

    done = {lbl: r for lbl, _, _, _, _, r in rows if r and r.get('status') == 'xong'}

    # ── so với bài báo ──
    if 'A' in done:
        a = done['A']
        print('\n' + '=' * 100)
        print('ĐỐI CHIẾU VỚI SỐ LIỆU BÀI BÁO')
        print('=' * 100)
        print(f"  tái hiện  MSE {a['mse']:.4f}   bài báo MSE {PAPER['conem']['mse']:.4f}"
              f"   -> lệch {a['mse'] / PAPER['conem']['mse']:.0f}x")
        if 'A-raw' in done:
            print(f"  (trên dữ liệu y nguyên còn -9999: MSE {done['A-raw']['mse']:.4f})")

    # ── kết luận khử rò rỉ ──
    if all(k in done for k in ('A', 'C', 'B')):
        A, C, B = done['A'], done['C'], done['B']
        print('\n' + '=' * 100)
        print('KIỂM CHỨNG KHỬ RÒ RỈ')
        print('=' * 100)
        gA = (C['mse'] - A['mse']) / C['mse'] * 100
        gB = (C['mse'] - B['mse']) / C['mse'] * 100
        print(f"  C  baseline, không ngoại sinh          MSE {C['mse']:10.4f}")
        print(f"  A  ConEm + 6 biến xác định target      MSE {A['mse']:10.4f}   ({gA:+.1f}% so với C)")
        print(f"  B  ConEm + wv/wd/rain (không thông tin) MSE {B['mse']:10.4f}   ({gB:+.1f}% so với C)")
        print()
        if gA > 2 * max(gB, 1e-9):
            print("  -> A cải thiện mạnh, B thì không: mức cải thiện trên Weather đến từ việc")
            print("     ĐƯỢC BIẾT TRƯỚC biến quyết định target, KHÔNG từ kiến trúc ConEm.")
        elif gB > 0.5 * gA:
            print("  -> B cũng cải thiện đáng kể: kiến trúc có đóng góp thật ngoài phần")
            print("     thông tin của biến ngoại sinh. Kết luận rò rỉ cần xét lại.")
        else:
            print("  -> kết quả trung gian; cần thêm run (seed khác, target khác) để kết luận.")
    else:
        missing = [k for k in ('A', 'C', 'B') if k not in done]
        print(f"\n(chưa đủ run để kết luận khử rò rỉ — còn thiếu: {', '.join(missing)})")


if __name__ == '__main__':
    main()
