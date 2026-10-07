#!/usr/bin/env python3
"""
Entry point tái hiện thực nghiệm ConEm (Knowledge-Based Systems 329 (2025) 114312).

PHẦN NÀY LÀ CỦA CHÚNG TA, không phải của tác giả.

Repo tác giả (github.com/anonymous7594/conem) KHÔNG có entry point nào — 0 file có
__main__ hay argparse, README ghi "Code will be released soon". File này dựng lại 68
tham số mà code tác giả đòi, bằng cách đối chiếu:
  - chữ ký Dataset_Train  (model/data/data_loader.py)
  - chữ ký Model          (model/model/model_*.py)
  - các args mà Exp_Main  (model/exp/exp_main_*.py) truy cập
  - siêu tham số công bố trong bài báo, Mục 4.2

File này KHÔNG chứa một dòng mạng neural nào: không nn.Module, không nn.Linear,
không forward(). Nó dựng args rồi gọi lớp Exp_Main CỦA TÁC GIẢ.

Giá trị siêu tham số lấy từ bài báo (Mục 4.2), thực nghiệm Weather:
  learning rate = 1e-5, dropout = 1e-5, số lớp encoder/decoder = 1 hoặc 2,
  FEDformer dùng bản Wavelet, chia train/valid = 0.8/0.2, I=96 -> O=96.

Ví dụ:
  # FEDformer + ConEm, target T (degC)
  python ours/runner/run.py --variant conem --backbone fedformer --target "T (degC)"
  # FEDformer gốc (không ConEm, không yếu tố ngoại sinh — đúng như tác giả chạy baseline)
  python ours/runner/run.py --variant backbone --backbone fedformer --target "T (degC)"
  # Ablation
  python ours/runner/run.py --variant ablation --ablate LoEm --backbone fedformer --target "T (degC)"
"""
import argparse
import os
import sys
import random
import numpy as np
import torch

# Cấu trúc: ours/runner/run.py -> repro/
#   upstream/conem  code tác giả nguyên trạng (không dùng trực tiếp)
#   build/conem     SINH RA bởi ours/build.py = upstream + patches  <- dùng cái này
HERE = os.path.dirname(os.path.abspath(__file__))
REPRO = os.path.dirname(os.path.dirname(HERE))
BUILD = os.path.join(REPRO, "build", "conem")
if not os.path.isdir(BUILD):
    sys.exit(f"Thiếu {BUILD}\n  Chạy trước:  python ours/build.py")
sys.path.insert(0, BUILD)
os.chdir(REPRO)                      # để root_path='data/weather/' hoạt động

# ---------------------------------------------------------------- dataset presets
# 6 chuỗi dùng làm YẾU TỐ NGOẠI SINH, theo Mục 4.1.4: p, sh, Tpot, H2OC, rho bị loại
# khỏi danh sách target vì có giá trị khuyết (-9999) giai đoạn 2022-01-21..01-23;
# SWDR bị loại vì ~50% giá trị bằng 0.
WEATHER_EXT = ['p (mbar)', 'sh (g/kg)', 'Tpot (K)', 'H2OC (mmol/mol)',
               'rho (g/m**3)', 'SWDR (W/m²)']
# 9 phép đo được dùng làm TARGET, theo Mục 4.2 và Bảng 11-13.
WEATHER_TARGETS = ['T (degC)', 'rh (%)', 'Tdew (degC)', 'VPmax (mbar)', 'VPact (mbar)',
                   'VPdef (mbar)', 'wv (m/s)', 'wd (deg)', 'rain (mm)']
# Bộ ngoại sinh "không mang thông tin": leakage_test.py cho thấy OLS từ 6 biến của bài báo
# giải thích R2 < 0.1 cho wv/wd/rain, nên 3 chuỗi này không xác định được các target khác.
WEATHER_NONINFO = ['wv (m/s)', 'wd (deg)', 'rain (mm)']

# backbone -> (file model ConEm, file model gốc)
BACKBONES = {
    'fedformer':  ('model_fed_based',  'model_fedformer'),
    'autoformer': ('model_auto_based', 'model_autoformer'),
    'informer':   ('model_info_based', 'model_informer'),
}
ABLATIONS = {'LoEm': 'model_without_LoEm',
             'FuEm': 'model_without_FuEm',
             'PaEm': 'model_without_PaEm'}


def build_args():
    p = argparse.ArgumentParser(description="Tái hiện ConEm")

    # --- chọn thực nghiệm ---
    p.add_argument('--variant', choices=['conem', 'backbone', 'ablation'], default='conem',
                   help="conem = +ConEm | backbone = mô hình gốc (không yếu tố ngoại sinh) | ablation = bỏ 1 thành phần")
    p.add_argument('--backbone', choices=list(BACKBONES), default='fedformer')
    p.add_argument('--ablate', choices=list(ABLATIONS), default=None)
    p.add_argument('--target', default='T (degC)')
    p.add_argument('--tag', default='', help="hậu tố tên run")
    # Chọn bộ yếu tố ngoại sinh. 'paper' = đúng 6 chuỗi bài báo dùng (p, sh, Tpot, H2OC,
    # rho, SWDR) — nhưng Tpot/p xác định T bằng hằng đẳng thức nhiệt động, xem
    # leakage_test.py. 'noninfo' = 3 chuỗi KHÔNG xác định target (wv, wd, rain), dùng để
    # tách xem mức cải thiện đến từ kiến trúc ConEm hay từ việc được biết trước biến
    # quyết định target.
    p.add_argument('--ext', choices=['paper', 'noninfo'], default='paper')

    # --- dữ liệu ---
    p.add_argument('--root_path', default='data/weather/')
    p.add_argument('--data_path', default='weather_all.csv')
    p.add_argument('--time_col', default='Date Time')
    p.add_argument('--freq', default='h',
                   help="'h' -> 5 đặc trưng thời gian (hour,dow,dom,doy,moy) đúng như Mục 4.2")
    # Bài báo: train 2022-01-01..2023-01-01, test 2023-01-01..2023-04-02, train/valid 0.8/0.2
    p.add_argument('--if_test', type=int, default=1)
    p.add_argument('--if_split_by_date', type=int, default=1)
    p.add_argument('--cut_off_time_test', default='2023-01-01')
    p.add_argument('--train_perc_split_by_date', type=float, default=0.8)
    p.add_argument('--train_perc', type=float, default=0.7)   # không dùng khi split-by-date
    p.add_argument('--test_perc', type=float, default=0.15)   # không dùng khi split-by-date
    # LƯU Ý: mặc định của tác giả là 0.8 — bộ lọc "loại nhóm nếu tỉ lệ (target>0) < cut_off".
    # Với dữ liệu khí tượng, 0.8 sẽ loại hết nhóm ở rain/VPdef (xem README_REPRO.md).
    p.add_argument('--cut_off', type=float, default=0.0)
    p.add_argument('--scale', type=int, default=1)
    # Loader gốc cắt cửa sổ trượt stride=1 -> 83.630 mẫu train, các cửa sổ liền nhau
    # trùng 95/96 giá trị, ~5 giờ/epoch trên M5. stride>1 chỉ áp cho train/val;
    # tập test luôn stride=1 để chỉ số so sánh được với bài báo.
    p.add_argument('--stride_train', type=int, default=1,
                   help="bước nhảy cửa sổ cho train/val (1 = y như tác giả)")
    p.add_argument('--RIN', type=int, default=1,
                   help="RIN=1: target KHÔNG bị StandardScaler -> MSE/MAE tính trên thang gốc")

    # --- độ dài chuỗi (Mục 4.2: Weather I=96, O=96) ---
    p.add_argument('--seq_len', type=int, default=96)
    p.add_argument('--label_len', type=int, default=48)
    p.add_argument('--pred_len', type=int, default=96)

    # --- siêu tham số mô hình ---
    p.add_argument('--d_model', type=int, default=512)
    p.add_argument('--n_heads', type=int, default=8)
    p.add_argument('--d_ff', type=int, default=1024)
    p.add_argument('--e_layers', type=int, default=2)
    p.add_argument('--d_layers', type=int, default=1)
    p.add_argument('--moving_avg', type=int, default=25)
    p.add_argument('--dropout', type=float, default=1e-5)      # Mục 4.2 (weather)
    p.add_argument('--activation', default='gelu')
    p.add_argument('--embed', default='timeF')
    p.add_argument('--output_attention', action='store_true')
    p.add_argument('--enc_in', type=int, default=1)
    p.add_argument('--dec_in', type=int, default=1)
    p.add_argument('--c_out', type=int, default=1)
    p.add_argument('--features', default='S')
    # FEDformer bản Wavelet (Mục 4.2)
    p.add_argument('--modes', type=int, default=64)
    p.add_argument('--L', type=int, default=3)
    p.add_argument('--base', default='legendre')
    p.add_argument('--cross_activation', default='tanh')
    # token ConEm (mặc định trong chữ ký Model của tác giả)
    p.add_argument('--static_d_token', type=int, default=30)
    p.add_argument('--contextual_encoder_d_token', type=int, default=30)
    p.add_argument('--contextual_decoder_d_token', type=int, default=30)
    p.add_argument('--init_weights', type=int, default=1)
    # TimesNet (không dùng cho FEDformer nhưng Exp_Main vẫn đọc)
    p.add_argument('--top_k', type=int, default=5)
    p.add_argument('--num_kernels', type=int, default=6)
    # quantile loss — bài báo dùng MSE nên tắt
    p.add_argument('--is_quantile', type=int, default=0)
    p.add_argument('--quantiles', type=float, nargs='+', default=[0.1, 0.3, 0.5, 0.7, 0.9])

    # --- huấn luyện ---
    p.add_argument('--batch_size', type=int, default=32)
    p.add_argument('--learning_rate', type=float, default=1e-5)  # Mục 4.2 (weather)
    p.add_argument('--train_epochs', type=int, default=10)
    p.add_argument('--patience', type=int, default=3)
    p.add_argument('--lradj', default='type1')
    p.add_argument('--num_workers', type=int, default=0)
    p.add_argument('--use_amp', action='store_true')
    # Vòng train của tác giả đánh giá lại toàn bộ tập test mỗi epoch, nhưng chỉ để IN RA
    # (early stopping dùng vali_loss). Ở stride 1 việc này chiếm ~75% thời gian mỗi epoch.
    p.add_argument('--eval_test_each_epoch', type=int, default=0)
    # THÊM BỞI BẢN TÁI HIỆN — repo tác giả KHÔNG có gradient clipping ở bất kỳ đâu.
    # Cấu hình đã công bố (lr=1e-5, Adam, không clipping) nổ gradient sau ~250 step.
    # 0 = tắt = y như bản gốc.
    p.add_argument('--clip_grad', type=float, default=0.0)
    p.add_argument('--checkpoints', default='./checkpoints/')
    # THÊM BỞI BẢN TÁI HIỆN (vá C5) — vòng train của tác giả không resume được:
    # EarlyStopping chỉ lưu state_dict, thiếu optimizer/epoch/counter/RNG.
    # Cần cho Colab free, nơi session bị cắt bất kỳ lúc nào. 0 = tắt = y như bản gốc.
    p.add_argument('--resume', type=int, default=0)
    p.add_argument('--seed', type=int, default=2021)

    # --- thiết bị ---
    p.add_argument('--use_gpu', type=int, default=0)   # không có CUDA trên macOS
    p.add_argument('--use_mps', type=int, default=1)
    p.add_argument('--gpu', type=int, default=0)
    p.add_argument('--use_multi_gpu', action='store_true')
    p.add_argument('--devices', default='0')

    # --- không dùng cho Weather nhưng data_factory vẫn đọc ---
    p.add_argument('--scaler_path', default=None)
    p.add_argument('--cat_encode_path', default=None)
    p.add_argument('--if_holiday_data', type=int, default=0)
    p.add_argument('--if_promo', type=int, default=0)

    a = p.parse_args()

    # --- suy ra cấu hình phụ thuộc ---
    a.device_ids = [int(x) for x in a.devices.split(',')]
    a.group_filter = ['location']            # 2 trạm Saaleaue / Beutenberg
    a.static_cat_vab = ['location']          # Mục 4.1.4: "Only one static feature (Station Location)"
    a.cat_vab = []                           # Weather không có biến phân loại biến thiên theo thời gian
    # num_vab PHẢI chứa target; loader tự đẩy target về cuối danh sách.
    ext_pool = WEATHER_EXT if a.ext == 'paper' else WEATHER_NONINFO
    a.num_vab = [c for c in ext_pool if c != a.target] + [a.target]

    # Biến thể backbone KHÔNG dùng yếu tố ngoại sinh:
    # data_loader_without_ext.py đặt seq_x_mark=None, và model_fedformer.forward đặt
    #     x_mark_enc = x_temporal_enc        (chỉ 5 đặc trưng thời gian)
    # vì hai dòng ghép ngoại sinh đã bị comment. Nhưng cùng file lại khai báo
    #     DataEmbedding_wo_pos(..., num_cat=len(cat_vab), num_num=len(num_vab)-1)
    # -> TimeFeatureEmbedding kỳ vọng freq_map['h'] + 0 + 6 = 11 đầu vào
    # -> RuntimeError: shapes cannot be multiplied (96x5 and 11x512).
    # Đặt num_vab = [target] cho biến thể này làm len(num_vab)-1 = 0 nên d_inp = 5, khớp
    # đúng những gì model thực sự nạp. Không cần sửa code tác giả, và không mất gì vì
    # loader without_ext vốn đã bỏ các cột ngoại sinh.
    if a.variant == 'backbone':
        a.num_vab = [a.target]

    if a.variant == 'conem':
        a.model_file, a.exp_module = BACKBONES[a.backbone][0], 'exp_main_compare'
    elif a.variant == 'backbone':
        a.model_file, a.exp_module = BACKBONES[a.backbone][1], 'exp_main_without_ext'
    else:
        if a.ablate is None:
            p.error("--variant ablation cần --ablate {LoEm,FuEm,PaEm}")
        a.model_file, a.exp_module = ABLATIONS[a.ablate], 'exp_main_ablation'

    for k in ('if_test', 'if_split_by_date', 'scale', 'RIN', 'is_quantile',
              'use_gpu', 'use_mps', 'if_holiday_data', 'if_promo'):
        setattr(a, k, bool(getattr(a, k)))
    return a


def main():
    args = build_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    short = {'T (degC)': 'T', 'rh (%)': 'rh', 'Tdew (degC)': 'Tdew', 'VPmax (mbar)': 'VPmax',
             'VPact (mbar)': 'VPact', 'VPdef (mbar)': 'VPdef', 'wv (m/s)': 'wv',
             'wd (deg)': 'wd', 'rain (mm)': 'rain'}.get(args.target, args.target)
    setting = (f"weather_{short}_{args.variant}"
               f"{'-' + args.ablate if args.ablate else ''}_{args.backbone}"
               f"_ext-{args.ext}_I{args.seq_len}_O{args.pred_len}"
               f"_el{args.e_layers}_dl{args.d_layers}"
               f"_lr{args.learning_rate}_st{args.stride_train}{args.tag}")

    from importlib import import_module
    Exp_Main = import_module(f"model.exp.{args.exp_module}").Exp_Main

    print("=" * 78)
    print(f"  setting     : {setting}")
    print(f"  variant     : {args.variant}  | backbone: {args.backbone}  | model: {args.model_file}")
    print(f"  exp module  : {args.exp_module}")
    print(f"  target      : {args.target}")
    print(f"  ngoại sinh  : {[c for c in args.num_vab if c != args.target]}")
    print(f"  static      : {args.static_cat_vab}")
    print(f"  I -> O      : {args.seq_len} -> {args.pred_len} (label_len {args.label_len})")
    print(f"  lr / dropout: {args.learning_rate} / {args.dropout}")
    print("=" * 78)

    exp = Exp_Main(args)
    exp.train(setting)
    exp.test(setting, test=1)


if __name__ == "__main__":
    main()
