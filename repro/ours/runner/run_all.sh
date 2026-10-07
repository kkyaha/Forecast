#!/bin/zsh
# Tái hiện + kiểm chứng thực nghiệm Weather của ConEm, target T (degC).
#
# Cách dùng:
#   ./run_all.sh raw     chạy trên weather_all.csv       (y nguyên dữ liệu tác giả, giữ -9999)
#   ./run_all.sh clean   chạy trên weather_all_clean.csv (đã nội suy giá trị khuyết)
#   ./run_all.sh         = clean
#
# Ba cấu hình:
#   A  conem    / ext=paper    -> tái hiện số liệu chủ đạo (bài báo: MSE 0,0343)
#   C  backbone / không ext    -> baseline bài báo đối chiếu (bài báo: MSE 1,5919)
#   B  conem    / ext=noninfo  -> KHỬ RÒ RỈ: cùng kiến trúc, nhưng ngoại sinh là
#                                 wv/wd/rain (R2 < 0,1 với target, không xác định target)
#
# Nếu A << C mà B ~ C thì mức cải thiện đến từ việc ĐƯỢC BIẾT TRƯỚC biến quyết định
# target, không từ kiến trúc ConEm.
#
# LƯU Ý về -9999: chỉ các biến thể +ConEm tiêu thụ yếu tố ngoại sinh, nên chỉ CHÚNG bị
# -9999 làm hỏng (loader fit StandardScaler trên cả -9999 -> std sai 42-177x -> phân kỳ).
# Backbone dùng data_loader_without_ext.py với seq_x_mark=None nên không bị ảnh hưởng.
# Vì vậy so sánh công bằng phải dùng bản clean.
#
# stride_train=12: loader gốc dùng stride 1 (83.630 mẫu = 5.226 step/epoch, ~3,2 giờ/epoch
# trên M5), cửa sổ liền nhau trùng 95/96. Stride 12 cho 6.969 mẫu = 435 step/epoch.
# Tập TEST luôn giữ stride 1 để định nghĩa metric không đổi.
#
# GIỚI HẠN: ngân sách gradient step ở đây nhỏ hơn tác giả ~12x mỗi epoch. Vì vậy các con số
# tuyệt đối KHÔNG so sánh trực tiếp được với bài báo. Giá trị khoa học nằm ở so sánh A/B/C
# dưới CÙNG ngân sách.

set -u
cd "$(dirname "$0")/../.."          # -> repro/
PY="${PY:-$(cd ../.. 2>/dev/null; pwd)/../.venv/bin/python}"
[ -x "$PY" ] || PY="/Users/ngkky/Data Science/.venv/bin/python"

MODE="${1:-clean}"
case "$MODE" in
  raw)   DATA="weather_all.csv" ;;        # y nguyên dữ liệu tác giả
  clean) DATA="weather_all_clean.csv" ;;
  *)     echo "Dùng: $0 [raw|clean]"; exit 2 ;;
esac

# RIN=1 (mặc định của tác giả) + vá A6fix: khôi phục dạng (sum+1) mà chính tác giả đã viết
# rồi comment. RIN=0 không dùng được vì bỏ mất instance norm: đo được epoch 1 train 1,29
# nhưng val 15,40 do dịch chuyển mức theo mùa. clip_grad=1.0 là bổ sung của bản tái hiện.
TARGET="T (degC)"
COMMON=(--target "$TARGET" --data_path "$DATA" --RIN 1 --clip_grad 1.0 --stride_train 12
        --train_epochs 4 --patience 2 --batch_size 16)

mkdir -p logs results
echo "########## CHẾ ĐỘ: $MODE  (data_path=$DATA) ##########"

run() {
  local name="${1}_${MODE}"; shift
  echo "===== [$(date '+%H:%M:%S')] BẮT ĐẦU $name ====="
  "$PY" -u -W ignore ours/runner/run.py "${COMMON[@]}" --tag "_$MODE" "$@" > "logs/$name.log" 2>&1
  local rc=$?
  if [[ $rc -eq 0 ]]; then
    echo "===== [$(date '+%H:%M:%S')] XONG $name ====="
    grep -E "^mse:" "logs/$name.log" | tail -1
  else
    echo "===== [$(date '+%H:%M:%S')] THẤT BẠI $name (rc=$rc) ====="
    tail -5 "logs/$name.log"
  fi
}

run A_conem_paper    --variant conem    --backbone fedformer --ext paper
run C_backbone       --variant backbone --backbone fedformer
run B_conem_noninfo  --variant conem    --backbone fedformer --ext noninfo

echo
echo "===== TỔNG HỢP ($MODE) ====="
for n in A_conem_paper C_backbone B_conem_noninfo; do
  printf "%-26s " "${n}_${MODE}"
  grep -E "^mse:" "logs/${n}_${MODE}.log" 2>/dev/null | tail -1 || echo "(chưa có kết quả)"
done
