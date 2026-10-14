#!/bin/bash
# Tải song song 4 kết nối (tốc độ tốt nhất đo được từ mạng này tới PhysioNet).
# curl -C - : tiếp tục nếu file đã tải dở, không tải lại từ đầu.
cd "$(dirname "$0")"
BASE="https://physionet.org/files/chbmit/1.0.0/chb01"
mkdir -p Dataset/chb01

download_one() {
  fn="$1"
  dest="Dataset/chb01/$fn"
  # kiểm tra đã tải đủ dung lượng chưa (so với server) trước khi coi là xong
  remote_size=$(curl -sI "$BASE/$fn" | grep -i content-length | tr -d '\r' | awk '{print $2}')
  local_size=$(stat -f%z "$dest" 2>/dev/null || echo 0)
  if [ -n "$remote_size" ] && [ "$local_size" = "$remote_size" ]; then
    echo "DA XONG: $fn"
    return
  fi
  curl -s -C - -o "$dest" "$BASE/$fn"
  echo "XONG: $fn"
}

export -f download_one
export BASE

cat /tmp/chb01_files.txt | xargs -P 4 -I{} bash -c 'download_one "$@"' _ {}

echo "TAT_CA_DA_XONG"
ls Dataset/chb01 | wc -l
du -sh Dataset/chb01
