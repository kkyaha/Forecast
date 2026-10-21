# Khung báo cáo tiểu luận (LaTeX)

Dùng lớp tài liệu `uetgraduation` của Lê Duy Quang
(https://github.com/leduyquang753/uet-graduation-latex, commit 86756f6, giấy phép MIT,
xem `LICENSE-uetgraduation.txt`). Lớp này trình bày theo quy định khóa luận của Trường ĐH Công nghệ.

## Dùng trên Overleaf

1. overleaf.com → New Project → Upload Project → chọn file zip này.
2. Menu (góc trên trái) → Compiler → chọn **LuaLaTeX** (không phải XeLaTeX hay pdfLaTeX).
   Main document: `main.tex`.
3. Bấm Recompile. Lần đầu mục lục có thể trống, bấm Recompile thêm một lần.

## Các file

| File | Nội dung |
|---|---|
| `main.tex` | File chính, ghép các phần theo thứ tự quy định |
| `thongtin.tex` | Họ tên, tên đề tài, giảng viên: **sửa ở đây trước** |
| `phandau/` | Tóm tắt, lời cam đoan, chữ viết tắt |
| `chuong/` | Mở đầu, 7 chương, kết luận |
| `tailieu.tex` | Tài liệu tham khảo |
| `hinh/` | Đặt ảnh, hình vẽ vào đây |
| `vi-du-cu-phap.tex` | Ví dụ công thức, bảng, hình, trích dẫn để chép cách viết |
| `uetgraduation.cls`, `sizes.clo`, `UET logo.pdf` | Lớp tài liệu, không cần sửa |

## Khác với LaTeX thông thường

Hình và bảng viết chú thích ngay sau `\begin`, không dùng `\caption`:

```latex
\begin{table}[H]{Tên bảng}
\label{bang:ten}
...
\end{table}
```

Xem đầy đủ trong `vi-du-cu-phap.tex`.

## Những chỗ đã thêm hoặc sửa so với lớp gốc

- `uetgraduation.cls`: đổi phông từ Times New Roman sang Latin Modern Roman (phông mặc định
  của LaTeX, giống khóa luận mẫu của trường). Đây là thay đổi duy nhất trong file lớp.
  Muốn dùng kiểu Times thì thay "Latin Modern Roman" bằng "TeX Gyre Termes" trong file .cls
  và trong 3 dòng `\mathfont`, `\newmathbold`, `\newmathrm` của `main.tex`.
- `main.tex`: sửa lỗi tên hàm (max, min, argmin) và chữ đậm trong công thức không hiện;
  thêm lệnh `\chuongkhongso` cho Mở đầu và Kết luận (lớp gốc không có);
  đánh số công thức theo chương; lệnh `\gichu`.

## Chỗ cần tự viết

Chữ xám nghiêng trong ngoặc vuông là chỗ trống, tạo bằng lệnh `\gichu{...}`.
Tìm `\gichu` (Ctrl+F) để biết còn chỗ nào chưa viết; viết xong thì xóa cả lệnh.

## Chỗ cần kiểm tra lại

- `thongtin.tex`: tên giảng viên.
- `tailieu.tex`: số tập và trang của bài báo trên tạp chí.
- Trang bìa ghi "Cán bộ hướng dẫn" (cố định trong lớp tài liệu) và không có dòng học phần.
- Phụ bìa và tóm tắt tiếng Anh đang tắt (chỉ bắt buộc với sinh viên CLC);
  bật trong `thongtin.tex` và `phandau/tomtat.tex`.
