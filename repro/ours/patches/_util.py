"""Tiện ích chung cho các bản vá."""
import re


def sub_once(text, old, new):
    """Thay chuỗi, đúng 1 lần. Trả về (text_mới, có_đổi_không)."""
    if old not in text:
        return text, False
    return text.replace(old, new, 1), True


def uncomment(text, pattern):
    """Bỏ dấu # ở đầu dòng khớp pattern (regex, không gồm dấu #)."""
    if re.search(rf'^\s*{pattern}\s*$', text, re.M):
        return text, False                      # đã không bị comment
    new = re.sub(rf'^(\s*)#\s*({pattern})\s*$', r'\1\2', text, count=1, flags=re.M)
    return new, new != text


def edit(root, rel, fn):
    """Đọc file, áp fn(text) -> (text, notes), ghi lại nếu đổi."""
    p = root / rel
    s = p.read_text()
    s2, notes = fn(s)
    if s2 != s:
        p.write_text(s2)
    return notes
