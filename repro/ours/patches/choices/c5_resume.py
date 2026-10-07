# -*- coding: utf-8 -*-
"""C5 — THÊM resume cho vòng train. Không có trong repo tác giả."""
ID = "C5"
TITLE = "exp_main_*.py: THÊM resume giữa các epoch (không có trong repo tác giả)"
CATEGORY = "choice"
FILES = ["model/exp/exp_main_compare.py",
         "model/exp/exp_main_without_ext.py",
         "model/exp/exp_main_ablation.py"]

WHY = """
ĐÂY LÀ THỨ CHÚNG TA THÊM VÀO, KHÔNG PHẢI CỦA TÁC GIẢ.
`grep -rn resume` trên bản gốc: 0 kết quả.

Lý do: chạy trên Colab free. Session bị cắt bất kỳ lúc nào (idle ~90 phút, usage
limit động, có lúc không được cấp GPU). Không có resume thì mỗi lần cắt là mất cả run.

Vòng train của tác giả KHÔNG resume được, vì EarlyStopping.save_checkpoint chỉ lưu
    torch.save(model.state_dict(), path + '/checkpoint.pth')
tức CHỈ trọng số, và chỉ lưu khi vali_loss giảm. Thiếu hết:
  - optimizer state (Adam moment m, v) -> tiếp tục từ weights không có m,v là một
    quỹ đạo huấn luyện KHÁC, không phải tiếp tục
  - số epoch đã xong    -> không biết chạy tiếp từ đâu
  - EarlyStopping state -> counter, best_score, val_loss_min reset về 0/None/Inf
  - RNG state           -> thứ tự batch và dropout mask lệch

Nên bản vá này lưu thêm `resume.pt` BÊN CẠNH `checkpoint.pth` của tác giả, sau MỖI
epoch (không chỉ khi vali_loss giảm). `checkpoint.pth` giữ nguyên ý nghĩa gốc là
"trọng số tốt nhất"; `resume.pt` là trạng thái huấn luyện hiện hành.

MẶC ĐỊNH TẮT (args.resume = 0). Khi tắt, code chạy y như bản gốc: không đọc, không
ghi, không một lệnh nào thêm trong vòng lặp nóng. Bật bằng --resume 1.

GHI CHÚ torch >= 2.6: torch.load mặc định weights_only=True, sẽ từ chối optimizer
state và numpy RNG state. Bản dưới truyền weights_only=False tường minh.
"""

EVIDENCE = """
Bản gốc, model/utils/tools.py:67-71 — toàn bộ phần lưu checkpoint của tác giả:
    def save_checkpoint(self, val_loss, model, path):
        if self.verbose: print(f'Validation loss decreased ...')
        torch.save(model.state_dict(), path + '/' + 'checkpoint.pth')
        self.val_loss_min = val_loss
-> không optimizer, không epoch, không counter. Không đủ để resume.

EarlyStopping (tools.py:42-50) có 4 trường trạng thái cần giữ:
    counter, best_score, val_loss_min, early_stop

Anchor áp vá, giống hệt trên cả 3 file exp:
    dòng 124  for epoch in range(self.args.train_epochs):
    dòng 238  early_stopping(vali_loss, self.model, path)
"""

OLD_LOOP = ("        for epoch in range(self.args.train_epochs):\n"
            "            iter_count = 0\n"
            "            train_loss = []")

NEW_LOOP = """        # [vá C5 — THÊM BỞI BẢN TÁI HIỆN, không có trong repo tác giả]
        # Resume cho session bị cắt giữa đường (Colab free). Mặc định TẮT.
        _start_epoch = 0
        _res_path = path + '/' + 'resume.pt'
        if getattr(self.args, 'resume', 0) and os.path.exists(_res_path):
            # weights_only=False: cần optimizer state + numpy RNG, torch>=2.6 mặc định True
            _st = torch.load(_res_path, map_location=self.device, weights_only=False)
            self.model.load_state_dict(_st['model'])
            model_optim.load_state_dict(_st['optim'])
            early_stopping.counter = _st['es_counter']
            early_stopping.best_score = _st['es_best_score']
            early_stopping.val_loss_min = _st['es_val_loss_min']
            early_stopping.early_stop = _st['es_early_stop']
            _start_epoch = _st['next_epoch']
            try:
                torch.set_rng_state(_st['rng_torch'].cpu().to(torch.uint8))
                np.random.set_state(_st['rng_numpy'])
            except Exception as _e:
                print('[C5] khong khoi phuc duoc RNG:', _e)
            if early_stopping.early_stop:
                print('[C5] run nay da early-stop truoc do, bo qua train')
                _start_epoch = self.args.train_epochs
            else:
                print('[C5] resume tu epoch %d | best vali %.7f | es_counter %d'
                      % (_start_epoch + 1, early_stopping.val_loss_min,
                         early_stopping.counter))

        for epoch in range(_start_epoch, self.args.train_epochs):
            iter_count = 0
            train_loss = []"""

OLD_SAVE = ("            early_stopping(vali_loss, self.model, path)\n"
            "            if early_stopping.early_stop:")

NEW_SAVE = """            early_stopping(vali_loss, self.model, path)
            # [vá C5] lưu trạng thái resume sau MỖI epoch, không chỉ khi vali giảm.
            # Ghi ra file tạm rồi os.replace -> không để lại file hỏng nếu bị cắt giữa lúc ghi.
            if getattr(self.args, 'resume', 0):
                _tmp = _res_path + '.tmp'
                torch.save({'next_epoch': epoch + 1,
                            'model': self.model.state_dict(),
                            'optim': model_optim.state_dict(),
                            'es_counter': early_stopping.counter,
                            'es_best_score': early_stopping.best_score,
                            'es_val_loss_min': early_stopping.val_loss_min,
                            'es_early_stop': early_stopping.early_stop,
                            'rng_torch': torch.get_rng_state(),
                            'rng_numpy': np.random.get_state()}, _tmp)
                os.replace(_tmp, _res_path)
            if early_stopping.early_stop:"""


def apply(root):
    notes = []
    for rel in FILES:
        p = root / rel
        s = p.read_text()
        if "vá C5" in s:
            continue
        s2 = s.replace(OLD_LOOP, NEW_LOOP, 1)
        if s2 == s:
            continue
        s3 = s2.replace(OLD_SAVE, NEW_SAVE, 1)
        if s3 == s2:
            continue
        p.write_text(s3)
        notes.append(f"{p.name}: thêm resume giữa các epoch (mặc định TẮT)")
    return notes
