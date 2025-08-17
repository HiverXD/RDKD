# src/data/soft_targets.py
import os, torch

class TeacherLogitsBank:
    """
    FP16 teacher logits cache on CPU, mmap-like lazy load.
    - path: .pt file
    - meta: {"n": num_samples, "c": num_classes}
    """
    def __init__(self, path, num_samples, num_classes, device="cuda"):
        self.path, self.device = path, device
        self.meta = {"n": num_samples, "c": num_classes}
        self.bank = None  # lazy load

    def exists(self):
        return os.path.exists(self.path)

    @torch.no_grad()
    def build(self, teacher_model, loader):
        teacher_model.eval()
        n, c = self.meta["n"], self.meta["c"]
        bank = torch.empty(n, c, dtype=torch.float16)
        for x, _, idx in loader:
            x = x.to(self.device, non_blocking=True)
            logits = teacher_model(x).to(torch.float16).cpu()
            bank[idx] = logits
        torch.save({"logits": bank, "meta": self.meta}, self.path)
        self.bank = bank  # keep in RAM for this run

    def _ensure_loaded(self):
        if self.bank is None:
            payload = torch.load(self.path, map_location="cpu")
            self.bank = payload["logits"]  # (N, C) fp16 on CPU

    def get(self, idx_tensor):
        self._ensure_loaded()
        out = self.bank[idx_tensor.cpu()].to(dtype=torch.float32, device=self.device)
        return out
