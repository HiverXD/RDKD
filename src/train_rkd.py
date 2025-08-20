# src/train_rkd.py
import os, math, time, yaml, argparse
from pathlib import Path
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import amp
from torch.utils.data import DataLoader

# ---------- imports from your repo ----------
# config snapshot
from src.utils.config import save_config_used, load_config, expand_softkd_templates

# teacher banks
from src.data.soft_targets import TeacherLogitsBank
from src.data.final_embeddings import TeacherEmbeddingBank  # you already added this module

# KD loss (prefer compute_* if available)
try:
    from src.losses.compute.compute_soft_kd_loss import make_compute_loss_kd
    HAVE_COMPUTE_KD = True
except Exception:
    from src.losses.soft_kd import SoftTargetKDLoss
    HAVE_COMPUTE_KD = False

from src.losses.soft_kd import SoftTargetKDLoss

# RKD loss (use both the core and compute wrapper)
from src.losses.relational_kd import cos_gram_mse as rkd_core
from src.losses.compute.compute_relational_kd_loss import compute_rkd_loss

# dataloaders/models/optimizers reused from 02
from src.train_soft_kd import (
    build_dataloaders_single,
    build_teacher,
    build_student,
    build_optimizer_scheduler,
)

# optional Trainer util
try:
    from src.utils.trainer import Trainer  # if exists
    HAVE_TRAINER = True
except Exception:
    HAVE_TRAINER = False


# --------- small helpers ----------
def robust_unpack_batch(batch):
    """Return (x,y,idx) from (x,y,idx) or (idx,x,y) or (x,idx)."""
    if isinstance(batch, (list, tuple)) and len(batch) == 3:
        a, b, c = batch
        if torch.is_tensor(a) and a.dim() == 4:
            x, y, idx = a, b, c
        elif torch.is_tensor(b) and b.dim() == 4:
            x, y, idx = b, c, a
        else:
            if torch.is_tensor(c) and c.dim() == 4:
                x, y, idx = c, a, b
            else:
                raise RuntimeError("Cannot infer (x,y,idx) from 3-tuple batch.")
    elif isinstance(batch, (list, tuple)) and len(batch) == 2:
        a, b = batch
        if torch.is_tensor(a) and a.dim() == 4:
            x, idx = a, b; y = None
        elif torch.is_tensor(b) and b.dim() == 4:
            x, idx = b, a; y = None
        else:
            raise RuntimeError("2-tuple batch must be (x,idx) or (idx,x).")
    else:
        raise RuntimeError("Unexpected batch structure.")
    if x.dim() != 4:
        raise RuntimeError(f"Images must be 4D, got {tuple(x.shape)}")
    return x, y, idx


def make_cache_loader_from(loader: DataLoader):
    """Deterministic loader for building banks."""
    return DataLoader(
        loader.dataset,
        batch_size=loader.batch_size,
        shuffle=False,
        num_workers=loader.num_workers,
        pin_memory=getattr(loader, "pin_memory", True),
        drop_last=False,
        collate_fn=getattr(loader, "collate_fn", None),
        persistent_workers=getattr(loader, "persistent_workers", False),
    )


# (torchvision) resnet18/34 penultimate features + logits
def extract_feats_and_logits(m: nn.Module, x: torch.Tensor):
    z = m.conv1(x); z = m.bn1(z); z = m.relu(z); z = m.maxpool(z)
    z = m.layer1(z); z = m.layer2(z); z = m.layer3(z); z = m.layer4(z)
    z = F.adaptive_avg_pool2d(z, 1)
    feat = torch.flatten(z, 1)     # (B,512)
    logits = m.fc(feat)            # (B,C)
    return feat, logits


# --------- beta scheduler (epoch-based; override via config) ----------
@dataclass
class BetaScheduleCfg:
    start_epoch: int = 21
    end_epoch: int = 30
    base: float = 50.0
    stepwise: bool = True

class BetaScheduler:
    def __init__(self, cfg: BetaScheduleCfg, steps_per_epoch: int):
        self.cfg = cfg
        self.spe = max(1, steps_per_epoch)

    def value(self, epoch: int, step_in_epoch: int = 0):
        s, e, base = self.cfg.start_epoch, self.cfg.end_epoch, self.cfg.base
        if epoch < s:
            return 0.0
        if epoch > e:
            return base
        # warm-up
        if self.cfg.stepwise:
            progress = (epoch - s) + (step_in_epoch / self.spe)
            total = max(1e-8, (e - s + 1))
            w = min(1.0, max(0.0, progress / total))
        else:
            w = (epoch - s + 1) / max(1, (e - s + 1))
            w = min(1.0, max(0.0, w))
        return base * w


# ----------------- main trainer -----------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", "-c", action="append", required=True,
                    help="root config.yaml + overrides (e.g., experiments/03_rkd/configs/<ds>.yaml)")
    ap.add_argument("--epochs", type=int, default=None)
    args = ap.parse_args()

    # load/merge config with repo util
    cfg = load_config(args.config[0], args.config[1:])  # first is root, others are overrides
    cfg = expand_softkd_templates(cfg)

    device = torch.device(cfg.get("device", "cuda" if torch.cuda.is_available() else "cpu"))
    precision = cfg.get("train", {}).get("precision", "fp16")
    scaler = amp.GradScaler('cuda', enabled=(precision == "fp16"))

    # dataloaders
    train_loader, val_loader = build_dataloaders_single(cfg)
    steps_per_epoch = len(train_loader)

    # teacher/student
    teacher = build_teacher(cfg, device)   # frozen
    student = build_student(cfg, device)

    # banks (KD logits & RKD embeddings)
    ds_key = cfg["data"]["dataset"]
    num_samples = len(train_loader.dataset)
    num_classes = cfg["model"]["num_classes"]

    kd_cache_path = cfg["kd"].get(
        "cache_path",
        cfg["kd"].get("cache_template", "src/model/ckpts/soft_targets/{dataset}_fp16.pt").format(dataset=ds_key)
    )
    logits_bank = TeacherLogitsBank(
        path=kd_cache_path, num_samples=num_samples, num_classes=num_classes, device=device
    )
    if not logits_bank.exists():
        cache_loader = make_cache_loader_from(train_loader)
        # 02 스타일의 build 시그니처를 따릅니다(teacher+loader)
        logits_bank.build(teacher_model=teacher, loader=cache_loader)

    embed_template = cfg.get("rkd", {}).get("embed_template",
        "src/model/ckpts/final_embeddings/{dataset}_fp16.pt")
    embed_cache_path = cfg.get("rkd", {}).get("embed_cache_path",
        embed_template.format(dataset=ds_key))

    feat_dim = int(cfg.get("rkd", {}).get("feat_dim", 512))
    emb_bank = TeacherEmbeddingBank(
        path=embed_cache_path, num_samples=num_samples, feat_dim=feat_dim, device=device
    )
    if not emb_bank.exists():
        cache_loader = make_cache_loader_from(train_loader)
        emb_bank.build(teacher_model=teacher, loader=cache_loader,
                       feature_extractor=lambda m, x: extract_feats_and_logits(m, x)[0])

    # optimizer/scheduler from 02
    optim, sched = build_optimizer_scheduler(cfg, student)

    # loss knobs
    alpha = float(cfg["kd"].get("alpha", 0.7))                # KD vs CE
    T = float(cfg["kd"].get("temperature", 3.0))
    # KD용 loss fn 먼저 만들고
    kd_loss_fn = SoftTargetKDLoss(alpha=alpha, temperature=T)

    # student / (teacher or bank)과 엮어서 compute 함수 생성
    kd_compute = make_compute_loss_kd(
        student=student,
        kd_loss_fn=kd_loss_fn,
        teacher=teacher if logits_bank is None else None,
        bank=logits_bank,  # soft target bank 쓰면 bank 전달, 아니면 None
    )

    # beta schedule
    beta_cfg = cfg.get("rkd", {}).get("beta_schedule", {})
    beta_sched = BetaScheduler(
        BetaScheduleCfg(
            start_epoch=int(beta_cfg.get("start_epoch", 21)),
            end_epoch=int(beta_cfg.get("end_epoch", 30)),
            base=float(beta_cfg.get("base", 50.0)),
            stepwise=bool(beta_cfg.get("stepwise", True)),
        ),
        steps_per_epoch=steps_per_epoch
    )

    # epochs (LP 1-20, FT 21-100 by default)
    lp_epochs = int(cfg.get("train", {}).get("linear_probe", {}).get("epochs", 20))
    total_epochs = int(cfg.get("train", {}).get("finetune", {}).get("epochs", 80)) + lp_epochs
    if args.epochs: total_epochs = int(args.epochs)

    # out
    exp_name = cfg.get("experiment", {}).get("name", "03_rkd")
    out_dir = Path(cfg["output"]["dir"]) / exp_name / ds_key
    out_dir.mkdir(parents=True, exist_ok=True)
    save_config_used(cfg, out_dir)

    # freeze for LP
    def set_train_mode_lp(m: nn.Module):
        for p in m.parameters(): p.requires_grad_(False)
        if hasattr(m, "fc"):
            for p in m.fc.parameters(): p.requires_grad_(True)

    def set_train_mode_ft(m: nn.Module):
        for p in m.parameters(): p.requires_grad_(True)

    # ------- define one step (used by either Trainer or fallback loop) -------
    def one_step(batch, epoch, phase):
        is_train = (phase == "train")
        x, y, idx = robust_unpack_batch(batch)
        x = x.to(device, non_blocking=True)
        if y is None:
            # 대부분의 파이프라인엔 y가 존재하지만 없을 경우 0 로스 처리
            y = None
        else:
            y = y.to(device, non_blocking=True)
        idx = idx.to(device, non_blocking=True).long()

        beta_t = beta_sched.value(epoch, 0 if not is_train else step_counters[epoch])

        with amp.autocast('cuda', enabled=(precision == "fp16")):
            # student forward
            s_feat, s_logits = extract_feats_and_logits(student, x)
            # teacher from banks
            t_logits = logits_bank.get(idx)
            t_feats  = emb_bank.get(idx)

            # CE
            loss_ce = F.cross_entropy(s_logits, y) if y is not None else torch.tensor(0., device=device)

            # KD (kd-only; CE는 위에서 별도)
            if HAVE_COMPUTE_KD:
                loss_kd = kd_compute(s_logits, t_logits)["loss_kd"]
            else:
                # SoftTargetKDLoss(alpha=1.0) == pure KD
                loss_kd, _ = kd_loss_fn(s_logits, t_logits, y)

            # RKD
            loss_rkd, _ = compute_rkd_loss(s_feat, t_feats, beta_t)

            # LP 구간에선 beta=0 (스케줄러가 이미 그렇게 줌)
            base = alpha * loss_kd + (1.0 - alpha) * loss_ce
            total = base + float(beta_t) * loss_rkd

        logs = {
            "loss_ce": loss_ce.detach(),
            "loss_kd": loss_kd.detach(),
            "loss_rkd": loss_rkd.detach(),
            "beta": torch.tensor(float(beta_t), device=device),
            "loss_total": total.detach(),
        }

        if is_train:
            optim.zero_grad(set_to_none=True)
            if scaler.is_enabled():
                scaler.scale(total).backward()
                scaler.step(optim)
                scaler.update()
            else:
                total.backward()
                optim.step()

        return total, logs

    # ------- run (Trainer or fallback) -------
    metrics_path = out_dir / "metrics_per_epoch.jsonl"

    def log_epoch(epoch, tr_avg, va_avg):
        payload = {"epoch": int(epoch)}
        payload.update({f"train/{k}": float(v) for k, v in tr_avg.items()})
        payload.update({f"val/{k}":   float(v) for k, v in va_avg.items()})
        with metrics_path.open("a", encoding="utf-8") as f:
            f.write(yaml.safe_dump(payload, sort_keys=False))

    # counters for beta(stepwise)
    step_counters = {}

    # LP
    set_train_mode_lp(student)

    if HAVE_TRAINER and hasattr(Trainer, "__call__") is False:
        # --- If your Trainer exposes a high-level API (example) ---
        # NOTE: This branch assumes a specific interface; if it differs in your repo,
        # keep the fallback loop below.
        trn = Trainer(model=student, optimizer=optim, scaler=scaler, device=device,
                      train_loader=train_loader, val_loader=val_loader,
                      scheduler=sched, precision=precision)

        for epoch in range(1, lp_epochs + 1):
            step_counters[epoch] = 0
            tr_avg = trn.run_epoch(lambda b: one_step(b, epoch, "train"),
                                   phase="train",
                                   step_counter=step_counters)
            va_avg = trn.run_epoch(lambda b: one_step(b, epoch, "val"),
                                   phase="val",
                                   step_counter=step_counters)
            log_epoch(epoch, tr_avg, va_avg)
    else:
        # --- Fallback simple loop (robust) ---
        for epoch in range(1, lp_epochs + 1):
            # train
            student.train(True); teacher.eval()
            step_counters[epoch] = 0
            sums_tr = {"loss_ce":0., "loss_kd":0., "loss_rkd":0., "beta":0., "loss_total":0.}; nt=0
            for step, batch in enumerate(train_loader):
                step_counters[epoch] = step
                _, logs = one_step(batch, epoch, "train")
                for k in sums_tr: sums_tr[k] += float(logs[k].item())
                nt += 1
            tr_avg = {k: v / max(1, nt) for k, v in sums_tr.items()}

            # val
            student.train(False)
            sums_va = {"loss_ce":0., "loss_kd":0., "loss_rkd":0., "beta":0., "loss_total":0.}; nv=0
            with torch.no_grad():
                for step, batch in enumerate(val_loader):
                    step_counters[epoch] = step
                    _, logs = one_step(batch, epoch, "val")
                    for k in sums_va: sums_va[k] += float(logs[k].item())
                    nv += 1
            va_avg = {k: v / max(1, nv) for k, v in sums_va.items()}
            log_epoch(epoch, tr_avg, va_avg)

    # FT
    set_train_mode_ft(student)

    for epoch in range(lp_epochs + 1, total_epochs + 1):
        # train
        student.train(True); teacher.eval()
        step_counters[epoch] = 0
        sums_tr = {"loss_ce":0., "loss_kd":0., "loss_rkd":0., "beta":0., "loss_total":0.}; nt=0
        for step, batch in enumerate(train_loader):
            step_counters[epoch] = step
            _, logs = one_step(batch, epoch, "train")
            for k in sums_tr: sums_tr[k] += float(logs[k].item())
            nt += 1
        tr_avg = {k: v / max(1, nt) for k, v in sums_tr.items()}

        # val
        student.train(False)
        sums_va = {"loss_ce":0., "loss_kd":0., "loss_rkd":0., "beta":0., "loss_total":0.}; nv=0
        with torch.no_grad():
            for step, batch in enumerate(val_loader):
                step_counters[epoch] = step
                _, logs = one_step(batch, epoch, "val")
                for k in sums_va: sums_va[k] += float(logs[k].item())
                nv += 1
        va_avg = {k: v / max(1, nv) for k, v in sums_va.items()}
        log_epoch(epoch, tr_avg, va_avg)


if __name__ == "__main__":
    main()
