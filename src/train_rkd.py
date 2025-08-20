# src/train_rkd.py
import os, math, time, yaml, argparse
from pathlib import Path
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import amp
from torch.utils.data import DataLoader

# --- banks ---
from src.data.soft_targets import TeacherLogitsBank
from src.data.final_embeddings import TeacherEmbeddingBank

# ----------------- small utils -----------------
def save_config_used(cfg, out_dir):
    out = Path(out_dir); out.mkdir(parents=True, exist_ok=True)
    (out / "config_used.yaml").write_text(
        yaml.safe_dump(cfg, sort_keys=False, allow_unicode=True),
        encoding="utf-8"
    )

def robust_unpack_batch(batch):
    """Return (x,y,idx) from (x,y,idx) or (x,idx) or (idx,x,y)."""
    if isinstance(batch, (list, tuple)) and len(batch) == 3:
        a,b,c = batch
        # 이미지가 어디에 있는지 모양으로 판별(4D)
        if torch.is_tensor(a) and a.dim()==4:
            x,y,idx = a,b,c
        elif torch.is_tensor(b) and b.dim()==4:
            x,y,idx = b,c,a
        else:
            # (x,idx,y)로 들어왔을 수도
            if torch.is_tensor(c) and c.dim()==4:
                x,y,idx = c,a,b
            else:
                raise RuntimeError(f"Cannot infer (x,y,idx) from shapes: {tuple(getattr(a,'shape',[]))}, {tuple(getattr(b,'shape',[]))}, {tuple(getattr(c,'shape',[]))}")
    elif isinstance(batch, (list, tuple)) and len(batch) == 2:
        a,b = batch
        if torch.is_tensor(a) and a.dim()==4:
            x,idx = a,b; y=None
        elif torch.is_tensor(b) and b.dim()==4:
            x,idx = b,a; y=None
        else:
            raise RuntimeError("2-tuple batch must be (x,idx) or (idx,x).")
    else:
        raise RuntimeError("Unexpected batch structure.")
    if x.dim()!=4: raise RuntimeError(f"Images must be 4D(B,C,H,W). got {tuple(x.shape)}")
    return x,y,idx

def make_cache_loader_from(loader: DataLoader):
    """Same dataset/bs/num_workers/collate but shuffle=False for deterministic caching."""
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

# KD only (no CE) — L_kd
def kd_only_loss(student_logits, teacher_logits, T: float):
    return F.kl_div(
        F.log_softmax(student_logits / T, dim=1),
        F.softmax(teacher_logits / T, dim=1),
        reduction="batchmean"
    ) * (T * T)

# feature extractor for (torchvision) resnet18/34
def extract_feats_and_logits(m: nn.Module, x: torch.Tensor):
    # torchvision resnet forward decomposition
    z = m.conv1(x); z = m.bn1(z); z = m.relu(z); z = m.maxpool(z)
    z = m.layer1(z); z = m.layer2(z); z = m.layer3(z); z = m.layer4(z)
    z = F.adaptive_avg_pool2d(z, 1)
    feat = torch.flatten(z, 1)     # (B,512)
    logits = m.fc(feat)            # (B,C)
    return feat, logits

# RKD cosine-gram with centering + off-diag MSE
def rkd_loss(student_feats: torch.Tensor, teacher_feats: torch.Tensor):
    s = F.normalize(student_feats.float(), dim=1)
    t = F.normalize(teacher_feats.float(), dim=1)
    Gs, Gt = (s @ s.T), (t @ t.T)
    Gs = Gs - Gs.mean(0, keepdim=True) - Gs.mean(1, keepdim=True) + Gs.mean()
    Gt = Gt - Gt.mean(0, keepdim=True) - Gt.mean(1, keepdim=True) + Gt.mean()
    B = Gs.size(0)
    off = ~torch.eye(B, dtype=torch.bool, device=Gs.device)
    return F.mse_loss(Gs[off], Gt[off], reduction="mean")

# -------- beta scheduler (epoch-based) ----------
@dataclass
class BetaSchedule:
    start_epoch: int = 21
    end_epoch: int = 30
    base: float = 50.0
    stepwise: bool = True

    def value(self, epoch: int, step_in_epoch: int = 0, steps_per_epoch: int = 1):
        if epoch < self.start_epoch: return 0.0
        if epoch > self.end_epoch:   return float(self.base)
        # warm-up
        if self.stepwise and steps_per_epoch > 0:
            progress = (epoch - self.start_epoch) + (step_in_epoch / steps_per_epoch)
            total    = max(1e-8, (self.end_epoch - self.start_epoch + 1))
            w = min(1.0, max(0.0, progress / total))
        else:
            w = (epoch - self.start_epoch + 1) / max(1, (self.end_epoch - self.start_epoch + 1))
            w = min(1.0, max(0.0, w))
        return float(self.base) * w

# ----------------- main trainer -----------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", "-c", action="append", required=True,
                    help="root config.yaml + overrides (e.g., experiments/03_rkd/configs/<ds>.yaml)")
    ap.add_argument("--epochs", type=int, default=None)
    args = ap.parse_args()

    # ---- load/merge config ----
    import yaml
    cfg = {}
    for i, p in enumerate(args.config):
        with open(p, "r", encoding="utf-8") as f:
            part = yaml.safe_load(f)
        if i == 0:
            cfg = part
        else:
            # shallow-merge convenience (repo의 load_config가 있다면 그걸 사용하세요)
            from copy import deepcopy
            def merge(a, b):
                out = deepcopy(a)
                for k,v in b.items():
                    if isinstance(v, dict) and k in out and isinstance(out[k], dict):
                        out[k] = merge(out[k], v)
                    else:
                        out[k] = v
                return out
            cfg = merge(cfg, part)

    device = torch.device(cfg.get("device", "cuda" if torch.cuda.is_available() else "cpu"))
    precision = cfg.get("train", {}).get("precision", "fp16")
    scaler = amp.GradScaler('cuda', enabled=(precision == "fp16"))

    # --- dataset/dataloader (02에서 쓰던 빌더를 그대로 사용) ---
    from src.train_soft_kd import build_dataloaders_single, build_teacher, build_student, build_optimizer_scheduler
    train_loader, val_loader = build_dataloaders_single(cfg)
    steps_per_epoch = len(train_loader)

    # --- models ---
    teacher = build_teacher(cfg, device)   # frozen eval in forward(no grad)
    student = build_student(cfg, device)

    # --- KD logits bank (reuse 02) ---
    ds_key = cfg["data"]["dataset"]
    num_classes = cfg["model"]["num_classes"]
    kd_cache_path = cfg["kd"].get(
        "cache_path",
        cfg["kd"].get("cache_template", "src/model/ckpts/soft_targets/{dataset}_fp16.pt").format(dataset=ds_key)
    )
    logits_bank = TeacherLogitsBank(
        path=kd_cache_path,
        num_samples=len(train_loader.dataset),
        num_classes=num_classes,
        device=device
    )
    if not logits_bank.exists():
        cache_loader = make_cache_loader_from(train_loader)
        logits_bank.build(teacher_model=teacher, loader=cache_loader, T=float(cfg["kd"].get("temperature", 4.0)))

    # --- RKD embedding bank (NEW) ---
    embed_template = cfg.get("rkd", {}).get("embed_template",
        "src/model/ckpts/final_embeddings/{dataset}_fp16.pt")
    embed_cache_path = cfg.get("rkd", {}).get("embed_cache_path",
        embed_template.format(dataset=ds_key))

    feat_dim = cfg.get("rkd", {}).get("feat_dim", 512)  # resnet18/34 penultimate
    emb_bank = TeacherEmbeddingBank(
        path=embed_cache_path,
        num_samples=len(train_loader.dataset),
        feat_dim=int(feat_dim),
        device=device
    )
    if not emb_bank.exists():
        cache_loader = make_cache_loader_from(train_loader)
        emb_bank.build(teacher_model=teacher, loader=cache_loader, feature_extractor=extract_feats_and_logits)

    # --- opt/sched: LP(1-20) → FT(21-100) ---
    lp_epochs = int(cfg.get("train", {}).get("linear_probe", {}).get("epochs", 20))
    total_epochs = int(cfg.get("train", {}).get("finetune", {}).get("epochs", 80)) + lp_epochs
    if args.epochs: total_epochs = int(args.epochs)

    # param freeze/unfreeze
    def set_train_mode_lp(m: nn.Module):
        for p in m.parameters(): p.requires_grad_(False)
        # 분류기 헤드만 학습
        if hasattr(m, "fc"):
            for p in m.fc.parameters(): p.requires_grad_(True)

    def set_train_mode_ft(m: nn.Module):
        for p in m.parameters(): p.requires_grad_(True)

    optim, sched = build_optimizer_scheduler(cfg, student)

    # --- loss coeffs ---
    alpha = float(cfg["kd"].get("alpha", 0.5))             # KD vs CE
    beta_cfg = cfg.get("rkd", {}).get("beta_schedule", {})
    beta_sched = BetaSchedule(
        start_epoch=int(beta_cfg.get("start_epoch", 21)),
        end_epoch=int(beta_cfg.get("end_epoch", 30)),
        base=float(beta_cfg.get("base", 50.0)),
        stepwise=bool(beta_cfg.get("stepwise", True)),
    )
    T = float(cfg["kd"].get("temperature", 4.0))

    # --- outputs ---
    exp_name = cfg.get("experiment", {}).get("name", "03_rkd")
    out_dir = Path(cfg["output"]["dir"]) / exp_name / ds_key
    out_dir.mkdir(parents=True, exist_ok=True)
    save_config_used(cfg, out_dir)

    def run_one_epoch(epoch, phase, optimizer=None):
        is_train = (phase == "train")
        student.train(is_train)
        teacher.eval()

        meter = {}
        nstep = 0
        ce_sum = kd_sum = rkd_sum = tot_sum = beta_sum = 0.0

        for step, batch in enumerate(train_loader if is_train else val_loader):
            x, y, idx = robust_unpack_batch(batch)
            x = x.to(device, non_blocking=True)
            if y is None:  # 안전
                # 02 파이프라인은 보통 y가 있음
                pass
            else:
                y = y.to(device, non_blocking=True)
            idx = idx.to(device, non_blocking=True).long()

            beta_t = beta_sched.value(epoch, step, steps_per_epoch)

            with amp.autocast('cuda', enabled=(precision=="fp16")):
                # student
                s_feat, s_logits = extract_feats_and_logits(student, x)
                # teacher from banks
                t_logits = logits_bank.get(idx)  # KD 용
                t_feats  = emb_bank.get(idx)     # RKD 용

                # losses
                loss_ce  = F.cross_entropy(s_logits, y) if y is not None else torch.tensor(0., device=device)
                loss_kd  = kd_only_loss(s_logits, t_logits, T=T)
                loss_rkd = rkd_loss(s_feat, t_feats)

                base = alpha * loss_kd + (1.0 - alpha) * loss_ce
                total = base + float(beta_t) * loss_rkd

            if is_train:
                optimizer.zero_grad(set_to_none=True)
                if scaler.is_enabled():
                    scaler.scale(total).backward()
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    total.backward()
                    optimizer.step()
                if sched is not None and getattr(sched, "step_per_iteration", False):
                    sched.step()

            ce_sum  += float(loss_ce.detach().cpu().item())
            kd_sum  += float(loss_kd.detach().cpu().item())
            rkd_sum += float(loss_rkd.detach().cpu().item())
            tot_sum += float(total.detach().cpu().item())
            beta_sum+= float(beta_t)
            nstep   += 1

        if sched is not None and not getattr(sched, "step_per_iteration", False) and is_train:
            sched.step()

        meter = {
            f"{phase}/loss_ce":  ce_sum / max(1,nstep),
            f"{phase}/loss_kd":  kd_sum / max(1,nstep),
            f"{phase}/loss_rkd": rkd_sum / max(1,nstep),
            f"{phase}/beta":     beta_sum/ max(1,nstep),
            f"{phase}/loss_total": tot_sum/ max(1,nstep),
        }
        return meter

    # ---- training loop ----
    # LP
    set_train_mode_lp(student)
    for epoch in range(1, lp_epochs+1):
        tr = run_one_epoch(epoch, "train", optimizer=optim)
        va = run_one_epoch(epoch, "val")
        # 간단 jsonl 로깅
        (out_dir / "metrics_per_epoch.jsonl").open("a", encoding="utf-8").write(
            yaml.safe_dump({"epoch": epoch, **tr, **va}, sort_keys=False)
        )

    # FT
    set_train_mode_ft(student)
    for epoch in range(lp_epochs+1, total_epochs+1):
        tr = run_one_epoch(epoch, "train", optimizer=optim)
        va = run_one_epoch(epoch, "val")
        (out_dir / "metrics_per_epoch.jsonl").open("a", encoding="utf-8").write(
            yaml.safe_dump({"epoch": epoch, **tr, **va}, sort_keys=False)
        )

if __name__ == "__main__":
    main()
