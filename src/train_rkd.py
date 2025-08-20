# src/train_rkd.py
import argparse
import json
import math
import os
from pathlib import Path
from typing import Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import torchvision
import torchvision.transforms as T
import yaml

# --- repo losses (고정 import, 분기 없음) ---
from src.losses.soft_kd import SoftTargetKDLoss
from src.losses.compute.compute_relational_kd_loss import compute_rkd_loss

# --- teacher banks ---
from src.data.soft_targets import TeacherLogitsBank as SoftTargetBank
from src.data.final_embeddings import TeacherEmbeddingBank as FinalEmbeddingBank
from src.data.indexed import collate_with_index

# ---------------------------------------------------------
# 유틸
# ---------------------------------------------------------
def load_yaml(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def deep_merge(base: dict, override: dict) -> dict:
    out = dict(base)
    for k, v in override.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = deep_merge(out[k], v)
        else:
            out[k] = v
    return out

def format_templates(cfg: dict) -> dict:
    """{dataset} 같은 템플릿 문자열을 config 값으로 치환"""
    ds = cfg.get("data", {}).get("dataset", "")
    def _fmt(x):
        if isinstance(x, str):
            try:
                return x.format(dataset=ds)
            except Exception:
                return x
        if isinstance(x, dict):
            return {k: _fmt(v) for k, v in x.items()}
        if isinstance(x, list):
            return [_fmt(i) for i in x]
        return x
    return _fmt(cfg)

def save_config_used(dst_dir: Path, cfg: dict):
    dst_dir.mkdir(parents=True, exist_ok=True)
    (dst_dir / "config_used.yaml").write_text(
        yaml.safe_dump(cfg, sort_keys=False, allow_unicode=True), encoding="utf-8"
    )

def set_seed(seed: int):
    import random
    import numpy as np
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

# ---------------------------------------------------------
# 데이터셋 (증강 없음, (img, label, idx) 반환)
# ---------------------------------------------------------
class IndexDataset(Dataset):
    def __init__(self, base: Dataset):
        self.base = base
    def __len__(self): return len(self.base)
    def __getitem__(self, idx):
        x, y = self.base[idx]
        return x, y, idx

def build_loaders(cfg: dict) -> Tuple[DataLoader, DataLoader, int]:
    name = cfg["data"]["dataset"].lower()
    root = cfg["data"]["root"]
    bs   = int(cfg["data"]["batch_size"])
    nw   = int(cfg["data"]["num_workers"])
    pin  = bool(cfg["data"].get("pin_memory", True))
    download = bool(cfg["data"].get("download", True))

    if name == "cifar10":
        normalize = T.Normalize((0.4914, 0.4822, 0.4465),
                                (0.2470, 0.2435, 0.2616))
        num_classes = 10
        train_base = torchvision.datasets.CIFAR10(root, train=True, transform=T.Compose([T.ToTensor(), normalize]),
                                                  download=download)
        val_base   = torchvision.datasets.CIFAR10(root, train=False, transform=T.Compose([T.ToTensor(), normalize]),
                                                  download=download)
    elif name == "cifar100":
        normalize = T.Normalize((0.5071, 0.4867, 0.4408),
                                (0.2675, 0.2565, 0.2761))
        num_classes = 100
        train_base = torchvision.datasets.CIFAR100(root, train=True, transform=T.Compose([T.ToTensor(), normalize]),
                                                   download=download)
        val_base   = torchvision.datasets.CIFAR100(root, train=False, transform=T.Compose([T.ToTensor(), normalize]),
                                                   download=download)
    elif name == "stl10":
        normalize = T.Normalize((0.4467, 0.4398, 0.4066),
                                (0.2603, 0.2566, 0.2713))
        num_classes = 10
        train_base = torchvision.datasets.STL10(root, split="train", transform=T.Compose([T.ToTensor(), normalize]),
                                                download=download)
        val_base   = torchvision.datasets.STL10(root, split="test", transform=T.Compose([T.ToTensor(), normalize]),
                                                download=download)
    elif name in ["tiny_imagenet", "tiny-imagenet-200", "tinyimagenet"]:
        # torchvision에 정식 Tiny-ImageNet이 없어서 폴더형태로 가정.
        # 리포에서 이미 동일 구조를 쓰고 있으므로 그대로 사용.
        # dataset/tiny-imagenet-200/train, /val 형태
        normalize = T.Normalize((0.4802, 0.4481, 0.3975),
                                (0.2770, 0.2691, 0.2821))
        num_classes = 200
        train_dir = os.path.join(root, "tiny-imagenet-200", "train")
        val_dir   = os.path.join(root, "tiny-imagenet-200", "val")
        transform = T.Compose([T.ToTensor(), normalize])
        train_base = torchvision.datasets.ImageFolder(train_dir, transform=transform)
        val_base   = torchvision.datasets.ImageFolder(val_dir, transform=transform)
    else:
        raise ValueError(f"Unknown dataset: {name}")

    train_ds = IndexDataset(train_base)
    val_ds   = IndexDataset(val_base)

    train_ld = DataLoader(train_ds, batch_size=bs, shuffle=True,  num_workers=nw, pin_memory=pin)
    val_ld   = DataLoader(val_ds,   batch_size=bs, shuffle=False, num_workers=nw, pin_memory=pin)
    return train_ld, val_ld, num_classes

# ---------------------------------------------------------
# 모델: torchvision ResNet + ckpt 로딩
# (repo에서 쓰는 ckpt 경로를 config에서 그대로 받아 사용)
# ---------------------------------------------------------
def build_resnet(arch: str, num_classes: int):
    arch = arch.lower()
    if arch == "resnet18":
        m = torchvision.models.resnet18(num_classes=num_classes)
        feat_dim = 512
    elif arch == "resnet34":
        m = torchvision.models.resnet34(num_classes=num_classes)
        feat_dim = 512
    else:
        raise ValueError(f"Unsupported arch: {arch}")
    return m, feat_dim

def load_ckpt_if_exists(model: nn.Module, path: str):
    if path and os.path.isfile(path):
        sd = torch.load(path, map_location="cpu")
        # state_dict or whole object 모두 지원
        if isinstance(sd, dict) and "state_dict" in sd:
            sd = sd["state_dict"]
        try:
            model.load_state_dict(sd, strict=True)
        except Exception:
            # 키가 앞에 'module.' 붙은 경우 등 완화
            new_sd = {}
            for k, v in sd.items():
                nk = k.replace("module.", "")
                new_sd[nk] = v
            model.load_state_dict(new_sd, strict=False)

# 백본 임베딩 추출(FC 이전)
def extract_feats(model: nn.Module, x: torch.Tensor) -> torch.Tensor:
    # torchvision resnet 기준 구현
    x = model.conv1(x)
    x = model.bn1(x)
    x = model.relu(x)
    x = model.maxpool(x)
    x = model.layer1(x)
    x = model.layer2(x)
    x = model.layer3(x)
    x = model.layer4(x)
    x = model.avgpool(x)
    x = torch.flatten(x, 1)
    return x  # (B, 512)

# ---------------------------------------------------------
# Beta 스케줄
# ---------------------------------------------------------
class BetaScheduler:
    def __init__(self, start_epoch: int, end_epoch: int, base: float):
        self.s = start_epoch
        self.e = end_epoch
        self.base = base
    def value(self, epoch: int) -> float:
        if epoch <= self.s - 1:
            return 0.0
        if self.s <= epoch <= self.e:
            # 선형 warm-up
            t = (epoch - self.s + 1) / (self.e - self.s + 1)
            return self.base * float(t)
        return self.base

# ---------------------------------------------------------
# 로그 유틸
# ---------------------------------------------------------
class JsonlLogger:
    def __init__(self, path: Path):
        self.f = open(path, "a", encoding="utf-8")
    def log(self, payload: Dict):
        self.f.write(json.dumps(payload, ensure_ascii=False) + "\n")
        self.f.flush()
    def close(self):
        self.f.close()

# ---------------------------------------------------------
# 은행 채우기(teacher 1회 추론)
# ---------------------------------------------------------
@torch.no_grad()
def ensure_teacher_banks(
    cfg: dict,
    teacher: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: torch.device,
    dtype: torch.dtype,
    num_classes: int,
):
    ds_name = cfg["data"]["dataset"]
    ckpt_root = cfg["paths"]["ckpt_root"]
    logits_path = cfg["kd"]["cache_template"]  # .../soft_targets/{dataset}_fp16.pt
    embed_path  = cfg["rkd"]["embed_template"] # .../final_embeddings/{dataset}_fp16.pt

    # Logit bank
    logit_bank = SoftTargetBank(logits_path, num_samples=len(train_loader.dataset), num_classes=num_classes)
    emb_bank   = FinalEmbeddingBank(embed_path, num_samples=len(train_loader.dataset), feat_dim=int(cfg["rkd"].get("feat_dim", 512)))

    if (not logit_bank.exists()) or (not emb_bank.exists()):
        teacher.eval()
        cache_loader = DataLoader(
                train_loader.dataset,
                batch_size=cfg["data"]["batch_size"],
                shuffle=False,
                num_workers=cfg["data"]["num_workers"],
                pin_memory=cfg["data"].get("pin_memory", True),
                collate_fn=collate_with_index,
            )
        logit_bank.build(teacher, cache_loader)
        emb_bank.build(teacher, cache_loader, extract_feats)

    # 항상 로드해 객체를 반환
    logit_bank._ensure_loaded()
    emb_bank._ensure_loaded()
    return logit_bank, emb_bank

# ---------------------------------------------------------
# 1 스텝 손실 계산
# ---------------------------------------------------------
def step_loss(
    student: nn.Module,
    batch: Tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    device: torch.device,
    kd_ce_fn: SoftTargetKDLoss,
    logit_bank: SoftTargetBank,
    emb_bank: FinalEmbeddingBank,
    beta_now: float,
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    x, y, idx = batch
    x = x.to(device, non_blocking=True)
    y = y.to(device, non_blocking=True)

    # student forward
    s_logits = student(x)
    with torch.no_grad():
        t_logits = logit_bank.get(idx)      # (B, C) on CPU float16/32
        t_logits = t_logits.to(device=device, dtype=s_logits.dtype)

    # KD + CE
    base_loss, parts = kd_ce_fn(s_logits, t_logits, y)  # keys: train_loss_ce, train_loss_kd, train_loss

    # RKD (임베딩)
    with torch.no_grad():
        t_emb = emb_bank.get(idx).to(device)   # (B, D) float32
    s_emb = extract_feats(student, x)          # (B, D)
    rkd_dict = compute_rkd_loss(s_emb, t_emb, beta_scalar=beta_now)  # {"loss_rkd", "loss_rkd_weighted"}

    # 최종 합
    loss_total = base_loss + rkd_dict["loss_rkd_weighted"]

    with torch.no_grad():
        acc = (s_logits.argmax(dim=1) == y).float().mean()

    log = {
        "loss_ce": parts["train_loss_ce"].detach(),
        "loss_kd": parts["train_loss_kd"].detach(),
        "loss_rkd": rkd_dict["loss_rkd"].detach(),
        "loss_total": loss_total.detach(),
        "acc": acc.detach(),
        "beta": torch.tensor(float(beta_now), device=device),
    }
    return loss_total, log

# ---------------------------------------------------------
# 학습 루프
# ---------------------------------------------------------
def run(cfg: dict):
    set_seed(int(cfg.get("seed", 42)))
    device = torch.device(cfg.get("device", "cuda" if torch.cuda.is_available() else "cpu"))
    dtype = torch.float16 if cfg.get("precision", "fp32") == "fp16" else torch.float32

    # loaders
    train_loader, val_loader, num_classes = build_loaders(cfg)

    # models
    st_arch = cfg["model"]["student"]["arch"]
    tc_arch = cfg["model"]["teacher"]["arch"]

    student, feat_dim = build_resnet(st_arch, num_classes)
    teacher, _        = build_resnet(tc_arch, num_classes)

    # ckpt 로딩 (학생/교사)
    load_ckpt_if_exists(student, cfg["model"]["student"].get("ckpt", ""))
    # teacher는 backone+classifier 템플릿 경로를 합쳐서 완모델 sd가 저장되어 있을 수도 있음
    # 우선 전체 ckpt 경로가 있으면 우선 사용
    if "ckpt" in cfg["model"]["teacher"] and os.path.isfile(cfg["model"]["teacher"]["ckpt"]):
        load_ckpt_if_exists(teacher, cfg["model"]["teacher"]["ckpt"])
    else:
        # 분리 저장된 경우(backbone, classifier)를 best-effort로 로드
        bb = cfg["model"]["teacher"].get("backbone_path", "")
        cl = cfg["model"]["teacher"].get("classifier_path", "")
        if os.path.isfile(bb):
            sd = torch.load(bb, map_location="cpu")
            # 백본 키 추정 불가 시, strict=False
            try: teacher.load_state_dict(sd, strict=False)
            except Exception: teacher.load_state_dict({k.replace("module.", ""): v for k, v in sd.items()}, strict=False)
        if os.path.isfile(cl):
            sd = torch.load(cl, map_location="cpu")
            try: teacher.load_state_dict(sd, strict=False)
            except Exception: teacher.load_state_dict({k.replace("module.", ""): v for k, v in sd.items()}, strict=False)

    student = student.to(device)
    teacher = teacher.to(device).eval()
    for p in teacher.parameters(): p.requires_grad_(False)

    # output dirs
    exp_name = cfg.get("experiment", {}).get("name", "03_relational_kd")
    ds_key   = cfg["data"]["dataset"]
    out_root = Path(cfg["output"]["dir"]) / exp_name / ds_key
    out_root.mkdir(parents=True, exist_ok=True)
    log_train = JsonlLogger(out_root / "train.jsonl")
    log_val   = JsonlLogger(out_root / "val.jsonl")
    save_config_used(out_root, cfg)

    # teacher banks (없으면 채우고 저장, 있으면 로드)
    logit_bank, emb_bank = ensure_teacher_banks(
        cfg, teacher, train_loader, val_loader, device, dtype, num_classes
    )

    # losses
    alpha = float(cfg["kd"].get("alpha", 0.7))
    T     = float(cfg["kd"].get("temperature", 3.0))
    kd_ce_fn = SoftTargetKDLoss(alpha=alpha, temperature=T)

    beta_cfg = cfg.get("rkd", {}).get("beta_schedule", {})
    beta_base  = float(beta_cfg.get("base", 50.0))
    warm_s    = int(beta_cfg.get("start_epoch", 21))
    warm_e    = int(beta_cfg.get("end_epoch",   30))
    beta_sched = BetaScheduler(warm_s, warm_e, beta_base)

    # 옵티마이저/스케줄러 (cosine, LP/FT 구간 동일 설정)
    opt_cfg = cfg["train"]["optimizer"]
    lr      = float(opt_cfg.get("lr", 3e-4))
    wd      = float(opt_cfg.get("weight_decay", 0.05))
    betas   = tuple(opt_cfg.get("betas", (0.9, 0.999)))
    eps     = float(opt_cfg.get("eps", 1e-8))

    # Epoch 설정
    ep_lp = int(cfg["train"]["linear_probe"]["epochs"])
    ep_ft = int(cfg["train"]["finetune"]["epochs"])
    total_epochs = ep_lp + ep_ft

    # Linear probe: fc만 학습
    for p in student.parameters(): p.requires_grad_(False)
    for p in student.fc.parameters(): p.requires_grad_(True)
    optimizer = torch.optim.AdamW(student.fc.parameters(), lr=lr, weight_decay=wd, betas=betas, eps=eps)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=ep_lp, eta_min=float(cfg["train"]["scheduler"].get("min_lr", 1e-6)))

    # ----- LP phase -----
    for epoch in range(1, ep_lp + 1):
        student.train()
        epoch_log = {"epoch": epoch}
        # beta=0 (강제)
        beta_now = 0.0

        # train
        agg = {"loss_ce": 0, "loss_kd": 0, "loss_rkd": 0, "loss_total": 0, "acc": 0}
        n = 0
        for batch in train_loader:
            optimizer.zero_grad(set_to_none=True)
            loss, parts = step_loss(student, batch, device, kd_ce_fn, logit_bank, emb_bank, beta_now)
            loss.backward()
            optimizer.step()

            bs = batch[0].size(0)
            for k in agg: agg[k] += float(parts[k]) * bs
            n += bs

        for k in agg: agg[k] /= max(1, n)
        agg["beta"] = beta_now
        agg["lr"]   = float(optimizer.param_groups[0]["lr"])
        log_train.log({"epoch": epoch, **{f"train_{k}": v for k, v in agg.items()}})

        # val
        student.eval()
        vagg = {"loss_ce": 0, "loss_kd": 0, "loss_rkd": 0, "loss_total": 0, "acc": 0}
        vn = 0
        with torch.no_grad():
            for batch in val_loader:
                loss, parts = step_loss(student, batch, device, kd_ce_fn, logit_bank, emb_bank, beta_now)
                bs = batch[0].size(0)
                for k in vagg: vagg[k] += float(parts[k]) * bs
                vn += bs
        for k in vagg: vagg[k] /= max(1, vn)
        vagg["beta"] = beta_now
        vagg["lr"]   = float(optimizer.param_groups[0]["lr"])
        log_val.log({"epoch": epoch, **{f"val_{k}": v for k, v in vagg.items()}})

        scheduler.step()

    # ----- FT phase -----
    # 모든 파라미터 학습
    for p in student.parameters(): p.requires_grad_(True)
    optimizer = torch.optim.AdamW(student.parameters(), lr=lr, weight_decay=wd, betas=betas, eps=eps)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=ep_ft, eta_min=float(cfg["train"]["scheduler"].get("min_lr", 1e-6)))

    for i, epoch in enumerate(range(ep_lp + 1, total_epochs + 1), start=1):
        student.train()
        beta_now = beta_sched.value(epoch)  # 21~30 warm-up, 이후 고정
        agg = {"loss_ce": 0, "loss_kd": 0, "loss_rkd": 0, "loss_total": 0, "acc": 0}
        n = 0
        for batch in train_loader:
            optimizer.zero_grad(set_to_none=True)
            loss, parts = step_loss(student, batch, device, kd_ce_fn, logit_bank, emb_bank, beta_now)
            loss.backward()
            optimizer.step()

            bs = batch[0].size(0)
            for k in agg: agg[k] += float(parts[k]) * bs
            n += bs
        for k in agg: agg[k] /= max(1, n)
        agg["beta"] = beta_now
        agg["lr"]   = float(optimizer.param_groups[0]["lr"])
        log_train.log({"epoch": epoch, **{f"train_{k}": v for k, v in agg.items()}})

        # val
        student.eval()
        vagg = {"loss_ce": 0, "loss_kd": 0, "loss_rkd": 0, "loss_total": 0, "acc": 0}
        vn = 0
        with torch.no_grad():
            for batch in val_loader:
                loss, parts = step_loss(student, batch, device, kd_ce_fn, logit_bank, emb_bank, beta_now)
                bs = batch[0].size(0)
                for k in vagg: vagg[k] += float(parts[k]) * bs
                vn += bs
        for k in vagg: vagg[k] /= max(1, vn)
        vagg["beta"] = beta_now
        vagg["lr"]   = float(optimizer.param_groups[0]["lr"])
        log_val.log({"epoch": epoch, **{f"val_{k}": v for k, v in vagg.items()}})

        scheduler.step()

    log_train.close()
    log_val.close()

# ---------------------------------------------------------
# CLI
# ---------------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, action="append", required=True,
                        help="하나 이상 지정 가능. 앞에서 뒤로 덮어씀")
    args = parser.parse_args()

    # 여러 yaml을 순서대로 deep merge
    cfg = {}
    for p in args.config:
        cfg = deep_merge(cfg, load_yaml(p))
    cfg = format_templates(cfg)

    run(cfg)

if __name__ == "__main__":
    main()
