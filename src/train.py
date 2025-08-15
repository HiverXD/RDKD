# src/train.py
from __future__ import annotations
import argparse, os, math
from pathlib import Path

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR

from utils.config import load_config, pretty
from data.data_setup import loaders
from model.backbones import build_classifier, ARCH2SAVE
from utils.trainer import Trainer

def _student_ckpt_path(cfg) -> str:
    # config에 명시가 없으면, 저장 규칙(ARCH2SAVE)로 추론
    ckpt = cfg["model"]["ckpts"].get("student")
    if ckpt and len(ckpt):
        return ckpt
    # 예: src/model/ckpts/ResNet18.pt
    save_dir = cfg["paths"]["ckpt_root"] if "paths" in cfg else "src/model/ckpts"
    fname = ARCH2SAVE[cfg["model"]["backbone"]]
    return str(Path(save_dir) / fname)

def build_student_for_stage(cfg: dict, device: torch.device, stage: str) -> nn.Module:
    """
    stage: "linear_probe" | "finetune"
    - linear_probe: backbone freeze, new fc 학습
    - finetune: backbone도 학습
    """
    num_classes = cfg["model"]["num_classes"]
    ckpt_path = _student_ckpt_path(cfg)
    if stage == "linear_probe":
        model = build_classifier(ckpt_path, num_classes, freeze_backbone=True, device=str(device))
    elif stage == "finetune":
        model = build_classifier(ckpt_path, num_classes, freeze_backbone=False, device=str(device))
    else:
        raise ValueError(stage)
    return model

def build_optimizer(model: nn.Module, cfg: dict) -> AdamW:
    opt_cfg = cfg["train"]["optimizer"]
    wd = opt_cfg["weight_decay"]
    decay, no_decay = [], []
    for n, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if p.ndim == 1 or n.endswith(".bias") or "bn" in n.lower() or "norm" in n.lower():
            no_decay.append(p)
        else:
            decay.append(p)
    groups = [
        {"params": decay, "weight_decay": wd},
        {"params": no_decay, "weight_decay": 0.0},
    ]
    return AdamW(groups, lr=opt_cfg["lr"],
                 betas=tuple(opt_cfg["betas"]), eps=opt_cfg["eps"])

def build_scheduler(optimizer, cfg: dict, total_epochs: int):
    sch_cfg = cfg["train"]["scheduler"]
    warmup = int(sch_cfg["warmup_epochs"])
    base_lr = cfg["train"]["optimizer"]["lr"]
    min_lr = float(sch_cfg["min_lr"])

    def lr_lambda(epoch):
        if epoch < warmup:
            return float(epoch + 1) / max(1, warmup)
        t = (epoch - warmup) / max(1, total_epochs - warmup)
        min_factor = min_lr / base_lr
        return min_factor + 0.5 * (1 - min_factor) * (1 + math.cos(math.pi * t))

    return LambdaLR(optimizer, lr_lambda)

def freeze_backbone(model: nn.Module, freeze_bn: bool = True):
    for n, p in model.named_parameters():
        if "fc" in n or "classifier" in n:   # 분류기는 열어둠
            p.requires_grad = True
        else:
            p.requires_grad = False
    if freeze_bn:
        for m in model.modules():
            if isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d)):
                m.eval()
                for p in m.parameters():
                    p.requires_grad = False

def unfreeze_all(model: nn.Module):
    for p in model.parameters():
        p.requires_grad = True

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cfg", type=str, default="config.yaml", help="root config path")
    ap.add_argument("--override", type=str, nargs="*", default=[], help="override yaml paths")
    args = ap.parse_args()

    cfg = load_config(args.cfg, args.override)
    print("\n=== Final Config ===")
    print(pretty(cfg))

    torch.manual_seed(cfg["seed"])
    device = torch.device(cfg["device"] if torch.cuda.is_available() else "cpu")

    # 데이터 로더 (공용 함수 사용)
    train_loaders, test_loaders = loaders(
        batch_size=cfg["data"]["batch_size"],
        root=cfg["data"]["root"],
        num_workers=cfg["data"]["num_workers"],
        pin_memory=cfg["data"]["pin_memory"],
        download=cfg["data"]["download"],
    )
    ds_key = cfg["data"]["dataset"]
    train_loader = train_loaders[ds_key]
    test_loader  = test_loaders[ds_key]

    # Linear Probe
    if cfg["train"]["linear_probe"]["enable"]:
        model = build_student_for_stage(cfg, device, "linear_probe")
        opt = build_optimizer(model, cfg)
        sch = build_scheduler(opt, cfg, total_epochs=cfg["train"]["linear_probe"]["epochs"])
        linear_probe_trainer = Trainer(
            model=model, optimizer=opt, scheduler=sch, schedule_mode="epoch", device=device,
            train_loader=train_loader, test_loader=test_loader,
            epochs=cfg["train"]["linear_probe"]["epochs"],
            output_dir=Path(cfg["output"]["dir"]) / ds_key / "linear_probe",
            amp=(cfg.get("precision","fp32")!="fp32"),
            compute_loss=None,  # CE 기본이면 None
            early_stopping={"enabled": False, "patience": 10},
        )
        linear_probe_trainer.fit()

    # Finetune
    if cfg["train"]["finetune"]["enable"]:
        model = build_student_for_stage(cfg, device, "finetune")
        opt = build_optimizer(model, cfg)
        sch = build_scheduler(opt, cfg, total_epochs=cfg["train"]["finetune"]["epochs"])
        finetune_trainer = Trainer(
            model=model, optimizer=opt, scheduler=sch, schedule_mode="epoch", device=device,
            train_loader=train_loader, test_loader=test_loader,
            epochs=cfg["train"]["finetune"]["epochs"],
            output_dir=Path(cfg["output"]["dir"]) / ds_key / "finetune",
            amp=(cfg.get("precision","fp32")!="fp32"),
            compute_loss=None,
            early_stopping={"enabled": False, "patience": 10},
        )
        finetune_trainer.fit()



if __name__ == "__main__":
    main()