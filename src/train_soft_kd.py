# src/train_soft_kd.py
import os, json, math, torch
from torch.utils.data import DataLoader
from torch.optim import AdamW, SGD
from torch.optim.lr_scheduler import LambdaLR
import torch.nn.functional as F

from src.utils.trainer import Trainer
from src.utils.config import load_config_merged
from src.data.indexed import WithIndex, collate_with_index
from src.data.soft_targets import TeacherLogitsBank
from src.losses.soft_kd import SoftTargetKDLoss
from src.losses.compute_soft_kd_loss import make_compute_loss_kd
from src.model.load_teacher import load_backbone_and_classifier, load_student_imagenet


# ----------------------------
# helpers: path/template resolve
# ----------------------------
def _expand_softkd_templates(cfg):
    """dataset만으로 teacher ckpt, cache 경로 자동 확장"""
    ds = cfg["data"]["dataset"]
    # teacher paths via template
    t = cfg["model"]["teacher"]
    t.setdefault("template_backbone", "src/model/ckpts/resnet34_{dataset}_backbone.pt")
    t.setdefault("template_classifier", "src/model/ckpts/resnet34_{dataset}_classifier.pt")
    t.setdefault("arch", "resnet34")

    if "backbone_path" not in t:
        t["backbone_path"] = t["template_backbone"].format(dataset=ds)
    if "classifier_path" not in t:
        t["classifier_path"] = t["template_classifier"].format(dataset=ds)

    # student
    s = cfg["model"]["student"]
    s.setdefault("arch", "resnet18")
    s.setdefault("ckpt", "src/model/ckpts/ResNet18.pt")

    # kd cache
    kd = cfg["kd"]
    kd.setdefault("cache_logits", True)
    kd.setdefault("cache_template", "src/model/ckpts/soft_targets/{dataset}_fp16.pt")
    if kd.get("cache_logits", False) and "cache_path" not in kd:
        kd["cache_path"] = kd["cache_template"].format(dataset=ds)
    return cfg


# ----------------------------
# dataloaders (증강 없음)
# ----------------------------
def build_datasets(cfg):
    """기존 data_setup 엔트리 사용 (증강 없음 가정)"""
    from src.data.data_setup import build_dataset_pair  # (train_ds, val_ds) 반환 가정
    return build_dataset_pair(cfg["data"])

def build_dataloaders(cfg, return_indices=True):
    base_train, base_val = build_datasets(cfg)
    train_ds = WithIndex(base_train) if return_indices else base_train
    val_ds   = WithIndex(base_val)   if return_indices else base_val

    train_loader = DataLoader(
        train_ds,
        batch_size=cfg["data"]["batch_size"],
        shuffle=True,
        num_workers=cfg["data"]["num_workers"],
        pin_memory=cfg["data"].get("pin_memory", True),
        collate_fn=collate_with_index if return_indices else None,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=cfg["data"]["batch_size"],
        shuffle=False,
        num_workers=cfg["data"]["num_workers"],
        pin_memory=cfg["data"].get("pin_memory", True),
        collate_fn=collate_with_index if return_indices else None,
    )
    return train_loader, val_loader


# ----------------------------
# model builders
# ----------------------------
def build_teacher(cfg, device):
    nc = cfg["model"]["num_classes"]
    tbb = cfg["model"]["teacher"]["backbone_path"]
    tcl = cfg["model"]["teacher"]["classifier_path"]
    teacher = load_backbone_and_classifier(tbb, tcl, cfg["model"]["teacher"]["arch"], nc, map_location="cpu").to(device)
    teacher.eval()
    for p in teacher.parameters():
        p.requires_grad_(False)
    return teacher

def build_student(cfg, device):
    nc = cfg["model"]["num_classes"]
    sckpt = cfg["model"]["student"]["ckpt"]
    student = load_student_imagenet(sckpt, nc, map_location="cpu").to(device)
    return student


# ----------------------------
# optimizer / scheduler
# ----------------------------
def build_optimizer_scheduler(cfg, model):
    opt_cfg = cfg["train"]["optimizer"]
    name = opt_cfg["name"].lower()
    lr = opt_cfg["lr"]
    wd = opt_cfg.get("weight_decay", 0.05)

    if name == "adamw":
        optim = AdamW(model.parameters(), lr=lr,
                      betas=tuple(opt_cfg.get("betas", [0.9, 0.999])),
                      eps=opt_cfg.get("eps", 1e-8),
                      weight_decay=wd)
    elif name == "sgd":
        optim = SGD(model.parameters(), lr=lr,
                    momentum=opt_cfg.get("momentum", 0.9),
                    weight_decay=wd, nesterov=True)
    else:
        optim = AdamW(model.parameters(), lr=lr, weight_decay=wd)

    sch_cfg = cfg.get("train", {}).get("scheduler", {"name": "cosine", "warmup_epochs": 0, "min_lr": 0.0})
    total_epochs = cfg["train"]["linear_probe"]["epochs"] * int(cfg["train"]["linear_probe"]["enable"]) + \
                   cfg["train"]["finetune"]["epochs"] * int(cfg["train"]["finetune"]["enable"])

    warm = sch_cfg.get("warmup_epochs", 0)
    min_lr = sch_cfg.get("min_lr", 0.0)

    def lr_lambda(epoch):
        if epoch < warm:
            return (epoch + 1) / max(1, warm)
        # cosine from warm..total
        t = (epoch - warm) / max(1, total_epochs - warm)
        return min_lr / lr + (1 - min_lr / lr) * (0.5 * (1 + math.cos(math.pi * t)))

    scheduler = LambdaLR(optim, lr_lambda=lr_lambda)
    return optim, scheduler


# ----------------------------
# evaluation (val 에서는 CE 기준)
# ----------------------------
@torch.no_grad()
def evaluate(student, val_loader, device, epoch, jsonl_path):
    student.eval()
    seen, loss_sum, acc_sum = 0, 0.0, 0.0
    for batch in val_loader:
        if len(batch) == 3:
            x, y, _ = batch
        else:
            x, y = batch
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        logits = student(x)
        loss = F.cross_entropy(logits, y)
        acc = (logits.argmax(1) == y).float().mean().item()
        bs = x.size(0)
        seen += bs
        loss_sum += loss.item() * bs
        acc_sum += acc * bs

    metrics = {
        "epoch": int(epoch),
        "mode": "val",
        "val_loss_total": loss_sum / max(1, seen),
        "val_acc": acc_sum / max(1, seen),
    }
    os.makedirs(os.path.dirname(jsonl_path), exist_ok=True)
    with open(jsonl_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(metrics, ensure_ascii=False) + "\n")
    return metrics


# ----------------------------
# run
# ----------------------------
def main():
    cfg = load_config_merged()               # 루트 + 실험 override 병합
    cfg = _expand_softkd_templates(cfg)      # 템플릿 경로 채움

    device = torch.device(cfg.get("device", "cuda" if torch.cuda.is_available() else "cpu"))

    # 출력 경로: runs/<experiment.name>/<dataset>
    exp_name = cfg["experiment"]["name"]
    dataset = cfg["data"]["dataset"]
    out = os.path.join(cfg["output"]["dir"], exp_name, dataset)
    os.makedirs(os.path.join(out, "checkpoints"), exist_ok=True)
    # 병합본 저장
    with open(os.path.join(out, "config_used.yaml"), "w", encoding="utf-8") as f:
        f.write(cfg.get("_merged_yaml_text", ""))

    # 데이터/모델
    train_loader, val_loader = build_dataloaders(cfg, return_indices=True)
    teacher = build_teacher(cfg, device)
    student = build_student(cfg, device)

    # 캐시
    bank = None
    if cfg["kd"].get("cache_logits", True):
        cache_path = cfg["kd"]["cache_path"]
        bank = TeacherLogitsBank(cache_path, num_samples=len(train_loader.dataset),
                                 num_classes=cfg["model"]["num_classes"], device=device)
        if not bank.exists():
            cache_loader = DataLoader(train_loader.dataset, batch_size=cfg["data"]["batch_size"],
                                      shuffle=False, num_workers=cfg["data"]["num_workers"],
                                      pin_memory=cfg["data"].get("pin_memory", True),
                                      collate_fn=collate_with_index)
            bank.build(teacher, cache_loader)

    # 손실/콜백/트레이너
    kd_loss = SoftTargetKDLoss(alpha=cfg["kd"]["alpha"], temperature=cfg["kd"]["temperature"])
    compute_loss = make_compute_loss_kd(student, kd_loss, teacher=None if bank else teacher, bank=bank)

    optim, sched = build_optimizer_scheduler(cfg, student)
    trainer = Trainer(model=student, optim=optim, sched=sched,
                      amp=(cfg.get("precision", "fp32") == "fp16"),
                      out_dir=out)

    best = -1.0
    ep_total = 0

    # Linear Probe (백본 freeze)
    if cfg["train"]["linear_probe"]["enable"]:
        for p in student.parameters():
            p.requires_grad_(False)
        for p in student.fc.parameters():
            p.requires_grad_(True)
        for epoch in range(cfg["train"]["linear_probe"]["epochs"]):
            trainer._epoch_loop(train_loader, "train", compute_loss, device, ep_total,
                                os.path.join(out, "metrics_per_epoch.jsonl"))
            val_m = evaluate(student, val_loader, device, ep_total,
                             os.path.join(out, "metrics_per_epoch.jsonl"))
            top1 = val_m.get("val_acc", 0.0)
            if top1 > best:
                best = top1
                torch.save(student.state_dict(), os.path.join(out, "checkpoints", "best.pt"))
                with open(os.path.join(out, "best_metrics.json"), "w") as f:
                    json.dump({"epoch": ep_total, "val_acc": best}, f)
            torch.save(student.state_dict(), os.path.join(out, "checkpoints", "last.pt"))
            if sched is not None:
                sched.step()
            ep_total += 1

    # Fine-tune (전체 unfreeze)
    if cfg["train"]["finetune"]["enable"]:
        for p in student.parameters():
            p.requires_grad_(True)
        for epoch in range(cfg["train"]["finetune"]["epochs"]):
            trainer._epoch_loop(train_loader, "train", compute_loss, device, ep_total,
                                os.path.join(out, "metrics_per_epoch.jsonl"))
            val_m = evaluate(student, val_loader, device, ep_total,
                             os.path.join(out, "metrics_per_epoch.jsonl"))
            top1 = val_m.get("val_acc", 0.0)
            if top1 > best:
                best = top1
                torch.save(student.state_dict(), os.path.join(out, "checkpoints", "best.pt"))
                with open(os.path.join(out, "best_metrics.json"), "w") as f:
                    json.dump({"epoch": ep_total, "val_acc": best}, f)
            torch.save(student.state_dict(), os.path.join(out, "checkpoints", "last.pt"))
            if sched is not None:
                sched.step()
            ep_total += 1


if __name__ == "__main__":
    main()
