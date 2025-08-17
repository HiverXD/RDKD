# src/train_soft_kd.py
import os, json, math, torch
from torch.utils.data import DataLoader
from torch.optim import AdamW, SGD
from torch.optim.lr_scheduler import LambdaLR
import torch.nn.functional as F

from src.utils.config import load_config_from_cli, expand_softkd_templates
from src.utils.trainer import Trainer
from src.data.indexed import WithIndex, collate_with_index
from src.data.soft_targets import TeacherLogitsBank
from src.losses.soft_kd import SoftTargetKDLoss
from src.losses.compute_soft_kd_loss import make_compute_loss_kd
from src.model.load_teacher import load_backbone_and_classifier, load_student_imagenet

# ----------------------------
# dataloaders (증강 없음)
# ----------------------------
def build_datasets(cfg):
    """
    returns: base_train_ds, base_val_ds (torch.utils.data.Dataset)
    data_setup.loaders(...)는 {name: DataLoader} 딕셔너리를 반환하므로,
    cfg["data"]["dataset"] 키로 해당 DataLoader를 선택 → .dataset 추출.
    """
    from src.data.data_setup import loaders
    dl_train_dict, dl_val_dict = loaders(
        batch_size=cfg["data"]["batch_size"],
        root=cfg["data"]["root"],
        num_workers=cfg["data"]["num_workers"],
        pin_memory=cfg["data"].get("pin_memory", True),
        download=cfg["data"].get("download", False),
    )
    ds_name = cfg["data"]["dataset"]
    if ds_name not in dl_train_dict or ds_name not in dl_val_dict:
        raise KeyError(f"Dataset '{ds_name}' not found in data_setup.loaders(...) return keys: "
                       f"{list(dl_train_dict.keys())}")
    base_train = dl_train_dict[ds_name].dataset
    base_val   = dl_val_dict[ds_name].dataset
    return base_train, base_val

def build_dataloaders(cfg, return_indices=True):
    base_train, base_val = build_datasets(cfg)
    if return_indices:
        train_ds = WithIndex(base_train)
        val_ds   = WithIndex(base_val)
        collate  = collate_with_index
    else:
        train_ds, val_ds, collate = base_train, base_val, None

    train_loader = DataLoader(
        train_ds,
        batch_size=cfg["data"]["batch_size"],
        shuffle=True,
        num_workers=cfg["data"]["num_workers"],
        pin_memory=cfg["data"].get("pin_memory", True),
        persistent_workers=(cfg["data"]["num_workers"] > 0),
        collate_fn=collate,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=cfg["data"]["batch_size"],
        shuffle=False,
        num_workers=cfg["data"]["num_workers"],
        pin_memory=cfg["data"].get("pin_memory", True),
        persistent_workers=(cfg["data"]["num_workers"] > 0),
        collate_fn=collate,
    )
    return train_loader, val_loader


# ----------------------------
# model builders
# ----------------------------
def build_teacher(cfg, device):
    nc  = cfg["model"]["num_classes"]
    arc = cfg["model"]["teacher"]["arch"]
    tbb = cfg["model"]["teacher"]["backbone_path"]
    tcl = cfg["model"]["teacher"]["classifier_path"]
    teacher = load_backbone_and_classifier(tbb, tcl, arc, nc, map_location="cpu").to(device)
    teacher.eval()
    for p in teacher.parameters():
        p.requires_grad_(False)
    return teacher

def build_student(cfg, device):
    nc   = cfg["model"]["num_classes"]
    sckp = cfg["model"]["student"]["ckpt"]
    student = load_student_imagenet(sckp, nc, map_location="cpu").to(device)
    return student

# ----------------------------
# optimizer / scheduler (AdamW 기본)
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
    total_epochs = (cfg["train"]["linear_probe"]["epochs"] if cfg["train"]["linear_probe"]["enable"] else 0) + \
                   (cfg["train"]["finetune"]["epochs"] if cfg["train"]["finetune"]["enable"] else 0)

    warm = sch_cfg.get("warmup_epochs", 0)
    min_lr = sch_cfg.get("min_lr", 0.0)

    def lr_lambda(epoch):
        if epoch < warm:
            return (epoch + 1) / max(1, warm)
        # cosine from warm..total
        if total_epochs <= warm:
            return min_lr / lr
        t = (epoch - warm) / max(1, total_epochs - warm)
        return min_lr / lr + (1 - min_lr / lr) * (0.5 * (1 + math.cos(math.pi * t)))

    scheduler = LambdaLR(optim, lr_lambda=lr_lambda)
    return optim, scheduler

# ----------------------------
# evaluation (val에서는 CE 기준)
# ----------------------------
@torch.no_grad()
def evaluate(student, val_loader, device, epoch, jsonl_path):
    student.eval()
    seen, loss_sum, acc_sum = 0, 0.0, 0.0
    for batch in val_loader:
        if len(batch) == 3: x, y, _ = batch
        else: x, y = batch
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        logits = student(x)
        loss = F.cross_entropy(logits, y)
        acc = (logits.argmax(1) == y).float().mean().item()
        bs = x.size(0); seen += bs
        loss_sum += loss.item() * bs
        acc_sum  += acc * bs

    metrics = {"epoch": int(epoch), "mode": "val",
               "val_loss_total": loss_sum / max(1, seen),
               "val_acc": acc_sum / max(1, seen)}
    os.makedirs(os.path.dirname(jsonl_path), exist_ok=True)
    with open(jsonl_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(metrics, ensure_ascii=False) + "\n")
    return metrics

# ----------------------------
# main
# ----------------------------
def main():
    cfg = load_config_from_cli()         # <-- 변경: load_config_merged -> load_config_from_cli
    cfg = expand_softkd_templates(cfg)   # 템플릿 경로 자동 확장

    device = torch.device(cfg.get("device", "cuda" if torch.cuda.is_available() else "cpu"))

    # 출력 경로: runs/<experiment.name>/<dataset>
    exp_name = cfg["experiment"]["name"]
    dataset  = cfg["data"]["dataset"]
    out = os.path.join(cfg["output"]["dir"], exp_name, dataset)
    os.makedirs(os.path.join(out, "checkpoints"), exist_ok=True)

    # config_used 저장(예쁘게)
    with open(os.path.join(out, "config_used.yaml"), "w", encoding="utf-8") as f:
        # 원본 pretty printer가 있으면 사용, 없으면 json으로라도 남김
        try:
            from src.utils.config import pretty
            f.write(pretty(cfg))
        except Exception:
            f.write(json.dumps(cfg, ensure_ascii=False, indent=2))

    # 데이터/모델
    train_loader, val_loader = build_dataloaders(cfg, return_indices=True)
    teacher = build_teacher(cfg, device)
    student = build_student(cfg, device)

    # 캐시
    bank = None
    if cfg["kd"].get("cache_logits", True):
        bank = TeacherLogitsBank(cfg["kd"]["cache_path"],
                                 num_samples=len(train_loader.dataset),
                                 num_classes=cfg["model"]["num_classes"],
                                 device=device)
        if not bank.exists():
            cache_loader = DataLoader(train_loader.dataset,
                                      batch_size=cfg["data"]["batch_size"],
                                      shuffle=False,
                                      num_workers=cfg["data"]["num_workers"],
                                      pin_memory=cfg["data"].get("pin_memory", True),
                                      collate_fn=collate_with_index)
            bank.build(teacher, cache_loader)

    # 손실/콜백/트레이너
    kd_loss = SoftTargetKDLoss(alpha=cfg["kd"]["alpha"], temperature=cfg["kd"]["temperature"])
    compute_loss = make_compute_loss_kd(student, kd_loss, teacher=None if bank else teacher, bank=bank)

    optim, sched = build_optimizer_scheduler(cfg, student)
    trainer = Trainer(model=student, optimizer=optim, scheduler=sched,
                      amp=(cfg.get("precision", "fp32") == "fp16"),
                      output_dir=out)

    best = -1.0
    ep_total = 0

    # Linear Probe (백본 freeze)
    if cfg["train"]["linear_probe"]["enable"]:
        for p in student.parameters(): p.requires_grad_(False)
        for p in student.fc.parameters(): p.requires_grad_(True)
        for _ in range(cfg["train"]["linear_probe"]["epochs"]):
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
            if sched is not None: sched.step()
            ep_total += 1

    # Fine-tune (전체 unfreeze)
    if cfg["train"]["finetune"]["enable"]:
        for p in student.parameters(): p.requires_grad_(True)
        for _ in range(cfg["train"]["finetune"]["epochs"]):
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
            if sched is not None: sched.step()
            ep_total += 1

if __name__ == "__main__":
    main()
