# src/data/data_setup.py
from torchvision import datasets, transforms
from torchvision.transforms import AutoAugment, AutoAugmentPolicy, RandAugment
from torch.utils.data import DataLoader
import torch
import math
from typing import Dict, Literal, Tuple

AugName = Literal["none", "light", "autoaugment", "randaugment"]

def _seed_worker(worker_id):
    # worker별 난수 시드 고정 (재현성)
    worker_seed = torch.initial_seed() % 2**32
    import numpy as np, random
    np.random.seed(worker_seed)
    random.seed(worker_seed)

def _build_transforms(dataset: str,
                      train: bool,
                      aug: AugName = "light",
                      rand_n: int = 2,
                      rand_m: int = 9,
                      random_erasing_p: float = 0.0) -> transforms.Compose:
    """
    dataset: one of {"mnist","fashion_mnist","svhn","cifar10","cifar100"}
    aug: "none" | "light" | "autoaugment" | "randaugment"
    """
    # --- stats ---
    mnist_mean, mnist_std = 0.1307, 0.3081
    fmnist_mean, fmnist_std = 0.2860, 0.3530
    svhn_mean  = (0.4377, 0.4438, 0.4728)
    svhn_std   = (0.1980, 0.2010, 0.1970)
    cifar10_mean = (0.4914, 0.4822, 0.4465)
    cifar10_std  = (0.2470, 0.2435, 0.2616)
    cifar100_mean = (0.5071, 0.4867, 0.4408)
    cifar100_std  = (0.2675, 0.2565, 0.2761)

    # dataset별 정규화 설정
    if dataset == "mnist":
        mean, std = (mnist_mean,)*3, (mnist_std,)*3
        to3ch = transforms.Grayscale(num_output_channels=3)
    elif dataset == "fashion_mnist":
        mean, std = (fmnist_mean,)*3, (fmnist_std,)*3
        to3ch = transforms.Grayscale(num_output_channels=3)
    elif dataset == "svhn":
        mean, std = svhn_mean, svhn_std
        to3ch = None
    elif dataset == "cifar10":
        mean, std = cifar10_mean, cifar10_std
        to3ch = None
    elif dataset == "cifar100":
        mean, std = cifar100_mean, cifar100_std
        to3ch = None
    else:
        raise ValueError(f"Unknown dataset: {dataset}")

    # --- 공통 후처리 ---
    post = [transforms.ToTensor(), transforms.Normalize(mean, std)]
    if train and random_erasing_p > 0:
        post.append(transforms.RandomErasing(p=random_erasing_p))

    # --- dataset별 학습 증강 ---
    if not train:
        # test/val: 정규화만
        return transforms.Compose(([to3ch] if to3ch else []) + post)

    aug_list = []
    if dataset in {"cifar10", "cifar100"}:
        # 기본 강건화
        aug_list += [transforms.RandomCrop(32, padding=4),
                     transforms.RandomHorizontalFlip(p=0.5)]
        if aug == "autoaugment":
            aug_list.append(AutoAugment(AutoAugmentPolicy.CIFAR10))
        elif aug == "randaugment":
            # torchvision 0.13+ : (N, M)
            aug_list.append(RandAugment(num_ops=rand_n, magnitude=rand_m))
        elif aug == "light":
            # 가벼운 색상 변형 (과하지 않게)
            aug_list.append(transforms.ColorJitter(0.2, 0.2, 0.2, 0.1))
        # "none"이면 아무 것도 추가하지 않음

    elif dataset == "svhn":
        # 숫자 왜곡 방지: flip/큰 rotation 피함
        aug_list += [transforms.RandomCrop(32, padding=4)]
        if aug in {"light", "autoaugment", "randaugment"}:
            # 약한 밝기/대비 정도만
            aug_list.append(transforms.ColorJitter(brightness=0.2, contrast=0.2))
        # flip/rotation은 기본 미적용

    elif dataset in {"mnist", "fashion_mnist"}:
        # 색상 증강 X, 기하학만 가볍게
        # 주의: Grayscale은 PIL 단계에서 적용되므로 augment 전에 위치
        aug_list += [
            transforms.RandomCrop(32, padding=4),
            transforms.RandomAffine(
                degrees=10, translate=(0.05, 0.05), scale=(0.95, 1.05), fill=0
            ),
        ]
        # autoaugment/randaugment는 grayscale 특성상 생략 or 향후 별도 정책

    pipeline = []
    # Grayscale→3ch는 PIL 단계에서 적용(augment 전에/후에 모두 가능하나,
    # 색상증강이 없으므로 MNIST류는 먼저 적용해도 무방)
    if to3ch:
        pipeline.append(to3ch)
    pipeline += aug_list
    pipeline += post
    return transforms.Compose(pipeline)

def loaders(batch_size=128,
            root="dataset",
            num_workers=4,
            pin_memory=True,
            download=True,
            seed: int = 42,
            aug_cfg: Dict[str, Dict] = None
            ) -> Tuple[Dict, Dict]:
    """
    aug_cfg 예시:
    {
      "cifar10":   {"aug": "autoaugment", "random_erasing_p": 0.1},
      "cifar100":  {"aug": "randaugment", "rand_n": 2, "rand_m": 9, "random_erasing_p": 0.1},
      "svhn":      {"aug": "light", "random_erasing_p": 0.05},
      "mnist":     {"aug": "light"},
      "fashion_mnist": {"aug": "light"}
    }
    """
    default_aug = {"aug": "light", "random_erasing_p": 0.0}
    aug_cfg = aug_cfg or {}

    # transforms 구성
    tf_train = {
        k: _build_transforms(k, train=True,  **{**default_aug, **aug_cfg.get(k, {})})
        for k in ["mnist","fashion_mnist","svhn","cifar10","cifar100"]
    }
    tf_test = {
        k: _build_transforms(k, train=False, **default_aug)
        for k in ["mnist","fashion_mnist","svhn","cifar10","cifar100"]
    }

    # datasets
    train = {
        "mnist":        datasets.MNIST(root=root, train=True,  download=download, transform=tf_train["mnist"]),
        "fashion_mnist":datasets.FashionMNIST(root=root, train=True,  download=download, transform=tf_train["fashion_mnist"]),
        "svhn":         datasets.SVHN(root=root, split="train", download=download, transform=tf_train["svhn"]),
        "cifar10":      datasets.CIFAR10(root=root, train=True,  download=download, transform=tf_train["cifar10"]),
        "cifar100":     datasets.CIFAR100(root=root, train=True,  download=download, transform=tf_train["cifar100"]),
    }
    test = {
        "mnist":        datasets.MNIST(root=root, train=False, download=download, transform=tf_test["mnist"]),
        "fashion_mnist":datasets.FashionMNIST(root=root, train=False, download=download, transform=tf_test["fashion_mnist"]),
        "svhn":         datasets.SVHN(root=root, split="test",  download=download, transform=tf_test["svhn"]),
        "cifar10":      datasets.CIFAR10(root=root, train=False, download=download, transform=tf_test["cifar10"]),
        "cifar100":     datasets.CIFAR100(root=root, train=False, download=download, transform=tf_test["cifar100"]),
    }

    # 재현성: DataLoader에 generator/worker_init_fn 지정
    g = torch.Generator()
    g.manual_seed(seed)

    train_loaders = {k: DataLoader(v, batch_size=batch_size, shuffle=True,
                                   num_workers=num_workers, pin_memory=pin_memory,
                                   persistent_workers=(num_workers > 0),
                                   worker_init_fn=_seed_worker,
                                   generator=g)
                     for k, v in train.items()}
    test_loaders  = {k: DataLoader(v, batch_size=batch_size, shuffle=False,
                                   num_workers=num_workers, pin_memory=pin_memory,
                                   persistent_workers=(num_workers > 0),
                                   worker_init_fn=_seed_worker,
                                   generator=g)
                     for k, v in test.items()}
    return train_loaders, test_loaders

if __name__ == "__main__":
    # 예: CIFAR-10 AutoAugment, CIFAR-100 RandAugment, SVHN 약한 증강 + RandomErasing 0.05
    aug_cfg = {
        "cifar10": {"aug": "autoaugment", "random_erasing_p": 0.1},
        "cifar100": {"aug": "randaugment", "rand_n": 2, "rand_m": 9, "random_erasing_p": 0.1},
        "svhn": {"aug": "light", "random_erasing_p": 0.05},
        "mnist": {"aug": "light"},
        "fashion_mnist": {"aug": "light"},
    }
    loaders(aug_cfg=aug_cfg)
