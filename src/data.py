"""
Data loading utilities.
- Optional stratified re-split (train -> train/val) to stabilize validation.
- Optional class-balanced sampler for imbalanced training.
- Optional grayscale->RGB conversion and mean/std computation.
"""
from __future__ import annotations
from typing import List, Tuple, Optional, Dict
from dataclasses import dataclass
from pathlib import Path

import random, numpy as np, torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
import torchvision.transforms as T
from torchvision import datasets
from sklearn.model_selection import StratifiedShuffleSplit

from .config import DataCfg

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]


class FlatImageDataset(Dataset):
    """Picklable dataset holding lists of (path, label)."""
    def __init__(self, paths: List[str], labels: List[int], transform):
        self.paths = list(paths)
        self.labels = list(labels)
        self.transform = transform
        self.samples = list(zip(self.paths, self.labels))
        self.classes = sorted(set(labels))
    def __len__(self): return len(self.paths)
    def __getitem__(self, i: int):
        import torchvision.io as io
        from torchvision.transforms import functional as TF
        p, y = self.paths[i], int(self.labels[i])
        img = io.read_image(p)  # [C,H,W], uint8
        img = TF.to_pil_image(img)
        img = self.transform(img)
        return img, y


def _compute_mean_std(paths: List[str], image_size: int, max_samples: int = 600, gray_to_rgb: bool = True):
    """Compute mean/std from a subset of training images (channel-wise)."""
    from PIL import Image
    import numpy as np
    random.seed(42)
    files = list(paths)
    random.shuffle(files)
    files = files[:max_samples]
    acc = []
    for f in files:
        try:
            im = Image.open(f)
            if gray_to_rgb:
                im = im.convert("L").resize((image_size, image_size))
                arr = np.array(im, dtype=np.float32) / 255.0
                arr = np.stack([arr, arr, arr], axis=0)  # [3,H,W]
            else:
                im = im.convert("RGB").resize((image_size, image_size))
                arr = (np.array(im, dtype=np.float32) / 255.0).transpose(2,0,1)  # [3,H,W]
            acc.append(arr)
        except Exception:
            pass
    if not acc:
        return IMAGENET_MEAN, IMAGENET_STD
    arr = np.stack(acc, axis=0)  # [N,3,H,W]
    mean = arr.mean(axis=(0,2,3)).tolist()
    std  = arr.std(axis=(0,2,3)).tolist()
    return mean, std


def _build_transforms(image_size: int, mean, std, gray_to_rgb: bool, inception_style: bool = True):
    rgb_or_gray = [T.Grayscale(3)] if gray_to_rgb else []
    if inception_style:
        train_tf = T.Compose([
            T.RandomResizedCrop(image_size, scale=(0.08, 1.0)),
            T.RandomHorizontalFlip(0.5),
            *rgb_or_gray,
            T.ToTensor(),
            T.Normalize(mean=mean, std=std),
        ])
    else:
        train_tf = T.Compose([
            T.Resize(256), T.CenterCrop(image_size),
            T.RandomHorizontalFlip(0.5),
            *rgb_or_gray,
            T.ToTensor(),
            T.Normalize(mean=mean, std=std),
        ])
    eval_tf = T.Compose([
        T.Resize(256), T.CenterCrop(image_size),
        *rgb_or_gray,
        T.ToTensor(),
        T.Normalize(mean=mean, std=std),
    ])
    return train_tf, eval_tf


def _sampler_if_needed(labels: List[int]) -> WeightedRandomSampler:
    """Return a WeightedRandomSampler that samples inverse-frequency per class."""
    counts = np.bincount(labels)
    weights_per_class = 1.0 / (counts + 1e-12)
    w = weights_per_class[np.array(labels)]
    return WeightedRandomSampler(weights=torch.as_tensor(w, dtype=torch.double),
                                 num_samples=len(labels), replacement=True)


def build_dataloaders(cfg: DataCfg, device: Optional[torch.device] = None):
    """
    Notebook-friendly signature:
        train_loader, val_loader, test_loader, info = build_dataloaders(cfg, device)
    """
    root = Path(cfg.dataset_dir)

    # Collect raw sets
    ds_train_raw = datasets.ImageFolder(root / "train")
    classes = ds_train_raw.classes
    ds_val_raw = datasets.ImageFolder(root / "val") if (root / "val").exists() else None
    ds_test = datasets.ImageFolder(root / "test")

    # Merge train + official val (tiny) if requested, then stratified split
    if cfg.use_stratified_val and ds_val_raw is not None:
        all_items = list(ds_train_raw.imgs) + list(ds_val_raw.imgs)
    else:
        all_items = list(ds_train_raw.imgs)

    paths = [p for p, _ in all_items]
    labels = [y for _, y in all_items]

    # Mean/std (optional; default to ImageNet)
    if cfg.compute_norm_stats:
        mean, std = _compute_mean_std(
            [p for p, _ in ds_train_raw.imgs], image_size=cfg.image_size, gray_to_rgb=cfg.gray_to_rgb
        )
    else:
        mean, std = IMAGENET_MEAN, IMAGENET_STD

    train_tf, eval_tf = _build_transforms(cfg.image_size, mean, std, cfg.gray_to_rgb, inception_style=True)

    # Stratified split or keep original train/val
    if cfg.use_stratified_val:
        sss = StratifiedShuffleSplit(n_splits=1, test_size=cfg.stratified_val_size, random_state=42)
        idx_tr, idx_va = next(sss.split(np.zeros(len(labels)), labels))
        ds_train = FlatImageDataset([paths[i] for i in idx_tr], [labels[i] for i in idx_tr], train_tf)
        ds_val   = FlatImageDataset([paths[i] for i in idx_va], [labels[i] for i in idx_va], eval_tf)
    else:
        ds_train = datasets.ImageFolder(root / "train", transform=train_tf)
        ds_val   = datasets.ImageFolder(root / "val",   transform=eval_tf) if (root / "val").exists() else ds_test

    # Apply transforms to the fixed test set
    ds_test.transform = eval_tf

    # Training class counts (for info / weighting)
    train_counts = np.bincount([y for _, y in ds_train.samples]).astype(int).tolist()

    # Dataloaders
    if cfg.balance_sampler:
        sampler = _sampler_if_needed([y for _, y in ds_train.samples])
        train_loader = DataLoader(ds_train, batch_size=cfg.batch_size, sampler=sampler,
                                  num_workers=cfg.num_workers, pin_memory=cfg.pin_memory)
    else:
        train_loader = DataLoader(ds_train, batch_size=cfg.batch_size, shuffle=True,
                                  num_workers=cfg.num_workers, pin_memory=cfg.pin_memory)

    val_loader   = DataLoader(ds_val,   batch_size=cfg.batch_size, shuffle=False,
                              num_workers=cfg.num_workers, pin_memory=cfg.pin_memory)
    test_loader  = DataLoader(ds_test,  batch_size=cfg.batch_size, shuffle=False,
                              num_workers=cfg.num_workers, pin_memory=cfg.pin_memory)

    info = {
        "classes": classes,
        "class_to_idx": ds_train_raw.class_to_idx,
        "train_counts": train_counts,
        "n_train": len(ds_train),
        "n_val": len(ds_val),
        "n_test": len(ds_test),
        "mean": mean,
        "std": std,
    }
    return train_loader, val_loader, test_loader, info
