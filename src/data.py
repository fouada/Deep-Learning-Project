"""
Data loading utilities.
- Always convert grayscale to RGB (3 channels) before normalization.
- Optional stratified re-split of the tiny official val/ to stabilize validation.
"""
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple
import numpy as np, torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T
from torchvision import datasets
from sklearn.model_selection import StratifiedShuffleSplit

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

def build_transforms(image_size: int, inception_style: bool = True):
    if inception_style:
        train_tf = T.Compose([
            T.RandomResizedCrop(image_size, scale=(0.08, 1.0)),
            T.RandomHorizontalFlip(0.5),
            T.Grayscale(3),
            T.ToTensor(),
            T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ])
    else:
        train_tf = T.Compose([
            T.Resize(256), T.CenterCrop(image_size),
            T.RandomHorizontalFlip(0.5),
            T.Grayscale(3),
            T.ToTensor(),
            T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ])
    eval_tf = T.Compose([
        T.Resize(256), T.CenterCrop(image_size),
        T.Grayscale(3),
        T.ToTensor(),
        T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])
    return train_tf, eval_tf

def build_dataloaders(dataset_dir: Path, image_size: int, batch_size: int, num_workers: int,
                      pin_memory: bool, use_stratified_val: bool, stratified_val_size: float):
    train_tf, eval_tf = build_transforms(image_size, inception_style=True)

    root = Path(dataset_dir)
    ds_train_raw = datasets.ImageFolder(root / "train", transform=train_tf)
    classes = ds_train_raw.classes
    ds_val_raw = datasets.ImageFolder(root / "val", transform=eval_tf) if (root / "val").exists() else None
    ds_test = datasets.ImageFolder(root / "test", transform=eval_tf)

    if use_stratified_val and ds_val_raw is not None:
        all_items = list(ds_train_raw.imgs) + list(ds_val_raw.imgs)
    else:
        all_items = list(ds_train_raw.imgs)
    paths = [p for p, _ in all_items]
    labels = [y for _, y in all_items]

    if use_stratified_val:
        sss = StratifiedShuffleSplit(n_splits=1, test_size=stratified_val_size, random_state=42)
        idx_tr, idx_va = next(sss.split(np.zeros(len(labels)), labels))
        ds_train = FlatImageDataset([paths[i] for i in idx_tr], [labels[i] for i in idx_tr], train_tf)
        ds_val   = FlatImageDataset([paths[i] for i in idx_va], [labels[i] for i in idx_va], eval_tf)
    else:
        ds_train = ds_train_raw
        ds_val   = ds_val_raw if ds_val_raw is not None else ds_test

    # Class counts for weights
    if hasattr(ds_train, "samples"):
        train_counts = np.bincount([y for _, y in ds_train.samples]).astype(int).tolist()
    else:
        train_counts = np.bincount([ds_train[i][1] for i in range(len(ds_train))]).astype(int).tolist()

    # MPS does not benefit from pin_memory
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        pin_memory = False

    train_loader = DataLoader(ds_train, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, pin_memory=pin_memory)
    val_loader   = DataLoader(ds_val,   batch_size=batch_size, shuffle=False,
                              num_workers=num_workers, pin_memory=pin_memory)
    test_loader  = DataLoader(ds_test,  batch_size=batch_size, shuffle=False,
                              num_workers=num_workers, pin_memory=pin_memory)
    info = {
        "classes": classes,
        "train_counts": train_counts,
        "n_train": len(ds_train),
        "n_val": len(ds_val),
        "n_test": len(ds_test),
    }
    return train_loader, val_loader, test_loader, info
