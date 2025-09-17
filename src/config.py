from __future__ import annotations
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

@dataclass
class DataCfg:
    dataset_dir: Path
    image_size: int = 224
    batch_size: int = 32
    num_workers: int = 0
    pin_memory: bool = True

    # fouad start change: add options used by notebooks
    use_stratified_val: bool = True
    stratified_val_size: float = 0.15
    val_ratio: Optional[float] = None          # alias for stratified_val_size
    compute_norm_stats: bool = False           # compute mean/std from train/
    gray_to_rgb: bool = True                   # convert grayscale images to 3ch
    balance_sampler: bool = False              # WeightedRandomSampler on training set
    # fouad end change

    def __post_init__(self):
        # fouad start change: honor alias and make MPS pin_memory-safe
        if self.val_ratio is not None:
            self.stratified_val_size = float(self.val_ratio)
        try:
            import torch
            if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
                self.pin_memory = False
        except Exception:
            pass
        # fouad end change

    # fouad change to support 1 channel start
    @property
    def in_chans(self) -> int:
        """
        Helper for model builders: returns 3 if we convert to RGB, else 1.
        Keeps data & model channels aligned.
        """
        return 3 if self.gray_to_rgb else 1
    # fouad change to support 1 channel end


@dataclass
class TrainCfg:
    # Common knobs
    epochs: int = 90
    patience: int = 10
    monitor: str = "va_auroc"  # 'va_auroc' (stable for screening), or 'va_f1', 'va_loss'
    base_lr: float = 3e-4       # For ViT/AdamW; ResNet scratch uses SGD defaults in optimizer builder
    weight_decay: float = 0.3   # ViT scratch WD ~0.3; ResNet scratch ~1e-3 (handled in builder)
    label_smoothing: float = 0.0
    clip_grad_norm: float = 1.0
    use_amp: bool = False       # AMP enabled only on CUDA internally

    # SAM
    use_sam: bool = True
    sam_rho: float = 0.2        # ViT likes larger rho (~0.2); ResNet uses smaller (~0.02–0.05)

    # LR schedule
    use_cosine: bool = True
    warmup_epochs: int = 3
    min_lr_mult: float = 0.01

    # Screening setting: pick threshold on val to hit at least target recall
    target_recall: float = 0.95

    # Pretraining (layer-wise LR decay for ViT)
    llrd: bool = False
    head_lr_mult: float = 10.0

    # Optional: path to Google .npz ViT weights (if not using timm)
    vit_npz_path: Path | None = None