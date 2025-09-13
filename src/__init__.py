# Re-export the public API from the submodules for concise imports in notebooks.

from .config import DataCfg, TrainCfg
from .utils import set_seed, get_device
from .data import build_dataloaders
from .models import (
    build_custom_cnn,
    build_resnet18,
    build_vit_scratch,
    build_vit_pretrained,
)

# fouad start change: add alias to match notebook import names
build_vit_b16_pretrained = build_vit_pretrained
# fouad end change

from .optimizers import SAM  # useful if you want to experiment directly
from .train import train_model
from .metrics import (
    evaluate_probs_targets, evaluate, choose_threshold_by_min_recall,
    summarize_at_threshold
)
from .plotting import plot_curve, plot_two_series, diag_plots
from .registry import add_result, result_registry

__all__ = [
    "DataCfg", "TrainCfg",
    "set_seed", "get_device",
    "build_dataloaders",
    "build_custom_cnn", "build_resnet18",
    "build_vit_scratch", "build_vit_pretrained", "build_vit_b16_pretrained",
    "SAM",
    "train_model",
    "evaluate_probs_targets", "evaluate", "choose_threshold_by_min_recall", "summarize_at_threshold",
    "plot_curve", "plot_two_series", "diag_plots",
    "add_result", "result_registry",
]
