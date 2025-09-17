# Pneumonia Classification from Chest X‑Rays (CNN vs. ViT, Scratch & Pretrained; 1‑ch and 3‑ch)

> **Goal.** Build and compare a Convolutional Neural Network (CNN), a ResNet‑18, and a Vision Transformer (ViT) for **binary chest X‑ray classification** (Normal vs. Pneumonia) on the public dataset from Kaggle. The project includes **scratch** and **pretrained** variants, with **1‑channel (true grayscale)** and **3‑channel (gray→RGB)** pipelines, and evaluates the impact of **Sharpness‑Aware Minimization (SAM)** on training from scratch.

Dataset: <https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia>

---

## Contents

- [Dataset & Folders](#dataset--folders)
- [Project Layout](#project-layout)
- [How to Run](#how-to-run)
- [Models Implemented](#models-implemented)
- [Training Strategy](#training-strategy)
- [Evaluation Protocol](#evaluation-protocol)
- [Hyperparameters & What They Do](#hyperparameters--what-they-do)
- [Key Findings (What the notebooks show)](#key-findings-what-the-notebooks-show)
- [Reproducibility](#reproducibility)
- [References](#references)

---

## Dataset & Folders

We use the **Chest X‑Ray Images (Pneumonia)** dataset by Kermany et al., hosted on Kaggle. The dataset already comes with a split: `train/`, `val/`, `test/`.  
**Important**: the **`test/` set is never touched during training or model selection**. All reporting in the notebooks’ “Final Results” cells is done on the untouched `test/` images.

A typical on‑disk layout:
```
data/chest_xray/
    train/
        NORMAL/
        PNEUMONIA/
    val/
        NORMAL/
        PNEUMONIA/
    test/
        NORMAL/
        PNEUMONIA/
```

You may optionally **re-split train+val** inside the notebooks using a **stratified** split to stabilize validation. The original `test/` is always preserved.

---

## Project Layout

```
src/
  __init__.py                 # unified API (DataCfg, TrainCfg, builders, training, metrics, plotting…)
  config.py                   # configuration dataclasses
  data.py                     # dataloaders, transforms (1‑ch and 3‑ch), stratified split, class weighting
  models.py                   # CustomCNN, ResNet‑18, ViT (scratch), ViT (via timm or JAX npz)
  optimizers.py               # SAM wrapper, cosine w/ warmup
  train.py                    # training loop, early stopping, AMP guards, metric tracking
  metrics.py                  # AUROC + standard metrics, threshold selection (recall ≥ target)
  plotting.py                 # training curves, ROC/PR, confusion matrix
  registry.py                 # light results registry to collect summaries
  vit_npz_loader.py           # (optional) load Google ViT‑B/16 npz weights into our PyTorch ViT
notebooks/
  CustomCNN_ResNet18_VIT_No-SAM-Scratch.ipynb
  CustomCNN_ResNet18_VIT_No-SAM-Scratch_1Channel.ipynb
  CustomCNN_ResNet18_VIT_With-SAM-Scratch.ipynb
  CustomCNN_ResNet18_VIT_With-SAM-Scratch_1Channel.ipynb
  ResNet18_ViT-B-16_No-SAM-Pretrained.ipynb
  ResNet18_ViT-B-16_With-SAM-Pretrained.ipynb
```

> **Per‑notebook presentation of results.** Each notebook ends with **two tables** that summarize all models trained **in that notebook** under two decision criteria:
> 1) **Recall ≥ 95%** – we **select the probability threshold** that achieves at least 0.95 recall on validation, then report metrics on the test set at that threshold.  
> 2) **Fixed threshold = 0.50** – we report test metrics at 0.5 to enable apples‑to‑apples comparison across models.
>
> The landing page (this README) summarizes the methodology and what you should expect to see; the **actual numbers** appear at the end of each notebook.

---

## How to Run

1. **Install** dependencies (PyTorch, torchvision, timm, scikit‑learn, matplotlib).  
2. **Set** the dataset root in the top cells of a notebook (e.g., `dataset_dir = "data/chest_xray"`).  
3. **Pick the pipeline**:  
   - **1‑channel** (true grayscale): set `DataCfg(gray_to_rgb=False)` and build models with `in_chans=1`.  
   - **3‑channel** (gray→RGB replication for compatibility with ImageNet‑style norms and pretrained models): set `DataCfg(gray_to_rgb=True)` and build models with `in_chans=3`.
4. **Run** the notebook end‑to‑end. Each model block trains, plots curves, evaluates on `val/` & `test/`, selects the threshold for recall≥95% and prints **both** result tables at the end.

---

## Models Implemented

- **CustomCNN (scratch)** – small baseline (≈ **0.48M** params). Stem + 3 stages, global‑avg‑pool, linear head. Good for ablations and sanity checks.
- **ResNet‑18 (scratch / pretrained)** – standard architecture (≈ **11.2M** params). Scratch uses SGD; pretrained variant uses torchvision weights and fine‑tunes the head.
- **Vision Transformer (ViT, scratch / pretrained)** – minimal ViT with patch embedding and Transformer encoder. Default scratch settings used in the notebooks:  
  `img_size=224, patch=16, embed_dim=384, depth=12, heads=6, mlp_ratio=4.0 (GELU), drop≈0.1–0.15`. (≈ **21–22M** params depending on head). The pretrained variant is loaded via `timm` when available.

---

## Training Strategy

- **Loss**: `CrossEntropyLoss`, optional **label smoothing** (`0.0–0.10`) and optional **class weights** if needed for imbalance (or a **WeightedRandomSampler**).  
- **Optimizers**:  
  - **CNN/ResNet scratch** → **SGD (mom=0.9)** with `base_lr≈0.05–0.1`, `weight_decay≈1e‑3`.  
  - **ViT scratch** → **AdamW** with `base_lr≈3e‑4`, `weight_decay≈0.1–0.3`.
- **Sharpness‑Aware Minimization (SAM)** (scratch notebooks that enable it): **ρ** set per model (ViT benefits from larger ρ, e.g., **0.2**; CNN/ResNet prefer smaller, e.g., **0.02–0.05**). SAM improves **loss‑landscape smoothness** and typically **helps ViT most** on small‑/medium‑scale supervised tasks. fileciteturn2file0 fileciteturn2file10
- **LR schedule**: **Cosine decay with warmup** (`warmup_epochs=3–5`, `min_lr_mult=0.01`).  
- **Gradient clipping**: `clip_grad_norm≈1.0`.  
- **AMP**: guarded to run when CUDA is available.  
- **Early stopping**: monitor **validation AUROC** (default) with **patience** (`10–17` epochs). We restore the **best model state** observed during training.

**Data pipeline & augmentation.** Inception‑style pre‑processing: `RandomResizedCrop(224, scale=(0.08,1.0))` + `RandomHorizontalFlip(0.5)`; normalization uses ImageNet mean/std (or computed from training data if `compute_norm_stats=True`). For **1‑ch** training we use `Grayscale(1)`; for **3‑ch**, `Grayscale(3)` replicates gray to RGB channels for compatibility with ImageNet statistics and pretrained weights.

---

## Evaluation Protocol

We evaluate each **trained model on the untouched test set**, reporting two complementary decision settings:

1) **Recall‑first (clinical screening)** – we **choose a threshold** on **validation** that achieves **≥ 95% recall**, then **freeze** it and report **test** metrics at that threshold. The implemented selector prefers the **highest specificity** (TNR) among thresholds meeting the recall target (ties broken by higher precision then F1).  
2) **Fixed threshold = 0.50** – we also report test metrics at a default 0.5 decision threshold for side‑by‑side comparison across models.

Both tables include **accuracy, precision, recall, F1‑score, AUROC, chosen threshold, train time and parameter count**. (See the final cells in each notebook.)

**Scoring functions** (in `src/metrics.py`):  
- `evaluate(...)` → returns per‑example probabilities and AUROC.  
- `choose_threshold_by_min_recall(y_true, y_prob, target_recall=0.95)` → returns the threshold that **meets** the recall target with **maximum specificity** (then precision, then F1) among feasible thresholds.  
- `summarize_at_threshold(...)` → computes Accuracy / Precision / Recall / F1 / AUROC for any threshold.

---

## Hyperparameters & What They Do

Below are the **defaults** used most often in the notebooks (tuned by architecture and whether **SAM** is enabled). Values are good starting points for chest X‑ray classification; feel free to sweep further.

### Data & Loader
| Setting | Typical Value | Notes |
|---|---|---|
| `image_size` | 224 | Standard for ViT‑B/16 and ResNet‑18 baselines |
| `batch_size` | 32 | Adjust to GPU memory; keep stable across models when comparing |
| `gray_to_rgb` | `True` (3‑ch) or `False` (1‑ch) | 3‑ch replicates grayscale to RGB for ImageNet norms & pretrained backbones |
| `use_stratified_val` | `True` | More stable validation split |
| `balance_sampler` | optional | Useful if class imbalance is large |

### Optimization (Scratch, **no** SAM)
| Model | Optimizer | LR | Weight Decay | Label Smoothing | Epochs | Warmup |
|---|---|---:|---:|---:|---:|---:|
| CustomCNN | SGD (mom=0.9) | 0.05 | 1e‑3 | 0.0–0.05 | 80–100 | 3 |
| ResNet‑18 | SGD (mom=0.9) | 0.10 | 1e‑3 | 0.0–0.05 | 80–100 | 3 |
| ViT | AdamW | 3e‑4 | 0.1–0.3 | 0.0–0.10 | 100–120 | 5 |

### Optimization (Scratch, **with** SAM)
| Model | Optimizer | LR | Weight Decay | **SAM ρ** | Label Smoothing | Epochs | Warmup |
|---|---|---:|---:|---:|---:|---:|---:|
| CustomCNN | SGD | 0.05 | 1e‑3 | **0.02** | 0.05 | 90–100 | 3 |
| ResNet‑18 | SGD | 0.10 | 1e‑3 | **0.05** | 0.05 | 90 | 3 |
| ViT | AdamW | 3e‑4 | 0.1–0.3 | **0.20** | 0.0–0.10 | 110–120 | 5 |

### Architecture (Scratch)
| Model | Key Knobs | Notes |
|---|---|---|
| CustomCNN | width=32, dropout≈0.0 | ~0.48M params; fast |
| ResNet‑18 | conv stem 7×7 stride 2; `in_chans` 1 or 3 | ~11.2M params |
| ViT | `patch=16`, `embed_dim=384`, `depth=12`, `heads=6`, `mlp_ratio=4.0`, `drop≈0.1–0.15`, `in_chans` 1 or 3 | ~21–22M params. Larger `drop` and SAM help on small data |

### Why these choices?
- **ViT’s data hunger.** ViT has **less image‑specific inductive bias** than CNNs and typically **relies on large‑scale pretraining or strong augmentation** when trained from scratch; otherwise ResNets often win on small datasets. fileciteturn2file15 fileciteturn2file16  
- **SAM helps Transformers disproportionately.** Adding SAM **smooths** the loss landscape and notably **boosts ViT** when trained from scratch, to the point of **outperforming ResNets** of similar size under comparable preprocessing. fileciteturn2file0 fileciteturn2file2

---

## Key Findings (What the notebooks show)

We run **two families** of experiments and **present results per notebook**:

1. **Scratch, no SAM** – CNN and ResNet baselines are **competitive** on this small medical dataset; ViT can match but often needs careful regularization. This is **consistent with the ViT paper’s observation** that, *without large‑scale pretraining or heavy augmentation*, scratch ViT tends to trail strong CNNs. fileciteturn2file15  
2. **Scratch, with SAM** – In our runs, **ViT + SAM** achieved **the best overall test performance** among the models trained **in this notebook**, under **both decision criteria** (threshold picked to achieve **recall ≥ 95%** and the **fixed 0.5 threshold**). This aligns with the SAM paper’s claim that smoothing the loss helps ViT the most, enabling **ViT to surpass ResNets when trained from scratch without large‑scale pretraining or strong augmentations**. fileciteturn2file0 fileciteturn2file12

> Each notebook concludes with **two result tables** (Recall‑first & Fixed‑0.5). Use them to compare the **three model families** side‑by‑side **within that notebook’s setting** (scratch/no‑SAM vs. scratch/with‑SAM; 1‑ch vs. 3‑ch).

**Complexity & time.** We report **parameter counts** and **wall‑clock training time** per model in the tables. Roughly: CustomCNN **~0.48M**, ResNet‑18 **~11.2M**, ViT (scratch config here) **~21–22M** parameters.

---

## Reproducibility

- Set a **random seed** (`src.utils.set_seed(42)`).  
- Fix **train/val/test** split policy (if you re‑split).  
- Keep **batch size**, **image size**, and **evaluation thresholds** identical across models when making comparisons.  
- Report **test‑set** metrics **once** per final chosen model/hyperparameters.
- Use the provided **results registry** (`src/registry.py`) to track runs; the notebooks collect results into the two tables automatically.

---

## References

- **Vision Transformer (ViT).** Dosovitskiy et al., “**An Image is Worth 16×16 Words: Transformers for Image Recognition at Scale**,” ICLR 2021. Key points: patch embedding, class token, reduced inductive bias; ViT typically relies on large‑scale **pretraining** or **strong augmentation** to match CNNs when trained from scratch. fileciteturn2file15 fileciteturn2file16  
- **Sharpness‑Aware Minimization (SAM) effects on ViT.** Chen, Hsieh, Gong, “**When Vision Transformers Outperform ResNets Without Pre‑training or Strong Data Augmentations**,” ICLR 2022. Key point: **smoothing the loss with SAM** markedly **improves ViT**, making it competitive or better than ResNets **when trained from scratch** under standard preprocessing. fileciteturn2file0 fileciteturn2file12

---

## Acknowledgements

- Kaggle dataset authors and maintainers.  
- Open‑source implementations in PyTorch, torchvision, and timm.
