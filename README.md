
# Chest X‑Ray Pneumonia Classification — CNN vs. ViT (+ SAM)

This project reproduces and analyzes **binary classification** of chest X‑ray images (Normal vs. Pneumonia) using:
- a compact **CustomCNN** trained from scratch,
- **ResNet‑18** (scratch & ImageNet‑pretrained),
- **Vision Transformer** (ViT; scratch & ImageNet‑pretrained),
- with and without **Sharpness‑Aware Minimization (SAM)**.

It follows the public dataset split from Kaggle’s *Chest X-Ray Pneumonia* collection (**train/val/test** directory layout is preserved). We **never touch the test set** until the very end to report final numbers.

> **Why this repo?** We provide a clean, modular codebase (see `src/`) and four reproducible notebooks that let you flip between **CNN vs. ViT**, **scratch vs. pretrained**, and **SAM on/off**—then compare learning curves, confusion matrices, AUROC/PR, and complexity (params/time).

---

## Dataset

- **Source**: [Chest X-Ray Pneumonia — Kaggle (Paul Mooney)](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia)
- **Classes**: `NORMAL`, `PNEUMONIA`
- **Split used in runs shown here**:  
  Train **4185**, Val **1047**, Test **624** (class counts are imbalanced; sampler/weights mitigate this).  
- **Preprocessing**: images are resized to **224×224**, single‑channel images are **replicated to 3‑channel** for CNN/ViT compatibility.

---

## What’s inside

```
.
├── notebooks/
│   ├── CustomCNN_ResNet18_VIT_No-SAM-Scratch.ipynb
│   ├── CustomCNN_ResNet18_VIT_With-SAM-Scratch.ipynb
│   ├── ResNet18_ViT-B-16_No-SAM-Pretrained.ipynb
│   └── ResNet18_ViT-B-16_With-SAM-Pretrained.ipynb
├── src/
│   ├── builders.py      # build_* helpers for CNN/ResNet/ViT
│   ├── data.py          # DataCfg + dataloaders, samplers, transforms
│   ├── train.py         # TrainCfg + training loop (SGD/AdamW, SAM, cosine, warmup)
│   ├── evaluate.py      # metrics, ROC/PR, thresholding @ target recall
│   ├── plots.py         # loss/val curves, diag plots, confusion matrix
│   └── registry.py      # result_registry, add_result, summary helpers
└── README.md
```

All moving parts are controlled by two simple configs:
- **`DataCfg`** — input size, batch size, sampler, workers, etc.
- **`TrainCfg`** — optimizer, schedule, SAM on/off, warmup, grad clip, etc.

---

## How we evaluate

We report on the **held‑out test set**:
- **Accuracy, Precision, Recall, F1, AUROC**  
- **Curves**: training loss, val Accuracy/F1, ROC/PR; and **Confusion Matrices** at:
  - **Fixed 0.5 threshold**, and
  - **Threshold picked to achieve target Recall = 0.95** (via `choose_threshold_by_min_recall`).

This mirrors clinical screening priorities (favor **high recall**) and exposes precision trade‑offs. We also log **parameter count** and **wall‑clock time** to compare complexity.

---

## Hyperparameters (single place to look)

Below are the **exact knobs** you will see in the notebooks (`TrainCfg` and `DataCfg`), with clear purposes and recommended values that reproduced the results quoted later.

### DataCfg (used both for SAM **ON** and **OFF**)

| Parameter | Purpose | Recommended value |
|---|---|---|
| `dataset_dir` | Path to Kaggle dataset root | `../Data/chest_xray` |
| `image_size` | Resize shorter side and center‑crop to this | `224` |
| `batch_size` | Samples per step (fits 8–12 GB GPUs) | `32` |
| `num_workers` | DataLoader workers (set `0` on Windows/Colab if issues) | `2` |
| `val_ratio` | Stratified split fraction for validation (keeps original test set untouched) | `0.20` |
| `balance_sampler` | Reweigh sampling to balance classes | `True` *(scratch & pretrained)* |
| `compute_norm_stats` | Compute dataset mean/std (we use ImageNet stats in transforms) | `False` |
| `gray_to_rgb` | Replicate 1‑channel X‑rays to 3 channels | `True` |

> **Class balancing:** If `balance_sampler=True`, **do not** also pass `class_weights` to the loss (to avoid “double‑counting” imbalance). If you set `balance_sampler=False`, compute inverse‑frequency weights per class and pass them to the criterion (code already provided).

### TrainCfg — **SAM OFF** (scratch & pretrained)

| Field | Purpose | Scratch: CustomCNN | Scratch: ResNet‑18 | Scratch: ViT (S/16) | Pretrained: ResNet‑18 | Pretrained: ViT‑B/16 |
|---|---|---:|---:|---:|---:|---:|
| `epochs` | Max epochs + early‑stopping on `monitor` | 100 | 90 | 120 | 20 | 20 |
| `patience` | Stop after N epochs without improvement | 15 | 15 | 15 | 5 | 5 |
| `monitor` | What to watch on val | `va_auroc` | `va_auroc` | `va_auroc` | `va_auroc` | `va_auroc` |
| `base_lr` | Initial LR (SGD for CNN/ResNet; AdamW for ViT) | **0.1** | **0.1** | **3e‑4** | **1e‑4** | **1e‑4** |
| `weight_decay` | L2 / AdamW decay | 1e‑3 | 1e‑3 | **0.3** | 1e‑4 | 1e‑4 |
| `label_smoothing` | Stabilize CE loss | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| `use_sam` | Toggle SAM | **False** | **False** | **False** | **False** | **False** |
| `sam_rho` | SAM perturbation radius ρ | – | – | – | – | – |
| `use_cosine` | Cosine LR decay | **True** | **True** | **True** | **True** | **True** |
| `warmup_epochs` | Warmup for stable starts | 3 | 3 | 5 | 2 | 2 |
| `min_lr_mult` | Final LR = `base_lr × min_lr_mult` | 0.01 | 0.01 | 0.01 | 0.05 | 0.05 |
| `clip_grad_norm` | Gradient‑norm clipping | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 |
| `use_amp` | Mixed precision | False | False | False | False | False |
| `llrd` | Layer‑wise LR decay (ViT finetune) | – | – | False | False | **True** |
| `head_lr_mult` | Classifier head LR multiplier | 10.0 | 10.0 | 10.0 | 10.0 | **10.0** |
| `target_recall` | Target recall for threshold selection | 0.95 | 0.95 | 0.95 | 0.95 | 0.95 |

> Notes: The ViT scratch settings (AdamW, **weight_decay=0.3**, **cosine+warmup**) mirror common ImageNet‑from‑scratch recipes and what we found stable in this medical‑imaging‑scale dataset.

### TrainCfg — **SAM ON** (scratch & pretrained)

SAM adds a *minimax* step that nudges training away from sharp minima; ViTs generally benefit from **larger** ρ than ResNets. The table below instantiates that (ρ ranges draw from the SAM/ViT literature and our runs).

| Field | Purpose | Scratch: CustomCNN + SAM | Scratch: ResNet‑18 + SAM | Scratch: ViT (S/16) + SAM | Pretrained: ResNet‑18 + SAM | Pretrained: ViT‑B/16 + SAM |
|---|---|---:|---:|---:|---:|---:|
| `epochs` | Max epochs + early‑stopping | 100 | 90 | 120 | 20 | 20 |
| `patience` | Early‑stop patience | 15 | 15 | 15 | 5 | 5 |
| `base_lr` | LR | 0.1 | 0.1 | 3e‑4 | 1e‑4 | 1e‑4 |
| `weight_decay` | Decay | 1e‑3 | 1e‑3 | **0.3** | 1e‑4 | 1e‑4 |
| `use_sam` | Enable SAM | **True** | **True** | **True** | **True** | **True** |
| `sam_rho` | **Perturb radius ρ** | **0.02** | **0.05** | **0.20** | **0.05** | **0.20** |
| `use_cosine / warmup / min_lr_mult` | Schedule | True / 3 / 0.01 | True / 3 / 0.01 | True / 5 / 0.01 | True / 2 / 0.05 | True / 2 / 0.05 |
| `clip_grad_norm` | Grad clip | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 |
| `llrd / head_lr_mult` | ViT finetune trick | – / 10.0 | – / 10.0 | – / 10.0 | – / 10.0 | **Yes / 10.0** |
| `other` | Misc | Target recall=0.95 | Target recall=0.95 | Target recall=0.95 | Target recall=0.95 | Target recall=0.95 |

**Why these ρ values?** The ICLR‑2022 paper *“When ViTs Outperform ResNets…”* reports that **ResNets prefer small ρ (≈ 0.02–0.05)** while **ViTs prefer larger ρ (≈ 0.1–0.2 for B/16)** and see bigger gains from SAM; it also lists ImageNet training defaults similar to ours (AdamW + weight decay 0.3 for ViT). See their Table of SAM strengths and training configs.  *(Cited in this repo’s README references.)*

---

## Class balancing — sampler vs. class weights

- Use **either** `balance_sampler=True` **or** `class_weights` (not both).
- If you disable the sampler, compute class weights via **inverse frequency** (our code computes `w = N / (K * count)` per class, normalized to mean≈1).

This keeps the **effective positive/negative ratio** stable during training while preserving calibrated probabilities for thresholding.

---

## Results (test set)

Below are the **test‑set** summaries captured by the notebooks. Each model is reported twice: at a fixed **0.5** threshold and at the **Recall‑tuned** threshold (target=0.95). We sort primarily by **F1** at the Recall‑tuned threshold.

### 1) Scratch — **SAM OFF** (`CustomCNN_ResNet18_VIT_No-SAM-Scratch.ipynb`)

| Model | Params (M) | Acc | Prec | Recall | F1 | AUROC | Thr |
|---|---:|---:|---:|---:|---:|---:|---:|
| **ResNet‑18 (scratch)** | 11.18 | 0.8830 | 0.8499 | **0.9872** | **0.9134** | 0.9511 | 0.076 |
| **ViT (scratch)** | 21.67 | 0.7869 | 0.7565 | 0.9718 | 0.8507 | 0.9063 | 0.048 |
| **CustomCNN (scratch)** | 0.48 | 0.7676 | 0.7379 | 0.9744 | 0.8398 | 0.9092 | 0.144 |

**Takeaways (scratch, no SAM):** ResNet‑18 is the **strongest from scratch** on this dataset; ViT requires more regularization to close the gap; the compact CNN is competitive but behind.

### 2) Scratch — **SAM ON** for ViT (`CustomCNN_ResNet18_VIT_With-SAM-Scratch.ipynb`)

> In this notebook, **SAM was enabled for ViT only** (ResNet/CustomCNN kept the same as above).

| Model | Params (M) | Acc | Prec | Recall | F1 | AUROC | Thr |
|---|---:|---:|---:|---:|---:|---:|---:|
| **ViT (scratch + SAM)** | 21.67 | 0.8558 | 0.8261 | **0.9744** | **0.8941** | 0.9332 | 0.046 |
| ResNet‑18 (scratch) | 11.18 | 0.8317 | 0.7975 | 0.9795 | 0.8792 | 0.9267 | 0.146 |
| CustomCNN (scratch) | 0.48 | 0.7660 | 0.7374 | 0.9718 | 0.8385 | 0.9074 | 0.065 |

**Takeaways (scratch, SAM on ViT):** **SAM lifts ViT** substantially (F1 **+4.3 points** vs. scratch without SAM), matching the literature that Transformers gain more from SAM than ConvNets. ResNet did **not** use SAM here.

### 3) Pretrained — **SAM ON** (`ResNet18_ViT-B-16_With-SAM-Pretrained.ipynb`)

| Model | Params (M) | Acc | Prec | Recall | F1 | AUROC | Thr |
|---|---:|---:|---:|---:|---:|---:|---:|
| **ViT‑B/16 (pretrained + SAM)** | 85.80 | 0.8510 | 0.8235 | **0.9692** | **0.8905** | 0.9298 | 0.145 |
| **ResNet‑18 (pretrained + SAM)** | 11.18 | 0.7821 | 0.7510 | 0.9744 | 0.8482 | 0.9073 | 0.155 |

**Takeaways (pretrained, SAM on):** Pretraining + SAM yields a **high‑recall ViT** with strong F1 and AUROC; ResNet‑18 improves vs. its compact scratch model but remains behind ViT‑B/16.

### 4) Pretrained — **SAM OFF** (`ResNet18_ViT-B-16_No-SAM-Pretrained.ipynb`)

> Run this notebook to populate numbers in the same format. Expect **pretraining** to boost both models; our ViT‑B/16 SAM‑ON run suggests the SAM‑OFF baseline will be slightly lower F1/AUROC.

---

## What the literature says and how it informed our choices

- **SAM helps ViT more than ResNet**: SAM explicitly **smooths sharp minima**; ViTs and Mixers converge to sharper regions than ResNets and therefore benefit more; **recommended ρ is larger for ViTs** (e.g., **0.2 for ViT‑B/16**, **0.02–0.05 for ResNets**).  
- **Training defaults** we mirror for ViT: **AdamW**, **weight_decay=0.3**, cosine schedule with warmup, gradient clipping; SAM often raises top‑1 by **≈+5 points** for ViT‑B/16 on ImageNet with basic preprocessing.  
These choices follow *When Vision Transformers Outperform ResNets without Pre‑training or Strong Augmentations* (ICLR 2022).

> See the reference at the end of this README for exact tables on SAM ρ per architecture and the training configurations we borrowed.

---

## Reproducing the runs

1. Download and unzip the Kaggle dataset so you have:
   ```
   Data/
     chest_xray/
       train/ NORMAL/ PNEUMONIA/
       val/   NORMAL/ PNEUMONIA/
       test/  NORMAL/ PNEUMONIA/
   ```
2. Open any notebook in `notebooks/` and set:
   ```python
   DATASET_DIR = Path("../Data/chest_xray")
   ```
3. Run all cells. The notebook will:
   - build loaders from `DataCfg`,
   - build the model via `build_*`,
   - train using `TrainCfg` (SGD/AdamW, cosine, warmup, optional SAM),
   - select threshold for **Recall=0.95**,
   - evaluate on the **untouched test set**,
   - add results to the registry for summary tables and plots.

---

## Practical tips

- **Class balancing**: Prefer `balance_sampler=True` for scratch training on imbalanced data; if you keep the sampler on, leave `class_weights=None`. If you disable the sampler, pass `class_weights` (inverse frequency) to the loss.
- **Thresholding**: High recall is clinically valuable—use our `choose_threshold_by_min_recall` to pick τ achieving **Recall≈0.95** and report the paired precision/F1.
- **SAM cost**: SAM roughly doubles the compute per step (extra fwd/bwd). Favor SAM for **ViT**; its impact on **ResNet‑18** is smaller.

---

## Reference

- Xiangning Chen et al., **“When Vision Transformers Outperform ResNets without Pre‑training or Strong Data Augmentations,”** ICLR 2022.  
  Key items we replicate: larger **ρ** for ViT than ResNet (e.g., **ViT‑B/16: ρ≈0.2**, **ResNet: ρ≈0.02–0.05**), and ViT training defaults (AdamW, weight decay 0.3, cosine+warmup).

---

## License

This repository is for academic use. Check the Kaggle dataset license for data usage terms.