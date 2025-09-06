# ANALYSIS.md — Executive Summary (ViT vs CNN Baseline Context)

**Task**: Binary classification of chest X‑rays (*NORMAL* vs *PNEUMONIA*) on the Kaggle dataset.  
**Goal**: Prioritize **recall** (sensitivity) for screening—catch as many pneumonia cases as possible—even if it slightly reduces precision.

**Pipelines compared**
- **ViT (scratch)** — custom implementation faithful to the paper (patch‑linear‑embed → [CLS]+pos → L×{Pre‑LN → MSA → residual → Pre‑LN → MLP(GELU) → residual} → LN → [CLS] → linear head), with **DropPath** and **label smoothing**.
- **ViT‑B/16 (pretrained)** — initialized offline from Google **JAX `.npz`** weights (`ViT-B_16.npz`), mapped into an equivalent PyTorch skeleton, then fine‑tuned.

Both use **stratified 20% validation** (from Kaggle train) to stabilize calibration and early stopping on **val AUROC**. Test set is Kaggle’s official **test/**.

---

## Key Findings

- **High recall** is achievable on both models with **VAL‑calibrated thresholds** (recall‑first).  
- **Pretrained ViT‑B/16** outperforms scratch in almost every metric and **converges faster**, at the cost of **~17× more parameters** and longer training time.  
- The initial instability seen with the tiny Kaggle `val/` (16 images) was resolved by **stratified splitting**; the learned thresholds became **reasonable** instead of extreme.

---

## Final Metrics (TEST set)

**Split used**: `(train, val, test) = (4172, 1044, 624)`; classes `['NORMAL', 'PNEUMONIA']`

### Operating points
- **@0.5** — fixed threshold = 0.5 (baseline view).  
- **@Rec** — **recall‑first**: choose the **highest‑precision** threshold on **VAL** s.t. recall ≥ **0.95**; apply it to **TEST**.

### Summary table (@Rec emphasized for screening)

| Model | Params (M) | Time (min) | Acc | Prec | **Recall** | F1 | AUROC | Thr (@Rec) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| **ViT (scratch)** | **4.99** | 28.39 | 0.7788 | 0.7520 | **0.9641** | 0.8449 | 0.8892 | 0.289 |
| **ViT‑B/16 (pretrained)** | **85.80** | 45.36 | 0.8413 | 0.7988 | **0.9974** | 0.8871 | 0.9007 | 0.216 |

For context (@0.5):
- **Scratch**: acc 0.7933 · prec 0.7843 · rec 0.9231 · F1 0.8481 · AUROC 0.8892  
- **Pretrained**: acc 0.8702 · prec 0.8337 · rec 0.9897 · F1 0.9050 · AUROC 0.9007

**Interpretation**: Under the recall‑first objective, **Pretrained ViT‑B/16** is the best choice; **Scratch ViT** is competitive for recall with far lower complexity.

---

## Training Curves, Convergence, and Generalization

- With the tiny built‑in `val/` (16 images), validation curves and thresholds were noisy/unreliable.  
  After switching to a **stratified 20% validation split**, curves became **smooth**, **val AUROC** rose consistently, and **threshold calibration stabilized**.
- **ViT (scratch)**: steady loss drop; minor overfitting mitigated by **DropPath** and **label smoothing**; AUROC stable around ~0.89 on TEST.  
- **ViT‑B/16 (pretrained)**: very fast convergence; **val** metrics are near‑ceiling, with a small but expected generalization gap on **TEST**, likely due to domain shift.

---

## Model Complexity

- **ViT (scratch)**: ~**5.0M** params; ~**28.4** minutes (MPS).  
- **ViT‑B/16 (pretrained)**: ~**85.8M** params; ~**45.4** minutes (MPS).
- Memory: both fit at **224×224** and batch size 32 on typical MPS/CPU; larger resolution requires more memory.

---

## Observations: Over/Under‑fitting

- The pretrained model, without heavy augmentation, can **over‑fit VAL** if VAL is too small; the stratified split cured most volatility.  
- The scratch model benefits from **regularization** (DropPath, smoothing) and enough **epochs**; performance improves with consistent validation and thresholding.

---

## Strengths & Weaknesses (ViT only)

**ViT (scratch)**  
**✔** Light (~5M), good recall after calibration, reproducible.  
**✖** Lower AUROC/precision than pretrained; benefits from more data and augmentation.

**ViT‑B/16 (pretrained)**  
**✔** Best metrics overall; near‑perfect recall under @Rec; fastest convergence.  
**✖** Heavy (~86M), longer training; needs offline weights under restricted networks.

---

## Techniques Applied / Alternatives to Try

- **Already used**: label smoothing (0.05), **DropPath** (0.05 linear schedule), class weights, early stopping on **val AUROC**, **recall‑first thresholding**.
- **Recommended next**:  
  - **Augmentations**: RandAugment / stronger random crop; **Mixup/CutMix**.  
  - **Loss**: focal loss for precision‑recall balance on imbalanced data.  
  - **Optimization**: cosine LR with warmup; **layer‑wise LR decay** (pretrained).  
  - **Resolution**: 384×384 with position‑embedding interpolation.  
  - **Calibration**: temperature scaling on VAL to improve calibrated probabilities.  
  - **Explainability**: attention roll‑out / Grad‑CAM to visualize pulmonary regions.

---

## Repro Tips

 
  Run top‑to‑bottom: **scratch** (train→eval) → **pretrained** (train→eval) → **comparison**.
- Data layout: `Data/chest_xray/{train,val,test}/...` (keep `Data/` out of Git; use `Data/.gitkeep`).
- Pretrained weights: `notebooks/weights/ViT-B_16.npz` (offline).  
- Use **RUNBOOK.md** for exact instructions; **EXPERIMENTS.md** for full details.

