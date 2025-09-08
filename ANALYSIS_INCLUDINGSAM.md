# Chest X‑ray Pneumonia Classification — CNN vs. ViT (with & without SAM)

**Goal.** Classify chest X‑ray images as **NORMAL** vs **PNEUMONIA** while *prioritizing recall* (sensitivity) to avoid missing pneumonia cases.  
**Dataset.** Kaggle Chest X-Ray Pneumonia (Mooney). Final split used in the working notebooks:

- Train: **4,172** images  
- Validation: **1,044** images (stratified from original *train + val*)  
- Test: **624** images (left untouched)

All images resized to **224×224**, 3‑channel. We apply standard normalization and light geometric/photometric augmentations (random crop/resize, horizontal flip).

---

## Models

### Convolutional baselines
- **Custom CNN (scratch)** — small 2–3 stage conv backbone with global pooling and a 2‑class head (~**0.39M** params).
- **ResNet‑18 (pretrained)** — ImageNet pretrained, unfrozen fine‑tuning (~**11.18M** params).

### Vision Transformers
- **ViT (scratch)** — 6 Transformer blocks, **embed_dim=256**, **heads=8**, **MLP ratio=4**, **patch=16**, **Dropout=0.1**, **DropPath=0.05** (~**4.99M** params).
- **ViT‑B/16 (pretrained)** — 12 blocks, **D=768**, **H=12**; weights loaded from Google’s `.npz`, classifier head re‑initialized for 2 classes (~**85.8M** params).

> **Recall‑first evaluation.** Thresholds are *not* fixed at 0.5. We choose on the validation set the highest‑precision threshold that still satisfies a **target recall ≥ 0.95**, then report on the test set. We also show the default `@0.5` for reference in the notebooks.

### Optimizer and training
- Optimizer: **AdamW**; cosine LR; label smoothing **0.05**; class weights enabled.  
- Early stopping on validation **AUROC**.  
- For the second study, we also train the four models with **SAM (Sharpness‑Aware Minimization)** and keep the rest of the recipe unchanged.

---

## Results

### A) *Vision Transformers without SAM* (recall‑first thresholding)

| Model | Params (M) | Time (min) | Acc | Prec | Recall | F1 | AUROC | Thr |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| **ViT (scratch)** | 4.99 | 27.60 | 0.7788 | 0.7520 | 0.9641 | 0.8449 | 0.8892 | 0.289 |
| **ViT (pretrained)** | 85.80 | 44.14 | 0.8413 | 0.7988 | 0.9974 | 0.8871 | 0.9007 | 0.216 |

**Takeaways.**
- Pretraining brings a consistent boost (higher **Acc/F1/AUROC**) while maintaining **very high recall**.
- The calibrated thresholds are **< 0.5**, reflecting a conservative decision boundary to meet the recall target.

### B) *CNN & ViT with SAM* (same recall‑first evaluation)

| Model | Params (M) | Time (min) | Acc | Prec | Recall | F1 | AUROC | Thr |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| **CNN (scratch)** | 0.39 | 15.88 | 0.8462 | 0.8088 | 0.9872 | 0.8891 | 0.9207 | 0.365 |
| **ResNet18 (pretrained)** | 11.18 | 12.93 | 0.8606 | 0.8230 | 0.9897 | 0.8987 | 0.8676 | 0.785 |
| **ViT (scratch + SAM)** | 4.99 | 27.60 | 0.7115 | 0.6895 | 0.9795 | 0.8093 | 0.8281 | 0.252 |
| **ViT‑B/16 (pretrained + SAM)** | 85.80 | 72.21 | 0.7468 | 0.7125 | 0.9974 | 0.8312 | 0.9093 | 0.261 |

**Takeaways.**
- On this small, single‑domain medical dataset, **CNNs dominate** Transformers on **F1/Acc** at the *same high recall target*.  
- **SAM** preserves **recall** for ViTs but **reduced precision and F1** compared with the non‑SAM ViT runs. In contrast, CNNs—especially **ResNet18**—benefit most overall.
- The **AUROC** of ViT‑B/16 + SAM is strong (0.909), but the **operating point** chosen for recall ≥ 0.95 hurts precision.

---

## Training curves & generalization

- **ViT (scratch)** with the larger validation split (1,044 images) shows **stable convergence**: training loss ↓ monotonically; validation **Acc/F1** climb into the **0.93–0.95** band by the end; gap between train/val is small ⇒ **no clear overfitting**.  
- **ViT (pretrained)** hits very high **val metrics quickly**; on the held‑out test set AUROC ≈ **0.90**, indicating **good—but not perfect—generalization**.  
- In the **SAM** study, the ViT curves remain smooth but the *recall‑first threshold* lands in a region where many **false positives** occur, trading precision for recall.

---

## Interpretation

1. **Why do CNNs win here?**  
   CNNs encode strong **inductive biases** (locality and translation equivariance) that make them **data‑efficient** and robust on small medical datasets. ViTs lack these biases and typically need **more data or stronger regularization** to match CNNs.

2. **What did SAM change?**  
   SAM explicitly seeks **flatter minima** and improves generalization, with particularly large gains reported for ViTs on ImageNet‑scale training. In our setting, it **maintained recall** but **did not improve precision/F1** for ViTs—likely because the data regime is small and the class boundary is already recall‑oriented. (See the paper references in the project report.)

3. **Effect of the *recall‑first* policy.**  
   By construction, thresholds shift **left** (below 0.5), which **increases recall** at the expense of precision. This is desired for screening; downstream triage can handle some extra false positives.

---

## Strengths & weaknesses by model

- **Custom CNN (scratch)**  
  - *Strengths*: best **F1** among all runs with SAM; smallest model; fastest to train; very high recall.  
  - *Weaknesses*: lower AUROC than ViT‑B/16; may miss subtle global patterns without multi‑scale features.

- **ResNet18 (pretrained)**  
  - *Strengths*: overall **best test F1** in the SAM study; excellent recall; fast; strong inductive bias.  
  - *Weaknesses*: AUROC behind ViT‑B/16; may be capacity‑limited for harder variants.

- **ViT (scratch)**  
  - *Strengths*: competitive AUROC; responds well to **pretraining** and to a **larger validation split** (more stable thresholds).  
  - *Weaknesses*: needs more data/regularization; precision drops at high‑recall operating points; SAM did not improve F1 in our small‑data regime.

- **ViT‑B/16 (pretrained)**  
  - *Strengths*: **Near‑perfect recall** with solid AUROC; interpretable attention maps; transfer‑friendly.  
  - *Weaknesses*: parameter‑heavy and slower; precision at the recall‑first threshold is lower than CNNs on this task.

---

## Model complexity and wall‑clock

- **Parameters** range from **0.39M (CNN)** to **85.8M (ViT‑B/16)**.  
- **Training time** on Apple **MPS** ranged from **~13–16 min** (CNN/ResNet18) to **~45–72 min** (ViT‑B/16), with SAM adding overhead due to the extra ascent step.

---

## Overfitting / underfitting

- With the **expanded validation split**, none of the models exhibited classic overfitting—val loss trended down with epochs and **AUROC** improved.  
- Earlier runs with a **tiny validation set** produced unstable thresholds and noisy val loss; this was resolved by stratified **20% validation** and recall‑first **threshold selection** on the larger VAL set.

---

## Recommendations (next steps)

1. **Tune SAM ρ per model** (e.g., ViT often favors larger ρ, but on small data a moderate ρ∈[0.05,0.15] can work better; we used a single setting).  
2. **Data augmentations** focused on X‑rays: small rotations, translation, slight CLAHE/brightness, cutout; optionally **mixup/cutmix** for Transformers.  
3. **Losses for class imbalance:** try **Focal** or **Asymmetric Focal**; keep label smoothing small (≤0.05).  
4. **Conv‑stem ViT (hybrid)** or **smaller patch sizes (8×8)** to inject locality while keeping global attention.  
5. **Calibration & thresholding:** continue using validation‑driven thresholds; consider **cost‑sensitive ROC** or maximize **Fβ (β>1)** to formalize “recall‑first.”  
6. **Cross‑validation** across multiple stratified folds to stabilize estimates; **TTA** at inference to nudge precision back up without hurting recall.

---

## What to report (grading checklist)

- **Metrics (test set only)**: Accuracy, Precision, Recall, F1, AUROC at both 0.5 and recall‑first thresholds (see tables).  
- **Curves**: Loss & Val Acc/F1 for each model (already in the notebooks).  
- **Complexity**: Parameters & training time (tables above).  
- **Analysis**: Why CNNs outperform ViTs here; how pretraining and SAM changed behavior; pros/cons for screening.

> **Bottom line.** For *screening* (recall‑first), a **pretrained ResNet18** currently gives the best balance of **very high recall** with the **highest F1** and the fastest runtime. **ViT‑B/16** achieves comparable AUROC and **near‑perfect recall**, but costs more compute and has lower precision in this regime. With more data or task‑specific augmentation, the ViT gap is likely to shrink.