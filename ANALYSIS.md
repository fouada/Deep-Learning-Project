
# Chest X‑ray Pneumonia Classification — CNN vs ViT (Scratch & Pretrained)

This document summarizes the experiments you ran across **two families of models**:

- **CNN family**
  - *Custom CNN* (from scratch)
  - *Pretrained CNN (ResNet‑XX)* — exact depth from your notebook
- **Vision Transformer (ViT) family**
  - *Custom ViT* (from scratch, faithful to Dosovitskiy et al., 2020)
  - *Pretrained ViT‑B/16* (Google JAX `.npz` weights, loaded offline)

It also captures the issues we faced, how they were fixed, and the final results under a **recall‑first** evaluation policy aligned with screening use‑cases.

> **Dataset**: Kaggle "Chest X‑Ray Images (Pneumonia)" (`paultimothymooney/chest-xray-pneumonia`).  
> **Classes**: `NORMAL`, `PNEUMONIA`.  
> **Final split used for ViT**: Stratified 80/20 split from Kaggle *train/* into **train=4,172**, **val=1,044**, with Kaggle *test/* kept **unchanged = 624**.  
> (Original Kaggle `val/` has only **16** images; we avoided it for calibration instability.)

---

## 1) What we built

### 1.1 Vision Transformer — **from scratch**
- **Architecture**: Patch embedding (P=16) → Pre‑LN Transformer encoder (depth=6, heads=8, D=256, MLP ratio=4) → `[CLS]` head.
- **Regularization**: dropout (0.1), **stochastic depth** (DropPath 0.05), label smoothing (0.05), early stopping by `val AUROC`.
- **Class imbalance**: class‑weighted CE.
- **Training**: AdamW; `lr=3e-4`, `wd=0.05`, 15 epochs.
- **Device**: Apple M‑series (MPS).

### 1.2 Vision Transformer — **pretrained ViT‑B/16**
- **Weights**: `notebooks/weights/ViT-B_16.npz` (Google JAX).  
  We implemented a robust **JAX→PyTorch weight loader** (qkv merge, out‑proj reshape, position embedding interpolation).
- **Head**: 2‑class linear head (randomly initialized).
- **Training**: AdamW; `lr=1e-4`, `wd=0.05`, 10 epochs; same loss/early‑stop/imbalance handling as scratch.
- **Device**: MPS.

### 1.3 CNN family
- **Custom CNN**: from your `pneumonia_cnn_colab.ipynb` (details depend on that code).
- **Pretrained CNN (ResNet‑XX)**: fine‑tuned on the same dataset.
- **Note about validation**: if you kept Kaggle’s `val/` (16 images), metrics will look jumpy and thresholds unstable.
  We recommend using the **same stratified 80/20 validation** protocol as in ViT for an apples‑to‑apples comparison.

---

## 2) Major issues we hit — and fixes

1. **Kaggle CLI blocked** by corporate proxy/SSL.  
   *Fix*: Provided offline zip/drag‑drop path and robust download helpers; later worked fully offline.

2. **PyTorch ≥2.6 safe load error** (`weights_only=True` default, `PosixPath` not whitelisted).  
   *Fix*: Added safe loader with `torch.serialization.safe_globals` and created a clean state‑dict file.

3. **timm pretrained ViT download blocked** (proxy).  
   *Fix*: Switched to **Google `.npz` ViT‑B/16** and wrote a JAX→PyTorch loader. Corrected q/k/v & out‑proj shapes and position embedding interpolation.

4. **Validation too small (16)** → highly discrete accuracy, unstable thresholds.  
   *Fix*: Performed **stratified split (20%)** from Kaggle train for validation; left Kaggle test intact for final metrics.

5. **Recall‑first requirement** for screening.  
   *Fix*: Added **threshold calibration on validation** to reach target recall (≥0.95), then evaluated this calibrated threshold on test (`@Rec`).

6. **ViT generalization from scratch** on small data.  
   *Fixes applied*: stronger augmentations, class weights, **DropPath**, label smoothing, early stopping on AUROC.

---

## 3) Results (ViT) — Final numbers from your runs

> **Split**: train=4,172 • val=1,044 • test=624.  
> **Target** (recall‑first calibration): pick **highest precision** on validation with **recall ≥ 0.95**, then evaluate on test.

### 3.1 **ViT (scratch)** — 4.99M params, ~28.4 min on MPS
- **Validation‑calibrated threshold**: `thr ≈ 0.289` (val recall ≈ 0.951, precision ≈ 0.961)
- **Test @0.5**:  Acc **0.7933**, Prec **0.7843**, Recall **0.9231**, F1 **0.8481**, AUROC **0.8892**
- **Test @Rec**:  Acc **0.7788**, Prec **0.7520**, Recall **0.9641**, F1 **0.8449**, AUROC **0.8892**

**Behavior & curves**  
- Stable convergence; training loss decreases smoothly; val loss steadily down with mild noise.
- F1/Acc on validation below pretrained but robust; AUROC ~0.89 indicates good separability.

### 3.2 **ViT (pretrained, ViT‑B/16)** — 85.8M params, ~45.4 min on MPS
- **Validation‑calibrated threshold**: `thr ≈ 0.216` (val recall ≈ 0.985, precision ≈ 0.999)
- **Test @0.5**:  Acc **0.8702**, Prec **0.8337**, Recall **0.9897**, F1 **0.9050**, AUROC **0.9007**
- **Test @Rec**:  Acc **0.8413**, Prec **0.7988**, Recall **0.9974**, F1 **0.8871**, AUROC **0.9007**

**Behavior & curves**  
- Very fast convergence; near‑saturation by ~epoch 4. AUROC ~0.90.  
- Recall is extremely high even at 0.5; recall‑first threshold increases false positives slightly but keeps recall ≳ 0.997.

> **Takeaway (ViT)**: Pretraining dominates — +~7–8 pts accuracy/F1 vs scratch at 0.5, **and** it preserves ultra‑high recall after calibration.

---

## 4) Results (CNN) — how to insert your numbers

We didn’t run your CNN notebook here, so we can’t quote exact numbers. Use this helper inside your **CNN notebook** (after you compute `probs_test` and `y_test`) to print the same report:

```python
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import numpy as np

def summarize_at_threshold(y, p1, thr):
    yhat = (p1 >= thr).astype(int)
    return {
        "acc": accuracy_score(y, yhat),
        "precision": precision_score(y, yhat, zero_division=0),
        "recall": recall_score(y, yhat, zero_division=0),
        "f1": f1_score(y, yhat, zero_division=0),
        "auroc": roc_auc_score(y, p1),
        "thr": float(thr),
    }

# If you use recall‑first calibration on validation:
#   thr_cnn = threshold picked on VAL for recall >= 0.95 (use same function as ViT)
# Then report:
print("CNN (custom) @0.5:", summarize_at_threshold(y_test, probs_test[:,1], 0.5))
print("CNN (custom) @Rec:", summarize_at_threshold(y_test, probs_test[:,1], thr_cnn))
```

Do the same for **pretrained ResNet**. Then paste the four rows below.

### 4.1 **Combined comparison table** (ViT filled, CNN placeholders)

| Family | Model | Params (M) | Train Time (min) | **Acc** | **Prec** | **Recall** | **F1** | **AUROC** | Threshold |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| **ViT** | Scratch (ours) | **4.99** | **28.4** | **0.7788** | **0.7520** | **0.9641** | **0.8449** | **0.8892** | **0.289** |
| **ViT** | Pretrained ViT‑B/16 | **85.8** | **45.4** | **0.8413** | **0.7988** | **0.9974** | **0.8871** | **0.9007** | **0.216** |
| **CNN** | Custom CNN | *TBD* | *TBD* | *TBD* | *TBD* | *TBD* | *TBD* | *TBD* | *0.5 / @Rec* |
| **CNN** | Pretrained ResNet‑XX | *TBD* | *TBD* | *TBD* | *TBD* | *TBD* | *TBD* | *TBD* | *0.5 / @Rec* |

> **Recommendation**: Re‑use the **same stratified 80/20 validation split** for CNN so that your recall‑first calibration is stable and directly comparable to ViT.

---

## 5) Training curves, convergence & generalization

- **ViT scratch**: clear learning signal. After we enlarged validation to 1,044 images and added **DropPath + label smoothing**, the model generalized well (val F1 ≈ 0.94 by the end). Test AUROC ≈ 0.89 suggests robust discrimination.
- **ViT pretrained**: converged very fast (≤5 epochs) with very high val metrics and excellent calibration behavior. Marginal AUROC gain (~0.90 vs 0.89) but substantial **recall** and **F1** gains at 0.5.

---

## 6) Model complexity

- **ViT scratch**: ~**5M** parameters; lighter & faster to train; less accurate than pretrained but good when compute is limited.
- **ViT‑B/16 pretrained**: ~**86M** params; higher memory/compute, better metrics.  
- **CNNs**: fill numbers from your notebook (typical ResNet‑18 ≈ 11.7M; ResNet‑50 ≈ 25.6M).

> **Time** on MPS is competitive; if you move to GPU (CUDA), ViT‑B/16 pretraining will speed up further.

---

## 7) Overfitting/Underfitting observations

- **With the tiny Kaggle `val/` (16 images)**: metrics jumped in 6.25% steps and thresholds were unstable — yielded misleading “perfect” val scores at times.  
- **With stratified 20% val**: curves smooth; no signs of severe overfit; early‑stopping by AUROC worked well.  
- **Pretrained ViT** shows slight over‑confidence; label smoothing + calibrating the decision threshold mitigated the impact.

---

## 8) Strengths & weaknesses

**ViT (scratch)**  
+ Learns with modest parameters; stable after regularization; recall can be pushed via thresholding.  
− Needs more data/augmentation to match pretrained; slightly lower AUROC; more sensitive to threshold.

**ViT (pretrained)**  
+ Best recall and F1 at both 0.5 and calibrated thresholds; converges fast; excellent separability.  
− Heavy (85M); depends on availability of pretrained weights.

**CNNs (expected)**  
+ Strong local inductive biases (translation equivariance/locality); small models can work well with less data.  
− May underperform on global context compared to ViT; performance depends on chosen depth and augmentation.

---

## 9) Improvements we explored / recommend

- **Already applied**: class weighting, label smoothing, **stochastic depth (DropPath)**, early stopping on AUROC, recall‑first threshold calibration, larger stratified validation.
- **Easy wins next**:
  - **Augmentations**: MixUp/CutMix, RandAugment, stronger geometric jitter (but not too aggressive for x‑rays).
  - **Sampler**: `WeightedRandomSampler` for minority balancing or light oversampling.
  - **Loss**: Focal loss for imbalance; try BCE with logits + class weights.
  - **Calibration**: temperature scaling or Platt scaling after training (affects threshold choice quality).
  - **Fine‑tuning schedule (pretrained)**: unfreeze progressively, layer‑wise LR decay, cosine schedule with warmup.
  - **Cross‑validation**: 5‑fold CV for threshold robustness; average per‑fold thresholds.
  - **Test‑Time Augmentation (TTA)**: average logits over 4–8 flips/crops.

---

## 10) Repro & extraction snippets

**Confirm split sizes** (both CNN & ViT notebooks):
```python
print(len(train_ds), len(val_ds), len(test_ds))
```

**Count per class**:
```python
import numpy as np
def class_counts(ds): 
    return np.bincount([y for _,y in ds.samples], minlength=len(ds.classes))
print(class_counts(train_ds), train_ds.classes)
```

**Pick recall‑first threshold on validation** (shared helper):
```python
import numpy as np
from sklearn.metrics import precision_recall_curve

def choose_threshold_by_min_recall(y_val, p1_val, min_recall=0.95):
    prec, rec, thr = precision_recall_curve(y_val, p1_val, pos_label=1)
    # Exclude the last point where threshold is undefined
    prec, rec, thr = prec[:-1], rec[:-1], thr
    mask = rec >= min_recall
    if not np.any(mask):
        # fall back to the single highest recall point
        idx = np.argmax(rec)
        return float(thr[idx]), float(rec[idx])
    # among eligible, choose the one with highest precision
    idx = np.argmax(prec[mask])
    thr_candidates = thr[mask]
    return float(thr_candidates[idx]), float(rec[mask][idx])
```

**Summarize at any threshold**:
```python
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

def summarize_at_threshold(y, p1, thr):
    yhat = (p1 >= thr).astype(int)
    return {
        "acc": accuracy_score(y, yhat),
        "precision": precision_score(y, yhat, zero_division=0),
        "recall": recall_score(y, yhat, zero_division=0),
        "f1": f1_score(y, yhat, zero_division=0),
        "auroc": roc_auc_score(y, p1),
        "thr": float(thr),
    }
```

---

## 11) Final recommendations

1. **Use the same (stratified) validation protocol** for CNN and ViT to calibrate thresholds fairly.  
2. Adopt the **recall‑first** threshold (`@Rec`) for clinical screening, and report @0.5 alongside for completeness.  
3. Keep **pretrained ViT‑B/16** as a strong baseline; use **scratch ViT** when compute/size matters or for ablations.  
4. Add **MixUp/CutMix + cosine LR**; consider **focal loss** to push recall with fewer FPs.  
5. If submitting or deploying, keep **AUROC, AUPRC, recall @ fixed FP rate** as key KPIs.

---

**Reference**: Dosovitskiy et al., *An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale*, 2020 (ViT‑B/16).
