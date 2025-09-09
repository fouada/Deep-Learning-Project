# Pneumonia X‑ray Classification — CNN vs ViT (with SAM)

**Project goal.** Classify chest X‑ray images as **Normal** vs **Pneumonia** using the Kaggle dataset (`chest_xray`). Keep the **test set untouched** and report final performance on that set. Implement both a **CNN** and a **Vision Transformer (ViT)**, compare them quantitatively (accuracy, precision, recall, F1, AUROC), study training dynamics and generalization, and discuss model complexity and inductive biases. Where appropriate, evaluate **data augmentation, transfer learning, and optimizer tricks**.

> **Literature anchor.** The analysis explicitly references the ICLR‑2022 paper **“When Vision Transformers Outperform ResNets without Pre‑training or Strong Data Augmentations”** (Chen *et al.* 2022) and the original **ViT** paper (Dosovitskiy *et al.* 2021). The ICLR paper argues that **ViTs and MLP‑Mixers converge to sharp minima** with conventional training and that **Sharpness‑Aware Minimization (SAM)** smooths the loss geometry, **improving generalization and robustness**, sometimes to the point where **ViTs trained from scratch can match/beat ResNets** under simple Inception‑style preprocessing.

---

## 1) Dataset & protocol

- **Data**: Kaggle *Chest X‑Ray Pneumonia* (train/val/test folders provided).  
- **Preprocessing** (“Inception‑style” as in the paper):  
  - Train: `RandomResizedCrop(224)`, `HorizontalFlip(0.5)`, `ToTensor`, `Normalize(ImageNet mean/std)`  
  - Val/Test: `Resize(256) → CenterCrop(224)`, `ToTensor`, `Normalize`
- **Model selection objective**: validation **AUROC** (tie‑breakers via F1).  
- **Decision threshold**: **recall‑first** (screening): pick the smallest threshold on **validation** achieving ≥ **0.95 recall**; then report **test** metrics at that threshold. We also show the plain 0.5 threshold for reference.
- **Hardware**: single machine; timings are wall‑clock minutes measured in the notebook cells (coarse).

---

## 2) Models & training recipes

- **CNN (scratch)**: compact 3‑block ConvNet (≈ **0.09M** params). Optimizer **SGD**/**SAM** optional.  
- **ResNet18 (pretrained)**: ImageNet‑pretrained backbone with a 2‑class head (≈ **11.18M** params).  
- **ViT (scratch)**: small ViT (≈ **5.0M** params).  
- **ViT‑B/16 (pretrained)**: Google JAX `.npz` checkpoint loaded into a PyTorch ViT‑B/16 head (≈ **85.8M** params).  
- **Optimizer**: AdamW for ViTs, SGD/AdamW for CNN/ResNet. **SAM** available via a base‑optimizer‑agnostic wrapper.  
- **Early stopping** on validation AUROC; label smoothing optional; class weights optional.

> **ICLR‑2022 alignment.** The paper trains with **Inception‑style preprocessing**, uses **SAM**, and reports **bigger gains for ViTs** than for ResNets; ViTs prefer **larger SAM ρ** (e.g., ViT‑B/16 around **0.2**), whereas ResNets use smaller ρ (≈ **0.02–0.05**). We followed this spirit (ρ≈0.2 for ViT‑B/16, ≈0.05 for CNN/ResNet).

---

## 3) Results (test set)

All entries below are **test** metrics. For each model, the row uses the **recall‑first** threshold (≥0.95 validation recall); the 0.5‑threshold snapshots are printed in the notebook for reference.

| Model | Params (M) | Time (min) | Acc | Prec | Recall | F1 | AUROC | Thr |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| **ViT (scratch + SAM)** | 4.99 | 27.36 | 0.7756 | 0.7500 | 0.9615 | 0.8427 | 0.8838 | 0.077 |
| **ViT‑B/16 (pretrained + SAM)** | 85.80 | 77.39 | **0.9279** | **0.9021** | **0.9923** | **0.9451** | **0.9859** | 0.288 |
| **CNN (scratch + SAM)** | 0.09 | 16.39 | 0.6090 | 0.6225 | 0.9513 | 0.7525 | 0.7076 | 0.456 |
| **ResNet18 (pretrained + SAM)** | 11.18 | 24.80 | 0.9247 | 0.8979 | 0.9923 | 0.9428 | 0.9842 | 0.299 |

**Context (without SAM, earlier runs)**  
- *ViT (scratch)*: F1 ≈ **0.8449**, AUROC ≈ **0.8892**  
- *ViT‑B/16 (pretrained)*: F1 ≈ **0.8871**, AUROC ≈ **0.9007**

**Takeaways.**
- **SAM + ViT‑B/16 (pretrained)** shows the **largest gain** over its non‑SAM counterpart (+**0.058 F1**, +**0.085 AUROC**), which is qualitatively in line with ICLR‑2022: **SAM helps ViTs more than ResNets**, because ViT’s loss landscape is sharper without inductive biases.  
- **ResNet18 (pretrained + SAM)** is very close to **ViT‑B/16 (pretrained + SAM)** on this task (F1 **0.943** vs **0.945**; AUROC **0.984** vs **0.986**) despite being ~**8× smaller** and **3× faster to train**. On medical X‑rays with limited data, convolutional inductive biases remain highly effective.  
- **ViT (scratch + SAM)** does **not** improve over earlier scratch runs without SAM (F1 ~0.843 vs ~0.845), likely because (a) **few training epochs** and (b) **small dataset** limit ViT’s data‑hungry optimization; the paper used **≈300 epochs** for ViTs even on ImageNet.  
- **CNN (scratch + SAM)** achieves **high recall** at the screening threshold but has the **lowest AUROC**, confirming that capacity and pretraining matter for this dataset.

---

## 4) Curves & generalization

- **Training curves** show **stable convergence** for ViT‑B/16 + SAM and ResNet18 + SAM: validation **loss steadily decreases** and **F1 rises** to ~0.95.  
- **ROC**/**PR** curves: ViT‑B/16 + SAM and ResNet18 + SAM both yield **high‑area** curves (AUROC ~**0.986** and **0.984**; AP ~**0.99**).  
- **Recall‑first operating point**: thresholds chosen on validation (e.g., **0.288** for ViT‑B/16+SAM) transfer well to **test**, preserving **≥0.99 recall** while maintaining **~0.90–0.91 precision**.

**Over/underfitting.**
- **ViT (scratch)**: early epochs show **larger train‑val gap**; performance improves with more data/epochs/regularization; still **data‑hungry**, consistent with the literature.  
- **ResNet18 (pretrained)**: **fast convergence** and **small gaps**, showing stronger data efficiency and **benefit from inductive biases** (locality & translation equivariance).  
- **SAM**: acts as a **geometry‑aware regularizer**, improving val metrics (especially for ViT‑B/16).

---

## 5) Complexity & efficiency

| Model | Params | Relative train time |
|---|---:|---:|
| CNN (scratch) | **0.09M** | **1×** (fastest) |
| ResNet18 (pretrained) | **11.18M** | ~**1.5×** |
| ViT (scratch) | **4.99M** | ~**1.7×** |
| ViT‑B/16 (pretrained) | **85.8M** | **3–5×** (slowest) |

**Implication.** On this binary X‑ray task, **ResNet18 (pretrained + SAM)** gives a **near‑ViT‑B/16** score **at a fraction of the cost**, making it a strong **default** for limited compute.

---

## 6) How do these results line up with ICLR‑2022?

**Paper’s core claims (abridged):**
1) With conventional (“Inception‑style”) preprocessing, **ViTs and Mixer converge to sharp minima**; **SAM** smooths the loss landscape, **improving accuracy & robustness**.  
2) **Improvement is larger for ViTs/Mixers** than for ResNets; with SAM, **ViTs trained from scratch** can **match/beat ResNets** (e.g., **+5.3% top‑1** for **ViT‑B/16** on ImageNet).  
3) Recommended **SAM ρ** is **larger for ViTs** (e.g., **ρ≈0.2** for ViT‑B/16) than for ResNets (ρ≈0.02–0.05).

**Our alignment on chest X‑rays:**
- ✅ **Inception‑style** preprocessing and **SAM** were used (ρ choices consistent with paper).  
- ✅ **Bigger SAM gain for ViT** vs ResNet: **ViT‑B/16** gains **substantially** with SAM (AUROC **+0.085**, F1 **+0.058** vs non‑SAM), while **ResNet18** sees **modest** gains.  
- ⚠️ **Training “from scratch”** for ViT was limited (epochs, data scale); consequently **ViT (scratch + SAM)** did **not** surpass ResNet18 here. The paper runs **~300 epochs** on ImageNet; with small medical data and few epochs, ResNet’s inductive biases remain advantageous.  
- ✅/⚠️ **No strong augmentations**: we followed the paper’s *no strong aug* philosophy; however, for medical imaging, modest domain‑aware augmentations (e.g., flips, small rotations) can help without deviating from the paper’s focus on optimizer‑driven gains.

**Bottom line.** On this medical dataset, **ViT‑B/16 + SAM (pretrained)** and **ResNet18 + SAM (pretrained)** are **neck‑and‑neck**, supporting the paper’s view that **SAM narrows the gap and can let ViTs compete with ResNets** under plain preprocessing. Limited data and shorter schedules explain why **scratch ViT** did not exceed ResNet here; the paper’s “ViTs > ResNets without pretraining” result appears at **ImageNet scale with long schedules**.

---

## 7) What we implemented from the paper vs. what’s missing

**Implemented**
- Inception‑style transforms (crop + flip).
- SAM with **ρ tailored per architecture** (ViT‑B/16 higher than ResNet/CNN).  
- Validation‑set model selection and early stopping (stability).  
- Clear **screening** operating point: **recall‑first** thresholding.
- Fair CNN/ResNet baselines (scratch and ImageNet‑pretrained).

**Not (yet) implemented / partial**
- **300‑epoch schedules** for ViT from scratch (we used ~10–15 epochs in several runs).  
- **SGD+SAM** for ViT as in the paper’s ablation (we used AdamW+SAM).  
- **Hessian/eigenvalue visualizations** to quantify sharpness.  
- **Robustness** evals (ImageNet‑C/R analogues); could approximate by adding common X‑ray corruptions.  
- **Exact architectural parity** (e.g., ViT‑S/16 vs ResNet‑50 trained from scratch).

---

## 8) Practical guidance (to get even closer to the paper)

1) **Longer schedules for scratch ViT**: 100–300 epochs with cosine decay & warmup often unlock SAM’s benefits.  
2) **Try SGD+SAM for ViT** (the paper reports large gains): start with LR ~0.1·(batch/256), momentum 0.9, weight decay 0.3, **ρ≈0.2** for ViT‑B/16.  
3) **ρ sweep**: ViT‑B/16 ρ∈[0.15, 0.25]; ResNet18 ρ∈[0.02, 0.05].  
4) **No strong augmentations** if you want a faithful reproduction; if you optimize for **screening recall**, modest medical augmentations (small rotations, brightness) can help AUROC without breaking comparability.  
5) **Conv‑stem ViT** (optional): a small 3×3 conv before patchify can improve low‑level structure on X‑rays without leaving the ViT family.  
6) **Calibration**: after fixing the recall‑first threshold, run **temperature scaling** on val to tighten precision at the chosen recall.

---

## 9) Strengths, weaknesses & model selection for screening

- **ResNet18 (pretrained + SAM)**: **Best efficiency–accuracy trade‑off**, excellent AUROC/F1, fast to train/deploy. **Great default**.  
- **ViT‑B/16 (pretrained + SAM)**: **Slightly best raw metrics**; more compute‑heavy. Consider when you have ample compute and want maximal AUC/F1.  
- **ViT (scratch)**: Can work, but on small data it needs **more epochs** and/or **regularization** to match pretraining baselines.  
- **CNN (scratch)**: Small and recall‑friendly but lowest AUROC; works as a **lightweight triage** model.

**For screening** (high recall): choose **ResNet18 (pretrained + SAM)** or **ViT‑B/16 (pretrained + SAM)**; set threshold from validation to **≥0.95 recall** and verify test precision/F1.

---

## 10) Reproducibility checklist

- Fix seeds, keep Inception‑style transforms, **log thresholds chosen on validation**, save per‑epoch curves, and report **test‑only** once at the end. Keep training/eval code for SAM identical between models to isolate architecture/optimizer effects.

---

## References

- Chen, Hsieh, Gong (2022). *When Vision Transformers Outperform ResNets without Pre‑training or Strong Data Augmentations*, ICLR 2022.  
- Dosovitskiy et al. (2021). *An Image is Worth 16×16 Words: Transformers for Image Recognition at Scale*, ICLR 2021.
