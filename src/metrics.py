from __future__ import annotations
import numpy as np, torch, torch.nn.functional as F
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix
)

@torch.no_grad()
def evaluate_probs_targets(model, loader, device):
    model.eval(); probs=[]; targs=[]
    for xb, yb in loader:
        xb = xb.to(device)
        out = model(xb)
        p = F.softmax(out, dim=1)[:,1].detach().cpu().numpy()
        probs.append(p)
        targs.append(yb.numpy())
    probs = np.concatenate(probs); targs = np.concatenate(targs)
    try:
        auroc = roc_auc_score(targs, probs)
    except Exception:
        auroc = float('nan')
    return probs, targs, auroc


def evaluate(model, loader, device):
    probs, targs, auroc = evaluate_probs_targets(model, loader, device)
    return {'targets': targs, 'probs': np.stack([1-probs, probs], axis=1), 'auroc': auroc}

# fouad start change
def _rates_from_cm(tn, fp, fn, tp):
    tpr = tp / (tp + fn + 1e-9)               # sensitivity / recall+
    tnr = tn / (tn + fp + 1e-9)               # specificity
    ppv = tp / (tp + fp + 1e-9)               # precision
    npv = tn / (tn + fn + 1e-9)
    return dict(tpr=tpr, tnr=tnr, ppv=ppv, npv=npv)


def _compute_cm(y_true, y_pred):
    # confusion matrix start
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    # confusion matrix end
    return tn, fp, fn, tp


def choose_threshold_by_min_recall(y_true, y_prob, target_recall=0.95):
    """
    Improved selection: among thresholds that achieve recall>=target_recall,
    pick the one with the **highest specificity** (TNR). If ties, prefer the
    one with higher precision, then higher F1.
    This prevents picking an ultra-low threshold that turns almost everything
    into positive (the "all-positive confusion matrix" issue).
    Returns: (threshold, recall, precision, f1)
    """
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob).astype(float)

    # Dense grid for stability (unique scores can be too coarse)
    grid = np.linspace(0.0, 1.0, 2001)  # step=0.0005
    best = None  # (spec, prec, f1, thr, rec)

    for thr in grid:
        yhat = (y_prob >= thr).astype(int)
        rec  = recall_score(y_true, yhat, zero_division=0)
        if rec + 1e-12 < float(target_recall):
            continue
        prec = precision_score(y_true, yhat, zero_division=0)
        f1   = f1_score(y_true, yhat, zero_division=0)
        tn, fp, fn, tp = _compute_cm(y_true, yhat)
        spec = (tn / (tn + fp + 1e-9))

        score = (spec, prec, f1)   # lexicographic tie-break
        if (best is None) or (score > best[:3]):
            best = (spec, prec, f1, thr, rec)

    # Fallback: if nothing meets the recall target, return 0.5 and defaults
    if best is None:
        thr = 0.5
        yhat = (y_prob >= thr).astype(int)
        return float(thr), float(recall_score(y_true, yhat, zero_division=0)), \
               float(precision_score(y_true, yhat, zero_division=0)), \
               float(f1_score(y_true, yhat, zero_division=0))

    spec, prec, f1, thr, rec = best
    return float(thr), float(rec), float(prec), float(f1)
# fouad end change


def summarize_at_threshold(y_true, y_prob, thr):
    yhat = (y_prob >= thr).astype(int)
    return {
        "acc": accuracy_score(y_true, yhat),
        "precision": precision_score(y_true, yhat, zero_division=0),
        "recall": recall_score(y_true, yhat, zero_division=0),
        "f1": f1_score(y_true, yhat, zero_division=0),
        "auroc": roc_auc_score(y_true, y_prob),
        "thr": float(thr),
    }
