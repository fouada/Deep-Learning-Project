from __future__ import annotations
import numpy as np, matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, precision_recall_curve, confusion_matrix

def plot_curve(tr, va, title):
    plt.figure(); plt.plot(tr); plt.plot(va); plt.title(title); plt.legend(["train","val"]); plt.show()

def plot_two_series(s1, l1, s2, l2, title):
    plt.figure(); plt.plot(s1, label=l1); plt.plot(s2, label=l2); plt.title(title); plt.legend(); plt.show()

def diag_plots(title, probs2d, y_true, thr, auroc=None):
    y_prob = probs2d[:,1]
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    prec, rec, _ = precision_recall_curve(y_true, y_prob)
    yhat = (y_prob >= thr).astype(int)

    # confusion matrix start
    cm = confusion_matrix(y_true, yhat)
    # confusion matrix end

    fig, ax = plt.subplots(1,3, figsize=(14,4))
    ax[0].plot(fpr, tpr); ax[0].set_title(f"ROC (AUROC={auroc:.3f})" if auroc is not None else "ROC")
    ax[0].set_xlabel("FPR"); ax[0].set_ylabel("TPR")

    ax[1].plot(rec, prec); ax[1].set_title("PR curve"); ax[1].set_xlabel("Recall"); ax[1].set_ylabel("Precision")

    im = ax[2].imshow(cm, cmap="Blues"); ax[2].set_title(f"Confusion @thr={thr:.3f}")
    ax[2].set_xlabel("Pred"); ax[2].set_ylabel("True")
    for (i,j), v in np.ndenumerate(cm):
        ax[2].text(j, i, int(v), ha='center', va='center')

    plt.suptitle(title); plt.tight_layout(); plt.show()
