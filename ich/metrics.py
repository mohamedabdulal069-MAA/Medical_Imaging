"""Slice-level multilabel metrics. Undefined rates are null, never perfect scores."""
import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score
from . import LABELS
from .dataset import validate_labels


def evaluate_predictions(y, probabilities, thresholds=0.5):
    y, p = np.asarray(y), np.asarray(probabilities)
    validate_labels(y)
    if p.shape != y.shape or not np.isfinite(p).all() or np.any((p < 0) | (p > 1)):
        raise ValueError("Expected N x 6 finite probabilities in [0,1]")
    t = np.asarray(thresholds)
    if t.shape not in ((), (6,)) or not np.isfinite(t).all() or np.any((t < 0) | (t > 1)):
        raise ValueError("Invalid thresholds")
    pred = p >= t
    rate = lambda a, b: float(a/b) if b else None
    classes = {}
    for i, label in enumerate(LABELS):
        a, b = y[:, i].astype(bool), pred[:, i]
        tp, tn = int((a & b).sum()), int((~a & ~b).sum())
        fp, fn = int((~a & b).sum()), int((a & ~b).sum())
        classes[label] = dict(tp=tp, tn=tn, fp=fp, fn=fn, positives=int(a.sum()),
            sensitivity=rate(tp, tp+fn), specificity=rate(tn, tn+fp), precision=rate(tp, tp+fp),
            f1=rate(2*tp, 2*tp+fp+fn),
            roc_auc=float(roc_auc_score(a, p[:, i])) if len(np.unique(a)) == 2 else None,
            average_precision=float(average_precision_score(a, p[:, i])) if a.any() else None)
    clipped = np.clip(p, 1e-7, 1-1e-7)
    bce = -(y*np.log(clipped)+(1-y)*np.log1p(-clipped))
    macro = {key: float(np.mean(values)) if (values := [v[key] for v in classes.values() if v[key] is not None]) else None
             for key in ("f1", "roc_auc", "average_precision")}
    return dict(unit="slice", n=len(y), labels=list(LABELS), thresholds=np.broadcast_to(t, (6,)).tolist(),
        per_label=classes, macro_defined_labels=macro,
        micro_f1=float(f1_score(y, pred, average="micro", zero_division=0)),
        exact_match=float(np.all(y == pred, axis=1).mean()), binary_accuracy=float((y == pred).mean()),
        bce=float(bce.mean()), weighted_log_loss=float(np.average(bce, axis=1, weights=[1,1,1,1,1,2]).mean()),
        hierarchy_violations=int((pred[:, :5].any(axis=1) & ~pred[:, 5]).sum()))
