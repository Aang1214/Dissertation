# scripts/evaluation.py
"""Reusable evaluation module for multiclass accident severity classification."""

import pandas as pd
from sklearn.metrics import (
    accuracy_score, f1_score, matthews_corrcoef, confusion_matrix
)


def evaluate(y_true, y_pred, model_name='model', verbose=True):
    """
    Compute accuracy, macro-F1, per-class F1, and MCC.

    Parameters
    ----------
    y_true : array-like 
        Ground truth labels.
    y_pred : array-like
        Predicted labels.
    model_name : str
        Identifier for the model.
    verbose : bool
        If True, print metrics and confusion matrix.

    Returns
    -------
    pd.DataFrame
        One-row DataFrame containing the metrics. Multiple results can be
        concatenated with pd.concat for cross-experiment comparison.
    """

    # Fixed class order — keeps per-class columns aligned across experiments
    classes = sorted(pd.Series(y_true).unique())
    per_class_f1 = f1_score(
        y_true, y_pred, labels=classes, average=None, zero_division=0
    )

    row = {
        'model':    model_name,
        'accuracy': accuracy_score(y_true, y_pred),
        'macro_f1': f1_score(y_true, y_pred, average='macro', zero_division=0),
        'mcc':      matthews_corrcoef(y_true, y_pred),
    }

    # Add one column per class
    for cls, f1 in zip(classes, per_class_f1):
        row[f'f1_{cls}'] = f1

    # Human-readable printout — turn off (verbose=False) when looping
    if verbose:
        print(f"\n{model_name}")
        print(f"Accuracy : {row['accuracy']:.4f}")
        print(f"Macro-F1 : {row['macro_f1']:.4f}")
        print(f"MCC      : {row['mcc']:.4f}")
        for cls, f1 in zip(classes, per_class_f1):
            print(f"  F1 [{cls}]: {f1:.4f}")

        # Rows = true class, columns = predicted class; diagonal = correct
        cm = confusion_matrix(y_true, y_pred, labels=classes)
        cm_df = pd.DataFrame(
            cm,
            index=[f"true_{c}"  for c in classes],
            columns=[f"pred_{c}" for c in classes],
        )
        print("\nConfusion matrix:")
        print(cm_df)
    
    # DataFrame (not dict) so Phase 2/3/4 can pd.concat results easily
    return pd.DataFrame([row])