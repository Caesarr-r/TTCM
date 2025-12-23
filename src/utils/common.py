import torch
import random
import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score, accuracy_score, roc_curve

def setup_seed(seed_val):
    """
    Set random seeds across libraries to ensure reproducibility.
    """
    torch.manual_seed(seed_val)
    torch.cuda.manual_seed(seed_val)
    torch.cuda.manual_seed_all(seed_val)
    random.seed(seed_val)
    np.random.seed(seed_val)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def calculate_metrics(binary_ood_labels, ood_scores, id_preds, id_true_labels):
    """
    Compute AUROC, AUPR, FPR95, and in-distribution accuracy/F1 for TTCM.
    """
    if ood_scores is None or np.isnan(ood_scores).any():
        return 0.0, 0.0, 1.0, 0.0, 0.0

    if len(np.unique(binary_ood_labels)) < 2:
        auroc, aupr, fpr95 = 0.0, 0.0, 1.0
    else:
        auroc = roc_auc_score(binary_ood_labels, ood_scores)
        aupr = average_precision_score(binary_ood_labels, ood_scores)
        fpr, tpr, _ = roc_curve(binary_ood_labels, ood_scores)
        fpr95 = fpr[np.where(tpr >= 0.95)[0][0]] if np.any(tpr >= 0.95) else 1.0
            
    ind_acc = accuracy_score(id_true_labels, id_preds) if len(id_true_labels) > 0 else 0.0
    ind_f1 = f1_score(id_true_labels, id_preds, average='macro', zero_division=0) if len(id_true_labels) > 0 else 0.0
    return auroc, aupr, fpr95, ind_acc, ind_f1

