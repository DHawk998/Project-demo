# lovasz_loss.py
# Based on Berman et al. "The Lovász-Softmax loss: A tractable surrogate
# for the optimisation of the intersection-over-union measure in neural networks"
# CVPR 2018 — https://arxiv.org/abs/1705.08790

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


# ──────────────────────────────────────────────
# Core Lovász extension (the mathematical heart)
# ──────────────────────────────────────────────

def lovasz_grad(gt_sorted):
    """
    Computes the Lovász extension weights for a sorted error vector.
    
    These weights encode how much each point (in sorted order) 
    changes the Jaccard / IoU when its label flips.
    
    Args:
        gt_sorted: 1D tensor of ground-truth labels sorted by 
                   descending prediction error (1 = wrong class, 0 = right class)
    Returns:
        1D tensor of Lovász weights, same length as gt_sorted
    """
    p = len(gt_sorted)
    
    # Cumulative sum of ground-truth labels in sorted order
    # gts[k] = number of ground-truth positives in the top-k errors
    gts = gt_sorted.sum()
    
    # intersection[k] = how many of the top-k errors are true positives
    intersection = gts - gt_sorted.float().cumsum(0)
    
    # union[k] = |predicted set up to k| + |remaining GT| - overlap
    union = gts + (1 - gt_sorted.float()).cumsum(0)
    
    # Jaccard score at each prefix of the sorted list
    jaccard = 1.0 - intersection / union
    
    # Lovász weight = marginal change in Jaccard when adding each point
    if p > 1:
        jaccard = torch.cat((jaccard[0:1], jaccard[1:] - jaccard[:-1]))
    
    return jaccard


# ──────────────────────────────────────────────
# Per-class Lovász-Softmax loss
# ──────────────────────────────────────────────

def lovasz_softmax_flat(probs, labels, classes='present', ignore_index=None):
    """
    Computes multi-class Lovász-Softmax loss on a flat (1D) set of points.
    
    Args:
        probs:        (N, C) float tensor — softmax probabilities
        labels:       (N,)   long tensor  — ground-truth class indices
        classes:      'all'     → average loss over all C classes
                      'present' → average only over classes that appear in labels
                      list      → average over specific class indices
        ignore_index: label value to exclude (e.g. unlabelled / void class)
    
    Returns:
        Scalar loss value
    """
    if ignore_index is not None:
        valid = labels != ignore_index
        probs  = probs[valid]
        labels = labels[valid]
    
    if probs.numel() == 0:
        return probs * 0.0  # no valid points — return zero with grad
    
    C = probs.size(1)
    losses = []
    
    # Determine which classes to include in the average
    if classes == 'all':
        class_indices = range(C)
    elif classes == 'present':
        class_indices = torch.unique(labels).tolist()
    else:
        class_indices = classes  # explicit list passed in
    
    for c in class_indices:
        # Binary indicator: 1 if this point belongs to class c, else 0
        fg = (labels == c).float()
        
        if fg.sum() == 0 and classes == 'present':
            continue  # class not present in this batch — skip
        
        # Error for class c:
        #   - True positive:  1 - p(c)  (how wrong we are about a GT positive)
        #   - False positive: p(c)       (how confidently wrong about a GT negative)
        class_pred  = probs[:, c]
        errors      = (fg - class_pred).abs()          # |fg - p(c)|, range [0, 1]
        errors_sorted, perm = torch.sort(errors, descending=True)
        
        # Sort ground-truth in the same order as errors
        fg_sorted = fg[perm]
        
        # Compute Lovász weights and dot product with sorted errors
        w = lovasz_grad(fg_sorted)
        losses.append(torch.dot(w, errors_sorted))
    
    if len(losses) == 0:
        return probs.sum() * 0.0
    
    return torch.stack(losses).mean()


# ──────────────────────────────────────────────
# Main loss class — drop-in for use in trainer
# ──────────────────────────────────────────────

class LovaszCELoss(nn.Module):
    """
    Blended Lovász-Softmax + Cross-Entropy loss.
    
    Cross-entropy stabilises early training (it has strong gradients
    even when predictions are near-uniform). Lovász takes over as the 
    model matures and starts refining class boundaries.
    
    Args:
        alpha:        weight on Lovász term  (1 - alpha on CE)
        ignore_index: void/unlabelled class index
        classes:      which classes to include — 'present', 'all', or list
    
    Usage:
        criterion = LovaszCELoss(alpha=0.5, ignore_index=0)
        loss = criterion(logits, labels)   # logits: (N, C), labels: (N,)
    """
    
    def __init__(self, alpha=0.5, ignore_index=None, classes='present'):
        super().__init__()
        self.alpha        = alpha
        self.ignore_index = ignore_index
        self.classes      = classes
        
        # Standard cross-entropy as the stabilising term
        if ignore_index is not None:
            self.ce = nn.CrossEntropyLoss(ignore_index=ignore_index, weight=None)
        else:
            self.ce = nn.CrossEntropyLoss(weight=None)
            
    def set_class_weights(self, weights):
        """Update CE loss weights dynamically."""
        self.ce.weight = weights  #D
    
    def forward(self, logits, labels):
        """
        Args:
            logits: (N, C) raw network outputs (before softmax)
            labels: (N,)   ground-truth class indices (long)
        """
        # Softmax probabilities for Lovász term
        probs = F.softmax(logits, dim=1)
        
        # Lovász loss directly approximates 1 - mean IoU
        l_loss = lovasz_softmax_flat(
            probs, labels,
            classes=self.classes,
            ignore_index=self.ignore_index
        )
        
        # Cross-entropy provides stable gradients throughout training
        ce_loss = self.ce(logits, labels)
        
        return self.alpha * l_loss + (1.0 - self.alpha) * ce_loss