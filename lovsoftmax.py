# lovasz_loss.py


import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


def lovasz_grad(gt_sorted):
    """
    Computes the Lovász extension weights for a sorted error vector.
    These weights encode how much each point (in sorted order) 
    changes the Jaccard / IoU when its label flips.
    
    """
    p = len(gt_sorted)
    gts = gt_sorted.sum()
    intersection = gts - gt_sorted.float().cumsum(0) 
    union = gts + (1 - gt_sorted.float()).cumsum(0)
    # Jaccard score at each prefix of the sorted list
    jaccard = 1.0 - intersection / union
    if p > 1:
        jaccard = torch.cat((jaccard[0:1], jaccard[1:] - jaccard[:-1]))
    
    return jaccard
#gts are just true positives in the data list, intersection is just the true positives left and union is the positives + false positives.
#Returns the changes in Jaccard loss.


# Per-class Lovász-Softmax loss.

def lovasz_softmax_flat(probs, labels, classes='present', ignore_index=None): 
    #Takes the points belonging to class c and seperates them with the point that don't- which become part of the error calculation.
    #This is the maths behind the Jaccards loss.
    #This pipeline removes ignored points, then loops through each selected class, converts it into a binary problem.
    #sorts prediction errors from largest to smallest, applies Lovász weights to approximate IoU loss, and finally averages the result across all classes.
    """
    Computes multi-class Lovász-Softmax loss on a flat (1D) set of points.
    
    Parameters:
        probs:        (N, C) tensor — predicted probabilities after softmax
        labels:       (N,)   long tensor  — ground-truth class indices
        classes:      'all'     → average loss over all C classes
                      'present' → average only over classes that appear in labels
                      list      → average over specific class indices
        ignore_index: label value to exclude (e.g. unlabelled class)
    
    Returns:
        The loss as a single scalar value.
    """
    if ignore_index is not None:
        valid = labels != ignore_index
        probs  = probs[valid]
        labels = labels[valid]
    
    if probs.numel() == 0:
        return probs * 0.0  
    
    C = probs.size(1)
    losses = []
    
    # Determines which classes to include in the average.
    if classes == 'all':
        class_indices = range(C)
    elif classes == 'present':
        class_indices = torch.unique(labels).tolist()
    else:
        class_indices = classes  # explicit list passed in.
    
    for c in class_indices:
        # Binary indicator: 1 if this point belongs to class c, else 0.
        fg = (labels == c).float()
        
        if fg.sum() == 0 and classes == 'present':
            continue  # class not present in this batch = skip.
        
        # Error for class c:
        #True positive:  1 - p(c)  (how wrong we are about a GT positive).
        #False positive: p(c) (how confidently wrong about a GT negative).
        class_pred  = probs[:, c]
        errors      = (fg - class_pred).abs()         
        errors_sorted, perm = torch.sort(errors, descending=True)
        
        # Sort ground-truth in the same order as errors.
        fg_sorted = fg[perm]
        
        # Compute Lovász weights and dot product with sorted errors.
        w = lovasz_grad(fg_sorted)
        losses.append(torch.dot(w, errors_sorted))
    
    if len(losses) == 0:
        return probs.sum() * 0.0
    
    return torch.stack(losses).mean()



class LovaszCELoss(nn.Module):
    #This class combines two loss functions: Cross-Entropy and Lovász-Softmax (to directly optimise IoU), blending them with alpha to control how much each contributes to the final loss signal.
    """
    Lovász-Softmax + Cross-Entropy loss.
    
    Cross-entropy stabilises early training (it has strong gradients
    even when predictions are near-uniform). Lovász takes over as the 
    model matures and starts refining class boundaries.
    
    Args:
        alpha:        weight on Lovász term  (1 - alpha on CE)
        ignore_index: void/unlabelled class index
        classes:      which classes to include — 'present', 'all', or list
    
    Usage:
        criterion = LovaszCELoss(alpha=0.5, ignore_index=0)
        loss = criterion(logits, labels) 
    """
    
    def __init__(self, alpha=0.5, ignore_index=None, classes='present'):
        super().__init__()
        self.alpha        = alpha
        self.ignore_index = ignore_index
        self.classes      = classes
        
        # Ce at the start to stabilise training.
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
        
        # Lovász loss directly approximates 1 - mean IoU.
        l_loss = lovasz_softmax_flat(
            probs, labels,
            classes=self.classes,
            ignore_index=self.ignore_index
        )
        
        # Cross-entropy provides stable gradients throughout training.
        ce_loss = self.ce(logits, labels)
        
        return self.alpha * l_loss + (1.0 - self.alpha) * ce_loss