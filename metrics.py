import torch
import torch.nn.functional as F
import numpy as np

def binary_dice_coeff(logits,true,threshold=None,eps=1e-7):
    """Computes the Sørensen-Dice coefficient for multi-class.
    Args:
        true: a tensor of shape [B, 1, H, W].
        logits: a tensor of shape [B, C, H, W]. Corresponds to
            the raw output or logits of the model.
        eps: added to the denominator for numerical stability.
    Returns:
        dice_coeff: the Sørensen-Dice coefficient.
    """
    probs = F.sigmoid(logits)
    if threshold != None:
        predictions = (probs >=threshold).float()
        # flatten to (Batch,WxH) shape
        true_flat =true.view(true.size(0),-1)
        preds_flat = predictions.view(predictions.size(0),-1)
        
        intersection = (true_flat*preds_flat).sum(dim=1)
        union = true_flat.sum(dim=1) + preds_flat.sum(dim=1)
        
        dice_score =  (2.0 * intersection + eps) / (union + eps)
            
        return dice_score.mean()
    else:
        # flatten to (Batch,WxH) shape
        true_flat =true.view(true.size(0),-1)
        preds_flat = probs.view(probs.size(0),-1)
        
        intersection = (true_flat*preds_flat).sum(dim=1)
        union = true_flat.sum(dim=1) + preds_flat.sum(dim=1)
        dice_score =  (2.0 * intersection + eps) / (union + eps)
            
        return dice_score.mean()
        
        
    
def binary_dice_loss(logits,true, eps=1e-7):
    """Computes the Sørensen-Dice loss, which is 1 minus the Dice coefficient.
    Args:
        true: a tensor of shape [B, 1, H, W].
        logits: a tensor of shape [B, C, H, W]. Corresponds to
            the raw output or logits of the model.
        eps: added to the denominator for numerical stability.
    Returns:
        dice_loss: the Sørensen-Dice loss.
    """
    return 1 - binary_dice_coeff(logits,true)