import os
import numpy as np
import torch
import torch.nn as nn
from functools import partial
from model.DSTformer import DSTformer

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
        
def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)
        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))
        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


def multilabel_accuracy(logits, targets, threshold=0.5):
    """
    Computes the accuracy for multi-label classification.
    """
    # Apply sigmoid to convert logits to probabilities
    probabilities = torch.sigmoid(logits)
    
    # Convert probabilities to binary predictions
    predictions = (probabilities > threshold).float()

    # correct_predictions = (predictions == targets).float()
    # sample_accuracy = correct_predictions.sum(dim=1) / targets.size(1)
    # return sample_accuracy.mean().item()*100

    correct_predictions = (predictions == targets).all(dim=1).float()
    return correct_predictions.mean().item() * 100


def compute_overall_accuracy(logit_action, batch_gt_action, logit_pattern, batch_gt_pattern, logit_muscle, batch_gt_muscle, threshold=0.5):
    # Action accuracy
    _, pred_action = logit_action.max(1)
    correct_action = (pred_action == batch_gt_action)

    # Pattern accuracy
    probabilities_pattern = torch.sigmoid(logit_pattern)
    predictions_pattern = (probabilities_pattern > threshold).float()
    correct_pattern = (predictions_pattern == batch_gt_pattern).all(dim=1)

    # Muscle accuracy
    logits_reshaped = logit_muscle.view(-1, 8, 3)
    pred_intensity_indices = logits_reshaped.argmax(dim=2)
    labels_reshaped = batch_gt_muscle.view(-1, 8, 3)
    true_intensity_indices = labels_reshaped.argmax(dim=2)
    correct_muscle = (pred_intensity_indices == true_intensity_indices).all(dim=1)

    # Overall accuracy
    overall_correct = correct_action & correct_pattern & correct_muscle
    overall_accuracy = overall_correct.float().mean().item() * 100

    return overall_accuracy



def load_pretrained_weights(model, checkpoint):
    """Load pretrianed weights to model
    Incompatible layers (unmatched in name or size) will be ignored
    Args:
    - model (nn.Module): network model, which must not be nn.DataParallel
    - weight_path (str): path to pretrained weights
    """
    import collections
    if 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
    else:
        state_dict = checkpoint
    model_dict = model.state_dict()
    new_state_dict = collections.OrderedDict()
    matched_layers, discarded_layers = [], []
    for k, v in state_dict.items():
        # If the pretrained state_dict was saved as nn.DataParallel,
        # keys would contain "module.", which should be ignored.
        if k.startswith('module.'):
            k = k[7:]
        if k in model_dict and model_dict[k].size() == v.size():
            new_state_dict[k] = v
            matched_layers.append(k)
        else:
            discarded_layers.append(k)
    model_dict.update(new_state_dict)
    model.load_state_dict(model_dict, strict=True)
    print('load_weight', len(matched_layers))
    return model

def partial_train_layers(model, partial_list):
    """Train partial layers of a given model."""
    for name, p in model.named_parameters():
        p.requires_grad = False
        for trainable in partial_list:
            if trainable in name:
                p.requires_grad = True
                break
    return model

def load_backbone(args):
    if not(hasattr(args, "backbone")):
        args.backbone = 'DSTformer' # Default
    if args.backbone=='DSTformer':
        model_backbone = DSTformer(dim_in=3, dim_out=3, dim_feat=args.dim_feat, dim_rep=args.dim_rep, 
                                   depth=args.depth, num_heads=args.num_heads, mlp_ratio=args.mlp_ratio, norm_layer=partial(nn.LayerNorm, eps=1e-6), 
                                   maxlen=args.maxlen, num_joints=args.num_joints)
    else:
        raise Exception("Undefined backbone type.")
    return model_backbone