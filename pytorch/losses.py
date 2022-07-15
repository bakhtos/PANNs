import torch
import torch.nn.functional as F

LOSS_FUNCS = {'clip_bce': clip_bce}

def clip_bce(output_dict, target):
    """Binary crossentropy loss.
    """
    return F.binary_cross_entropy(
        output_dict['clipwise_output'], target)
