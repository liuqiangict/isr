
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Sequential
from models.transformers import BertForTokenClassification
from models.transformers import LongformerForTokenClassification 
from torchvision import models

__all__ = [ "transformer", "longformer" ]

class TRANSFORMER(nn.Module):
    def __init__(self, args, model):
        super(TRANSFORMER, self).__init__()

        self.args = args
        self.scale = args.scale
        self.patch_size = args.patch_size
        self.input_size = (args.patch_size // self.scale) * (args.patch_size // self.scale)
        self.output_size = args.patch_size * args.patch_size

        self.model = model


    def forward(self, lr, is_train=True):
        b, c, h, w = lr.shape
        lr_input_ids = lr.view(b, c, h * w)
        lr_input_ids = lr_input_ids.transpose(1, 2).long()
        lr_input_ids = lr_input_ids.view(b, -1)
        
        output = self.model(lr_input_ids).reshape(b, h * w, c)
        output.clamp_(0., 255.)
        output = output.transpose(1, 2).reshape(b, c, h, w)

        return output


def _transformer(arch, args):
    if arch == 'TRANSFORMER':
        m = BertForTokenClassification.from_pretrained(args.model_name_or_path)
    elif arch == 'LONGFORMER':
        m = LongformerForTokenClassification.from_pretrained(args.model_name_or_path)
    else:
        raise('Incorrect model architecture called in transformer as {}'.format(arch))

    model = TRANSFORMER(args, m)
    return model

def transformer(args):
    return _transformer('TRANSFORMER', args)

def longformer(args):
    return _transformer('LONGFORMER', args)
