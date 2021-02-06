
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Sequential
from models.transformers import BertForTokenClassification
from models.transformers import LongformerForTokenClassification 
from models.transformers import SparseForTokenClassification 
from torchvision import models

__all__ = [ "transformer", "longformer", "sparse" ]

class TRANSFORMER(nn.Module):
    def __init__(self, model):
        super(TRANSFORMER, self).__init__()
        self.model = model


    def forward(self, lr, is_train=True):
        b, c, h, w = lr.shape
        lr_input_ids = lr.reshape(b, c, h * w)
        lr_input_ids = lr_input_ids.transpose(1, 2).long()
        #lr_input_ids = lr_input_ids.reshape(b, -1)
        
        output = self.model(lr_input_ids).reshape(b, h * w, c)
        output.clamp_(0., 255.)
        output = output.transpose(1, 2).reshape(b, c, h, w)

        return output


def _transformer(arch, model_name_or_path):
    if arch == 'TRANSFORMER':
        m = BertForTokenClassification.from_pretrained(model_name_or_path)
    elif arch == 'LONGFORMER':
        m = LongformerForTokenClassification.from_pretrained(model_name_or_path)
    elif arch == 'SPARSE':
        m = SparseForTokenClassification.from_pretrained(model_name_or_path)
    else:
        raise('Incorrect model architecture called in transformer as {}'.format(arch))

    model = TRANSFORMER(m)
    return model

def transformer(args):
    return _transformer('TRANSFORMER', args)

def longformer(args):
    return _transformer('LONGFORMER', args)

def sparse(args):
    return _transformer('SPARSE', args)
