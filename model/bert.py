import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
from transformers import BertModel

from utils import constant, torch_utils

class BERTclassifier(nn.Module):
    """ A wrapper classifier for GCNRelationModel. """
    def __init__(self, opt, emb_matrix=None):
        super().__init__()
        in_dim = 768
        self.model = BertModel.from_pretrained("bert-base-cased")
        self.classifier1 = nn.Linear(in_dim*3, 400)
        self.classifier2 = nn.Linear(400, opt['num_class'])
        self.opt = opt

    def forward(self, inputs):
        words, masks, pos, ner, deprel, head, subj_mask, obj_mask, subj_type, obj_type = inputs
        subj_mask, obj_mask = subj_mask.eq(0).unsqueeze(2), obj_mask.eq(0).unsqueeze(2)
        outputs = self.model(**words)
        h = outputs.last_hidden_state
        pool_type = self.opt['pooling']
        subj_out = pool(h, subj_mask, type=pool_type)
        obj_out = pool(h, obj_mask, type=pool_type)
        cls_out = h.transpose(1,0)[0]
        outputs = torch.cat([cls_out, subj_out, obj_out], dim=1)
        logits = self.classifier2(F.tanh(self.classifier1(outputs)))
        return logits, None, None, None

def pool(h, mask, type='max'):
    if type == 'max':
        h = h.masked_fill(mask, -constant.INFINITY_NUMBER)
        return torch.max(h, 1)[0]
    elif type == 'avg':
        h = h.masked_fill(mask, 0)
        return h.sum(1) / (mask.size(1) - mask.float().sum(1))
    else:
        h = h.masked_fill(mask, 0)
        return h.sum(1)