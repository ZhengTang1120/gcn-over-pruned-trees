import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
from transformers import BertModel

from utils import constant, torch_utils

class BERTclassifier(nn.Module):
    def __init__(self, opt, emb_matrix=None):
        super().__init__()
        in_dim = 1024
        self.model = BertModel.from_pretrained("mrm8488/spanbert-large-finetuned-tacred")
        # self.linear = nn.Linear(in_dim, in_dim)
        # self.classifier1 = nn.Linear(in_dim*3, 400)
        # self.classifier2 = nn.Linear(400, opt['num_class'])
        self.classifier = nn.Linear(in_dim, opt['num_class'])
        self.opt = opt
        self.tagger = nn.Linear(in_dim, 1)

    def forward(self, inputs):
        words, masks, pos, ner, deprel, head, subj_pos, obj_pos, subj_type, obj_type = inputs
        subj_mask, obj_mask = subj_pos.eq(1000).eq(0).unsqueeze(2), obj_pos.eq(1000).eq(0).unsqueeze(2)
        outputs = self.model(words)
        h = outputs.last_hidden_state
        pool_type = self.opt['pooling']
        out_mask = masks.unsqueeze(2).eq(0) + subj_mask.eq(0) + obj_mask.eq(0)
        cls_out = pool(h, out_mask.eq(0), type=pool_type)
        # logits = self.classifier2(F.tanh(self.classifier1(outputs)))
        logits = self.classifier(cls_out)
        tag_logits = self.tagger(h)
        return logits, tag_logits

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