import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
from transformers import BertModel

from utils import constant, torch_utils

class BERTencoder(nn.Module):
    def __init__(self):
        super().__init__()
        in_dim = 1024
        self.model = BertModel.from_pretrained("mrm8488/spanbert-large-finetuned-tacred")

    def forward(self, inputs):
        words, masks, pos, ner, deprel, head, subj_pos, obj_pos, subj_type, obj_type = inputs
        outputs = self.model(words)
        h = outputs.last_hidden_state

        return h

class BERTclassifier(nn.Module):
    def __init__(self, opt):
        super().__init__()
        in_dim = 1024
        self.classifier = nn.Linear(3 * in_dim, opt['num_class'])
        self.opt = opt

    def forward(self, h, masks, subj_pos, obj_pos):
        subj_mask, obj_mask = subj_pos.eq(1000).eq(0).unsqueeze(2), obj_pos.eq(1000).eq(0).unsqueeze(2)
        
        pool_type = self.opt['pooling']
        subj_out = pool(h, subj_mask, type=pool_type)
        obj_out = pool(h, obj_mask, type=pool_type)
        cls_out = pool(h, masks.unsqueeze(2), type=pool_type)
        outputs = torch.cat([cls_out, subj_out, obj_out], dim=1)
        logits = self.classifier(outputs)
        return logits

class Tagger(nn.Module):
    def __init__(self):
        super().__init__()
        in_dim = 1024

        self.tagger = nn.Linear(in_dim, 1)
        self.threshold = 0.8

    def forward(self, h):

        tag_logits = F.sigmoid(self.tagger(h))
        
        return tag_logits

    def generate_cand_tags(self, tag_logits):
        print (tag_logits)
        cand_tags = [[]]
        for t in tag_logits.gt(self.threshold):
            if t:
                temp = []
                for ct in cand_tags:
                    temp.append(ct+[0])
                    ct.append(1)
                cand_tags += temp
            else:
                for ct in cand_tags:
                    ct.append(0)
        print (cand_tags)
        return torch.BoolTensor(cand_tags).cuda(), len(cand_tags)

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
