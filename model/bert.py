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
        self.classifier = nn.Linear(in_dim, 1)
        self.pos_emb = nn.Embedding(6, 5, padding_idx=constant.PAD_ID)

    def forward(self, inputs):
        words, masks, pos, ner, deprel, ent_pos, subj_pos, obj_pos, subj_type, obj_type = inputs
        outputs = self.model(words)
        h = outputs.last_hidden_state
        h = torch.cat([h, self.pos_emb(ent_pos)], dim=2)
        out = torch.sigmoid(self.classifier(outputs.pooler_output))

        return h, out

class BERTclassifier(nn.Module):
    def __init__(self, opt):
        super().__init__()
        in_dim = 1029
        self.classifier = nn.Linear(in_dim, opt['num_class'])
        self.opt = opt

    def forward(self, h, masks, subj_pos, obj_pos):
        subj_mask, obj_mask = subj_pos.eq(1000).unsqueeze(2), obj_pos.eq(1000).unsqueeze(2)
        
        pool_type = self.opt['pooling']
        out_mask = masks.unsqueeze(2).eq(0) + subj_mask + obj_mask
        cls_out = pool(h, out_mask.eq(0), type=pool_type)
        logits = self.classifier(cls_out)
        return logits

class Tagger(nn.Module):
    def __init__(self):
        super().__init__()
        in_dim = 1029

        self.tagger = nn.Linear(in_dim, 1)
        self.threshold1 = 0.8
        self.threshold2 = 0.2

    def forward(self, h):

        tag_logits = torch.sigmoid(self.tagger(h))
        
        return tag_logits

    def generate_cand_tags(self, tag_logits, device):
        cand_tags = [[]]
        for t in tag_logits:
            if t < self.threshold1 and t > self.threshold2:
                temp = []
                for ct in cand_tags:
                    temp.append(ct+[0])
                    ct.append(1)
                cand_tags += temp
                if len(cand_tags) > 2048:
                    return None, -1
            elif t > self.threshold1:
                for ct in cand_tags:
                    ct.append(1)
            else:
                for ct in cand_tags:
                    ct.append(0)
        with torch.cuda.device(device):
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
