import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
from transformers import BertModel

class BERTclassifier(nn.Module):
    """ A wrapper classifier for GCNRelationModel. """
    def __init__(self, opt, emb_matrix=None):
        super().__init__()
        in_dim = 768
        self.model = BertModel.from_pretrained("bert-base-cased")
        self.classifier = nn.Linear(in_dim, 42)
        self.opt = opt

    def forward(self, words):
        # words, masks, pos, ner, deprel, head, subj_pos, obj_pos, subj_type, obj_type = inputs

        outputs = self.model(**words)
        print (outputs.last_hidden_state[:,-1,:])
        print (outputs.last_hidden_state[:,-1,:].size())
        outputs = outputs.pooler_output
        print (outputs)
        print (outputs.size())
        logits = self.classifier(outputs)
        return logits, None, None, None