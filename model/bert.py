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
        self.model = BertModel.from_pretrained("bert-base-uncased")
        self.classifier = nn.Linear(in_dim, opt['num_class'])
        self.opt = opt

    def forward(self, inputs):
        words, masks, pos, ner, deprel, head, subj_pos, obj_pos, subj_type, obj_type = inputs
        outputs = self.model(words)
        print (outputs[0].size())
        outputs = outputs.last_hidden_state
        print (outputs)
        logits = self.classifier(outputs)
        return logits, encode_outputs, hidden