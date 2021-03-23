"""
A trainer class.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np

from model.bert import BERTencoder, BERTclassifier, Tagger
from model.decoder import Decoder
from utils import constant, torch_utils

from transformers import AdamW

class Trainer(object):
    def __init__(self, opt, emb_matrix=None):
        raise NotImplementedError

    def update(self, batch):
        raise NotImplementedError

    def predict(self, batch):
        raise NotImplementedError

    def update_lr(self, new_lr):
        torch_utils.change_lr(self.optimizer, new_lr)

    def load(self, filename):
        try:
            checkpoint = torch.load(filename)
        except BaseException:
            print("Cannot load model from {}".format(filename))
            exit()
        self.classifier.load_state_dict(checkpoint['classifier'])
        self.opt = checkpoint['config']

    def save(self, filename, epoch):
        params = {
                'classifier': self.classifier.state_dict(),
                'config': self.opt,
                }
        try:
            torch.save(params, filename)
            print("model saved to {}".format(filename))
        except BaseException:
            print("[Warning: Saving failed... continuing anyway.]")


def unpack_batch(batch, cuda):
    rules = None
    if cuda:
        inputs = [batch[0].to('cuda')] + [Variable(b.cuda()) for b in batch[1:10]]
        labels = Variable(batch[10].cuda())
        rules  = Variable(batch[12]).cuda()
    else:
        inputs = [Variable(b) for b in batch[:10]]
        labels = Variable(batch[10])
        rules  = Variable(batch[12])
    tokens = batch[0]
    head = batch[5]
    subj_pos = batch[6]
    obj_pos = batch[7]
    tagged = batch[-1]
    lens = batch[1].eq(0).long().sum(1).squeeze()
    return inputs, labels, rules, tokens, head, subj_pos, obj_pos, lens, tagged

class BERTtrainer(Trainer):
    def __init__(self, opt, emb_matrix=None):
        self.opt = opt
        self.emb_matrix = emb_matrix
        self.encoder = BERTencoder()
        self.classifier = BERTclassifier(opt)
        self.tagger = Tagger()
        self.criterion = nn.CrossEntropyLoss(ignore_index=0)
        self.criterion2 = nn.BCELoss()
        self.parameters = [p for p in self.classifier.parameters() if p.requires_grad] + [p for p in self.encoder.parameters() if p.requires_grad]+ [p for p in self.tagger.parameters() if p.requires_grad]
        if opt['cuda']:
            self.encoder.cuda()
            self.tagger.cuda()
            self.classifier.cuda()
            self.criterion.cuda()
        #self.optimizer = torch_utils.get_optimizer(opt['optim'], self.parameters, opt['lr'])
        self.optimizer = AdamW(
            self.parameters,
            lr=opt['lr'],
        )
    
    def update(self, batch, epoch):
        inputs, labels, rules, tokens, head, subj_pos, obj_pos, lens, tagged = unpack_batch(batch, self.opt['cuda'])

        # step forward
        self.encoder.train()
        self.classifier.train()
        self.tagger.train()
        self.optimizer.zero_grad()

        loss = 0
        h, b_out = self.encoder(inputs)
        tagging_output = self.tagger(h)
        loss = self.criterion2(b_out, (~(labels.eq(0))).to(torch.float32).unsqueeze(1))
        if epoch <= 1:
            logits = self.classifier(h, inputs[1], inputs[6], inputs[7])
            loss += self.criterion(logits, labels)
            for i, f in enumerate(tagged):
                if f:
                    loss += self.criterion2(tagging_output[i], rules[i].unsqueeze(1).to(torch.float32))
        else:
            for i, f in enumerate(tagged):
                if f:
                    if loss == 0:
                        loss = self.criterion2(tagging_output[i], rules[i].unsqueeze(1).to(torch.float32))
                    else:
                        loss += self.criterion2(tagging_output[i], rules[i].unsqueeze(1).to(torch.float32))
                    logits = self.classifier(h[i], inputs[1][i].unsqueeze(0), inputs[6][i].unsqueeze(0), inputs[7][i].unsqueeze(0))
                    loss += self.criterion(logits, labels.unsqueeze(1)[i])
                elif torch.round(b_out)[i] == 1:
                    tag_cands, n = self.tagger.generate_cand_tags(tagging_output[i])
                    print (n)
                    logits = self.classifier(h[i], tag_cands, torch.cat(n*[inputs[6][i].unsqueeze(0)], dim=0), torch.cat(n*[inputs[7][i].unsqueeze(0)], dim=0))
                    best = np.argmax(logits.data.cpu().numpy(), axis=0).tolist()[labels[i]]
                    if loss == 0:
                        loss = self.criterion2(tagging_output[i], tag_cands[best].unsqueeze(1).to(torch.float32))
                    else:
                        loss += self.criterion2(tagging_output[i], tag_cands[best].unsqueeze(1).to(torch.float32))
                    loss += self.criterion(logits[best].unsqueeze(0), labels.unsqueeze(1)[i])
        if loss != 0:
            loss_val = loss.item()
            # backward
            loss.backward()
            self.optimizer.step()
        else:
            loss_val = 0
        return loss_val

    def predict(self, batch, unsort=True):
        inputs, labels, rules, tokens, head, subj_pos, obj_pos, lens, tagged = unpack_batch(batch, self.opt['cuda'])
        rules = rules.data.cpu().numpy().tolist()
        tokens = tokens.data.cpu().numpy().tolist()
        orig_idx = batch[11]
        # forward
        self.encoder.eval()
        self.classifier.eval()
        self.tagger.eval()
        h, b_out = self.encoder(inputs)
        tagging_output = self.tagger(h)
        tagging_mask = torch.round(tagging_output).squeeze(2).eq(0)
        logits = self.classifier(h, tagging_mask, inputs[6], inputs[7])
        loss = self.criterion(logits, labels)
        probs = F.softmax(logits, 1) * torch.round(b_out)#.data.cpu().numpy().tolist()
        predictions = np.argmax(probs.data.cpu().numpy(), axis=1).tolist()
        tags = []
        for i, p in enumerate(predictions):
            if p != 0:
                t = np.argmax(tagging_output[i].data.cpu().numpy(), axis=1).tolist()
                tags += [t]
            else:
                tags += [[]]
        if unsort:
            _, predictions, probs, tags, rules, tokens = [list(t) for t in zip(*sorted(zip(orig_idx,\
                    predictions, probs, tags, rules, tokens)))]
        return predictions, tags, rules, tokens#, probs, decoded, loss.item()