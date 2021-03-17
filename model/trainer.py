"""
A trainer class.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np

from model.bert import BERTclassifier
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
        self.classifier = BERTclassifier(opt, emb_matrix=emb_matrix)
        self.criterion = nn.CrossEntropyLoss()
        self.criterion2 = nn.BCEWithLogitsLoss()
        self.parameters = [p for p in self.classifier.parameters() if p.requires_grad]# + [p for p in self.decoder.parameters() if p.requires_grad]
        if opt['cuda']:
            self.classifier.cuda()
            self.decoder.cuda()
            self.criterion.cuda()
            self.criterion_d.cuda()
        #self.optimizer = torch_utils.get_optimizer(opt['optim'], self.parameters, opt['lr'])
        self.optimizer = AdamW(
            self.parameters,
            lr=opt['lr'],
        )
    def update(self, batch):
        inputs, labels, rules, tokens, head, subj_pos, obj_pos, lens, tagged = unpack_batch(batch, self.opt['cuda'])

        # step forward
        self.classifier.train()
        self.optimizer.zero_grad()

        loss = 0
        # classifier
        logits, tagging_output, encoder_outputs, hidden = self.classifier(inputs)
        if self.opt['classifier']:
            loss = self.criterion(logits, labels)
        if self.opt['decoder']:
            # decoder
            for i, f in enumerate(tagged):
                if f:
                    loss += self.criterion2(tagging_output[i], rules[i].unsqueeze(2).to(torch.float32))
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
        self.classifier.eval()
        self.decoder.eval()
        logits, tagging_output, encoder_outputs, hidden = self.classifier(inputs)
        loss = self.criterion(logits, labels)
        probs = F.softmax(logits, 1).data.cpu().numpy().tolist()
        predictions = np.argmax(logits.data.cpu().numpy(), axis=1).tolist()
        tags = []
        for i, p in enumerate(predictions):
            if p != 0:
                t = torch.round(torch.sigmoid(tagging_output[i])).data.cpu().numpy().tolist()
                tags += [t]
            else:
                tags += [[]]
        if unsort:
            _, predictions, probs, tags, rules, tokens = [list(t) for t in zip(*sorted(zip(orig_idx,\
                    predictions, probs, tags, rules, tokens)))]
        # ids = [orig_idx.index(i) for i in range(len(inputs))]
        # decoder
        # batch_size = labels.size(0)
        # decoded = []
        # masks = inputs[1]
        # output = Variable(torch.LongTensor([constant.SOS_ID] * batch_size)) # sos
        # output = output.cuda() if self.opt['cuda'] else output
        # decoded = torch.zeros(constant.MAX_RULE_LEN, batch_size)
        # decoded[0] = output
        # if self.opt['cuda']:
        #         decoded = decoded.cuda()
        # h0 = hidden.view(self.opt['num_layers'], batch_size, -1)
        # c0 = hidden.view(self.opt['num_layers'], batch_size, -1)
        # decoder_hidden = (h0, c0)
        # for t in range(1, constant.MAX_RULE_LEN):
        #     output, decoder_hidden, attn_weights = self.decoder(
        #             output, masks, decoder_hidden, encoder_outputs)
        #     topv, topi = output.data.topk(1)
        #     output = topi.view(-1)
        #     decoded[t] = output
        # decoded = decoded.transpose(0, 1).tolist()
        # if unsort:
        #     _, decoded, probs = [list(t) for t in zip(*sorted(zip(orig_idx,\
        #             decoded, probs)))]
        return predictions, tags, rules, tokens#, probs, decoded, loss.item()