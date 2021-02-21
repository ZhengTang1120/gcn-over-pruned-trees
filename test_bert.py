import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np

from model.bert import BERTclassifier
from transformers import BertTokenizer

from utils import constant, torch_utils, scorer
import json

tokenizer = BertTokenizer.from_pretrained('bert-base-cased')
special_tokens_dict = {'additional_special_tokens': constant.ENTITY_TOKENS}
num_added_toks = tokenizer.add_special_tokens(special_tokens_dict)
id2label = dict([(v,k) for k,v in constant.LABEL_TO_ID.items()])
def process_data(filename):
    with open(filename) as infile:
        j = json.load(infile)
    data = list()
    batch_size = 30
    for c, d in enumerate(j):
        tokens = list(d['token'])
        ss, se = d['subj_start'], d['subj_end']
        os, oe = d['obj_start'], d['obj_end']
        tokens[ss:se+1] = ['[SUBJ-'+d['subj_type']+']'] * (se-ss+1)
        tokens[os:oe+1] = ['[OBJ-'+d['obj_type']+']'] * (oe-os+1)
        tokens = [t for i, t in enumerate(tokens) if i not in range(ss, se) and i not in range(os, oe)]
        words = ' '.join(tokens)
        relation = constant.LABEL_TO_ID[d['relation']]
        data += [(words, relation)]
    data = [data[i:i+batch_size] for i in range(0, len(data), batch_size)]
    batches = list()
    for batch in data:
        batch = list(zip(*batch))
        words = tokenizer(batch[0], padding=True, return_tensors="pt").to('cuda')
        rels = torch.LongTensor(batch[1]).cuda()
        batches += [(words, rels)]
    return batches
train_batches = process_data('dataset/tacred/train.json')
dev_batches = process_data('dataset/tacred/dev.json')
classifier = BERTclassifier(None, emb_matrix=None)
classifier.cuda()
classifier.model.resize_token_embeddings(len(tokenizer)) 
criterion = nn.CrossEntropyLoss()
parameters = [p for p in classifier.parameters() if p.requires_grad]
optimizer = torch_utils.get_optimizer('adam', parameters, 1e-5)
for i in range(100):
    classifier.train()
    for words, labels in train_batches:
        logits, pooling_output, encoder_outputs, hidden = classifier(words)
        loss = criterion(logits, labels)
        # print (loss.item())
        loss.backward()
        optimizer.step()
    classifier.eval()
    preds = []
    golds = []
    for words, labels in dev_batches:
        logits, hidden, encoder_outputs, hidden = classifier(words)
        loss = criterion(logits, labels)
        probs = F.softmax(logits, 1).data.cpu().numpy().tolist()
        preds += [id2label[t] for t in np.argmax(logits.data.cpu().numpy(), axis=1).tolist()]
        golds += [id2label[t] for t in labels.cpu().numpy().tolist()]
    print (scorer.score(golds, preds))
