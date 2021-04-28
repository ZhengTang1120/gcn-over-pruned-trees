"""
Data loader for TACRED json files.
"""

import json
import random
import torch
import numpy as np
import string

from utils import constant, helper, vocab
from collections import defaultdict
from statistics import mean


class DataLoader(object):
    """
    Load data from json files, preprocess and prepare batches.
    """
    def __init__(self, filename, batch_size, opt, vocab, intervals, patterns, tokenizer, odin, evaluation=False):
        self.batch_size = batch_size
        self.opt = opt
        self.vocab = vocab
        self.eval = evaluation
        self.label2id = constant.LABEL_TO_ID
        self.intervals = intervals
        self.patterns = patterns
        self.tokenizer = tokenizer
        self.odin = odin

        with open(filename) as infile:
            data = json.load(infile)
        data = self.preprocess(data, vocab, opt)

        # shuffle for training
        if not evaluation:
            indices = list(range(len(data)))
            random.shuffle(indices)
            data = [data[i] for i in indices]
        self.id2label = dict([(v,k) for k,v in self.label2id.items()])
        self.labels = [self.id2label[d[-4]] for d in data]
        self.words = [d[-1] for d in data]
        self.num_examples = len(data)
        
        # chunk into batches
        data = [data[i:i+batch_size] for i in range(0, len(data), batch_size)]
        self.data = data
        print("{} batches created for {}".format(len(self.data), filename))

    def preprocess(self, data, vocab, opt):
        """ Preprocess the data and convert to ids. """
        processed = []
        processed_rule = []
        with open(self.intervals) as f:
            intervals = f.readlines()
        with open(self.patterns) as f:
            patterns = f.readlines()
        with open(self.odin) as f:
            odin = f.readlines()
        for c, d in enumerate(data):
            tokens = list(d['token'])
            words  = list(d['token'])
            for i in range(len(words)):
                if words[i] == '-LRB-':
                    words[i] = '('
                if words[i] == '-RRB-':
                    words[i] = ')'
            if opt['lower']:
                tokens = [t.lower() for t in tokens]
            # anonymize tokens
            ss, se = d['subj_start'], d['subj_end']
            os, oe = d['obj_start'], d['obj_end']
            tokens[ss:se+1] = ['[SUBJ-'+d['subj_type']+']'] * (se-ss+1)
            tokens[os:oe+1] = ['[OBJ-'+d['obj_type']+']'] * (oe-os+1)
            rl, masked = intervals[c].split('\t')
            rl, pattern = patterns[c].split('\t')
            ol, tagged = odin[c].split('\t')
            masked = eval(masked)
            tagged = eval(tagged)
            ner = d['stanford_ner']
            if tagged and d['relation'] != 'no_relation' and d['relation'] == ol:
                for i in range(len(tagged)):
                    if tagged[i] < min(os, ss):
                        tagged[i] += 1
                    elif tagged[i] <= min(se,oe):
                        tagged[i] += 2
                    elif tagged[i] < max(os, ss):
                        tagged[i] += 3
                    elif tagged[i] <= max(se, oe):
                        tagged[i] += 4
                    else:
                        tagged[i] += 5
                has_tag = True
            elif masked and d['relation'] != 'no_relation' and d['relation'] == rl:
                tagged = []
                masked = [i for i in range(masked[0], masked[1]) if i not in range(ss, se+1) and i not in range(os, os+1)]
                for i in range(len(masked)):
                    if masked[i] < min(os, ss):
                        masked[i] += 1
                    elif masked[i] <= min(se,oe):
                        masked[i] += 2
                    elif masked[i] < max(os, ss):
                        masked[i] += 3
                    elif masked[i] <= max(se, oe):
                        masked[i] += 4
                    else:
                        masked[i] += 5
                has_tag = True
            else:
                tagged = []
                pattern = ''
                masked = range(min(oe, se)+4, max(os, ss)+3)
                has_tag = False
            if ss<os:
                os = os + 2
                oe = oe + 2
                tokens.insert(ss, '[SUBJ-SEP]')
                tokens.insert(se+2, '[SUBJ-SEP]')
                tokens.insert(os, '[OBJ-SEP]')
                tokens.insert(oe+2, '[OBJ-SEP]')
                words.insert(ss, '[SUBJ-SEP]')
                words.insert(se+2, '[SUBJ-SEP]')
                words.insert(os, '[OBJ-SEP]')
                words.insert(oe+2, '[OBJ-SEP]')
                ner.insert(ss, '[SUBJ-SEP]')
                ner.insert(se+2, '[SUBJ-SEP]')
                ner.insert(os, '[OBJ-SEP]')
                ner.insert(oe+2, '[OBJ-SEP]')
            else:
                ss = ss + 2
                se = se + 2
                tokens.insert(os, '[OBJ-SEP]')
                tokens.insert(oe+2, '[OBJ-SEP]')
                tokens.insert(ss, '[SUBJ-SEP]')
                tokens.insert(se+2, '[SUBJ-SEP]')
                words.insert(os, '[SUBJ-SEP]')
                words.insert(oe+2, '[SUBJ-SEP]')
                words.insert(ss, '[OBJ-SEP]')
                words.insert(se+2, '[OBJ-SEP]')
                ner.insert(os, '[OBJ-SEP]')
                ner.insert(oe+2, '[OBJ-SEP]')
                ner.insert(ss, '[SUBJ-SEP]')
                ner.insert(se+2, '[SUBJ-SEP]')
            tokens = ['[CLS]'] + tokens
            words = ['[CLS]'] + words
            ner = ['CLS'] + ner
            relation = self.label2id[d['relation']]
            if has_tag and relation!=0:
                if tagged:
                    tagging = [0 if i not in tagged else 1 for i in range(len(tokens))]
                else:
                    tagging = [0 if i not in masked else 1 if (tokens[i] in pattern or (ner[i] in pattern and ner[i]!='O')) and (tokens[i] not in string.punctuation) else 0 for i in range(len(tokens))]
            # elif relation!=0:
            #     tagging = [1 if i !=0 else 0 for i in range(len(tokens))]
            else:
                tagging = [0 for i in range(len(tokens))]
            l = len(tokens)
            for i in range(l):
                if tokens[i] == '-LRB-':
                    tokens[i] = '('
                if tokens[i] == '-RRB-':
                    tokens[i] = ')'
            if ss<os:
                entity_positions = get_positions2(ss+2, se+2, os+2, oe+2, l)
            else:
                entity_positions = get_positions2(os+2, oe+2, ss+2, se+2, l)
            tokens = self.tokenizer.convert_tokens_to_ids(tokens)
            pos = map_to_ids(d['stanford_pos'], constant.POS_TO_ID)
            ner = map_to_ids(d['stanford_ner'], constant.NER_TO_ID)
            deprel = map_to_ids(d['stanford_deprel'], constant.DEPREL_TO_ID)
            head = [int(x) for x in d['stanford_head']]
            assert any([x == 0 for x in head])
            subj_positions = get_positions(ss+2, se+2, l)
            obj_positions = get_positions(os+2, oe+2, l)
            subj_type = [constant.SUBJ_NER_TO_ID[d['subj_type']]]
            obj_type = [constant.OBJ_NER_TO_ID[d['obj_type']]]
            processed += [(tokens, pos, ner, deprel, entity_positions, subj_positions, obj_positions, subj_type, obj_type, relation, tagging, has_tag, words)]
        return processed

    def gold(self):
        """ Return gold labels as a list. """
        return self.labels

    def __len__(self):
        return len(self.data)

    def __getitem__(self, key):
        """ Get a batch with index. """
        if not isinstance(key, int):
            raise TypeError
        if key < 0 or key >= len(self.data):
            raise IndexError
        batch = self.data[key]
        batch_size = len(batch)
        batch = list(zip(*batch))
        

        # sort all fields by lens for easy RNN operations
        lens = [len(x) for x in batch[0]]
        batch, orig_idx = sort_all(batch, lens)
        # word dropout
        if not self.eval:
            words = [word_dropout(sent, self.opt['word_dropout']) for sent in batch[0]]
        else:
            words = batch[0]
        # convert to tensors
        words = get_long_tensor(words, batch_size)
        # words = self.tokenizer(batch[0], is_split_into_words=True, padding=True, truncation=True, return_tensors="pt")
        pos = get_long_tensor(batch[1], batch_size)
        ner = get_long_tensor(batch[2], batch_size)
        deprel = get_long_tensor(batch[3], batch_size)
        entity_positions = get_long_tensor(batch[4], batch_size)
        subj_positions = get_long_tensor(batch[5], batch_size)
        obj_positions = get_long_tensor(batch[6], batch_size)
        # subj_mask = torch.ge(words.input_ids, 28996) * torch.lt(words.input_ids, 28998)
        # obj_mask = torch.ge(words.input_ids, 28998)
        subj_type = get_long_tensor(batch[7], batch_size)
        obj_type = get_long_tensor(batch[8], batch_size)

        rels = torch.LongTensor(batch[9])#

        rule = get_long_tensor(batch[10], batch_size)
        masks = torch.eq(rule, 0)
        return (words, masks, pos, ner, deprel, entity_positions, subj_positions, obj_positions, subj_type, obj_type, rels, orig_idx, rule, batch[-2])

    def __iter__(self):
        for i in range(self.__len__()):
            yield self.__getitem__(i)

def map_to_ids(tokens, vocab):
    ids = [constant.SOS_ID] + [vocab[t] if t in vocab else constant.UNK_ID for t in tokens] + [constant.EOS_ID]
    return ids

def get_positions(start_idx, end_idx, length):
    """ Get subj/obj position sequence. """
    return list(range(-start_idx, 0)) + [1000]*(end_idx - start_idx + 1) + \
            list(range(1, length-end_idx))

def get_positions2(s1, e1, s2, e2, length):
    """ Get subj&obj position sequence. """
    return [1] + [3] * (s1 - 2) + \
            [2] * (e1 - s1 + 3) + \
            [4] * (s2 - e1 - 3) +\
            [2] * (e2 - s2 + 3) + \
            [5] * (length - e2 - 2)

def get_long_tensor(tokens_list, batch_size):
    """ Convert list of list of tokens to a padded LongTensor. """
    token_len = max(len(x) for x in tokens_list)
    tokens = torch.LongTensor(batch_size, token_len).fill_(constant.PAD_ID)
    for i, s in enumerate(tokens_list):
        tokens[i,:len(s)] = torch.LongTensor(s)
    return tokens

def sort_all(batch, lens):
    """ Sort all fields by descending order of lens, and return the original indices. """
    unsorted_all = [lens] + [range(len(lens))] + list(batch)
    sorted_all = [list(t) for t in zip(*sorted(zip(*unsorted_all), reverse=True))]
    return sorted_all[2:], sorted_all[1]

def word_dropout(tokens, dropout):
    """ Randomly dropout tokens (IDs) and replace them with <UNK> tokens. """
    return [constant.UNK_ID if x != constant.UNK_ID and np.random.random() < dropout \
            else x for x in tokens]