"""
Data loader for TACRED json files.
"""

import json
import random
import torch
import numpy as np

from utils import constant, helper, vocab


class DataLoader(object):
    """
    Load data from json files, preprocess and prepare batches.
    """
    def __init__(self, filename, batch_size, opt, vocab, mappings, tokenizer, evaluation=False):
        self.batch_size = batch_size
        self.opt = opt
        self.vocab = vocab
        self.eval = evaluation
        self.label2id = constant.LABEL_TO_ID
        self.mappings = mappings
        self.tokenizer = tokenizer

        with open(filename) as infile:
            data = json.load(infile)
        self.raw_data = data
        data = self.preprocess(data, vocab, opt)

        # shuffle for training
        if not evaluation:
            indices = list(range(len(data)))
            random.shuffle(indices)
            data = [data[i] for i in indices]
        self.refs = list()
        for d in data:
            temp = []
            for rule in d[11]:
                temp += [[vocab.id2rule[r] for r in rule if r not in [0,2,3]]]
            self.refs.append(temp)

        self.id2label = dict([(v,k) for k,v in self.label2id.items()])
        self.labels = [self.id2label[d[-3]] for d in data]
        self.num_examples = len(data)
        
        # chunk into batches
        data = [data[i:i+batch_size] for i in range(0, len(data), batch_size)]
        self.data = data
        print("{} batches created for {}".format(len(self.data), filename))

    def preprocess(self, data, vocab, opt):
        """ Preprocess the data and convert to ids. """
        processed = []
        processed_rule = []
        with open(self.mappings) as f:
            mappings = f.readlines()
        # with open('dataset/tacred/rules.json') as f:
        #     rules = json.load(f)
        a = 0
        b = 0
        for c, d in enumerate(data):
            tokens = list(d['token'])
            if opt['lower']:
                tokens = [t.lower() for t in tokens]
            # anonymize tokens
            ss, se = d['subj_start'], d['subj_end']
            os, oe = d['obj_start'], d['obj_end']
            tokens[ss:se+1] = ['[SUBJ-'+d['subj_type']+']'] * (se-ss+1)
            tokens[os:oe+1] = ['[OBJ-'+d['obj_type']+']'] * (oe-os+1)
            rl, masked = mappings[c].split('\t')
            masked = eval(masked)
            if masked:
                if (rl == d['relation']):
                    if (masked!=(min(os, ss), max(oe, se)+1)):
                        a += 1
                        print (masked, (min(os, ss), max(oe, se)+1))
                    else:
                        b += 1
            rule = [[]]
            tk_map = {}
            if ss<os:
                os = os + 2
                oe = oe + 2
                tokens.insert(ss, '#')
                tokens.insert(se+2, '#')
                tokens.insert(os, '$')
                tokens.insert(oe+2, '$')
            else:
                ss = ss + 2
                se = se + 2
                tokens.insert(os, '$')
                tokens.insert(oe+2, '$')
                tokens.insert(ss, '#')
                tokens.insert(se+2, '#')
            
            tokens = ['[CLS]'] + tokens
            tokens = self.tokenizer.convert_tokens_to_ids(tokens)
            # tokens = map_to_ids(tokens, vocab.word2id)
            pos = map_to_ids(d['stanford_pos'], constant.POS_TO_ID)
            ner = map_to_ids(d['stanford_ner'], constant.NER_TO_ID)
            deprel = map_to_ids(d['stanford_deprel'], constant.DEPREL_TO_ID)
            head = [int(x) for x in d['stanford_head']]
            assert any([x == 0 for x in head])
            l = len(tokens)
            subj_positions = get_positions(ss+2, se+2, l)
            obj_positions = get_positions(os+2, oe+2, l)
            subj_type = [constant.SUBJ_NER_TO_ID[d['subj_type']]]
            obj_type = [constant.OBJ_NER_TO_ID[d['obj_type']]]
            relation = self.label2id[d['relation']]
            processed += [(tokens, pos, ner, deprel, head, subj_positions, obj_positions, subj_type, obj_type, relation, rule[0], rule)]
        print (a, b)
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
        masks = torch.eq(words, 0)
        pos = get_long_tensor(batch[1], batch_size)
        ner = get_long_tensor(batch[2], batch_size)
        deprel = get_long_tensor(batch[3], batch_size)
        head = get_long_tensor(batch[4], batch_size)
        subj_positions = get_long_tensor(batch[5], batch_size)
        obj_positions = get_long_tensor(batch[6], batch_size)
        # subj_mask = torch.ge(words.input_ids, 28996) * torch.lt(words.input_ids, 28998)
        # obj_mask = torch.ge(words.input_ids, 28998)
        subj_type = get_long_tensor(batch[7], batch_size)
        obj_type = get_long_tensor(batch[8], batch_size)

        rels = torch.LongTensor(batch[9])

        rule = get_long_tensor(batch[10], batch_size)
        
        return (words, masks, pos, ner, deprel, head, subj_positions, obj_positions, subj_type, obj_type, rels, orig_idx, rule)

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
