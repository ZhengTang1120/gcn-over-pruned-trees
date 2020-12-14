"""
Run evaluation with saved models.
"""
import random
import argparse
from tqdm import tqdm
import torch

from data.loader import DataLoader
from model.trainer import GCNTrainer
from utils import torch_utils, scorer, constant, helper
from utils.vocab import Vocab

from nltk.translate.bleu_score import corpus_bleu, sentence_bleu

parser = argparse.ArgumentParser()
parser.add_argument('model_dir', type=str, help='Directory of the model.')
parser.add_argument('--model', type=str, default='best_model.pt', help='Name of the model file.')
parser.add_argument('--data_dir', type=str, default='dataset/tacred')
parser.add_argument('--dataset', type=str, default='test', help="Evaluate on dev or test.")

parser.add_argument('--seed', type=int, default=1234)
parser.add_argument('--cuda', type=bool, default=torch.cuda.is_available())
parser.add_argument('--cpu', action='store_true')
args = parser.parse_args()

torch.manual_seed(args.seed)
random.seed(1234)
if args.cpu:
    args.cuda = False
elif args.cuda:
    torch.cuda.manual_seed(args.seed)

# load opt
model_file = args.model_dir + '/' + args.model
print("Loading model from {}".format(model_file))
opt = torch_utils.load_config(model_file)
trainer = GCNTrainer(opt)
trainer.load(model_file)

# load vocab
vocab_file = args.model_dir + '/' +'vocab.pkl'
vocab = Vocab(vocab_file, load=True)
assert opt['vocab_size'] == vocab.size, "Vocab size must match that in the saved model."

# load data
data_file = opt['data_dir'] + '/{}.json'.format(args.dataset)
print("Loading data from {} with batch size {}...".format(data_file, opt['batch_size']))
batch = DataLoader(data_file, opt['batch_size'], opt, vocab, opt['data_dir'] + '/mappings_{}.txt'.format(args.dataset), evaluation=True)

helper.print_config(opt)
label2id = constant.LABEL_TO_ID
id2label = dict([(v,k) for k,v in label2id.items()])

predictions = []
references = []
candidates = []
all_probs = []
batch_iter = tqdm(batch)

import json
from collections import defaultdict
with open('dataset/tacred/mappings_train.txt') as f:
    mappings = f.readlines()

with open('dataset/tacred/rules.json') as f:
    rules = json.load(f)
rule_dict = defaultdict(int)
for m in mappings:
    if 't_' in m or 's_' in m:
        for l, r in eval(m):
            r = ''.join(helper.word_tokenize(rules[r]))
            rule_dict[r] += 1

whole = set(rule_dict.keys())

x = 0
exact_match = 0
other = 0
rule_set = set()
rule_set2 = set()
for b in batch:#enumerate(batch_iter):
    preds, probs, decoded, loss = trainer.predict(b)
    predictions += preds
    all_probs += probs

    batch_size = len(preds)
    for i in range(batch_size):
        if id2label[preds[i]] != 'no_relation':
            output = decoded[i]
            candidate = []
            candidate = helper.parse_rule(output, vocab, b[0].view(batch_size, -1)[i])
            reference = [helper.parse_rule(batch.refs[x], vocab, b[0].view(batch_size, -1)[i])]
            if len(reference)!=0:
                if candidate not in reference:
                    rule_set.add(''.join(candidate))
                    rule_set2.add(''.join(reference[0]))
                    other += 1
                else:
                    exact_match += 1

                references.append(reference)
                candidates.append(candidate)
        x += 1
print (exact_match, other, len(rule_set), len(rule_set2))
for line in rule_set.intersection(rule_set2):
    print (line, rule_dict[line])
print (1)
for line in rule_set.difference(rule_set2):
    print (line, rule_dict[line])
print (2)
for line in rule_set2.difference(rule_set):
    print (line, rule_dict[line])
print (3)
for line in rule_set.difference(whole):
    print (line, rule_dict[line])
print (4)
predictions = [id2label[p] for p in predictions]
# for pred in predictions:
#     print (pred)
p, r, f1 = scorer.score(batch.gold(), predictions, verbose=True)
bleu = corpus_bleu(references, candidates)
print("{} set evaluate result: {:.2f}\t{:.2f}\t{:.2f}\t{:.4f}".format(args.dataset,p,r,f1,bleu))

print("Evaluation ended.")

