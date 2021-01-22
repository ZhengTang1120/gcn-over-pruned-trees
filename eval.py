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

d_wrong = list()
r_wrong = list()

d_correct = list()
r_correct = list()

d_wrong_no = list()
r_wrong_no = list()

d_correct_no = list()
r_correct_no = list()

for c, b in enumerate(batch_iter):
    preds, probs, decoded, loss = trainer.predict(b)
    predictions += preds
    all_probs += probs

    batch_size = len(preds)
    for i in range(batch_size):
        if len(batch.refs[x][0])!=0:
            if preds[i]!=batch.gold[x]:
                d_wrong.append(''.join(candidate))
                r_wrong.append(''.join(batch.refs[x][0]))
            else:
                d_correct.append(''.join(candidate))
                r_correct.append(''.join(batch.refs[x][0]))
        elif id2label[preds[i]] != 'no_relation':
            if preds[i]!=batch.gold[x]:
                d_wrong_no.append(''.join(candidate))
                r_wrong_no.append(''.join(batch.refs[x][0]))
            else:
                d_correct_no.append(''.join(candidate))
                r_correct_no.append(''.join(batch.refs[x][0]))
        x += 1

l = random.sample(range(len(d_wrong)), 50)
for i in l:
    print (d_wrong[i])
    print (r_wrong[i])
    print ()
print ()
print ()
print ()
print ()
l = random.sample(range(len(d_correct)), 50)
for i in l:
    print (d_correct[i])
    print (r_correct[i])
    print ()
print ()
print ()
print ()
print ()
l = random.sample(range(len(d_wrong_no)), 50)
for i in l:
    print (d_wrong_no[i])
    print (r_wrong_no[i])
    print ()
print ()
print ()
print ()
print ()
l = random.sample(range(len(d_correct_no)), 50)
for i in l:
    print (d_correct_no[i])
    print (r_correct_no[i])
    print ()

predictions = [id2label[p] for p in predictions]
# for pred in predictions:
#     print (pred)
p, r, f1 = scorer.score(batch.gold(), predictions, verbose=True)
bleu = corpus_bleu(references, candidates)
print("{} set evaluate result: {:.2f}\t{:.2f}\t{:.2f}\t{:.4f}".format(args.dataset,p,r,f1,bleu))

print("Evaluation ended.")

