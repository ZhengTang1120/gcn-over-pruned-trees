"""
Run evaluation with saved models.
"""
import random
import argparse
from tqdm import tqdm
import torch

from data.loader import DataLoader
from model.trainer import BERTtrainer
from utils import torch_utils, scorer, constant, helper
from utils.vocab import Vocab

from nltk.translate.bleu_score import corpus_bleu, sentence_bleu

from transformers import BertTokenizer

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

tokenizer = BertTokenizer.from_pretrained('bert-large-cased')
special_tokens_dict = {'additional_special_tokens': constant.ENTITY_TOKENS}
num_added_toks = tokenizer.add_special_tokens(special_tokens_dict)

# load opt
model_file = args.model_dir + '/' + args.model
print("Loading model from {}".format(model_file))
opt = torch_utils.load_config(model_file)
trainer = BERTtrainer(opt)
trainer.encoder.model.resize_token_embeddings(len(tokenizer)) 
trainer.load(model_file)

# load vocab
vocab_file = args.model_dir + '/' +'vocab.pkl'
vocab = Vocab(vocab_file, load=True)
assert opt['vocab_size'] == vocab.size, "Vocab size must match that in the saved model."

# load data
data_file = opt['data_dir'] + '/{}.json'.format(args.dataset)
print("Loading data from {} with batch size {}...".format(data_file, opt['batch_size']))
batch = DataLoader(data_file, opt['batch_size'], opt, vocab, opt['data_dir'] + '/interval_{}.txt'.format(args.dataset), opt['data_dir'] + '/pattern_{}.txt'.format(args.dataset), tokenizer, evaluation=True)

helper.print_config(opt)
label2id = constant.LABEL_TO_ID
id2label = dict([(v,k) for k,v in label2id.items()])

predictions = []
tags = []
goldt = []
inputs = []

x = 0
exact_match = 0
other = 0
for c, b in enumerate(batch):
    preds, ts, tagged, ids = trainer.predict(b, id2label, tokenizer)
    predictions += preds
    tags += ts
    goldt += tagged
    batch_size = len(preds)
    for i in range(batch_size):
        inputs += [[tokenizer.convert_ids_to_tokens(j) for j in ids[i]]]
for i, p in enumerate(predictions):
        predictions[i] = id2label[p]
        if p!=0:
            if sum(goldt[i])!=0:
                print (id2label[p], batch.gold()[i])
                pass
                # print ([(goldt[i][j], tags[i][j], inputs[i][j])for j in range(len(inputs[i])) if inputs[i][j] != '[PAD]'])
                # print ([(goldt[i][j], tags[i][j], batch.words[i][j])for j in range(len(inputs[i])) if inputs[i][j] != '[PAD]'])
            elif sum(tags[i])!=0:
                print (id2label[p], batch.gold()[i])
                pass
                print ([(tags[i][j], inputs[i][j])for j in range(len(inputs[i])) if inputs[i][j] != '[PAD]'])
                print ([(tags[i][j], batch.words[i][j])for j in range(len(inputs[i])) if inputs[i][j] != '[PAD]'])
            print ()
p, r, f1 = scorer.score(batch.gold(), predictions, verbose=True)
print("{} set evaluate result: {:.2f}\t{:.2f}\t{:.2f}".format(args.dataset,p,r,f1))

print("Evaluation ended.")

