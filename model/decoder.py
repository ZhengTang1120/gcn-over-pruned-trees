import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np

from utils import constant, torch_utils

class Attention(nn.Module):
    """
    A GCN layer with attention on deprel as edge weights.
    """
    
    def __init__(self, input_size, query_size):
        super(Attention, self).__init__()
        self.input_size = input_size
        self.query_size = query_size
        # self.ulinear = nn.Linear(input_size, attn_size)
        # self.vlinear = nn.Linear(query_size, attn_size, bias=False)
        # self.tlinear = nn.Linear(attn_size, 1)
        self.weight = nn.Parameter(torch.Tensor(input_size, query_size))
        self.init_weights()

    def init_weights(self):
        # self.ulinear.weight.data.normal_(std=0.001)
        # self.vlinear.weight.data.normal_(std=0.001)
        # self.tlinear.weight.data.zero_() # use zero to give uniform attention at the beginning
        self.weight.data.normal_(std=0.001)

    def forward(self, x, x_mask, q):
        """
        x : batch_size * seq_len * input_size
        q : batch_size * query_size
        """
        batch_size, seq_len, _ = x.size()

        # x_proj = self.ulinear(x.contiguous().view(-1, self.input_size)).view(
        #     batch_size, seq_len, self.attn_size)
        # q_proj = self.vlinear(q.view(-1, self.query_size)).contiguous().view(
        #     batch_size, self.attn_size).unsqueeze(1).expand(
        #         batch_size, seq_len, self.attn_size)
        # projs = [x_proj, q_proj]
        # scores = self.tlinear(torch.tanh(sum(projs)).view(-1, self.attn_size)).view(
        #     batch_size, seq_len)

        x_proj = torch.matmul(x, self.weight)
        scores = torch.bmm(x_proj, q.view(batch_size, self.query_size, 1)).view(batch_size, seq_len)

        # mask padding
        scores.data.masked_fill_(x_mask.data, -float('inf'))
        weights = F.softmax(scores, dim=1)

        return weights

class Decoder(nn.Module):
    def __init__(self, opt):
        super(Decoder, self).__init__()
        self.embed_size = 100
        self.hidden_size = opt['hidden_dim']
        self.output_size = opt['rule_size']
        self.n_layers = opt['num_layers']

        self.embed = nn.Embedding(self.output_size+constant.MAX_SEN_LEN, self.embed_size, padding_idx=constant.PAD_ID)
        self.dropout = nn.Dropout(opt['input_dropout'], inplace=True)
        self.attention = Attention(self.hidden_size, self.embed_size + 2 * self.hidden_size)
        self.rnn = nn.LSTM(self.embed_size + self.hidden_size, self.hidden_size,
                          self.n_layers, dropout=opt['input_dropout'])
        self.out = nn.Linear(self.hidden_size, self.output_size)

        self.p_gen_linear = nn.Linear(self.embed_size + 2 * self.hidden_size, 1)

    def forward(self, input, masks, last_hidden, encoder_outputs, extend_vocab):

        # Get the embedding of the current input word (last output word)
        embedded = self.embed(input).unsqueeze(0)  # (1,B,N)
        embedded = self.dropout(embedded)

        batch_size = encoder_outputs.size(0)
        # Calculate attention weights and apply to encoder outputs
        query = torch.cat((last_hidden[0].view(batch_size, -1), embedded.squeeze(0)), 1)
        attn_weights = self.attention(encoder_outputs, masks, query).view(batch_size, 1, -1)
        context = attn_weights.bmm(encoder_outputs)  # (B,1,N)
        context = context.transpose(0, 1)  # (1,B,N)
        # Combine embedded input word and attended context, run through RNN
        rnn_input = torch.cat([embedded, context], 2)
        rnn_output, hidden = self.rnn(rnn_input, last_hidden)
        rnn_output = rnn_output.squeeze(0)  # (1,B,N) -> (B,N)
        output = self.out(rnn_output) #torch.cat([output, context], 1))
        output = F.softmax(output, dim=1)

        #pointer generator
        p_gen_input = torch.cat((rnn_output, context.squeeze(0), embedded.squeeze(0)), 1)
        p_gen = F.sigmoid(self.p_gen_linear(p_gen_input))
        output_ = p_gen * output
        attn_weights_ = (1 - p_gen) * attn_weights.view(batch_size, -1)
        extra_zeros = torch.zeros(attn_weights_.size()).cuda()
        print (output_)
        print (extend_vocab)
        output_ = torch.cat((output_, extra_zeros), 1)
        output_ += 1e-10
        final_output = output_.scatter_add(1, extend_vocab, attn_weights_)
        print (attn_weights_)
        print (final_output)
        print ("-----------------")
        final_output = torch.log(final_output+1e-7)

        return final_output, hidden





