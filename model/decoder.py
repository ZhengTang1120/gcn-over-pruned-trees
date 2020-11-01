import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np

class Decoder(nn.Module):
    def __init__(self, opt):
        super(Decoder, self).__init__()
        self.embed_size = 100
        self.hidden_size = opt['hidden_dim']
        self.output_size = opt['rule_size']
        self.n_layers = opt['num_layers']

        self.embed = nn.Embedding(self.output_size, self.embed_size, padding_idx=constant.PAD_ID)
        self.dropout = nn.Dropout(opt['dropout'], inplace=True)
        self.attention = Attention(2 * self.hidden_size, self.embed_size + 2 * self.hidden_size)
        self.rnn = nn.LSTM(self.embed_size + 2 * self.hidden_size, self.hidden_size,
                          self.n_layers, dropout=opt['dropout'])
        self.out = nn.Linear(self.hidden_size, self.output_size)

    def forward(self, input, masks, last_hidden, encoder_outputs):

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
        output, hidden = self.rnn(rnn_input, last_hidden)
        output = output.squeeze(0)  # (1,B,N) -> (B,N)
        # context = context.squeeze(0)
        output = self.out(output) #torch.cat([output, context], 1))
        output = F.log_softmax(output, dim=1)
        return output, hidden, attn_weights