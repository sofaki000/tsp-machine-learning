import math
import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Categorical
from modules import Attention, GraphEmbedding


class RNNTSP(nn.Module):
    def __init__(self,   embedding_size, hidden_size,  seq_len,  n_glimpses,   tanh_exploration):
        super(RNNTSP, self).__init__()

        self.embedding_size = embedding_size
        self.hidden_size = hidden_size
        self.n_glimpses = n_glimpses
        self.seq_len = seq_len

        self.embedding = GraphEmbedding(2, embedding_size)
        self.encoder = nn.LSTM(embedding_size, hidden_size, batch_first=True)
        self.decoder = nn.LSTM(embedding_size, hidden_size, batch_first=True)
        self.pointer = Attention(hidden_size, C=tanh_exploration)
        self.glimpse = Attention(hidden_size)

        self.decoder_start_input = nn.Parameter(torch.FloatTensor(embedding_size))
        self.decoder_start_input.data.uniform_(-(1. / math.sqrt(embedding_size)), 1. / math.sqrt(embedding_size))
    def forward(self, inputs):
        """ Args:  inputs: [batch_size x seq_len x 2]   """
        batch_size = inputs.shape[0]
        seq_len = inputs.shape[1]

        embedded = self.embedding(inputs) #batch_size * seq_len * emb_size
        encoder_outputs, (hidden, context) = self.encoder(embedded) #enc_out: b*seq_len*emb_size

        prev_chosen_logprobs = []
        preb_chosen_indices = []
        mask = torch.zeros(batch_size, self.seq_len, dtype=torch.bool)
        decoder_input = self.decoder_start_input.unsqueeze(0).repeat(batch_size, 1)
        for index in range(seq_len):
            _, (hidden, context) = self.decoder(decoder_input.unsqueeze(1), (hidden, context))

            query = hidden.squeeze(0) # apo ton decoder epikentrwnetai sto hidden
            for _ in range(self.n_glimpses):
                ref, logits = self.glimpse(query, encoder_outputs)
                _mask = mask.clone()
                logits[_mask] = -100000.0
                query = torch.matmul(ref.transpose(-1, -2), torch.softmax(logits, dim=-1).unsqueeze(-1)).squeeze(-1)

            _, logits = self.pointer(query, encoder_outputs)

            _mask = mask.clone()
            logits[_mask] = -100000.0
            probs = torch.softmax(logits, dim=-1)
            cat = Categorical(probs)
            chosen_city = cat.sample()
            mask[[i for i in range(batch_size)], chosen_city] = True
            log_probs = cat.log_prob(chosen_city)
            decoder_input = embedded.gather(1, chosen_city[:, None, None].repeat(1, 1, self.hidden_size)).squeeze(1)
            prev_chosen_logprobs.append(log_probs)
            preb_chosen_indices.append(chosen_city) #
        # me to stack ta kanei mia seira-> ena sequence swsto (gt htan mperdemena prin)
        return torch.stack(prev_chosen_logprobs, 1), torch.stack(preb_chosen_indices, 1) # num_nodes list (px 30 nodes) pou to kathena exei ta coordinates