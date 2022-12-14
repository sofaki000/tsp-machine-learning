from torch.autograd import Variable
import torch
import torch.nn as nn
import torch.nn.functional as F


class PointerNetwork(nn.Module):
    def __init__(self, input_size, emb_size, weight_size, answer_seq_len, hidden_size=512):
        super(PointerNetwork, self).__init__()
        self.hidden_size = hidden_size
        self.input_size = input_size
        self.answer_seq_len = answer_seq_len
        self.weight_size = weight_size
        self.emb_size = emb_size

        self.emb = nn.Embedding(num_embeddings =input_size+1,embedding_dim = emb_size)  # embed inputs

        self.enc = nn.LSTM(emb_size, hidden_size, batch_first=True)
        self.dec = nn.LSTMCell(emb_size, hidden_size) # LSTMCell's input is always batch first

        self.W1 = nn.Linear(hidden_size, weight_size, bias=False) # blending encoder
        self.W2 = nn.Linear(hidden_size, weight_size, bias=False) # blending decoder
        self.vt = nn.Linear(weight_size, 1, bias=False) # scaling sum of enc and dec by v.T

    def forward(self, input):
        batch_size = input.size(0)
        input = self.emb(input) # (bs, sequence_length, embd_size)-> this seems to depend on the max number in input
        # max_num_in_input = input.max().item()
        #  input = nn.Embedding(max_num_in_input+1, self.emb_size)(input)

        # Encoding
        encoder_states, (hidden_state, cell_state) = self.enc(input) # encoder_state: (bs, sequence_length, H)
        encoder_last_state = encoder_states.transpose(1, 0) # (sequence_length, bs, H)

        # Decoding states initialization
        decoder_input = Variable(torch.zeros(batch_size, self.emb_size)) # (bs, embd_size)
        hidden = Variable(torch.zeros([batch_size, self.hidden_size]))   # (bs, h)
        cell_state = encoder_last_state[-1]   # cell state: last encoder state                           # (bs, h)

        probs = []
        # Decoding
        for i in range(self.answer_seq_len): # range(M)
            hidden, cell_state = self.dec(decoder_input, (hidden, cell_state)) # (bs, h), (bs, h)
            # Compute blended representation at each decoder time step
            blend1 = self.W1(encoder_last_state)          # (L, bs, W)
            blend2 = self.W2(hidden)                  # (bs, W)
            blend_sum = F.tanh(blend1 + blend2)    # (L, bs, W) # typos apo to paper
            out = self.vt(blend_sum).squeeze()        # (L, bs)
            if len(out.size())==1:
                out = F.log_softmax(out.contiguous(), -1)
            else:
                out = F.log_softmax(out.transpose(0, 1).contiguous(), -1) # (bs, L)
            probs.append(out) #gia ton epomeno node (gia kathe batch), dinei ena probability poio tha einai
            # apo ta available nodes
        probs = torch.stack(probs, dim=1) # (bs, M, L)
        return probs