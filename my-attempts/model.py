import torch
import torch.nn as nn
from torch.autograd import Variable

# Linear Embedding
from torch.distributions import Categorical


class GraphEmbedding(nn.Module):
    def __init__(self, input_size, embedding_size):
        super(GraphEmbedding, self).__init__()
        self.embedding = nn.Linear(input_size, embedding_size)
    def forward(self, inputs):
        return self.embedding(inputs)

class Encoder(nn.Module):
    def __init__(self, n_embedded, n_hidden):
        super(Encoder, self).__init__()
        self.encoder = nn.LSTM(n_embedded, n_hidden, batch_first=True)
    def forward(self, embedded_current_city):
        encoded, (encoder_hidden, encoder_context) = self.encoder(embedded_current_city)
        return encoded, encoder_hidden, encoder_context

class Decoder(nn.Module): #gyrnaei probability distribution gia to epomeno city pou tha episkeftoume
  def __init__(self, embedding_size, hidden_size):
      super(Decoder, self).__init__()
      self.decoder=nn.LSTM(embedding_size, hidden_size, batch_first=True)
  def forward(self, current_city, hidden_state,context):
      _, (decoder_hidden, decoder_context)  = self.decoder(current_city, (hidden_state, context))
      return decoder_hidden,decoder_context

class Attention(nn.Module):
    def __init__(self, hidden_size, C=10):
        super(Attention, self).__init__()
        self.C = C
        self.W_q = nn.Linear(hidden_size, hidden_size)
        self.W_k = nn.Linear(hidden_size, hidden_size)
        self.W_v = nn.Linear(hidden_size, 1)
    def forward(self, query, target):
        """  Args:  query: [batch_size x hidden_size]  target:   [batch_size x seq_len x hidden_size]  """
        batch_size, seq_len, _ = target.shape
        query = self.W_q(query).unsqueeze(1).repeat(1, seq_len, 1)  # [batch_size x seq_len x hidden_size]
        target = self.W_k(target)  # [batch_size x seq_len x hidden_size]
        logits = self.W_v(torch.tanh(query + target)).squeeze(-1)
        logits = self.C * torch.tanh(logits)
        return target, logits

class EncoderDecoderModel(nn.Module):
    def __init__(self, hidden_size,  embedding_size,tanh_exploration=10):
        super(EncoderDecoderModel, self).__init__()
        self.hidden_size = hidden_size
        self.embedding = GraphEmbedding(2, embedding_size)
        self.encoder = Encoder(n_embedded=embedding_size, n_hidden=hidden_size)
        self.decoder = Decoder(embedding_size=embedding_size, hidden_size=hidden_size)
        self.decoder_start_input = nn.Parameter(torch.FloatTensor(embedding_size))
        self.pointer = Attention(hidden_size, C=tanh_exploration)
    def reward(self, sample_solution):
        """
        Args:  sample_solution seq_len of [batch_size]
            torch.LongTensor [batch_size x seq_len x 2]
        """
        batch_size, seq_len, _ = sample_solution.size()
        tour_len = Variable(torch.zeros([batch_size]))
        if isinstance(sample_solution, torch.cuda.FloatTensor):
            tour_len = tour_len.cuda()
        for i in range(seq_len - 1):
            tour_len += torch.norm(sample_solution[:, i, :] - sample_solution[:, i + 1, :], dim=-1)

        tour_len += torch.norm(sample_solution[:, seq_len - 1, :] - sample_solution[:, 0, :], dim=-1)

        return tour_len
    def forward(self, current_batch):
        """ Args:  inputs: [batch_size x seq_len x 2]   """
        batch_size = current_batch.shape[0]
        seq_len = current_batch.shape[1]

        # gia ola ta cities
        embedded_cities = self.embedding(current_batch)
        encoder_outputs, hidden,context = self.encoder(embedded_cities)
        prev_chosen_logprobs = []
        preb_chosen_indices = []
        mask = torch.zeros(batch_size, seq_len, dtype=torch.bool)

        decoder_input = self.decoder_start_input.unsqueeze(0).repeat(batch_size, 1)

        for index in range(seq_len): #tha broume gia kathe batch sample poio einai to epomeno city
            hidden, context = self.decoder(torch.nan_to_num(decoder_input).unsqueeze(1), hidden, context)# dinei to next city probability: batch_size * num_nodes

            query = hidden.squeeze()

            _, logits = self.pointer(query.unsqueeze(-1).transpose(1,0), encoder_outputs)
            _mask = mask.clone()
            logits[_mask] = -100000.0
            probs = torch.softmax(logits, dim=-1)
            cat = Categorical(probs)
            chosen_city = cat.sample()
            mask[[i for i in range(batch_size)], chosen_city] = True
            log_probs = cat.log_prob(chosen_city)
            decoder_input = embedded_cities.gather(1, chosen_city[:, None, None].repeat(1, 1, self.hidden_size)).squeeze(1)
            prev_chosen_logprobs.append(log_probs)
            preb_chosen_indices.append(chosen_city)


        probs = torch.stack(prev_chosen_logprobs, 1)
        actions = torch.stack(preb_chosen_indices, 1)
        R = self.reward(current_batch.gather(1, actions.unsqueeze(2).repeat(1, 1, 2)))

        return R, probs, actions