import torch
import cont_time_cell
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np
from SA.encoder import Encoder, EncoderLayer
from SA.attn import FullAttention, AttentionLayer


class TokenEmbedding(nn.Module):
    def __init__(self, num_item, d_model):
        super(TokenEmbedding, self).__init__()

        self.item_emb = nn.Embedding(num_item + 1, d_model, padding_idx=num_item)

    def forward(self, x):
        x = self.item_emb(x)
        return x


class PositionalEmbedding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEmbedding, self).__init__()
        pe = torch.zeros(max_len, d_model).float()
        pe.require_grad = False

        position = torch.arange(0, max_len).float().unsqueeze(1)
        div_term = (torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model)).exp()

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return self.pe[:, :x.size(1)]


class Fusion(nn.Module):
    def __init__(self, settings):
        super(Fusion, self).__init__()

        self.num_item = settings['num_item']
        self.num_user = settings['num_user']
        self.device = settings['device']
        self.num_duration = settings['num_duration']
        self.beta = settings['beta']
        self.hid_dim = settings['hid_dim']
        self.n_heads = settings['n_heads']
        self.dropout_p = settings['dropout']
        self.e_layers = settings['e_layers']
        self.activation = settings['activation']
        self.output_attention = settings['output_attention']
        self.d_ff = settings['d_ff']
        self.batch_size = settings['batch_size']
        self.d_fcn = settings['d_fcn']
        self.activation_out = settings['activation_out']
        self.dl = settings['dl']
        self.mix = settings['mix']
        self.use_duration = settings['use_duration']

        self.side_dim = self.hid_dim
        if self.use_duration:
            self.side_dim = self.hid_dim * 2

        self.item_emb = TokenEmbedding(num_item=self.num_item, d_model=self.hid_dim).to(self.device)
        self.position_emb = PositionalEmbedding(d_model=self.hid_dim).to(self.device)
        self.lstm_cell = cont_time_cell.CTLSTMCell(self.hid_dim, self.beta, device=self.device)
        self.duration_emb = nn.Embedding(self.num_duration + 1, self.hid_dim, padding_idx=self.num_duration).to(
            self.device)
        self.dropout = nn.Dropout(p=self.dropout_p)
        self.hid_out = nn.Linear(self.hid_dim * 2, self.d_fcn).to(self.device)
        self.activation_out = F.relu if settings['activation_out'] == 'relu' else F.sigmoid
        self.out = nn.Linear(self.d_fcn, self.hid_dim, bias=True).to(self.device)
        self.gate_out = nn.Linear(self.hid_dim * 2, self.hid_dim, bias=True).to(self.device)
        self.side = nn.Linear(self.side_dim, self.hid_dim, bias=True).to(self.device)

        self.encoder = Encoder(
            [
                EncoderLayer(
                    AttentionLayer(FullAttention(False, attention_dropout=self.dropout_p,
                                        output_attention=self.output_attention), self.hid_dim, self.n_heads,
                                   self.device),
                    self.hid_dim, self.d_ff, dropout=self.dropout_p, activation=self.activation, device=self.device)
                for l in range(self.e_layers)
            ],
        )

    def dynamic_interest(self, items, time_interval, type_duration):
        numb_seq, seq_len = time_interval.shape

        self.hid_layer_minus = torch.zeros(numb_seq, self.hid_dim, dtype=torch.float32).to(self.device)
        self.cell_minus = torch.zeros(numb_seq, self.hid_dim, dtype=torch.float32).to(self.device)
        self.cell_bar_minus = torch.zeros(numb_seq, self.hid_dim, dtype=torch.float32).to(self.device)

        h_list = []
        for i in range(seq_len):
            item_input = self.item_emb(items[:, i])
            duration_input = self.duration_emb(type_duration[:, i])
            cell_i, cell_bar_updated, gate_decay, gate_output = self.lstm_cell(item_input, duration_input,
                                                                               self.hid_layer_minus, self.cell_minus,
                                                                               self.cell_bar_minus)

            self.cell_minus, self.hid_layer_minus = self.lstm_cell.decay(cell_i, cell_bar_updated, gate_decay,
                                                                         gate_output, time_interval[:, i])

            h_list.append(self.hid_layer_minus)

        dynamic_interest = h_list[-1]

        return dynamic_interest

    def static_interest(self, items, type_duration):

        pure_items = self.item_emb(items)
        side_infos = self.position_emb(items)
        side_infos = side_infos.expand(items.shape[0], side_infos.shape[1], side_infos.shape[2])

        if self.use_duration:
            side_duration = self.duration_emb(type_duration)

            side_infos = torch.cat((side_infos, side_duration), -1)

        side_infos = F.relu(self.side(side_infos))
        enc_out, attns = self.encoder(pure_items, side_infos, self.mix)
        attns = attns[-1].sum(1) / self.n_heads * 1.0
        seq_len = enc_out.size(1)
        W_s = (torch.ones(seq_len, dtype=torch.float32) / seq_len).unsqueeze(0).unsqueeze(0)
        W_s = W_s.expand(attns.shape[0], W_s.shape[1], W_s.shape[2]).to(self.device)

        static_interest = torch.matmul(attns, enc_out)
        static_interest = torch.matmul(W_s, static_interest).squeeze()

        return static_interest

    def train_batch(self, batch, predict_items, interest_using):
        items, time_interval, type_duration = batch
        if interest_using == 'dynamic':
            fusion_interest = self.dynamic_interest(items[:, -self.dl:], time_interval[:, -self.dl:],
                                                  type_duration[:, -self.dl:])
        elif interest_using == 'static':
            fusion_interest = self.static_interest(items, type_duration)

        else:
            static_interest = self.static_interest(items, type_duration)

            dynamic_interest = self.dynamic_interest(items[:, -self.dl:], time_interval[:, -self.dl:],
                                                 type_duration[:, -self.dl:])

            temp_interest = torch.cat((static_interest, dynamic_interest), -1)

            g = torch.sigmoid(self.gate_out(temp_interest))
            fusion_interest = torch.mul(static_interest, g) + torch.mul(dynamic_interest, 1 - g)

        predict_embedding = self.item_emb(predict_items)

        score = torch.matmul(predict_embedding, fusion_interest.unsqueeze(dim=2)).squeeze()

        return score
