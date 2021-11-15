import torch
import cont_time_cell
import torch.nn as nn
import numpy as np


class Conttime(nn.Module):
    def __init__(self, settings):
        super(Conttime, self).__init__()

        self.num_item = settings['num_item']
        self.num_user = settings['num_user']
        self.device = settings['device']
        self.num_duration = settings['num_duration']
        self.beta = settings['beta']
        self.hid_dim = settings['hid_dim']

        self.item_emb = nn.Embedding(self.num_item + 1, self.hid_dim).to(self.device)
        self.lstm_cell = cont_time_cell.CTLSTMCell(self.hid_dim, self.beta, device=self.device)
        self.duration_emb = nn.Embedding(self.num_duration + 1, self.hid_dim).to(self.device)
        self.hidden_lambda = nn.Linear(self.hid_dim, self.num_item).to(self.device)

    def forward(self, items, time_interval, type_duration, seq_len_lists):
        numb_seq, seq_len = time_interval.shape

        self.hid_layer_minus = torch.zeros(numb_seq, self.hid_dim, dtype=torch.float32).to(self.device)
        self.cell_minus = torch.zeros(numb_seq, self.hid_dim, dtype=torch.float32).to(self.device)
        self.cell_bar_minus = torch.zeros(numb_seq, self.hid_dim, dtype=torch.float32).to(self.device)

        h_list = []
        h_final_list = []

        for i in range(seq_len):
            item_input = self.item_emb(items[:, i])
            duration_input = self.duration_emb(type_duration[:, i])
            cell_i, cell_bar_updated, gate_decay, gate_output = self.lstm_cell(item_input, duration_input,
                                                                               self.hid_layer_minus, self.cell_minus,
                                                                               self.cell_bar_minus)

            self.cell_minus, self.hid_layer_minus = self.lstm_cell.decay(cell_i, cell_bar_updated, gate_decay,
                                                                         gate_output, time_interval[:, i])

            h_list.append(self.hid_layer_minus)

        for idx in range(numb_seq):
            h_final_list.append(h_list[seq_len_lists[idx] - 1][0])

        h_out = torch.stack(h_list)
        h_final_out = torch.stack(h_final_list)

        return h_out, h_final_out

    def train_batch(self, batch, seq_len_lists, predict_items):
        items, interval, duration = batch
        h_out, h_final_out = self.forward(items, interval, duration, seq_len_lists)
        predict_embedding = self.item_emb(predict_items)

        score = torch.matmul(predict_embedding, h_final_out.unsqueeze(dim=2)).squeeze()

        return score
