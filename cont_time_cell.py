import torch
import torch.nn as nn
import torch.nn.functional as F


class CTLSTMCell(nn.Module):

    def __init__(self, hid_dim, beta=1.0, device=None):
        super(CTLSTMCell, self).__init__()

        self.device = device
        self.hid_dim = hid_dim

        self.linear1 = nn.Linear(hid_dim * 2, hid_dim * 7, bias=True).to(self.device)
        self.linear_du1 = nn.Linear(hid_dim, hid_dim, bias=False).to(self.device)
        self.linear_du2 = nn.Linear(hid_dim, hid_dim, bias=True).to(self.device)
        self.linear_du3 = nn.Linear(hid_dim, hid_dim, bias=False).to(self.device)
        self.beta = beta

    def forward(self, type_input, duration_input, hidden_t_i_minus, cell_t_i_minus, cell_bar_im1):
        dim_of_hidden = type_input.dim() - 1
        input_i = torch.cat((type_input, hidden_t_i_minus), dim=dim_of_hidden)
        output_i = self.linear1(input_i)

        gate_input, gate_forget, gate_output, gate_pre_c, \
        gate_input_bar, gate_forget_bar, gate_decay = output_i.chunk(7, dim_of_hidden)
        # todo:为gate_pre_duration1增加激活函数并观察结果
        gate_pre_duration1 = self.linear_du1(duration_input)
        gate_pre_duration2 = self.linear_du2(type_input)


        gate_input = torch.sigmoid(gate_input)
        gate_forget = torch.sigmoid(gate_forget)
        gate_output = torch.sigmoid(gate_output + self.linear_du3(duration_input))
        gate_duration = torch.sigmoid(gate_pre_duration1 + gate_pre_duration2)
        gate_input_bar = torch.sigmoid(gate_input_bar)
        gate_forget_bar = torch.sigmoid(gate_forget_bar)
        gate_decay = F.softplus(gate_decay, beta=self.beta)

        cell_i = gate_forget * cell_t_i_minus + gate_input * gate_pre_c * gate_duration
        cell_bar_i = gate_forget_bar * cell_bar_im1 + gate_input_bar * gate_pre_c * gate_duration

        return cell_i, cell_bar_i, gate_decay, gate_output

    def decay(self, cell_i, cell_bar_i, gate_decay, gate_output, interval):
        if interval.dim() < cell_i.dim():
            interval = interval.unsqueeze(cell_i.dim() - 1).expand_as(cell_i)
        cell_t_ip1_minus = cell_bar_i + (cell_i - cell_bar_i) * torch.exp(
            -gate_decay * interval)
        hidden_t_ip1_minus = gate_output * torch.tanh(cell_t_ip1_minus)

        return cell_t_ip1_minus, hidden_t_ip1_minus
