import torch
import torch.nn as nn
import torch.nn.functional as F


class EncoderLayer(nn.Module):
    def __init__(self, attention, d_model, d_ff=None, dropout=0.1, device=None, activation="relu"):
        super(EncoderLayer, self).__init__()
        d_ff = d_ff or 4 * d_model
        self.attention = attention
        self.conv1 = nn.Conv1d(in_channels=d_model, out_channels=d_ff, kernel_size=1).to(device)
        self.conv2 = nn.Conv1d(in_channels=d_ff, out_channels=d_model, kernel_size=1).to(device)
        self.norm1 = nn.LayerNorm(d_model).to(device)
        self.norm2 = nn.LayerNorm(d_model).to(device)
        self.dropout = nn.Dropout(dropout).to(device)
        self.activation = F.relu if activation == "relu" else F.gelu

    def forward(self, pure_x, x, attn_mask=None):
        new_x, attn = self.attention(
            x, x, pure_x,
            attn_mask=attn_mask
        )
        pure_x = pure_x + self.dropout(new_x)

        y = pure_x = self.norm1(pure_x)
        y = self.dropout(self.activation(self.conv1(y.transpose(-1, 1))))
        y = self.dropout(self.conv2(y).transpose(-1, 1))

        return self.norm2(pure_x + y), attn


class Encoder(nn.Module):
    def __init__(self, attn_layers, conv_layers=None, norm_layer=None):
        super(Encoder, self).__init__()
        self.attn_layers = nn.ModuleList(attn_layers)
        self.conv_layers = nn.ModuleList(conv_layers) if conv_layers is not None else None
        self.norm = norm_layer

    def forward(self, pure, side, mix, attn_mask=None):
        # x [B, L, D]
        attns = []
        if mix:
            x = pure + side
            if self.conv_layers is not None:
                for attn_layer, conv_layer in zip(self.attn_layers, self.conv_layers):
                    x, attn = attn_layer(x, x, attn_mask=attn_mask)
                    x = conv_layer(x)
                    attns.append(attn)
                x, attn = self.attn_layers[-1](x, x)
                attns.append(attn)
            else:
                for attn_layer in self.attn_layers:
                    x, attn = attn_layer(x, x, attn_mask=attn_mask)
                    attns.append(attn)

            if self.norm is not None:
                x = self.norm(x)

            return x, attns
        else:
            if self.conv_layers is not None:
                # print((pure + side).shape)
                for attn_layer, conv_layer in zip(self.attn_layers, self.conv_layers):
                    pure, attn = attn_layer(pure, pure + side, attn_mask=attn_mask)
                    pure = conv_layer(pure)
                    side = conv_layer(side)
                    attns.append(attn)
                pure, attn = self.attn_layers[-1](pure, pure + side)
                attns.append(attn)
            else:
                for attn_layer in self.attn_layers:
                    pure, attn = attn_layer(pure, pure + side, attn_mask=attn_mask)
                    attns.append(attn)

            if self.norm is not None:
                pure = self.norm(pure)

            return pure, attns
