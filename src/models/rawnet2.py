import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math

class SincConv(nn.Module):
    def __init__(self, out_channels=80, kernel_size=251, sample_rate=16000):
        super(SincConv, self).__init__()
        self.out_channels = out_channels
        self.kernel_size = kernel_size if kernel_size % 2 != 0 else kernel_size + 1
        self.sample_rate = sample_rate
        
        self.low_hz = 30
        self.high_hz = sample_rate / 2 - (self.low_hz + 50)
        mel_low = 2595 * np.log10(1 + self.low_hz / 700)
        mel_high = 2595 * np.log10(1 + self.high_hz / 700)
        mel_points = np.linspace(mel_low, mel_high, self.out_channels + 1)
        hz_points = 700 * (10 ** (mel_points / 2595) - 1)
        
        self.band_low = nn.Parameter(torch.Tensor(hz_points[:-1]).view(-1, 1))
        self.band_high = nn.Parameter(torch.Tensor(hz_points[1:]).view(-1, 1))
        
        window = 0.54 - 0.46 * torch.cos(2 * math.pi * torch.arange(self.kernel_size, dtype=torch.float32) / (self.kernel_size - 1))
        self.register_buffer('window', window.view(1, -1))
        t = (torch.arange(self.kernel_size, dtype=torch.float32) - (self.kernel_size - 1) / 2) / self.sample_rate
        self.register_buffer('t', t.view(1, -1))

    def forward(self, x):
        low, high = self.band_low, self.band_high
        band_pass_left = 2 * high * torch.sinc(2 * high * self.t)
        band_pass_right = 2 * low * torch.sinc(2 * low * self.t)
        filters = (band_pass_left - band_pass_right) * self.window
        filters = filters.view(self.out_channels, 1, self.kernel_size)
        return F.conv1d(x, filters, padding=self.kernel_size // 2)

class ResidualBlock(nn.Module):
    def __init__(self, channels, conv_kernel=3):
        super(ResidualBlock, self).__init__()
        padding = conv_kernel // 2
        self.bn1 = nn.BatchNorm1d(channels)
        self.conv1 = nn.Conv1d(channels, channels, kernel_size=conv_kernel, padding=padding)
        self.bn2 = nn.BatchNorm1d(channels)
        self.conv2 = nn.Conv1d(channels, channels, kernel_size=conv_kernel, padding=padding)
        self.leaky_relu = nn.LeakyReLU(0.3)

    def forward(self, x):
        residual = x
        out = self.bn1(x)
        out = self.leaky_relu(out)
        out = self.conv1(out)
        out = self.bn2(out)
        out = self.leaky_relu(out)
        out = self.conv2(out)
        return out + residual

class RawNet2(nn.Module):
    """ Table 7 Configurations implementation """
    def __init__(self, sinc_filters=80, sinc_kernel=251, res_blocks=4, 
                 channel_scale=1.0, conv_kernel=3, dropout=0.3):
        super(RawNet2, self).__init__()
        scaled_filters = int(sinc_filters * channel_scale)
        
        self.sinc = SincConv(out_channels=scaled_filters, kernel_size=sinc_kernel)
        self.pool1 = nn.MaxPool1d(3)
        self.bn1 = nn.BatchNorm1d(scaled_filters)
        self.leaky_relu = nn.LeakyReLU(0.3)
        
        self.res_layers = nn.Sequential(
            *[ResidualBlock(scaled_filters, conv_kernel) for _ in range(res_blocks)]
        )
        
        expanded_ch = int(512 * channel_scale)
        self.conv_expand = nn.Conv1d(scaled_filters, expanded_ch, kernel_size=conv_kernel, padding=conv_kernel//2)
        self.pool2 = nn.MaxPool1d(3)
        self.dropout = nn.Dropout(dropout)
        
        self.gru = nn.GRU(input_size=expanded_ch, hidden_size=1024, num_layers=3, batch_first=True)
        self.fc = nn.Linear(1024, 2)

    def forward(self, x):
        if len(x.shape) == 2: x = x.unsqueeze(1)
        x = self.sinc(x)
        x = self.pool1(x)
        x = self.bn1(x)
        x = self.leaky_relu(x)
        x = self.res_layers(x)
        x = self.conv_expand(x)
        x = self.pool2(x)
        
        x = x.transpose(1, 2)
        x = self.dropout(x) # Enabled for MC Dropout later
        x, _ = self.gru(x)
        x = x[:, -1, :]
        return self.fc(x)