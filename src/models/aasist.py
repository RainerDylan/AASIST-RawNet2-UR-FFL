import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio

class GraphAttentionLayer(nn.Module):
    def __init__(self, in_features, out_features, n_heads, dropout):
        super(GraphAttentionLayer, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.n_heads = n_heads
        self.dropout = dropout
        
        self.W = nn.Parameter(torch.zeros(size=(in_features, out_features * n_heads)))
        self.a = nn.Parameter(torch.zeros(size=(2 * out_features, n_heads)))
        nn.init.xavier_uniform_(self.W.data, gain=1.414)
        nn.init.xavier_uniform_(self.a.data, gain=1.414)
        self.leakyrelu = nn.LeakyReLU(0.2)

    def forward(self, h):
        B, N, feat_dim = h.size()
        h_prime = torch.matmul(h, self.W).view(B, N, self.n_heads, self.out_features)
        
        a_src = self.a[:self.out_features, :].permute(1, 0)
        scores = (h_prime * a_src.unsqueeze(0).unsqueeze(0)).sum(dim=-1)
        e = self.leakyrelu(scores)
        
        attention = F.softmax(e, dim=1).unsqueeze(-1) 
        attention = F.dropout(attention, p=self.dropout, training=self.training)
        
        out = h_prime * attention
        return out.sum(dim=2)

class AASIST(nn.Module):
    """ Table 8 Configurations implementation with Spectrogram Front-end """
    def __init__(self, stft_window=512, stft_hop=160, freq_bins=128, 
                 gat_layers=3, heads=4, head_dim=64, hidden_dim=256, dropout=0.3):
        super(AASIST, self).__init__()
        
        # Spectrogram Extraction Matrix
        self.spectrogram = torchaudio.transforms.MelSpectrogram(
            sample_rate=16000,
            n_fft=stft_window,
            win_length=stft_window,
            hop_length=stft_hop,
            n_mels=freq_bins
        )
        
        # Spectrogram Encoder mapping freq_bins to hidden_dim
        self.encoder = nn.Sequential(
            nn.Conv1d(freq_bins, hidden_dim//2, kernel_size=3, padding=1),
            nn.BatchNorm1d(hidden_dim//2), nn.LeakyReLU(0.3),
            nn.Conv1d(hidden_dim//2, hidden_dim, kernel_size=3, padding=1),
            nn.BatchNorm1d(hidden_dim), nn.LeakyReLU(0.3)
        )
        
        # Configurable GAT Layers Stack
        self.gat_layers = nn.ModuleList([
            GraphAttentionLayer(hidden_dim if i==0 else head_dim, head_dim, heads, dropout)
            for i in range(gat_layers)
        ])
        
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(head_dim, 2)

    def forward(self, x):
        if len(x.shape) == 2: x = x.unsqueeze(1) # (B, 1, L)
        x = x.squeeze(1)
        
        # STFT to Spectrogram (B, Freq_Bins, Temporal_Frames)
        spec = self.spectrogram(x) 
        spec = torch.log(spec + 1e-9) # Log scaling
        
        x = self.encoder(spec)
        x = x.transpose(1, 2) # Nodes as Temporal Frames
        
        for gat in self.gat_layers:
            x = gat(x)
            
        x = x.transpose(1, 2)
        x = self.pool(x).squeeze(-1)
        return self.fc(x)