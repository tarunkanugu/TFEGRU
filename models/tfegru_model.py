import torch
import torch.nn as nn
import torch.nn.functional as F


class TimeFrequencyEnhancedBlock(nn.Module):
    def __init__(self, seq_len, d_model):
        super().__init__()
        self.seq_len = seq_len
        self.d_model = d_model
        self.temporal_conv = nn.Sequential(
            nn.Conv1d(d_model, d_model, kernel_size=3, padding=1, groups=d_model),
            nn.BatchNorm1d(d_model),
            nn.GELU(),
        )
        freq_dim = seq_len // 2 + 1
        self.freq_proj = nn.Sequential(
            nn.Linear(freq_dim, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model),
        )
        self.fusion_gate = nn.Linear(d_model * 2, d_model)
        self.layer_norm = nn.LayerNorm(d_model)

    def forward(self, x):
        # x: (B, T, D)
        t = x.permute(0, 2, 1)           # (B, D, T)
        t = self.temporal_conv(t)
        t = t.permute(0, 2, 1)           # (B, T, D)

        f = torch.fft.rfft(x.permute(0, 2, 1), dim=-1)  # (B, D, F)
        f = f.abs()
        f = self.freq_proj(f)            # (B, D, D)
        f = f.permute(0, 2, 1)          # (B, D, D)

        if f.size(1) != t.size(1):
            f = F.interpolate(f.permute(0, 2, 1), size=t.size(1),
                              mode='linear', align_corners=False).permute(0, 2, 1)

        gate = torch.sigmoid(self.fusion_gate(torch.cat([t, f], dim=-1)))
        out = gate * t + (1 - gate) * f
        return self.layer_norm(out + x)


class MultiHeadSelfAttention(nn.Module):
    def __init__(self, d_model, num_heads, dropout=0.1):
        super().__init__()
        self.attn = nn.MultiheadAttention(
            embed_dim=d_model, num_heads=num_heads,
            dropout=dropout, batch_first=True)
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        out, _ = self.attn(x, x, x)
        return self.norm(x + self.dropout(out))


class TFEGRU(nn.Module):
    def __init__(self, input_dim=1, seq_len=32, d_model=64,
                 gru_hidden=128, gru_layers=2, num_heads=4,
                 output_dim=1, dropout=0.1):
        super().__init__()
        self.embed = nn.Sequential(nn.Linear(input_dim, d_model), nn.LayerNorm(d_model))
        self.tfeb = TimeFrequencyEnhancedBlock(seq_len, d_model)
        self.gru = nn.GRU(d_model, gru_hidden, gru_layers,
                          batch_first=True,
                          dropout=dropout if gru_layers > 1 else 0)
        self.attn = MultiHeadSelfAttention(gru_hidden, num_heads, dropout)
        self.fc = nn.Sequential(
            nn.Linear(gru_hidden, gru_hidden // 2), nn.GELU(),
            nn.Dropout(dropout), nn.Linear(gru_hidden // 2, output_dim))

    def forward(self, x):
        x = self.embed(x)
        x = self.tfeb(x)
        x, _ = self.gru(x)
        x = self.attn(x)
        return self.fc(x[:, -1, :])


class HybridTFEGRU_BiGRU(nn.Module):
    def __init__(self, input_dim=1, seq_len=32, d_model=64,
                 gru_hidden=128, gru_layers=2, num_heads=4,
                 output_dim=1, dropout=0.1):
        super().__init__()
        self.embed = nn.Sequential(nn.Linear(input_dim, d_model), nn.LayerNorm(d_model))
        self.tfeb = TimeFrequencyEnhancedBlock(seq_len, d_model)
        self.bigru = nn.GRU(d_model, gru_hidden, gru_layers,
                            batch_first=True, bidirectional=True,
                            dropout=dropout if gru_layers > 1 else 0)
        out_dim = gru_hidden * 2
        self.attn = MultiHeadSelfAttention(out_dim, num_heads, dropout)
        self.fc = nn.Sequential(
            nn.Linear(out_dim, out_dim // 2), nn.GELU(),
            nn.Dropout(dropout), nn.Linear(out_dim // 2, output_dim))

    def forward(self, x):
        x = self.embed(x)
        x = self.tfeb(x)
        x, _ = self.bigru(x)
        x = self.attn(x)
        return self.fc(x[:, -1, :])


class HybridTFEGRU_LSTM(nn.Module):
    def __init__(self, input_dim=1, seq_len=32, d_model=64,
                 gru_hidden=128, lstm_hidden=64, gru_layers=2,
                 num_heads=4, output_dim=1, dropout=0.1):
        super().__init__()
        self.embed = nn.Sequential(nn.Linear(input_dim, d_model), nn.LayerNorm(d_model))
        self.tfeb = TimeFrequencyEnhancedBlock(seq_len, d_model)
        self.gru = nn.GRU(d_model, gru_hidden, gru_layers,
                          batch_first=True,
                          dropout=dropout if gru_layers > 1 else 0)
        self.lstm = nn.LSTM(gru_hidden, lstm_hidden, 1, batch_first=True)
        self.attn = MultiHeadSelfAttention(lstm_hidden, max(1, num_heads // 2), dropout)
        self.fc = nn.Sequential(
            nn.Linear(lstm_hidden, lstm_hidden // 2), nn.GELU(),
            nn.Dropout(dropout), nn.Linear(lstm_hidden // 2, output_dim))

    def forward(self, x):
        x = self.embed(x)
        x = self.tfeb(x)
        x, _ = self.gru(x)
        x, _ = self.lstm(x)
        x = self.attn(x)
        return self.fc(x[:, -1, :])


class BaselineGRU(nn.Module):
    def __init__(self, input_dim=1, seq_len=32, hidden=128,
                 layers=2, output_dim=1, dropout=0.1):
        super().__init__()
        self.gru = nn.GRU(input_dim, hidden, layers, batch_first=True,
                          dropout=dropout if layers > 1 else 0)
        self.fc = nn.Linear(hidden, output_dim)

    def forward(self, x):
        out, _ = self.gru(x)
        return self.fc(out[:, -1, :])


MODEL_REGISTRY = {
    "TFEGRU": TFEGRU,
    "Hybrid-BiGRU": HybridTFEGRU_BiGRU,
    "Hybrid-LSTM": HybridTFEGRU_LSTM,
    "Baseline-GRU": BaselineGRU,
}


def build_model(name, **kwargs):
    if name not in MODEL_REGISTRY:
        raise ValueError(f"Unknown model: {name}")
    return MODEL_REGISTRY[name](**kwargs)
