"""CNN–wavelet-enhanced recurrent regression and benchmark ablations."""
from __future__ import annotations

import torch
from torch import nn


def morlet(x: torch.Tensor, frequency: float = 2.1, width_divisor: float = 3.0) -> torch.Tensor:
    """The real Morlet form from Eq. (8): cos(2.1*x) * exp(-x²/3)."""
    return torch.cos(frequency * x) * torch.exp(-x.square() / width_divisor)


class AMCNNWaveletBiLSTM(nn.Module):
    """Paper-specified convolution sizes and a documented wavelet approximation.

    Standard PyTorch LSTM gates are preserved. The Morlet nonlinearity acts on
    the recurrent output states, as the paper does not provide modified gate
    equations for an exact wavelet LSTM cell.
    """
    def __init__(self, features: int, conv_filters=(32, 64, 128),
                 kernels=(3, 3, 3), pools=(2, 2, 2), lstm_hidden=156,
                 attention_units=312, dropout=0.3, use_wavelet=True,
                 use_attention=True, bidirectional=True):
        super().__init__()
        self.use_wavelet = use_wavelet
        self.use_attention = use_attention
        blocks = []
        channels = features
        for count, width, pool in zip(conv_filters, kernels, pools):
            blocks += [nn.Conv1d(channels, count, kernel_size=width, padding=width // 2),
                       nn.ReLU(), nn.MaxPool1d(kernel_size=pool, stride=pool)]
            channels = count
        self.cnn = nn.Sequential(*blocks)
        self.dropout = nn.Dropout(dropout)
        self.lstm = nn.LSTM(channels, lstm_hidden, batch_first=True,
                            bidirectional=bidirectional)
        output_dim = lstm_hidden * (2 if bidirectional else 1)
        self.attention = nn.Sequential(nn.Linear(output_dim, attention_units),
                                       nn.Tanh(), nn.Linear(attention_units, 1))
        self.head = nn.Linear(output_dim, 1)
        self.last_attention = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 3 or x.shape[1] < 8:
            raise ValueError('Expected [batch, at-least-8-cycles, features]')
        h = self.cnn(x.transpose(1, 2)).transpose(1, 2)
        h, _ = self.lstm(self.dropout(h))
        if self.use_wavelet:
            h = morlet(h)
        if self.use_attention:
            # Table I specifies sigmoid; normalizing the gates produces
            # a convex temporal weighting, consistent with Eqs. (9)–(11).
            a = torch.sigmoid(self.attention(h)).squeeze(-1)
            a = a / a.sum(dim=1, keepdim=True).clamp_min(1e-9)
            self.last_attention = a.detach()
            h = (h * a.unsqueeze(-1)).sum(dim=1)
        else:
            h = h.mean(dim=1)
        return self.head(h).squeeze(-1)


def make_model(kind: str, features: int, **kwargs):
    opts = dict(use_wavelet=True, use_attention=True, bidirectional=True)
    ablations = {
        'AM-CNN-WBiLSTM': opts,
        'AM-CNN-BiLSTM': dict(use_wavelet=False, use_attention=True, bidirectional=True),
        'AM-CNN-LSTM': dict(use_wavelet=False, use_attention=True, bidirectional=False),
        'CNN-BiLSTM': dict(use_wavelet=False, use_attention=False, bidirectional=True),
        'CNN-LSTM': dict(use_wavelet=False, use_attention=False, bidirectional=False),
    }
    if kind not in ablations:
        raise ValueError(f'Unsupported variant {kind}; choose from {list(ablations)}')
    return AMCNNWaveletBiLSTM(features, **kwargs, **ablations[kind])
