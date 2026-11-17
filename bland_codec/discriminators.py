"""
Adversarial Discriminators for Neural Audio Codec Training.
Implements:
1. Multi-Period Discriminator (MPD): Captures harmonic pitch structures by reshaping 1D audio into 2D periodic grids.
2. Multi-Scale STFT Discriminator (MS-STFTD): Operates on complex STFT spectrograms across multiple resolutions.
"""

from typing import List, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F


class PeriodDiscriminator(nn.Module):
    """Discriminator for a single prime period."""
    def __init__(self, period: int, kernel_size: int = 5, stride: int = 3):
        super().__init__()
        self.period = period
        self.convs = nn.ModuleList([
            nn.utils.weight_norm(nn.Conv2d(1, 32, (kernel_size, 1), (stride, 1), padding=(2, 0))),
            nn.utils.weight_norm(nn.Conv2d(32, 128, (kernel_size, 1), (stride, 1), padding=(2, 0))),
            nn.utils.weight_norm(nn.Conv2d(128, 256, (kernel_size, 1), (stride, 1), padding=(2, 0))),
            nn.utils.weight_norm(nn.Conv2d(256, 512, (kernel_size, 1), 1, padding=(2, 0))),
        ])
        self.post_conv = nn.utils.weight_norm(nn.Conv2d(512, 1, (3, 1), 1, padding=(1, 0)))

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        """
        x: [B, 1, T]
        Returns: logits, list of intermediate feature maps
        """
        fmaps = []
        B, C, T = x.shape
        # Pad to multiple of period
        if T % self.period != 0:
            pad = self.period - (T % self.period)
            x = F.pad(x, (0, pad), "reflect")
            T = x.shape[-1]
            
        # Reshape to 2D: [B, 1, T // period, period]
        x_2d = x.view(B, C, T // self.period, self.period)

        out = x_2d
        for conv in self.convs:
            out = conv(out)
            out = F.leaky_relu(out, 0.1)
            fmaps.append(out)

        out = self.post_conv(out)
        fmaps.append(out)
        return out, fmaps


class MultiPeriodDiscriminator(nn.Module):
    """Multi-Period Discriminator spanning prime periods [2, 3, 5, 7, 11]."""
    def __init__(self, periods: List[int] = [2, 3, 5, 7, 11]):
        super().__init__()
        self.discriminators = nn.ModuleList([
            PeriodDiscriminator(p) for p in periods
        ])

    def forward(self, x: torch.Tensor) -> Tuple[List[torch.Tensor], List[List[torch.Tensor]]]:
        logits = []
        fmaps = []
        for disc in self.discriminators:
            logit, fmap = disc(x)
            logits.append(logit)
            fmaps.append(fmap)
        return logits, fmaps


class SingleScaleSTFTDiscriminator(nn.Module):
    """Discriminator operating on complex STFT spectrogram at a specific FFT scale."""
    def __init__(self, n_fft: int = 1024, hop_length: int = 256, win_length: int = 1024):
        super().__init__()
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.win_length = win_length
        self.register_buffer("window", torch.hann_window(win_length))

        self.convs = nn.ModuleList([
            nn.utils.weight_norm(nn.Conv2d(2, 32, kernel_size=3, stride=1, padding=1)),
            nn.utils.weight_norm(nn.Conv2d(32, 64, kernel_size=3, stride=(2, 2), padding=1)),
            nn.utils.weight_norm(nn.Conv2d(64, 128, kernel_size=3, stride=(2, 2), padding=1)),
            nn.utils.weight_norm(nn.Conv2d(128, 256, kernel_size=3, stride=(2, 2), padding=1)),
        ])
        self.post_conv = nn.utils.weight_norm(nn.Conv2d(256, 1, kernel_size=3, padding=1))

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        # x: [B, 1, T] -> STFT
        x_flat = x.squeeze(1)
        stft = torch.stft(
            x_flat,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.win_length,
            window=self.window,
            return_complex=True
        ) # [B, F, T]
        # Real and imaginary components as 2 channels: [B, 2, F, T]
        x_spec = torch.stack([stft.real, stft.imag], dim=1)

        fmaps = []
        out = x_spec
        for conv in self.convs:
            out = conv(out)
            out = F.leaky_relu(out, 0.1)
            fmaps.append(out)

        out = self.post_conv(out)
        fmaps.append(out)
        return out, fmaps


class MultiScaleSTFTDiscriminator(nn.Module):
    """Ensemble of STFT discriminators across multiple window scales (e.g., 512, 1024, 2048)."""
    def __init__(self, scales: List[Tuple[int, int, int]] = [(512, 128, 512), (1024, 256, 1024), (2048, 512, 2048)]):
        super().__init__()
        self.discriminators = nn.ModuleList([
            SingleScaleSTFTDiscriminator(n_fft, hop, win) for n_fft, hop, win in scales
        ])

    def forward(self, x: torch.Tensor) -> Tuple[List[torch.Tensor], List[List[torch.Tensor]]]:
        logits = []
        fmaps = []
        for disc in self.discriminators:
            logit, fmap = disc(x)
            logits.append(logit)
            fmaps.append(fmap)
        return logits, fmaps
