"""
Loss Functions for Neural Audio Codec Training.
Implements:
1. Multi-Scale STFT Loss (Spectral Convergence + Log Magnitude)
2. Discriminator Adversarial Loss (LS-GAN / Hinge)
3. Feature Matching Loss across MPD and MS-STFTD layers
4. Codebook Commitment Loss
"""

from typing import List, Tuple, Dict
import torch
import torch.nn as nn
import torch.nn.functional as F


class SingleScaleSTFTLoss(nn.Module):
    """Spectral convergence and log STFT magnitude loss at a single scale."""
    def __init__(self, n_fft: int = 1024, hop_length: int = 256, win_length: int = 1024):
        super().__init__()
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.win_length = win_length
        self.register_buffer("window", torch.hann_window(win_length))

    def forward(self, x: torch.Tensor, x_hat: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # x, x_hat: [B, 1, T]
        x_flat = x.squeeze(1)
        x_hat_flat = x_hat.squeeze(1)

        mag_x = torch.abs(torch.stft(
            x_flat, self.n_fft, self.hop_length, self.win_length, self.window, return_complex=True
        ))
        mag_x_hat = torch.abs(torch.stft(
            x_hat_flat, self.n_fft, self.hop_length, self.win_length, self.window, return_complex=True
        ))

        # Spectral convergence: Frobenius norm ratio
        sc_loss = torch.norm(mag_x - mag_x_hat, p="fro") / (torch.norm(mag_x, p="fro") + 1e-7)

        # Log magnitude loss (L1)
        log_mag_x = torch.log(mag_x + 1e-7)
        log_mag_x_hat = torch.log(mag_x_hat + 1e-7)
        mag_loss = F.l1_loss(log_mag_x_hat, log_mag_x)

        return sc_loss, mag_loss


class MultiScaleSTFTLoss(nn.Module):
    """Multi-Scale STFT Loss across window sizes [512, 1024, 2048]."""
    def __init__(self, scales: List[Tuple[int, int, int]] = [(512, 128, 512), (1024, 256, 1024), (2048, 512, 2048)]):
        super().__init__()
        self.losses = nn.ModuleList([
            SingleScaleSTFTLoss(n_fft, hop, win) for n_fft, hop, win in scales
        ])

    def forward(self, x: torch.Tensor, x_hat: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        total_sc = torch.tensor(0.0, device=x.device)
        total_mag = torch.tensor(0.0, device=x.device)
        for loss_fn in self.losses:
            sc, mag = loss_fn(x, x_hat)
            total_sc = total_sc + sc
            total_mag = total_mag + mag
        return total_sc / len(self.losses), total_mag / len(self.losses)


class CodecLoss(nn.Module):
    """
    Unified loss for neural audio codec training.
    Balances time-frequency reconstruction, adversarial fidelity, and latent discretization.
    """
    def __init__(
        self,
        stft_scales: List[Tuple[int, int, int]] = [(512, 128, 512), (1024, 256, 1024), (2048, 512, 2048)],
        lambda_sc: float = 1.0,
        lambda_mag: float = 1.0,
        lambda_feat: float = 2.0,
        lambda_adv: float = 1.0,
        lambda_commit: float = 0.25
    ):
        super().__init__()
        self.stft_loss = MultiScaleSTFTLoss(stft_scales)
        self.lambda_sc = lambda_sc
        self.lambda_mag = lambda_mag
        self.lambda_feat = lambda_feat
        self.lambda_adv = lambda_adv
        self.lambda_commit = lambda_commit

    def generator_loss(
        self,
        x: torch.Tensor,
        x_hat: torch.Tensor,
        fmaps_real: List[List[torch.Tensor]],
        fmaps_fake: List[List[torch.Tensor]],
        logits_fake: List[torch.Tensor],
        commit_loss: torch.Tensor
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        # 1. Multi-scale STFT loss
        sc_loss, mag_loss = self.stft_loss(x, x_hat)

        # 2. Adversarial generator loss (Least-Squares GAN: target 1)
        adv_loss = torch.tensor(0.0, device=x.device)
        for logit in logits_fake:
            adv_loss = adv_loss + F.mse_loss(logit, torch.ones_like(logit))
        adv_loss = adv_loss / len(logits_fake)

        # 3. Feature matching loss across discriminator layers
        feat_loss = torch.tensor(0.0, device=x.device)
        total_fmaps = 0
        for fmap_r, fmap_f in zip(fmaps_real, fmaps_fake):
            for layer_r, layer_f in zip(fmap_r, fmap_f):
                feat_loss = feat_loss + F.l1_loss(layer_f, layer_r.detach())
                total_fmaps += 1
        if total_fmaps > 0:
            feat_loss = feat_loss / total_fmaps

        # Total generator loss
        total_loss = (
            self.lambda_sc * sc_loss
            + self.lambda_mag * mag_loss
            + self.lambda_adv * adv_loss
            + self.lambda_feat * feat_loss
            + self.lambda_commit * commit_loss
        )

        metrics = {
            "loss_stft_sc": sc_loss.item(),
            "loss_stft_mag": mag_loss.item(),
            "loss_adv_gen": adv_loss.item(),
            "loss_feat_match": feat_loss.item(),
            "loss_commit": commit_loss.item(),
            "loss_gen_total": total_loss.item()
        }
        return total_loss, metrics

    def discriminator_loss(
        self,
        logits_real: List[torch.Tensor],
        logits_fake: List[torch.Tensor]
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        # Least-Squares GAN: Real -> 1, Fake -> 0
        loss_d = torch.tensor(0.0, device=logits_real[0].device)
        for r, f in zip(logits_real, logits_fake):
            loss_d = loss_d + F.mse_loss(r, torch.ones_like(r)) + F.mse_loss(f, torch.zeros_like(f))
        loss_d = loss_d / len(logits_real)

        return loss_d, {"loss_discriminator": loss_d.item()}
