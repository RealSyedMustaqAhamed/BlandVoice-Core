"""
Vector Quantization Suite for Extreme Audio Compression & Generative Speech Modeling.
Includes:
1. Residual Vector Quantization (RVQ) with EMA updates, codebook restart, and commitment loss.
2. Finite Scalar Quantization (FSQ) for discrete-continuous latent bridge without codebook collapse.
3. Codebook perplexity and utilization metrics for experimental tracking.
"""

from typing import List, Tuple, Dict, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F


class VectorQuantizer(nn.Module):
    """
    Single-layer Vector Quantizer with EMA codebook updates and dead-code restart.
    """
    def __init__(
        self,
        dim: int,
        codebook_size: int = 1024,
        decay: float = 0.99,
        epsilon: float = 1e-5,
        threshold_ema_dead_code: float = 2.0
    ):
        super().__init__()
        self.dim = dim
        self.codebook_size = codebook_size
        self.decay = decay
        self.epsilon = epsilon
        self.threshold_ema_dead_code = threshold_ema_dead_code

        # Codebook embedding
        init_bound = 1.0 / codebook_size
        embedding = torch.randn(codebook_size, dim) * init_bound
        self.register_buffer("embedding", embedding)
        self.register_buffer("cluster_size", torch.zeros(codebook_size))
        self.register_buffer("embedding_avg", embedding.clone())

    @torch.no_grad()
    def _expire_codes(self, x: torch.Tensor):
        """
        Re-initializes codes with low utilization to random inputs from the current batch.
        Prevents codebook collapse in low-entropy regimes.
        """
        dead_codes = self.cluster_size < self.threshold_ema_dead_code
        if not dead_codes.any():
            return
            
        n_dead = dead_codes.sum().item()
        # Sample n_dead vectors from input x
        flat_x = x.reshape(-1, self.dim)
        if flat_x.size(0) >= n_dead:
            perm = torch.randperm(flat_x.size(0), device=x.device)[:n_dead]
            sampled = flat_x[perm]
            self.embedding[dead_codes] = sampled
            self.embedding_avg[dead_codes] = sampled
            self.cluster_size[dead_codes] = self.threshold_ema_dead_code

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        x: [B, C, T] where C == self.dim
        Returns:
            quantized: [B, C, T]
            indices: [B, T]
            loss: scalar commitment loss
        """
        B, C, T = x.shape
        # Flatten to [N, C] where N = B * T
        x_flat = x.permute(0, 2, 1).contiguous().reshape(-1, C)

        # Compute L2 distances: ||x - e||^2 = ||x||^2 + ||e||^2 - 2 * x @ e^T
        distances = (
            torch.sum(x_flat ** 2, dim=1, keepdim=True)
            + torch.sum(self.embedding ** 2, dim=1)
            - 2 * torch.matmul(x_flat, self.embedding.t())
        )

        indices_flat = torch.argmin(distances, dim=-1)
        indices = indices_flat.view(B, T)

        # Quantize
        quantized_flat = F.embedding(indices_flat, self.embedding)
        quantized = quantized_flat.view(B, T, C).permute(0, 2, 1).contiguous()

        # Update EMA codebook during training
        if self.training:
            encodings = F.one_hot(indices_flat, self.codebook_size).type(x_flat.dtype)
            
            # EMA for cluster size
            n_tokens = encodings.sum(0)
            self.cluster_size.data.mul_(self.decay).add_(n_tokens, alpha=1 - self.decay)
            
            # EMA for embedding sum
            embed_sum = torch.matmul(encodings.t(), x_flat)
            self.embedding_avg.data.mul_(self.decay).add_(embed_sum, alpha=1 - self.decay)
            
            # Laplace smoothing for cluster size
            n = self.cluster_size.sum()
            smoothed_cluster_size = (
                (self.cluster_size + self.epsilon)
                / (n + self.codebook_size * self.epsilon)
                * n
            )
            
            # Normalized embedding update
            self.embedding.data.copy_(self.embedding_avg / smoothed_cluster_size.unsqueeze(1))
            self._expire_codes(x)

        # Straight-Through Estimator (STE)
        loss = F.mse_loss(quantized.detach(), x)
        quantized = x + (quantized - x).detach()

        return quantized, indices, loss

    def decode_indices(self, indices: torch.Tensor) -> torch.Tensor:
        """
        indices: [B, T]
        Returns: [B, C, T]
        """
        quantized = F.embedding(indices, self.embedding) # [B, T, C]
        return quantized.permute(0, 2, 1).contiguous()


class ResidualVectorQuantizer(nn.Module):
    """
    Residual Vector Quantizer (RVQ) chaining multiple VQ stages.
    Achieves high bitrate resolution with low-complexity codebooks.
    e.g., 4 codebooks x 1024 entries = 40 bits/frame. At 50 Hz frame rate = 2.0 kbps.
    """
    def __init__(
        self,
        dim: int,
        num_quantizers: int = 4,
        codebook_size: int = 1024,
        decay: float = 0.99
    ):
        super().__init__()
        self.dim = dim
        self.num_quantizers = num_quantizers
        self.codebook_size = codebook_size
        self.layers = nn.ModuleList([
            VectorQuantizer(dim, codebook_size, decay=decay)
            for _ in range(num_quantizers)
        ])

    def forward(
        self, 
        x: torch.Tensor, 
        n_q: Optional[int] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Dict[str, float]]:
        """
        x: [B, C, T]
        n_q: Number of quantizers to use (supports variable bitrate inference)
        Returns:
            quantized: [B, C, T]
            indices: [B, n_q, T]
            total_loss: scalar commitment loss
            metrics: dict containing perplexity and active code ratio
        """
        n_q = n_q or self.num_quantizers
        residual = x
        quantized_out = torch.zeros_like(x)
        all_indices: List[torch.Tensor] = []
        total_loss = torch.tensor(0.0, device=x.device)
        perplexities: List[float] = []

        for i in range(n_q):
            layer = self.layers[i]
            quantized_i, indices_i, loss_i = layer(residual)
            residual = residual - quantized_i
            quantized_out = quantized_out + quantized_i
            all_indices.append(indices_i)
            total_loss = total_loss + loss_i

            # Compute perplexity for monitoring codebook utilization
            counts = torch.bincount(indices_i.view(-1), minlength=self.codebook_size).float()
            probs = counts / counts.sum().clamp(min=1.0)
            entropy = -(probs * torch.log(probs + 1e-9)).sum()
            perplexity = torch.exp(entropy).item()
            perplexities.append(perplexity)

        indices = torch.stack(all_indices, dim=1) # [B, n_q, T]
        avg_perplexity = sum(perplexities) / len(perplexities)
        active_ratio = avg_perplexity / self.codebook_size

        bits_per_code = int(torch.log2(torch.tensor(float(self.codebook_size))).item())
        metrics = {
            "avg_codebook_perplexity": avg_perplexity,
            "codebook_utilization_ratio": active_ratio,
            "bitrate_kbps": (n_q * bits_per_code * 50) / 1000.0
        }

        return quantized_out, indices, total_loss, metrics

    def decode_indices(self, indices: torch.Tensor) -> torch.Tensor:
        """
        indices: [B, n_q, T]
        Returns: [B, C, T]
        """
        B, n_q, T = indices.shape
        quantized = torch.zeros(B, self.dim, T, device=indices.device)
        for i in range(n_q):
            quantized = quantized + self.layers[i].decode_indices(indices[:, i, :])
        return quantized


class FiniteScalarQuantizer(nn.Module):
    """
    Finite Scalar Quantization (FSQ) (Mentovai et al., 2023).
    Maps continuous latents to bounded integers without a codebook.
    - Zero codebook collapse
    - No commitment loss needed
    - Fully deterministic and highly compressible
    Levels e.g., [8, 5, 5, 5] yields 8*5*5*5 = 1000 codes in dim=4.
    """
    def __init__(self, levels: List[int] = [8, 6, 5, 5]):
        super().__init__()
        self.levels = levels
        self.dim = len(levels)
        self.codebook_size = 1
        for l in levels:
            self.codebook_size *= l
            
        self.register_buffer("_levels", torch.tensor(levels, dtype=torch.float32))
        self.register_buffer("_basis", torch.cumprod(
            torch.tensor([1] + levels[:-1]), dim=0
        ).long())

    def bound(self, z: torch.Tensor) -> torch.Tensor:
        """Applies tanh and scales to level ranges."""
        return torch.tanh(z)

    def quantize(self, z: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        z: [B, dim, T]
        Returns:
            z_q: [B, dim, T]
            indices: [B, T]
        """
        B, D, T = z.shape
        z_bounded = self.bound(z)
        
        # Scale to [-0.5, levels - 0.5]
        half_width = (self._levels.view(1, D, 1) - 1) / 2.0
        z_scaled = (z_bounded + 1.0) * half_width
        
        # Rounding with STE
        z_rounded = torch.round(z_scaled)
        z_q = z_scaled + (z_rounded - z_scaled).detach()
        
        # Normalize back to [-1, 1] range for decoder stability
        z_norm = (z_q / half_width) - 1.0
        
        # Compute scalar discrete indices: sum(z_i * basis_i)
        indices_int = z_rounded.long()
        flat_indices = torch.sum(indices_int * self._basis.view(1, D, 1), dim=1) # [B, T]
        
        return z_norm, flat_indices

    def forward(self, z: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Dict[str, float]]:
        z_q, indices = self.quantize(z)
        loss = torch.tensor(0.0, device=z.device) # FSQ needs no commitment loss!
        
        counts = torch.bincount(indices.view(-1), minlength=self.codebook_size).float()
        probs = counts / counts.sum().clamp(min=1.0)
        entropy = -(probs * torch.log(probs + 1e-9)).sum()
        perplexity = torch.exp(entropy).item()
        
        metrics = {
            "avg_codebook_perplexity": perplexity,
            "codebook_utilization_ratio": perplexity / self.codebook_size,
            "bitrate_kbps": (torch.log2(torch.tensor(self.codebook_size)).item() * 50) / 1000.0
        }
        return z_q, indices.unsqueeze(1), loss, metrics
