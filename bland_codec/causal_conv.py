"""
Causal Convolutional Layers with Streaming State Buffering
Designed for sub-millisecond lookahead in streaming speech codecs.

At Bland.com, telephony agents require true zero-lookahead causality:
- No symmetric padding (zero future frames leaked into latent representations).
- Explicit streaming state buffers for chunk-by-chunk real-time inference without boundary clicks.
"""

from typing import Optional, Tuple, Dict, Any
import torch
import torch.nn as nn
import torch.nn.functional as F


class CausalConv1d(nn.Module):
    """
    1D Causal Convolution that pads only on the left (past).
    Supports stateful chunk-by-chunk streaming inference via `streaming_forward`.
    """
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        dilation: int = 1,
        groups: int = 1,
        bias: bool = True
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.dilation = dilation
        self.groups = groups
        
        # Effective receptive field padding on the left
        self.left_padding = (kernel_size - 1) * dilation
        
        self.conv = nn.Conv1d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=0,
            dilation=dilation,
            groups=groups,
            bias=bias
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Offline forward pass (batch mode with zero-init causal padding).
        x shape: [B, C, T]
        """
        if self.left_padding > 0:
            x = F.pad(x, (self.left_padding, 0), mode="constant", value=0.0)
        return self.conv(x)

    def streaming_forward(
        self, 
        x: torch.Tensor, 
        state: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Online streaming forward pass for chunk-by-chunk processing.
        x shape: [B, C, T_chunk]
        state: Cached past inputs of shape [B, C, left_padding]
        Returns: (output, next_state)
        """
        B, C, T = x.shape
        if self.left_padding == 0:
            return self.conv(x), torch.empty(B, C, 0, device=x.device, dtype=x.dtype)
            
        if state is None:
            state = torch.zeros(B, C, self.left_padding, device=x.device, dtype=x.dtype)
            
        # Concatenate cached history with new chunk
        x_padded = torch.cat([state, x], dim=-1)
        
        # New state is the last `left_padding` samples of x_padded
        next_state = x_padded[:, :, -self.left_padding:]
        
        out = self.conv(x_padded)
        return out, next_state


class CausalConvTranspose1d(nn.Module):
    """
    1D Causal Transposed Convolution for streaming upsampling (decoder).
    Trims future samples produced by transposed convolution filter span.
    """
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        bias: bool = True
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        
        self.conv_tr = nn.ConvTranspose1d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            bias=bias
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Offline forward pass.
        Trims right padding to enforce causal alignment:
        Output length = T * stride
        """
        out = self.conv_tr(x)
        # Trim right tail introduced by kernel_size - stride
        trim = self.kernel_size - self.stride
        if trim > 0:
            out = out[:, :, :-trim]
        return out

    def streaming_forward(
        self, 
        x: torch.Tensor, 
        state: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Streaming forward pass with overlap-add carryover buffer.
        """
        B, _, T = x.shape
        raw_out = self.conv_tr(x)
        expected_len = T * self.stride
        trim = self.kernel_size - self.stride
        
        if trim <= 0:
            return raw_out[:, :, :expected_len], torch.empty(B, self.out_channels, 0, device=x.device)
            
        valid_out = raw_out[:, :, :expected_len]
        carry_out = raw_out[:, :, expected_len:]
        
        if state is not None and state.shape[-1] > 0:
            state_len = min(state.shape[-1], valid_out.shape[-1])
            valid_out[:, :, :state_len] += state[:, :, :state_len]
            
        next_state = carry_out
        return valid_out, next_state


class SnakeBeta(nn.Module):
    """
    SnakeBeta activation function for high-fidelity periodic audio modeling (Ziyue et al.).
    x + (1/b) * sin^2(a * x)
    Enables neural vocoders to generalize pitch, harmonic richness, and high-frequency details.
    """
    def __init__(self, channels: int):
        super().__init__()
        self.alpha = nn.Parameter(torch.zeros(1, channels, 1))
        self.beta = nn.Parameter(torch.zeros(1, channels, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        alpha = torch.exp(self.alpha)
        beta = torch.exp(self.beta)
        return x + (1.0 / (beta + 1e-7)) * torch.pow(torch.sin(alpha * x), 2)


class CausalResnetBlock(nn.Module):
    """
    Dilated residual block with causal convolutions and SnakeBeta activations.
    Features stateful caching for real-time audio streams.
    """
    def __init__(self, channels: int, kernel_size: int = 3, dilation: int = 1):
        super().__init__()
        self.act1 = SnakeBeta(channels)
        self.conv1 = CausalConv1d(channels, channels, kernel_size, dilation=dilation)
        self.act2 = SnakeBeta(channels)
        self.conv2 = CausalConv1d(channels, channels, kernel_size=1, dilation=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        res = x
        x = self.act1(x)
        x = self.conv1(x)
        x = self.act2(x)
        x = self.conv2(x)
        return res + x

    def streaming_forward(
        self, 
        x: torch.Tensor, 
        state_dict: Optional[Dict[str, torch.Tensor]] = None
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        if state_dict is None:
            state_dict = {}
            
        res = x
        x_act = self.act1(x)
        x_conv1, next_s1 = self.conv1.streaming_forward(x_act, state_dict.get("conv1"))
        x_act2 = self.act2(x_conv1)
        x_conv2, next_s2 = self.conv2.streaming_forward(x_act2, state_dict.get("conv2"))
        
        new_states = {"conv1": next_s1, "conv2": next_s2}
        return res + x_conv2, new_states
