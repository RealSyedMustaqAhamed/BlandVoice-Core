"""
Causal Neural Audio Encoder for Streaming Telephony & Conversational Voice.
Downsamples 16 kHz PCM audio by 320x to 50 Hz latent representations (20ms/frame).
Zero lookahead ensures strictly causal inference for sub-100ms conversational turn-taking.
"""

from typing import List, Tuple, Dict, Optional, Any
import torch
import torch.nn as nn
from .causal_conv import CausalConv1d, CausalResnetBlock


class CausalAudioEncoder(nn.Module):
    """
    Downsamples 16kHz audio to 50Hz latent representations.
    Strides: (2, 4, 5, 8) -> Total downsampling factor = 320.
    1 second (16000 samples) -> 50 latent frames (each frame = 20ms).
    """
    def __init__(
        self,
        in_channels: int = 1,
        latent_dim: int = 128,
        base_channels: int = 32,
        strides: Tuple[int, ...] = (2, 4, 5, 8),
        kernel_sizes: Tuple[int, ...] = (5, 8, 10, 16)
    ):
        super().__init__()
        self.latent_dim = latent_dim
        self.total_downsample_factor = 1
        for s in strides:
            self.total_downsample_factor *= s

        # Initial input projection
        self.init_conv = CausalConv1d(in_channels, base_channels, kernel_size=7)

        # Downsampling stages
        self.down_blocks = nn.ModuleList()
        curr_channels = base_channels
        
        for stride, k_size in zip(strides, kernel_sizes):
            out_ch = min(curr_channels * 2, 256)
            block = nn.ModuleDict({
                "res": CausalResnetBlock(curr_channels, kernel_size=3, dilation=1),
                "down": CausalConv1d(curr_channels, out_ch, kernel_size=k_size, stride=stride)
            })
            self.down_blocks.append(block)
            curr_channels = out_ch

        # Final projection to latent dimension
        self.final_res = CausalResnetBlock(curr_channels, kernel_size=3)
        self.proj = CausalConv1d(curr_channels, latent_dim, kernel_size=3)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Offline batch forward pass.
        x: [B, 1, T_audio] (16kHz audio)
        Returns: [B, latent_dim, T_audio // 320]
        """
        out = self.init_conv(x)
        for stage in self.down_blocks:
            out = stage["res"](out)
            out = stage["down"](out)
        out = self.final_res(out)
        out = self.proj(out)
        return out

    def streaming_forward(
        self, 
        x_chunk: torch.Tensor, 
        states: Optional[Dict[str, Any]] = None
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """
        Chunk-by-chunk streaming encoder.
        x_chunk: [B, 1, T_chunk] where T_chunk is a multiple of 320 samples (e.g. 320, 640, 1600)
        states: dictionary of previous layer states
        Returns: (latent_chunk, updated_states)
        """
        if states is None:
            states = {}
            
        new_states = {}
        out, new_states["init_conv"] = self.init_conv.streaming_forward(
            x_chunk, states.get("init_conv")
        )

        for i, stage in enumerate(self.down_blocks):
            res_state = states.get(f"res_{i}")
            down_state = states.get(f"down_{i}")
            
            out, new_res_state = stage["res"].streaming_forward(out, res_state)
            out, new_down_state = stage["down"].streaming_forward(out, down_state)
            
            new_states[f"res_{i}"] = new_res_state
            new_states[f"down_{i}"] = new_down_state

        final_res_state = states.get("final_res")
        out, new_states["final_res"] = self.final_res.streaming_forward(out, final_res_state)
        
        proj_state = states.get("proj")
        out, new_states["proj"] = self.proj.streaming_forward(out, proj_state)

        return out, new_states
