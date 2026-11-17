"""
Causal Neural Audio Decoder (Vocoder) with Streaming Receptive Field Management.
Upsamples 50 Hz quantized latent tokens by 320x back to 16 kHz high-fidelity PCM audio.
Zero lookahead allows instant generation of audio chunks as tokens are predicted by the TTS model.
"""

from typing import List, Tuple, Dict, Optional, Any
import torch
import torch.nn as nn
from .causal_conv import CausalConv1d, CausalConvTranspose1d, CausalResnetBlock, SnakeBeta


class CausalAudioDecoder(nn.Module):
    """
    Decodes 50Hz latent tokens back to 16kHz audio waveform.
    Upsampling stages: strides (8, 5, 4, 2) -> Total factor 320.
    1 latent frame (20ms) -> 320 audio PCM samples.
    """
    def __init__(
        self,
        latent_dim: int = 128,
        out_channels: int = 1,
        base_channels: int = 256,
        strides: Tuple[int, ...] = (8, 5, 4, 2),
        kernel_sizes: Tuple[int, ...] = (16, 10, 8, 5)
    ):
        super().__init__()
        self.latent_dim = latent_dim
        self.total_upsample_factor = 1
        for s in strides:
            self.total_upsample_factor *= s

        # Project from latent space to vocoder channel dimension
        self.init_conv = CausalConv1d(latent_dim, base_channels, kernel_size=7)

        # Upsampling stages
        self.up_blocks = nn.ModuleList()
        curr_channels = base_channels
        
        for stride, k_size in zip(strides, kernel_sizes):
            out_ch = max(curr_channels // 2, 32)
            block = nn.ModuleDict({
                "up": CausalConvTranspose1d(curr_channels, out_ch, kernel_size=k_size, stride=stride),
                "res": CausalResnetBlock(out_ch, kernel_size=3, dilation=1)
            })
            self.up_blocks.append(block)
            curr_channels = out_ch

        # Final projection to audio PCM
        self.final_act = SnakeBeta(curr_channels)
        self.final_conv = CausalConv1d(curr_channels, out_channels, kernel_size=7)

    def forward(self, z_q: torch.Tensor) -> torch.Tensor:
        """
        Offline batch forward pass.
        z_q: [B, latent_dim, T_latent]
        Returns: [B, 1, T_latent * 320]
        """
        out = self.init_conv(z_q)
        for stage in self.up_blocks:
            out = stage["up"](out)
            out = stage["res"](out)
        out = self.final_act(out)
        out = self.final_conv(out)
        return torch.tanh(out)

    def streaming_forward(
        self, 
        z_chunk: torch.Tensor, 
        states: Optional[Dict[str, Any]] = None
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """
        Chunk-by-chunk streaming vocoder.
        z_chunk: [B, latent_dim, T_chunk_tokens]
        states: cached filter memory from previous audio chunks
        Returns: (audio_chunk, updated_states)
        """
        if states is None:
            states = {}

        new_states = {}
        out, new_states["init_conv"] = self.init_conv.streaming_forward(
            z_chunk, states.get("init_conv")
        )

        for i, stage in enumerate(self.up_blocks):
            up_state = states.get(f"up_{i}")
            res_state = states.get(f"res_{i}")
            
            out, new_up_state = stage["up"].streaming_forward(out, up_state)
            out, new_res_state = stage["res"].streaming_forward(out, res_state)
            
            new_states[f"up_{i}"] = new_up_state
            new_states[f"res_{i}"] = new_res_state

        out = self.final_act(out)
        final_conv_state = states.get("final_conv")
        out, new_states["final_conv"] = self.final_conv.streaming_forward(out, final_conv_state)

        return torch.tanh(out), new_states
