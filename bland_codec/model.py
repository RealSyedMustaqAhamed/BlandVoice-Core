"""
BlandStreamingCodec: End-to-End Real-Time Neural Audio Codec.
Unified Causal Encoder -> Multi-Rate RVQ/FSQ Quantizer -> Causal Decoder.
Enables sub-100ms conversational turn-taking and streaming TTS audio synthesis.
"""

from typing import Tuple, Dict, Optional, Any, Union
import torch
import torch.nn as nn

from .encoder import CausalAudioEncoder
from .decoder import CausalAudioDecoder
from .quantization import ResidualVectorQuantizer, FiniteScalarQuantizer


class BlandStreamingCodec(nn.Module):
    """
    Production-ready Causal Neural Audio Codec designed for enterprise phone agents.
    - Sample rate: 16,000 Hz
    - Downsampling factor: 320x (50 Hz frame rate = 20ms per token)
    - Quantization: Residual Vector Quantizer (RVQ) or Finite Scalar Quantizer (FSQ)
    - Bitrates: 1.5 kbps (3 quantizers), 2.0 kbps (4 quantizers), 3.0 kbps (6 quantizers)
    - Receptive field: 100% causal (zero lookahead into future audio)
    """
    def __init__(
        self,
        sample_rate: int = 16000,
        latent_dim: int = 128,
        quantizer_type: str = "rvq",  # 'rvq' or 'fsq'
        num_quantizers: int = 4,
        codebook_size: int = 1024,
        encoder_base_channels: int = 32,
        decoder_base_channels: int = 256,
        strides: Tuple[int, ...] = (2, 4, 5, 8)
    ):
        super().__init__()
        self.sample_rate = sample_rate
        self.latent_dim = latent_dim
        self.quantizer_type = quantizer_type.lower()
        self.downsample_factor = 1
        for s in strides:
            self.downsample_factor *= s

        # 1. Causal Audio Encoder
        self.encoder = CausalAudioEncoder(
            in_channels=1,
            latent_dim=latent_dim,
            base_channels=encoder_base_channels,
            strides=strides
        )

        # 2. Latent Quantizer
        if self.quantizer_type == "rvq":
            self.quantizer = ResidualVectorQuantizer(
                dim=latent_dim,
                num_quantizers=num_quantizers,
                codebook_size=codebook_size
            )
        elif self.quantizer_type == "fsq":
            # 4-dim FSQ: [8, 6, 5, 5] = 1200 codes per frame
            self.fsq_proj_in = nn.Conv1d(latent_dim, 4, kernel_size=1)
            self.fsq_proj_out = nn.Conv1d(4, latent_dim, kernel_size=1)
            self.quantizer = FiniteScalarQuantizer(levels=[8, 6, 5, 5])
        else:
            raise ValueError(f"Unsupported quantizer: {quantizer_type}")

        # 3. Causal Audio Decoder (Vocoder)
        # Mirror strides for symmetric upsampling
        rev_strides = tuple(reversed(strides))
        self.decoder = CausalAudioDecoder(
            latent_dim=latent_dim,
            out_channels=1,
            base_channels=decoder_base_channels,
            strides=rev_strides
        )

    def encode(
        self, 
        audio: torch.Tensor, 
        n_q: Optional[int] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Dict[str, float]]:
        """
        Encodes raw 16kHz audio into quantized latents and discrete token IDs.
        audio: [B, 1, T_audio]
        Returns:
            quantized_latent: [B, latent_dim, T_frames]
            discrete_indices: [B, n_q, T_frames]
            commit_loss: scalar tensor
            metrics: dict
        """
        continuous_latent = self.encoder(audio)
        
        if self.quantizer_type == "rvq":
            z_q, indices, loss, metrics = self.quantizer(continuous_latent, n_q=n_q)
        else:
            z_projected = self.fsq_proj_in(continuous_latent)
            z_q_sub, indices, loss, metrics = self.quantizer(z_projected)
            z_q = self.fsq_proj_out(z_q_sub)

        return z_q, indices, loss, metrics

    def decode(self, indices_or_latent: torch.Tensor) -> torch.Tensor:
        """
        Decodes either discrete token IDs or continuous quantized latents to PCM audio.
        If indices: shape [B, n_q, T_frames] (long)
        If latent: shape [B, latent_dim, T_frames] (float)
        Returns: [B, 1, T_audio]
        """
        if indices_or_latent.dtype == torch.long:
            if self.quantizer_type == "rvq":
                z_q = self.quantizer.decode_indices(indices_or_latent)
            else:
                raise NotImplementedError("FSQ index decode requires sub-projection")
        else:
            z_q = indices_or_latent
            
        return self.decoder(z_q)

    def forward(
        self, 
        audio: torch.Tensor, 
        n_q: Optional[int] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Dict[str, float]]:
        """
        End-to-end forward pass.
        audio: [B, 1, T_audio]
        Returns:
            reconstructed_audio: [B, 1, T_audio]
            discrete_indices: [B, n_q, T_frames]
            commit_loss: scalar tensor
            metrics: dict
        """
        z_q, indices, loss, metrics = self.encode(audio, n_q=n_q)
        audio_recon = self.decoder(z_q)
        
        # Ensure exact sample-level length match
        if audio_recon.shape[-1] != audio.shape[-1]:
            min_len = min(audio_recon.shape[-1], audio.shape[-1])
            audio_recon = audio_recon[:, :, :min_len]
            
        return audio_recon, indices, loss, metrics

    def streaming_step_encode(
        self, 
        audio_chunk: torch.Tensor, 
        enc_state: Optional[Dict[str, Any]] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, Any]]:
        """
        Stream a 20ms or 40ms audio chunk -> get latent token indices immediately.
        audio_chunk: [1, 1, T_chunk] (e.g. 320 samples = 20ms @ 16kHz)
        """
        continuous_chunk, next_enc_state = self.encoder.streaming_forward(audio_chunk, enc_state)
        if self.quantizer_type == "rvq":
            z_q, indices, _, _ = self.quantizer(continuous_chunk)
        else:
            z_proj = self.fsq_proj_in(continuous_chunk)
            z_q_sub, indices, _, _ = self.quantizer(z_proj)
            z_q = self.fsq_proj_out(z_q_sub)
            
        return z_q, indices, next_enc_state

    def streaming_step_decode(
        self, 
        z_chunk: torch.Tensor, 
        dec_state: Optional[Dict[str, Any]] = None
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """
        Stream newly predicted latent tokens -> immediately get 16kHz PCM audio chunk.
        z_chunk: [1, latent_dim, T_token_chunk] (e.g. 1 frame = 20ms)
        """
        audio_chunk, next_dec_state = self.decoder.streaming_forward(z_chunk, dec_state)
        return audio_chunk, next_dec_state
