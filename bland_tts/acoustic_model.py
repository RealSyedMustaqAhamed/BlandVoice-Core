"""
Causal Acoustic Model for Low-Latency Codec Latent Generation.
Maps linguistic phonemes and prosodic conditioning to 50Hz continuous codec representations.
Designed for chunk-by-chunk autoregressive or flow generation.
"""

from typing import List, Tuple, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F

from .phonemizer import VOCAB
from .prosody import ProsodyContourGenerator


class CausalAcousticModel(nn.Module):
    """
    Acoustic Transformer predicting 50Hz neural codec latents from phonemes and prosody.
    Employs length regulation with conversational duration scaling.
    """
    def __init__(
        self,
        vocab_size: int = len(VOCAB),
        latent_dim: int = 128,
        hidden_dim: int = 192,
        num_layers: int = 4,
        num_heads: int = 4
    ):
        super().__init__()
        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim

        # Phoneme embedding
        self.phoneme_embed = nn.Embedding(vocab_size, hidden_dim)

        # Duration predictor: predicts frame count per phoneme
        self.duration_predictor = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Softplus()
        )

        # Causal Transformer Decoder layers
        decoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 2,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(decoder_layer, num_layers=num_layers)

        # Conditioning fusion
        self.fusion = nn.Linear(hidden_dim + latent_dim, hidden_dim)

        # Projection to codec latent space
        self.out_proj = nn.Linear(hidden_dim, latent_dim)

    def forward(
        self,
        token_ids: torch.Tensor,
        prosody_latent: torch.Tensor,
        durations: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        token_ids: [1, L_phonemes]
        prosody_latent: [1, latent_dim, T_frames]
        durations: Optional ground-truth frame durations
        Returns:
            predicted_codec_latents: [1, latent_dim, T_frames]
            predicted_durations: [1, L_phonemes]
        """
        device = token_ids.device
        ph_emb = self.phoneme_embed(token_ids) # [1, L, H]

        pred_dur = self.duration_predictor(ph_emb).squeeze(-1) # [1, L]
        
        # If durations not provided, use rounded predictions (minimum 2 frames per phoneme = 40ms)
        if durations is None:
            dur_counts = torch.clamp(torch.round(pred_dur + 3.0), min=2).long()
        else:
            dur_counts = durations.long()

        # Length regulator: expand phonemes to frames
        expanded_list = []
        for i in range(ph_emb.shape[1]):
            count = dur_counts[0, i].item()
            repeated = ph_emb[:, i:i+1, :].repeat(1, count, 1)
            expanded_list.append(repeated)
        expanded_ph = torch.cat(expanded_list, dim=1) # [1, T_expanded, H]

        # Match frame length with prosody condition
        T_target = prosody_latent.shape[-1]
        if expanded_ph.shape[1] < T_target:
            pad_len = T_target - expanded_ph.shape[1]
            expanded_ph = F.pad(expanded_ph, (0, 0, 0, pad_len), mode="replicate")
        elif expanded_ph.shape[1] > T_target:
            expanded_ph = expanded_ph[:, :T_target, :]

        # Fuse phonetic representation with prosody conditioning
        prosody_perm = prosody_latent.permute(0, 2, 1) # [1, T, latent_dim]
        fused = self.fusion(torch.cat([expanded_ph, prosody_perm], dim=-1)) # [1, T, H]

        # Causal mask for the transformer to ensure no future lookahead
        mask = nn.Transformer.generate_square_subsequent_mask(T_target).to(device)

        # Transformer forward
        hidden = self.transformer(fused, mask=mask, is_causal=True)

        # Project to codec latent dimension
        latents = self.out_proj(hidden).permute(0, 2, 1) # [1, latent_dim, T]
        return latents, pred_dur
