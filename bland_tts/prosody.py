"""
Conversational Prosody & Pitch Contour Modeling.
Specialized in question inflection (rising F0 contours), emotional styling, and conversational micro-prosody.
"""

from typing import Tuple, Dict
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class ProsodyContourGenerator(nn.Module):
    """
    Synthesizes explicit F0 (fundamental frequency) and energy trajectories
    conditioned on sentence semantics, question markers, and emotional tone.
    """
    def __init__(
        self,
        base_f0: float = 180.0,     # Standard human conversational pitch baseline (Hz)
        f0_std: float = 30.0,
        latent_dim: int = 128
    ):
        super().__init__()
        self.base_f0 = base_f0
        self.f0_std = f0_std
        
        # Style embedding table: neutral, empathetic, urgent, sales
        self.style_embedding = nn.Embedding(4, 32)
        self.style_map = {"neutral": 0, "empathetic": 1, "urgent": 2, "sales": 3}
        
        # Projection from style + F0 + energy to latent condition vector
        self.prosody_proj = nn.Linear(32 + 2, latent_dim)

    def generate_f0_contour(
        self,
        num_frames: int,
        is_question: bool = False,
        emotion: str = "neutral",
        device: torch.device = torch.device("cpu")
    ) -> torch.Tensor:
        """
        Generates realistic pitch trajectory across time frames.
        Returns tensor: [1, 1, num_frames] in Hz.
        """
        t = torch.linspace(0, 1, num_frames, device=device)
        
        # Base pitch adjustment according to emotional tone
        pitch_shift = 0.0
        if emotion == "urgent":
            pitch_shift = 25.0
        elif emotion == "empathetic":
            pitch_shift = -10.0
        elif emotion == "sales":
            pitch_shift = 15.0

        f0_start = self.base_f0 + pitch_shift

        # Declination slope: natural statements drop ~15-25% over the phrase
        if not is_question:
            f0_contour = f0_start - 25.0 * t + 8.0 * torch.sin(2 * math.pi * 3 * t)
        else:
            # Question inflection: initial flat/gentle dip followed by a sharp rising upturn in the last 30%
            rising_point = 0.7
            slope_mask = torch.clamp((t - rising_point) / (1.0 - rising_point), min=0.0)
            rise_boost = slope_mask ** 1.8 * 65.0  # +65 Hz terminal upturn
            f0_contour = (f0_start - 10.0 * t) + rise_boost + 5.0 * torch.sin(2 * math.pi * 3 * t)

        return f0_contour.view(1, 1, num_frames)

    def generate_energy_contour(
        self,
        num_frames: int,
        emotion: str = "neutral",
        device: torch.device = torch.device("cpu")
    ) -> torch.Tensor:
        """
        Generates normalized energy contour [0.0 - 1.0] across frames.
        """
        t = torch.linspace(0, 1, num_frames, device=device)
        # Phrase envelope (attack, sustain, decay)
        attack = torch.clamp(t / 0.1, max=1.0)
        decay = torch.clamp((1.0 - t) / 0.15, max=1.0)
        envelope = attack * decay
        
        # Conversational syllable stress variation
        syllable_pulses = 0.8 + 0.2 * torch.sin(2 * math.pi * 4 * t)
        energy = envelope * syllable_pulses
        
        if emotion == "urgent":
            energy = energy * 1.2
        elif emotion == "empathetic":
            energy = energy * 0.9

        return energy.view(1, 1, num_frames)

    def forward(
        self,
        num_frames: int,
        is_question: bool = False,
        emotion: str = "neutral",
        device: torch.device = torch.device("cpu")
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Produces:
        - f0: [1, 1, num_frames]
        - energy: [1, 1, num_frames]
        - prosody_latent: [1, latent_dim, num_frames]
        """
        f0 = self.generate_f0_contour(num_frames, is_question, emotion, device)
        energy = self.generate_energy_contour(num_frames, emotion, device)
        
        # Style conditioning
        style_idx = torch.tensor([self.style_map.get(emotion, 0)], device=device)
        style_vec = self.style_embedding(style_idx) # [1, 32]
        style_expanded = style_vec.unsqueeze(-1).expand(-1, -1, num_frames) # [1, 32, T]
        
        # Normalize f0
        f0_norm = (f0 - self.base_f0) / self.f0_std
        
        # Concatenate features
        combined = torch.cat([style_expanded, f0_norm, energy], dim=1) # [1, 34, T]
        combined_perm = combined.permute(0, 2, 1) # [1, T, 34]
        
        prosody_latent = self.prosody_proj(combined_perm).permute(0, 2, 1) # [1, latent_dim, T]
        
        return f0, energy, prosody_latent
