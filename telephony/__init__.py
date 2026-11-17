"""
Telephony Simulation & Conversational Robustness Package.
"""

from .codec_g711 import (
    linear_to_mulaw,
    mulaw_to_linear,
    linear_to_alaw,
    alaw_to_linear,
    apply_g711_telephony_channel
)
from .channel_effects import (
    apply_telephony_bandpass,
    simulate_packet_loss_bursts,
    apply_packet_loss_concealment,
    add_telephony_noise,
    full_telephony_channel_pipeline
)
from .vad_turn_taking import StreamingVAD, TurnTakingBargeInManager

__all__ = [
    "linear_to_mulaw",
    "mulaw_to_linear",
    "linear_to_alaw",
    "alaw_to_linear",
    "apply_g711_telephony_channel",
    "apply_telephony_bandpass",
    "simulate_packet_loss_bursts",
    "apply_packet_loss_concealment",
    "add_telephony_noise",
    "full_telephony_channel_pipeline",
    "StreamingVAD",
    "TurnTakingBargeInManager",
]
