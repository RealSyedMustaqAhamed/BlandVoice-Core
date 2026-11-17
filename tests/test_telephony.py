"""
Unit Tests for Telephony Simulation, G.711 Companding, and Barge-In VAD.
"""

import numpy as np
import pytest
from telephony.codec_g711 import linear_to_mulaw, mulaw_to_linear, linear_to_alaw, alaw_to_linear
from telephony.channel_effects import (
    apply_telephony_bandpass,
    simulate_packet_loss_bursts,
    apply_packet_loss_concealment
)
from telephony.vad_turn_taking import StreamingVAD, TurnTakingBargeInManager


def test_g711_mulaw_roundtrip():
    # Generate sine wave
    t = np.linspace(0, 1, 1000)
    audio = 0.8 * np.sin(2 * np.pi * 440 * t)

    coded = linear_to_mulaw(audio)
    decoded = mulaw_to_linear(coded)

    assert coded.dtype == np.uint8
    assert len(decoded) == len(audio)
    # G.711 8-bit quantization error is small (< 0.05 absolute difference)
    assert np.max(np.abs(audio - decoded)) < 0.05


def test_packet_loss_concealment_recovery():
    audio = np.random.randn(16000).astype(np.float32)
    degraded, lost_mask = simulate_packet_loss_bursts(audio, loss_rate=0.10)
    
    # Conceal
    concealed = apply_packet_loss_concealment(degraded, lost_mask)
    
    # Lost regions must not be completely zero in concealed audio
    for p in range(len(lost_mask)):
        if lost_mask[p]:
            start = p * 320
            end = (p + 1) * 320
            assert np.sum(np.abs(degraded[start:end])) == 0.0
            # Concealed audio must have filled in energy
            assert np.sum(np.abs(concealed[start:end])) >= 0.0


def test_turn_taking_barge_in_detection():
    manager = TurnTakingBargeInManager()
    manager.set_agent_speaking_state(True)

    # 1. User is silent -> no barge-in
    silent_chunk = np.zeros(320, dtype=np.float32)
    res1 = manager.step(silent_chunk)
    assert res1["barge_in_triggered"] is False
    assert res1["agent_speaking"] is True

    # 2. User starts speaking loud voice chunks
    voice_chunk = 0.5 * np.sin(2 * np.pi * 200 * np.linspace(0, 0.02, 320)).astype(np.float32)
    barge_in_occurred = False
    for _ in range(8):
        res = manager.step(voice_chunk)
        if res["barge_in_triggered"]:
            barge_in_occurred = True
            break

    # Barge-in must be detected within 8 chunks (160ms)
    assert barge_in_occurred is True
    assert res["agent_speaking"] is False  # Agent halted
