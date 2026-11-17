"""
Telephony Channel Degradation & Robustness Simulation Suite.
Implements:
1. ITU-T G.712 Narrowband Bandpass Filtering (300 Hz - 3400 Hz)
2. Gilbert-Elliott Burst Packet Loss & Jitter Simulator
3. G.711 Appendix I Packet Loss Concealment (PLC)
4. Telephony Background Noise Injection (AC hum, line hiss, ambient chatter)
"""

from typing import Tuple, List, Optional
import numpy as np
from scipy.signal import butter, sosfilt


def apply_telephony_bandpass(audio: np.ndarray, sample_rate: int = 16000) -> np.ndarray:
    """
    Applies ITU-T G.712 telephone frequency bandpass filter (300Hz - 3400Hz).
    Removes low-end chest resonance and high-frequency sibilants, exactly matching PSTN voice.
    """
    nyquist = sample_rate / 2.0
    low_cutoff = 300.0 / nyquist
    high_cutoff = min(3400.0 / nyquist, 0.99)
    
    # 4th order Butterworth bandpass
    sos = butter(4, [low_cutoff, high_cutoff], btype="bandpass", output="sos")
    filtered = sosfilt(sos, audio)
    return filtered.astype(np.float32)


def simulate_packet_loss_bursts(
    audio: np.ndarray,
    sample_rate: int = 16000,
    packet_ms: int = 20,
    loss_rate: float = 0.08,
    burst_probability: float = 0.4
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Simulates realistic VoIP / SIP RTP packet drops using a 2-state Gilbert-Elliott Markov model.
    Returns:
        degraded_audio: audio with zeroed / dropped packets
        lost_packet_mask: boolean array indicating which packets were dropped
    """
    packet_size = int(sample_rate * (packet_ms / 1000.0))
    total_packets = int(np.ceil(len(audio) / packet_size))
    
    lost_mask = np.zeros(total_packets, dtype=bool)
    
    # Gilbert-Elliott Markov chain
    # State 0: Good, State 1: Bad (loss state)
    in_bad_state = False
    p_good_to_bad = loss_rate * (1.0 - burst_probability)
    p_bad_to_bad = burst_probability
    
    for p in range(total_packets):
        if not in_bad_state:
            if np.random.rand() < p_good_to_bad:
                in_bad_state = True
        else:
            if np.random.rand() > p_bad_to_bad:
                in_bad_state = False
                
        lost_mask[p] = in_bad_state

    # Zero out lost packets
    degraded = audio.copy()
    for p in range(total_packets):
        if lost_mask[p]:
            start = p * packet_size
            end = min((p + 1) * packet_size, len(audio))
            degraded[start:end] = 0.0

    return degraded.astype(np.float32), lost_mask


def apply_packet_loss_concealment(
    degraded_audio: np.ndarray,
    lost_mask: np.ndarray,
    sample_rate: int = 16000,
    packet_ms: int = 20,
    decay_rate: float = 0.8
) -> np.ndarray:
    """
    ITU-T G.711 App I style Packet Loss Concealment (PLC).
    Replaces dropped frames with energy-attenuated pitch extrapolation from previous frame.
    """
    packet_size = int(sample_rate * (packet_ms / 1000.0))
    total_packets = len(lost_mask)
    concealed = degraded_audio.copy()
    
    last_good_packet = np.zeros(packet_size, dtype=np.float32)
    consecutive_losses = 0

    for p in range(total_packets):
        start = p * packet_size
        end = min((p + 1) * packet_size, len(degraded_audio))
        actual_len = end - start

        if not lost_mask[p]:
            # Cache last valid frame
            last_good_packet[:actual_len] = degraded_audio[start:end]
            consecutive_losses = 0
        else:
            # Conceal packet with attenuated repetition + cross-fade
            consecutive_losses += 1
            attenuation = max(decay_rate ** consecutive_losses, 0.1)
            
            # Smooth windowing to avoid boundary pops
            window = np.hanning(actual_len)
            extrapolated = last_good_packet[:actual_len] * attenuation * window
            concealed[start:end] = extrapolated

    return concealed.astype(np.float32)


def add_telephony_noise(
    audio: np.ndarray,
    sample_rate: int = 16000,
    snr_db: float = 22.0,
    include_mains_hum: bool = True
) -> np.ndarray:
    """
    Injects realistic telephony noise:
    - Thermal / quantization white noise
    - 50Hz/60Hz AC power mains hum with 2nd harmonic (120Hz)
    """
    # Calculate signal power
    sig_power = np.mean(audio ** 2)
    if sig_power <= 1e-9:
        return audio

    noise_power = sig_power / (10.0 ** (snr_db / 10.0))
    noise = np.random.normal(0.0, np.sqrt(noise_power), size=len(audio))

    if include_mains_hum:
        t = np.arange(len(audio)) / sample_rate
        # 60Hz hum + 120Hz harmonic
        hum = 0.3 * np.sin(2 * np.pi * 60 * t) + 0.15 * np.sin(2 * np.pi * 120 * t)
        hum_scaled = hum * np.sqrt(noise_power * 0.25)
        noise = noise + hum_scaled

    noisy_audio = audio + noise
    return np.clip(noisy_audio, -1.0, 1.0).astype(np.float32)


def full_telephony_channel_pipeline(
    clean_audio: np.ndarray,
    sample_rate: int = 16000,
    packet_loss_rate: float = 0.05,
    snr_db: float = 24.0
) -> Tuple[np.ndarray, np.ndarray]:
    """
    End-to-end phone call degradation pipeline:
    Clean -> Bandpass -> G.711 8kHz companding -> Noise -> Packet Loss -> PLC.
    Returns (raw_degraded, plc_concealed).
    """
    from .codec_g711 import apply_g711_telephony_channel

    # 1. Bandpass filter
    bp_audio = apply_telephony_bandpass(clean_audio, sample_rate)
    
    # 2. G.711 μ-law companding & 8kHz resampling
    g711_audio = apply_g711_telephony_channel(bp_audio, standard="mulaw")
    
    # 3. Ambient telephony noise
    noisy_audio = add_telephony_noise(g711_audio, sample_rate, snr_db=snr_db)
    
    # 4. Burst packet loss
    degraded, lost_mask = simulate_packet_loss_bursts(
        noisy_audio, sample_rate, packet_ms=20, loss_rate=packet_loss_rate
    )
    
    # 5. Packet Loss Concealment
    concealed = apply_packet_loss_concealment(degraded, lost_mask, sample_rate, packet_ms=20)
    
    return degraded, concealed
