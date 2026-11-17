"""
ITU-T G.711 μ-law and A-law Audio Companding for Telephony Simulation.
Replicates legacy PSTN / SIP network quantization at 8kHz, 64 kbps (8-bit PCM).
Essential for training and evaluating speech models under real enterprise phone call conditions.
"""

from typing import Tuple
import numpy as np


def linear_to_mulaw(x: np.ndarray, mu: float = 255.0) -> np.ndarray:
    """
    Compresses linear PCM audio [-1.0, 1.0] to μ-law 8-bit quantized values [0, 255].
    Formula: F(x) = sgn(x) * ln(1 + mu * |x|) / ln(1 + mu)
    """
    x = np.clip(x, -1.0, 1.0)
    mu_compressed = np.sign(x) * np.log1p(mu * np.abs(x)) / np.log1p(mu)
    # Quantize to 8-bit [0, 255]
    quantized = np.floor((mu_compressed + 1.0) / 2.0 * 255.0 + 0.5)
    return np.clip(quantized, 0, 255).astype(np.uint8)


def mulaw_to_linear(y: np.ndarray, mu: float = 255.0) -> np.ndarray:
    """
    Expands 8-bit μ-law values [0, 255] back to linear PCM [-1.0, 1.0].
    Formula: F^-1(y) = sgn(y) * (1/mu) * ((1 + mu)^|y| - 1)
    """
    y_norm = (y.astype(np.float32) / 255.0) * 2.0 - 1.0
    linear = np.sign(y_norm) * (1.0 / mu) * (np.power(1.0 + mu, np.abs(y_norm)) - 1.0)
    return np.clip(linear, -1.0, 1.0)


def linear_to_alaw(x: np.ndarray, a: float = 87.6) -> np.ndarray:
    """
    Compresses linear PCM audio to European A-law 8-bit quantized values.
    """
    x = np.clip(x, -1.0, 1.0)
    abs_x = np.abs(x)
    sign_x = np.sign(x)
    
    compressed = np.zeros_like(x)
    thresh = 1.0 / a
    
    # Region 1: |x| < 1/A
    mask1 = abs_x < thresh
    compressed[mask1] = sign_x[mask1] * (a * abs_x[mask1]) / (1.0 + np.log(a))
    
    # Region 2: 1/A <= |x| <= 1
    mask2 = ~mask1
    compressed[mask2] = sign_x[mask2] * (1.0 + np.log(a * abs_x[mask2])) / (1.0 + np.log(a))
    
    quantized = np.floor((compressed + 1.0) / 2.0 * 255.0 + 0.5)
    return np.clip(quantized, 0, 255).astype(np.uint8)


def alaw_to_linear(y: np.ndarray, a: float = 87.6) -> np.ndarray:
    """
    Expands 8-bit A-law values back to linear PCM.
    """
    y_norm = (y.astype(np.float32) / 255.0) * 2.0 - 1.0
    abs_y = np.abs(y_norm)
    sign_y = np.sign(y_norm)
    
    thresh = 1.0 / (1.0 + np.log(a))
    linear = np.zeros_like(y_norm)
    
    mask1 = abs_y < thresh
    linear[mask1] = sign_y[mask1] * (abs_y[mask1] * (1.0 + np.log(a))) / a
    
    mask2 = ~mask1
    linear[mask2] = sign_y[mask2] * (np.exp(abs_y[mask2] * (1.0 + np.log(a)) - 1.0)) / a
    
    return np.clip(linear, -1.0, 1.0)


def apply_g711_telephony_channel(audio_16k: np.ndarray, standard: str = "mulaw") -> np.ndarray:
    """
    Simulates full G.711 telephony transport:
    1. Downsample 16kHz -> 8kHz
    2. Quantize to 8-bit μ-law / A-law
    3. Reconstruct linear PCM
    4. Upsample back to 16kHz
    """
    from scipy.signal import resample_poly

    # Downsample to 8kHz (standard telephony sampling rate)
    audio_8k = resample_poly(audio_16k, up=1, down=2)

    # Apply 8-bit companding
    if standard.lower() == "alaw":
        coded = linear_to_alaw(audio_8k)
        decoded_8k = alaw_to_linear(coded)
    else:
        coded = linear_to_mulaw(audio_8k)
        decoded_8k = mulaw_to_linear(coded)

    # Upsample back to 16kHz
    audio_restored_16k = resample_poly(decoded_8k, up=2, down=1)
    
    # Match length
    if len(audio_restored_16k) > len(audio_16k):
        audio_restored_16k = audio_restored_16k[:len(audio_16k)]
    elif len(audio_restored_16k) < len(audio_16k):
        audio_restored_16k = np.pad(audio_restored_16k, (0, len(audio_16k) - len(audio_restored_16k)))

    return audio_restored_16k.astype(np.float32)
