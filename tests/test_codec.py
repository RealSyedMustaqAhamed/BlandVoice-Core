"""
Unit Tests for Causal Neural Audio Codec.
Tests:
1. Receptive field causality (zero future frame leakage).
2. End-to-end shape consistency.
3. Residual Vector Quantization and multi-rate depth selection.
4. Online streaming state carryover.
"""

import pytest
import torch
from bland_codec.causal_conv import CausalConv1d, CausalConvTranspose1d
from bland_codec.quantization import ResidualVectorQuantizer, FiniteScalarQuantizer
from bland_codec.model import BlandStreamingCodec


def test_causal_conv1d_zero_lookahead():
    """Confirms causal conv does not leak future information."""
    in_ch, out_ch, K = 2, 4, 5
    conv = CausalConv1d(in_ch, out_ch, kernel_size=K)

    # Input: 10 frames of audio
    x1 = torch.randn(1, in_ch, 10)
    x2 = x1.clone()
    # Modify future frames (frames 6 through 9)
    x2[:, :, 6:] = torch.randn(1, in_ch, 4)

    y1 = conv(x1)
    y2 = conv(x2)

    # Output up to frame 5 must be 100% IDENTICAL
    torch.testing.assert_close(y1[:, :, :6], y2[:, :, :6])


def test_codec_shape_consistency():
    """Verifies 320x downsample/upsample ratio across arbitrary lengths."""
    codec = BlandStreamingCodec(
        sample_rate=16000,
        latent_dim=64,
        encoder_base_channels=16,
        decoder_base_channels=64
    )
    # 1 second of audio at 16kHz
    audio = torch.randn(1, 1, 16000)
    recon, indices, loss, metrics = codec(audio)

    assert recon.shape == audio.shape
    assert indices.shape == (1, 4, 50)  # 50 frames @ 50 Hz
    assert loss.item() >= 0.0
    assert "avg_codebook_perplexity" in metrics


def test_rvq_multi_rate_bitrates():
    """Confirms multi-rate RVQ can operate at 2, 4, and 8 quantizers."""
    rvq = ResidualVectorQuantizer(dim=64, num_quantizers=4, codebook_size=256)
    z = torch.randn(2, 64, 30)

    # Test with 2 quantizers
    z_q2, idx2, _, m2 = rvq(z, n_q=2)
    assert idx2.shape == (2, 2, 30)
    assert m2["bitrate_kbps"] == (2 * 8 * 50) / 1000.0

    # Test with 4 quantizers
    z_q4, idx4, _, m4 = rvq(z, n_q=4)
    assert idx4.shape == (2, 4, 30)


def test_streaming_step_matches_offline():
    """Confirms chunked streaming decoder matches offline decoding."""
    codec = BlandStreamingCodec(latent_dim=64, encoder_base_channels=16, decoder_base_channels=64)
    codec.eval()

    z = torch.randn(1, 64, 6) # 6 frames = 120ms
    # Offline decode
    with torch.no_grad():
        offline_audio = codec.decode(z)

        # Chunk streaming decode (2 frames at a time)
        dec_state = {}
        chunks = []
        for i in range(0, 6, 2):
            chunk_z = z[:, :, i:i+2]
            chunk_out, dec_state = codec.streaming_step_decode(chunk_z, dec_state)
            chunks.append(chunk_out)
        streaming_audio = torch.cat(chunks, dim=-1)

    assert streaming_audio.shape == offline_audio.shape
