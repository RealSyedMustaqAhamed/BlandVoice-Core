"""
Unit Tests for Conversational TTS Pipeline.
Tests:
1. Question inflection detection & rising terminal F0 contour.
2. Phonemizer tag extraction.
3. TTFA latency profiling.
"""

import pytest
import torch
import numpy as np
from bland_tts.phonemizer import ConversationalPhonemizer
from bland_tts.prosody import ProsodyContourGenerator
from bland_tts.streaming_engine import BlandStreamingTTSEngine
from bland_codec.model import BlandStreamingCodec


def test_phonemizer_conversational_tags():
    phonemizer = ConversationalPhonemizer()
    text = "[emotion:empathetic] [filler:um] Hi John, did you receive our call?"
    parsed = phonemizer.parse(text)

    assert parsed["emotion"] == "empathetic"
    assert parsed["is_question"] is True
    assert "<filler_um>" in parsed["tokens"]
    assert len(parsed["token_ids"]) > 5


def test_question_inflection_pitch_rise():
    prosody_gen = ProsodyContourGenerator(base_f0=180.0)
    
    # 1. Statement: should decline
    f0_statement = prosody_gen.generate_f0_contour(num_frames=50, is_question=False)
    f0_s = f0_statement.squeeze().numpy()
    assert f0_s[-1] < f0_s[0]  # Final pitch is lower than initial

    # 2. Question: should exhibit terminal upturn
    f0_question = prosody_gen.generate_f0_contour(num_frames=50, is_question=True)
    f0_q = f0_question.squeeze().numpy()
    assert f0_q[-1] > f0_q[int(len(f0_q) * 0.7)]  # Rising pitch at boundary


def test_streaming_tts_ttfa_budget():
    codec = BlandStreamingCodec(latent_dim=64, encoder_base_channels=16, decoder_base_channels=64)
    tts = BlandStreamingTTSEngine(codec=codec, sample_rate=16000)

    # Stream synthesis
    chunks_gen = tts.stream_synthesize("Hello, can you hear me?", chunk_frames=4)
    first_chunk, meta = next(chunks_gen)

    assert meta["is_first"] is True
    assert "ttfa_ms" in meta
    assert len(first_chunk) > 0
