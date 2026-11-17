"""
Interactive Command-Line Demonstration of Bland Voice Core.
Allows instant testing of Expressive TTS, Causal Neural Codec, and Telephony Simulator.
"""

import os
import sys
import time

# Ensure project root is in sys.path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import soundfile as sf
import torch

from bland_codec.model import BlandStreamingCodec
from bland_tts.streaming_engine import BlandStreamingTTSEngine
from telephony.channel_effects import full_telephony_channel_pipeline
from evaluation.metrics import comprehensive_audio_evaluation


def main():
    print("=" * 70)
    print(" BLAND.COM AUDIO RESEARCH PROTOTYPE - INTERACTIVE CLI DEMO ")
    print("=" * 70)

    output_dir = "demo_outputs"
    os.makedirs(output_dir, exist_ok=True)

    # 1. Initialize Neural Codec
    print("\n[+] Initializing Causal Neural Audio Codec (16kHz, 50Hz latents)...")
    codec = BlandStreamingCodec(
        sample_rate=16000,
        latent_dim=128,
        quantizer_type="rvq",
        num_quantizers=4,
        codebook_size=1024
    )
    codec.eval()

    # 2. Initialize Streaming TTS Engine
    print("[+] Initializing Expressive Conversational TTS Engine...")
    tts_engine = BlandStreamingTTSEngine(codec=codec, sample_rate=16000)

    # Test Sentences
    test_prompts = [
        ("question_inflection", "Hi John, this is Sarah from Bland AI. Did you have a quick minute to chat?"),
        ("empathetic_support", "[emotion:empathetic] I completely understand your frustration, and I am here to help you fix this."),
        ("conversational_fillers", "[filler:um] Let me check that account for you... [filler:uh_huh] Yes, it looks active.")
    ]

    print("\n--- [A] RUNNING STREAMING CONVERSATIONAL TTS SYNTHESIS ---")
    for name, prompt in test_prompts:
        print(f"\nPrompt: \"{prompt}\"")
        res = tts_engine.synthesize_full(prompt)
        
        wav_path = os.path.join(output_dir, f"{name}.wav")
        sf.write(wav_path, res["audio"], res["sample_rate"])

        print(f"  -> Generated: {wav_path}")
        print(f"  -> Duration: {res['duration_sec']:.2f}s | TTFA Latency: {res['latency_ms']:.1f}ms | RTF: {res['rtf']:.3f}")
        print(f"  -> Question inflection detected: {res['parsed_info']['is_question']}")
        print(f"  -> Emotional tone: {res['parsed_info']['emotion']}")

    # 3. Telephony Simulation & PLC Recovery
    print("\n--- [B] TELEPHONY CHANNEL SIMULATION (G.711 + 5% BURST PACKET LOSS) ---")
    ref_audio = res["audio"]
    degraded, concealed = full_telephony_channel_pipeline(
        clean_audio=ref_audio,
        sample_rate=16000,
        packet_loss_rate=0.06,
        snr_db=22.0
    )

    deg_path = os.path.join(output_dir, "telephony_degraded_packetloss.wav")
    plc_path = os.path.join(output_dir, "telephony_concealed_plc.wav")
    sf.write(deg_path, degraded, 16000)
    sf.write(plc_path, concealed, 16000)

    eval_deg = comprehensive_audio_evaluation(ref_audio, degraded)
    eval_plc = comprehensive_audio_evaluation(ref_audio, concealed)

    print(f"  -> Saved degraded audio: {deg_path}")
    print(f"     MCD: {eval_deg['mcd_db']:.2f} dB | SI-SDR: {eval_deg['si_sdr_db']:.2f} dB (Severe drops)")
    print(f"  -> Saved PLC recovered audio: {plc_path}")
    print(f"     MCD: {eval_plc['mcd_db']:.2f} dB | SI-SDR: {eval_plc['si_sdr_db']:.2f} dB (+{eval_plc['si_sdr_db'] - eval_deg['si_sdr_db']:.1f} dB recovery)")

    print("\n" + "=" * 70)
    print(" DEMO COMPLETED. All audio artifacts saved to 'demo_outputs/'.")
    print("=" * 70)


if __name__ == "__main__":
    main()
