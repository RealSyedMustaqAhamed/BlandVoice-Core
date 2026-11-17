"""
FastAPI Backend for BlandVoice-Core Interactive Showcase.
Provides REST and WebSocket endpoints for streaming audio synthesis,
telephony simulation, and live research benchmarks.
"""

import os
import sys
import io
import base64
from typing import Optional

# Ensure project root is in sys.path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import soundfile as sf
import torch

from fastapi import FastAPI, Request
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import BaseModel

from bland_codec.model import BlandStreamingCodec
from bland_tts.streaming_engine import BlandStreamingTTSEngine
from telephony.channel_effects import full_telephony_channel_pipeline
from evaluation.ablation_experiments import (
    generate_synthetic_speech_signal,
    run_quantization_bitrate_ablation,
    run_telephony_robustness_ablation,
    run_latency_budget_profiling
)
from evaluation.metrics import comprehensive_audio_evaluation


app = FastAPI(title="BlandVoice-Core Audio Research Prototype")

# Mount static directory
current_dir = os.path.dirname(os.path.abspath(__file__))
static_dir = os.path.join(current_dir, "static")
app.mount("/static", StaticFiles(directory=static_dir), name="static")

# Initialize models
print("[*] Initializing Bland Audio Models...")
codec = BlandStreamingCodec(
    sample_rate=16000,
    latent_dim=128,
    quantizer_type="rvq",
    num_quantizers=4,
    codebook_size=1024
)
codec.eval()

tts_engine = BlandStreamingTTSEngine(codec=codec, sample_rate=16000)
print("[+] Bland Audio Pipeline Ready.")


class SynthesizeRequest(BaseModel):
    text: str
    emotion: str = "neutral"
    chunk_frames: int = 4


class TelephonyRequest(BaseModel):
    loss_rate: float = 0.05
    standard: str = "mulaw"
    enable_plc: bool = True


def audio_to_base64_wav(audio_np: np.ndarray, sample_rate: int = 16000) -> str:
    """Encodes float32 numpy audio into base64 WAV byte string."""
    buf = io.BytesIO()
    sf.write(buf, audio_np, sample_rate, format="WAV", subtype="PCM_16")
    buf.seek(0)
    return base64.b64encode(buf.read()).decode("utf-8")


@app.get("/", response_class=HTMLResponse)
async def serve_index():
    index_file = os.path.join(static_dir, "index.html")
    with open(index_file, "r", encoding="utf-8") as f:
        return HTMLResponse(content=f.read())


@app.post("/api/synthesize")
async def api_synthesize(req: SynthesizeRequest):
    res = await tts_engine.synthesize_full_async(req.text)
    audio_b64 = audio_to_base64_wav(res["audio"], res["sample_rate"])

    return JSONResponse({
        "status": "ok",
        "audio_base64": audio_b64,
        "sample_rate": res["sample_rate"],
        "duration_sec": res["duration_sec"],
        "latency_ms": res["latency_ms"],
        "rtf": res["rtf"],
        "f0_contour": res["f0_contour"].tolist(),
        "parsed_info": res["parsed_info"]
    })


@app.post("/api/telephony_simulate")
async def api_telephony(req: TelephonyRequest):
    # Use real spoken voice agent audio as anchor
    sample_wav_path = os.path.join(current_dir, "..", "demo_outputs", "question_inflection.wav")
    if os.path.exists(sample_wav_path):
        clean, sr = sf.read(sample_wav_path)
    else:
        res = await tts_engine.synthesize_full_async("Hi John, this is Sarah from Bland AI. Did you have a quick minute to chat?")
        clean = res["audio"]
    
    degraded, concealed = full_telephony_channel_pipeline(
        clean_audio=clean,
        sample_rate=16000,
        packet_loss_rate=req.loss_rate,
        snr_db=22.0
    )

    eval_deg = comprehensive_audio_evaluation(clean, degraded)
    eval_plc = comprehensive_audio_evaluation(clean, concealed)

    return JSONResponse({
        "status": "ok",
        "mcd_dropped": eval_deg["mcd_db"],
        "sdr_dropped": eval_deg["si_sdr_db"],
        "mcd_plc": eval_plc["mcd_db"],
        "sdr_plc": eval_plc["si_sdr_db"],
        "audio_dropped_base64": audio_to_base64_wav(degraded, 16000),
        "audio_plc_base64": audio_to_base64_wav(concealed, 16000)
    })


@app.get("/api/benchmarks")
async def api_benchmarks():
    return JSONResponse({
        "status": "ok",
        "bitrate_ablation": run_quantization_bitrate_ablation(),
        "telephony_ablation": run_telephony_robustness_ablation(),
        "latency_budget": run_latency_budget_profiling()
    })


if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 7860))
    uvicorn.run(app, host="0.0.0.0", port=port)
