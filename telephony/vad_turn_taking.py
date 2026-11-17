"""
Low-Latency Streaming Voice Activity Detection (VAD) & Turn-Taking Barge-In Engine.
Enables sub-100ms conversational interruption detection for real-time AI phone agents.
"""

from typing import Tuple, Dict, Any, List
import numpy as np


class StreamingVAD:
    """
    Sub-50ms streaming Voice Activity Detector based on short-time energy,
    spectral flatness, and adaptive background noise estimation.
    """
    def __init__(
        self,
        sample_rate: int = 16000,
        frame_ms: int = 20,
        energy_threshold_db: float = -38.0,
        speech_pad_frames: int = 3
    ):
        self.sample_rate = sample_rate
        self.frame_size = int(sample_rate * (frame_ms / 1000.0))
        self.energy_threshold_db = energy_threshold_db
        self.speech_pad_frames = speech_pad_frames

        self.noise_floor_energy = 1e-5
        self.consecutive_speech_frames = 0
        self.consecutive_silence_frames = 0
        self.is_speaking = False

    def process_chunk(self, chunk: np.ndarray) -> Dict[str, Any]:
        """
        Processes a single audio chunk (e.g. 20ms = 320 samples).
        Returns VAD decision and confidence metrics.
        """
        if len(chunk) == 0:
            return {"is_speech": False, "energy_db": -100.0, "state_changed": False}

        # 1. Short-time RMS energy
        rms = np.sqrt(np.mean(chunk ** 2) + 1e-10)
        energy_db = 20.0 * np.log10(rms + 1e-9)

        # 2. Spectral Flatness Measure (SFM)
        # Speech has harmonic formant peaks (low flatness); white noise has high flatness
        fft_mag = np.abs(np.fft.rfft(chunk * np.hanning(len(chunk)))) + 1e-9
        geometric_mean = np.exp(np.mean(np.log(fft_mag)))
        arithmetic_mean = np.mean(fft_mag)
        spectral_flatness = geometric_mean / (arithmetic_mean + 1e-9)

        # 3. Dynamic noise floor tracking (exponential moving average during silence)
        if energy_db < self.energy_threshold_db and not self.is_speaking:
            self.noise_floor_energy = 0.95 * self.noise_floor_energy + 0.05 * rms

        snr_estimate = 20.0 * np.log10((rms / (self.noise_floor_energy + 1e-9)) + 1e-9)

        # Decision rule: High energy + structured spectrum (low flatness)
        frame_is_voice = (energy_db > self.energy_threshold_db) and (snr_estimate > 6.0) and (spectral_flatness < 0.65)

        previous_state = self.is_speaking
        if frame_is_voice:
            self.consecutive_speech_frames += 1
            self.consecutive_silence_frames = 0
            if self.consecutive_speech_frames >= self.speech_pad_frames:
                self.is_speaking = True
        else:
            self.consecutive_silence_frames += 1
            if self.consecutive_silence_frames >= (self.speech_pad_frames * 2):
                self.consecutive_speech_frames = 0
                self.is_speaking = False

        state_changed = (self.is_speaking != previous_state)

        return {
            "is_speech": self.is_speaking,
            "raw_voice_frame": frame_is_voice,
            "energy_db": float(energy_db),
            "snr_db": float(snr_estimate),
            "spectral_flatness": float(spectral_flatness),
            "state_changed": state_changed
        }


class TurnTakingBargeInManager:
    """
    Manages full-duplex conversational turn-taking on telephone calls.
    Detects when a customer interrupts the agent, halting agent speech output within 60-100ms.
    """
    def __init__(
        self,
        sample_rate: int = 16000,
        barge_in_latency_chunks: int = 3  # 3 * 20ms = 60ms interruption threshold
    ):
        self.vad = StreamingVAD(sample_rate=sample_rate)
        self.barge_in_latency_chunks = barge_in_latency_chunks
        self.agent_speaking = False
        self.user_speech_chunk_count = 0

    def set_agent_speaking_state(self, speaking: bool):
        self.agent_speaking = speaking
        if not speaking:
            self.user_speech_chunk_count = 0

    def step(self, incoming_user_chunk: np.ndarray) -> Dict[str, Any]:
        """
        Receives incoming 20ms user audio chunk from telephony socket.
        Returns whether a barge-in interruption occurred.
        """
        vad_res = self.vad.process_chunk(incoming_user_chunk)
        barge_in_triggered = False

        if vad_res["is_speech"]:
            self.user_speech_chunk_count += 1
            if self.agent_speaking and (self.user_speech_chunk_count >= self.barge_in_latency_chunks):
                barge_in_triggered = True
                self.agent_speaking = False  # Immediately cancel agent speech playout
        else:
            self.user_speech_chunk_count = max(0, self.user_speech_chunk_count - 1)

        return {
            "user_speaking": vad_res["is_speech"],
            "agent_speaking": self.agent_speaking,
            "barge_in_triggered": barge_in_triggered,
            "interruption_latency_ms": self.user_speech_chunk_count * 20,
            "vad_metrics": vad_res
        }
