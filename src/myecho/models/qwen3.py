"""Qwen-Audio-TTS (Qwen3) voice cloning via Alibaba Cloud DashScope."""

import io
import time
from pathlib import Path
import soundfile as sf
import dashscope
from dashscope.audio.tts_v2 import VoiceEnrollmentService, SpeechSynthesizer, AudioFormat
from myecho.config import QWEN3_API_KEY, QWEN3_BASE_HTTP_API_URL, QWEN3_BASE_WS_API_URL


class Qwen3TTS:
    def __init__(self, voice_id=None, voice_name="myecho", sample_audio_url=None):
        # Region config must happen before any client is instantiated.
        dashscope.api_key = QWEN3_API_KEY
        dashscope.base_http_api_url = QWEN3_BASE_HTTP_API_URL
        dashscope.base_websocket_api_url = QWEN3_BASE_WS_API_URL

        self.model = "qwen-audio-3.0-tts-flash"
        self.voice_id = voice_id
        self.voice_name = voice_name  # used as `prefix`, max 10 alphanumeric chars
        self.sample_audio_url = sample_audio_url  # must be a PUBLICLY reachable URL
        self._enrollment = VoiceEnrollmentService()

    def create_voice(self, audio_url, language_hints=["de"], enable_preprocess=True):
        """Create a custom voice from a publicly reachable audio URL.

        NOTE: unlike the Mistral API, DashScope does not accept raw audio
        bytes here - the reference sample must already be hosted at a URL
        the service can fetch itself (e.g. an OSS bucket link).
        """
        self.voice_id = self._enrollment.create_voice(
            target_model=self.model,
            prefix=self.voice_name,
            url=audio_url,
            language_hints=language_hints,
            enable_preprocess=enable_preprocess,  # denoising/normalization - useful given background noise in the source recordings
        )
        return self.voice_id

    async def synthesize(self, text, output_path=None):
        """Generate speech from text using the cloned voice."""
        if not self.voice_id:
            raise ValueError("No voice_id set. Call create_voice() first or pass voice_id to __init__")

        synthesizer = SpeechSynthesizer(
            model=self.model,
            voice=self.voice_id,
            format=AudioFormat.WAV_24000HZ_MONO_16BIT,  # WAV needed for SIM/WER; verify exact enum name against installed SDK version
        )

        t_request = time.perf_counter()
        audio_data = synthesizer.call(text)
        t_finish = time.perf_counter()

        # DashScope reports first-packet delay natively via the WebSocket
        # protocol - use it directly instead of approximating TTFA.
        ttfa_ms = synthesizer.get_first_package_delay()

        metrics = {
            "latency": t_finish - t_request,
            "ttfa": (ttfa_ms / 1000) if ttfa_ms is not None else None,
            "audio_duration_s": None,
            "rtf": None,
            "request_id": synthesizer.get_last_request_id(),
        }

        try:
            array, sr = sf.read(io.BytesIO(audio_data))
            duration = len(array) / sr
            metrics["audio_duration_s"] = duration
            metrics["rtf"] = metrics["latency"] / duration
        except Exception:
            pass

        if output_path:
            Path(output_path).parent.mkdir(parents=True, exist_ok=True)
            Path(output_path).write_bytes(audio_data)

        return {"audio": audio_data, "metrics": metrics}