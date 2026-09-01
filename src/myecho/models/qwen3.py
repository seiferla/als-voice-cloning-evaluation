"""Qwen3-TTS-VC voice cloning via Alibaba Cloud DashScope (realtime WebSocket)."""

import base64
import io
import threading
import time
import wave
from pathlib import Path

import dashscope
from dashscope.audio.qwen_tts_realtime import (
    AudioFormat,
    QwenTtsRealtime,
    QwenTtsRealtimeCallback,
)
from dashscope.audio.tts_v2 import VoiceEnrollmentService

from myecho.config import (
    QWEN3_API_KEY,
    QWEN3_BASE_HTTP_API_URL,
    QWEN3_BASE_WS_API_URL,
    QWEN3_BASE_WS_REALTIME_URL,
    QWEN3_LANGUAGE_TYPE,
    QWEN3_REALTIME_MODEL,
    QWEN3_WORKSPACE_ID,
)


class _AudioCollector(QwenTtsRealtimeCallback):
    """Collects base64 PCM chunks from the realtime event stream."""

    def __init__(self):
        self.chunks = []
        self.error = None
        self.done = threading.Event()
        self.t_first_audio = None

    def on_open(self):
        pass

    def on_close(self, close_status_code, close_msg):
        # The server closes the socket after session.finish() - treat as EOS.
        self.done.set()

    def on_event(self, message):
        # NOTE: despite the base-class type hint, the SDK hands us a parsed dict.
        event_type = message.get("type") if isinstance(message, dict) else None

        if event_type == "response.audio.delta":
            payload = message.get("delta") or message.get("audio")
            if payload:
                if self.t_first_audio is None:
                    self.t_first_audio = time.perf_counter()
                self.chunks.append(base64.b64decode(payload))
        elif event_type in ("session.finished", "response.done"):
            # response.done still leaves the socket open in server_commit mode;
            # only session.finished is a hard end of stream.
            if event_type == "session.finished":
                self.done.set()
        elif event_type == "error":
            self.error = message.get("error", message)
            self.done.set()


class Qwen3TTS:
    def __init__(
        self,
        voice_id=None,
        voice_name="myecho",
        sample_audio_url=None,
        language_type=QWEN3_LANGUAGE_TYPE,
    ):
        # Region config must happen before any client is instantiated -
        # QwenTtsRealtime reads dashscope.api_key in its constructor.
        dashscope.api_key = QWEN3_API_KEY
        dashscope.base_http_api_url = QWEN3_BASE_HTTP_API_URL
        dashscope.base_websocket_api_url = QWEN3_BASE_WS_API_URL

        self.model = QWEN3_REALTIME_MODEL
        self.voice_id = voice_id
        self.voice_name = voice_name  # used as `prefix`, max 10 alphanumeric chars
        self.sample_audio_url = sample_audio_url  # must be a PUBLICLY reachable URL
        self.language_type = language_type
        self._enrollment = VoiceEnrollmentService()

    def create_voice(self, audio_url, language_hints=None, enable_preprocess=False):
        """Create a custom voice from a publicly reachable audio URL.

        enable_preprocess defaults to False, against the API's own default:
        True denoises server-side, which would hand qwen3 cleaner audio than the
        other two services and confound the comparison. Do the preprocessing
        once for all three with evaluation/prepare_enrollment.py.

        NOTE: unlike the Mistral API, DashScope does not accept raw audio
        bytes here - the reference sample must already be hosted at a URL
        the service can fetch itself (e.g. an OSS bucket link). Use
        qwen_voice_enroll.py if you need to upload a local file as data URI.
        """
        self.voice_id = self._enrollment.create_voice(
            target_model=self.model,
            prefix=self.voice_name,
            url=audio_url,
            language_hints=language_hints if language_hints is not None else ["de"],
            enable_preprocess=enable_preprocess,
        )
        return self.voice_id

    async def synthesize(self, text, output_path=None, timeout=120):
        """Generate speech from text using the cloned voice.

        The realtime API streams raw PCM (24 kHz mono 16 bit) as base64 chunks;
        we concatenate them and wrap the result in a WAV container.
        """
        if not self.voice_id:
            raise ValueError("No voice_id set. Call create_voice() first or pass voice_id to __init__")

        collector = _AudioCollector()
        client = QwenTtsRealtime(
            model=self.model,
            callback=collector,
            url=QWEN3_BASE_WS_REALTIME_URL,
            workspace=QWEN3_WORKSPACE_ID,
        )

        response_format = AudioFormat.PCM_24000HZ_MONO_16BIT
        session_kwargs = {}
        if self.language_type:
            session_kwargs["language_type"] = self.language_type

        t_request = time.perf_counter()
        client.connect()
        t_connected = time.perf_counter()
        try:
            client.update_session(
                voice=self.voice_id,
                response_format=response_format,
                mode="server_commit",
                **session_kwargs,
            )
            client.append_text(text)
            client.finish()  # server synthesizes the buffer, then closes the socket

            if not collector.done.wait(timeout):
                raise TimeoutError(f"No end-of-stream from DashScope within {timeout}s")
        finally:
            client.close()
        t_finish = time.perf_counter()

        if collector.error:
            raise RuntimeError(f"DashScope realtime error: {collector.error}")

        pcm = b"".join(collector.chunks)
        if not pcm:
            raise RuntimeError("DashScope returned no audio data")

        sample_rate = response_format.sample_rate
        buffer = io.BytesIO()
        with wave.open(buffer, "wb") as wav:
            wav.setnchannels(1)
            wav.setsampwidth(2)  # 16 bit
            wav.setframerate(sample_rate)
            wav.writeframes(pcm)
        audio_data = buffer.getvalue()

        duration = len(pcm) / (2 * sample_rate)
        latency = t_finish - t_request

        # TTFA is measured from t_request (i.e. including handshake +
        # session.update) so it is comparable with S2-Pro. The SDK's own
        # get_first_audio_delay() only counts append_text -> first audio,
        # so we keep it separately as the server-side view.
        ttfa_sdk_ms = client.get_first_audio_delay()

        metrics = {
            "latency": latency,
            "ttfa": (collector.t_first_audio - t_request) if collector.t_first_audio else None,
            "ttfa_sdk": (ttfa_sdk_ms / 1000) if ttfa_sdk_ms is not None else None,
            "audio_duration_s": duration,
            "rtf": (latency / duration) if duration else None,
            "connect_s": t_connected - t_request,
            "session_id": client.get_session_id(),
            "response_id": client.get_last_response_id(),
        }

        if output_path:
            Path(output_path).parent.mkdir(parents=True, exist_ok=True)
            Path(output_path).write_bytes(audio_data)

        return {"audio": audio_data, "metrics": metrics}
