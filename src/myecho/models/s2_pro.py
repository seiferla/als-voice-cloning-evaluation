"""S2-Pro TTS via Fish Audio WebSocket API."""

import io
import time
from pathlib import Path

import msgpack
import websockets

from myecho.config import (
    S2PRO_API_KEY,
    S2PRO_LATENCY_MODE,
    S2PRO_MODEL,
    S2PRO_WS_URI,
)

# Fish Audio sends a whole WAV in a single frame in "normal" mode; a ~20 s
# sentence exceeds the 1 MiB default of the websockets client and the server
# closes with 1009 (message too big). No limit -> no truncated eval runs.
WS_MAX_FRAME_SIZE = None


def _audio_duration(audio_data, audio_format):
    """Duration in seconds, or None if the container cannot be parsed."""
    try:
        if audio_format == "mp3":
            from mutagen.mp3 import MP3
            return MP3(io.BytesIO(audio_data)).info.length
        import soundfile as sf
        return sf.info(io.BytesIO(audio_data)).duration
    except Exception:
        return None


class S2ProTTS:
    def __init__(self, reference_id=None):
        self.api_key = S2PRO_API_KEY
        self.uri = S2PRO_WS_URI
        self.model = S2PRO_MODEL
        self.reference_id = reference_id
        self.latency_mode = S2PRO_LATENCY_MODE

    async def synthesize(self, text, output_path=None, speed=1.0, volume=0,
                         audio_format="wav", latency_mode=None):
        """Generate speech from text.

        Defaults to WAV (not MP3) so the output is lossless and directly
        comparable with Voxtral/Qwen3 in the SIM/WER/N-MOS evaluation.

        latency_mode: "balanced" (default) streams the audio in several frames,
        which is what makes TTFA a real measurement. Set S2PRO_LATENCY_MODE=normal
        to get the non-streaming best-quality mode instead - TTFA then equals
        latency, because the whole file arrives at once.
        """
        latency_mode = latency_mode or self.latency_mode
        headers = {"Authorization": f"Bearer {self.api_key}", "model": self.model}
        audio_data = b""
        metrics = {
            "ttfa": None,
            "latency": None,
            "rtf": None,
            "audio_duration_s": None,
            "connect_s": None,
            "latency_mode": latency_mode,
        }

        # The clock starts BEFORE the handshake: Qwen3 also pays a WebSocket
        # connect and Voxtral an HTTP round-trip, so only an end-to-end
        # measurement makes latency/TTFA comparable across the three models.
        t_request = time.perf_counter()

        async with websockets.connect(
            self.uri, additional_headers=headers, max_size=WS_MAX_FRAME_SIZE
        ) as ws:
            metrics["connect_s"] = time.perf_counter() - t_request

            await ws.send(msgpack.packb({
                "event": "start",
                "request": {
                    "text": "",
                    "format": audio_format,
                    "latency": latency_mode,
                    "reference_id": self.reference_id,
                    "prosody": {"speed": speed, "volume": volume},
                },
            }))
            await ws.send(msgpack.packb({"event": "text", "text": text}))
            await ws.send(msgpack.packb({"event": "flush"}))
            await ws.send(msgpack.packb({"event": "stop"}))

            t_finish = None
            async for message in ws:
                msg = msgpack.unpackb(message)
                if msg.get("event") == "audio":
                    if metrics["ttfa"] is None:
                        metrics["ttfa"] = time.perf_counter() - t_request
                    audio_data += msg["audio"]
                elif msg.get("event") == "finish":
                    t_finish = time.perf_counter()
                    break

        if t_finish:
            metrics["latency"] = t_finish - t_request
            duration = _audio_duration(audio_data, audio_format)
            if duration:
                metrics["audio_duration_s"] = duration
                metrics["rtf"] = metrics["latency"] / duration

        if output_path:
            Path(output_path).parent.mkdir(parents=True, exist_ok=True)
            Path(output_path).write_bytes(audio_data)

        return {"audio": audio_data, "metrics": metrics}
