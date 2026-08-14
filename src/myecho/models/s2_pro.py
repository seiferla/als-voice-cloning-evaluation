"""S2-Pro TTS via Fish Audio WebSocket API."""

import asyncio
import time
import websockets
import msgpack
from pathlib import Path
from myecho.config import S2PRO_API_KEY, S2PRO_WS_URI, S2PRO_MODEL


class S2ProTTS:
    def __init__(self, reference_id=None):
        self.api_key = S2PRO_API_KEY
        self.uri = S2PRO_WS_URI
        self.model = S2PRO_MODEL
        self.reference_id = reference_id

    async def synthesize(self, text, output_path=None, speed=1.0, volume=0):
        """Generate speech from text."""
        headers = {"Authorization": f"Bearer {self.api_key}", "model": self.model}
        audio_data = b""
        metrics = {"ttfa": None, "latency": None, "rtf": None}

        async with websockets.connect(self.uri, additional_headers=headers) as ws:
            await ws.send(msgpack.packb({
                "event": "start",
                "request": {
                    "text": "",
                    "format": "mp3",
                    "latency": "normal",
                    "reference_id": self.reference_id,
                    "prosody": {"speed": speed, "volume": volume},
                },
            }))

            t_request = time.perf_counter()
            await ws.send(msgpack.packb({"event": "text", "text": text}))
            await ws.send(msgpack.packb({"event": "flush"}))
            await ws.send(msgpack.packb({"event": "stop"}))

            t_first_audio = None
            t_finish = None

            async for message in ws:
                msg = msgpack.unpackb(message)
                if msg.get("event") == "audio":
                    if t_first_audio is None:
                        t_first_audio = time.perf_counter()
                        metrics["ttfa"] = t_first_audio - t_request
                    audio_data += msg["audio"]
                elif msg.get("event") == "finish":
                    t_finish = time.perf_counter()
                    break

        if t_finish:
            metrics["latency"] = t_finish - t_request
            try:
                from mutagen.mp3 import MP3
                import io
                duration = MP3(io.BytesIO(audio_data)).info.length
                metrics["audio_duration_s"] = duration
                metrics["rtf"] = metrics["latency"] / duration
            except Exception:
                pass

        if output_path:
            Path(output_path).write_bytes(audio_data)

        return {"audio": audio_data, "metrics": metrics}
