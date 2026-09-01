"""Voxtral TTS via Mistral API."""

import base64
import io
import time
from pathlib import Path
import soundfile as sf

try:
    from mistralai import Mistral  # neuere/ältere SDKs
except ImportError:  # namespace layout in mistralai>=2.x
    from mistralai.client.sdk import Mistral

from myecho.config import VOXTRAL_API_KEY


class VoxtralTTS:
    def __init__(self, voice_id=None, voice_name="VoiceClone", sample_audio_path=None):
        self.client = Mistral(api_key=VOXTRAL_API_KEY)
        self.model = "voxtral-mini-tts-2603"
        self.voice_id = voice_id
        self.voice_name = voice_name
        self.sample_audio_path = sample_audio_path

    def create_voice(self, audio_path, languages=None, gender=None):
        """Create a custom voice from an audio sample.

        gender defaults to None, not to a fixed value. A gender hint is an extra
        input that neither of the other two services receives, and the evaluation
        compares the three against each other - anything one of them gets and the
        others do not is a confound, not a feature. Pass it explicitly if you
        want it, and then say so in the write-up.
        """
        audio_b64 = base64.b64encode(Path(audio_path).read_bytes()).decode()
        payload = {
            "name": self.voice_name,
            "sample_audio": audio_b64,
            "sample_filename": Path(audio_path).name,
            "languages": languages if languages is not None else ["de"],
        }
        if gender is not None:
            payload["gender"] = gender
        voice = self.client.audio.voices.create(**payload)
        self.voice_id = voice.id
        return voice.id

    async def synthesize(self, text, output_path=None):
        """Generate speech from text using custom voice."""
        if not self.voice_id:
            raise ValueError("No voice_id set. Call create_voice() first or pass voice_id to __init__")

        t_request = time.perf_counter()
        response = self.client.audio.speech.complete(
            model=self.model,
            input=text,
            voice_id=self.voice_id,
            response_format="wav",
        )
        t_finish = time.perf_counter()

        audio_data = base64.b64decode(response.audio_data)
        # latency: Sekunden (Request -> Response)
        # rtf:     Real-Time-Factor = latency_s / audio_duration_s (dimensionslos)
        # audio_duration_s: Sekunden
        # ttfa: not measurable - the Mistral endpoint is not streaming, so the
        # first audio byte arrives together with the last one. Stays None so the
        # eval CSV shows an honest gap instead of a value equal to latency.
        metrics = {
            "latency": t_finish - t_request,
            "ttfa": None,
            "rtf": None,
            "audio_duration_s": None,
            "connect_s": None,
        }

        # Calculate audio duration and RTF
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
