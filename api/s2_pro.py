import asyncio
import time
import subprocess
import websockets
import msgpack

API_KEY = "0f8930e0371d4c1a805481ec757ae9c9"


async def test_tts():
    uri = "wss://api.fish.audio/v1/tts/live"
    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "model": "s1"
    }

    async with websockets.connect(uri, additional_headers=headers) as ws:
        # 1. Send StartEvent
        await ws.send(msgpack.packb({
            "event": "start",
            "request": {
                "text": "",
                "format": "mp3",
                "latency": "normal",
                "reference_id": "daaad6c2df01439cae59d4f8e3a08284",
                "prosody": {
                    "speed": 0.95,   # 1.0 = normal, 0.5–2.0 möglich
                    "volume": 2      # 0 = normal
                }
            }
        }))

        # 2. Send text chunk
        text = "Ich hab in letzter Zeit immer mal wieder mit Claudi geschrieben. Sie hat erzählt dass sie in diesem Jahr in Kahla Stadtjubiliäum feiern und viele verschiedene Veranstaltungen dazu haben."

        t_request = time.perf_counter()

        await ws.send(msgpack.packb({
            "event": "text",
            "text": text
        }))

        # 3. Flush + stop
        await ws.send(msgpack.packb({"event": "flush"}))
        await ws.send(msgpack.packb({"event": "stop"}))

        # 4. ffplay starten — spielt sofort ab sobald Chunks ankommen
        ffplay = subprocess.Popen(
            ["ffplay", "-nodisp", "-autoexit", "-i", "pipe:0"],
            stdin=subprocess.PIPE,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL
        )

        # 5. Collect + stream audio chunks
        audio_data = b""
        t_first_audio = None
        chunk_count = 0

        async for message in ws:
            msg = msgpack.unpackb(message)
            event = msg.get("event")
            print(f"Received event: {event}")

            if event == "audio":
                if t_first_audio is None:
                    t_first_audio = time.perf_counter()
                    ttfa = t_first_audio - t_request
                    print(f"  → TTFA (Time To First Audio): {ttfa * 1000:.0f} ms")
                audio_data += msg["audio"]
                chunk_count += 1
                # sofort an ffplay weitergeben
                ffplay.stdin.write(msg["audio"])
                ffplay.stdin.flush()

            elif event == "finish":
                t_finish = time.perf_counter()
                print(f"Done. Reason: {msg.get('reason')}")
                break

        # 6. ffplay sauber beenden
        ffplay.stdin.close()
        ffplay.wait()

        # 7. Save
        with open("output_slower_0.95_volume_4.mp3", "wb") as f:
            f.write(audio_data)

        # 8. Metrics
        total_latency = t_finish - t_request
        audio_duration = estimate_mp3_duration(audio_data)
        rtf = total_latency / audio_duration if audio_duration > 0 else None

        print("\n─── Performance ───────────────────────")
        print(f"  TTFA  (Time To First Audio) : {ttfa * 1000:.0f} ms")
        print(f"  Total latency               : {total_latency * 1000:.0f} ms")
        print(f"  Audio duration (estimated)  : {audio_duration:.2f} s")
        print(f"  RTF   (Real-Time Factor)    : {rtf:.3f}  {'✓ faster than realtime' if rtf and rtf < 1 else '✗ slower than realtime'}")
        print(f"  Chunks received             : {chunk_count}")
        print(f"  Audio size                  : {len(audio_data) / 1024:.1f} KB")
        print("────────────────────────────────────────")


def estimate_mp3_duration(mp3_bytes: bytes) -> float:
    try:
        from mutagen.mp3 import MP3
        import io
        audio = MP3(io.BytesIO(mp3_bytes))
        return audio.info.length
    except ImportError:
        bitrate_kbps = 128
        return (len(mp3_bytes) * 8) / (bitrate_kbps * 1000)


asyncio.run(test_tts())