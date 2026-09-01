"""Local server for running the listening test in person, rater after rater.

Usage:
    uv run python evaluation/listening_test_server.py

Open http://localhost:8765 for each rater in turn. The rater enters their own
scores; when they submit, the answers are appended to
responses/all_ratings.jsonl and the button turns into a reset for the next
person. Add ?plan=2 to force a specific rater plan.

Summaries are recomputed in a background thread: the bootstrap takes a few
seconds and no rater should wait for it.
"""

import errno
import functools
import http.server
import json
import threading
from pathlib import Path

from analyze_listening_test import load_key, load_responses, write_summaries

OUT_DIR = Path(__file__).parent / "results" / "listening_test"
RESPONSES_DIR = OUT_DIR / "responses"
PORT = 8765

# One writer at a time: two submissions arriving close together would otherwise
# recompute and rewrite the same CSVs concurrently.
_summary_lock = threading.Lock()


def update_summaries():
    with _summary_lock:
        try:
            write_summaries(load_key(), load_responses(), quiet=True)
            print("  summaries updated")
        except Exception as e:
            print(f"  (summary update failed: {type(e).__name__}: {e})")


class Handler(http.server.SimpleHTTPRequestHandler):
    def do_POST(self):
        if self.path != "/submit":
            self.send_error(404)
            return

        length = int(self.headers.get("Content-Length", 0))
        try:
            payload = json.loads(self.rfile.read(length))
        except json.JSONDecodeError:
            self.send_error(400, "invalid json")
            return

        RESPONSES_DIR.mkdir(parents=True, exist_ok=True)
        with open(RESPONSES_DIR / "all_ratings.jsonl", "a", encoding="utf-8") as f:
            f.write(json.dumps(payload, ensure_ascii=False) + "\n")

        answered = sum(1 for r in payload.get("nmos", []) if r.get("rating") is not None)
        chosen = sum(1 for t in payload.get("turing", []) if t.get("choice") is not None)
        print(f"\nSaved {payload.get('rater')} (plan {payload.get('plan_id')}): "
              f"{answered} naturalness ratings, {chosen} pair answers")

        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(b'{"status": "ok"}')

        threading.Thread(target=update_summaries, daemon=True).start()

    def end_headers(self):
        # The stimulus set is rebuilt between studies and browsers happily serve
        # a stale index.html or a stale sample from cache, which would silently
        # desync a rater from key.json.
        self.send_header("Cache-Control", "no-store")
        super().end_headers()

    def log_message(self, format, *args):
        pass  # our own print()s are enough


def serve(port):
    handler = functools.partial(Handler, directory=str(OUT_DIR))
    try:
        return http.server.ThreadingHTTPServer(("localhost", port), handler)
    except OSError as e:
        if e.errno != errno.EADDRINUSE:
            raise
        return None


def main():
    if not (OUT_DIR / "index.html").exists():
        raise SystemExit("Run evaluation/prepare_listening_test.py first.")

    port = PORT
    httpd = serve(port)
    if httpd is None:
        # Usually an earlier session of this script still running. Don't fail
        # with a traceback in the middle of a rater session - say what holds
        # the port and move to the next one.
        print(f"Port {PORT} is already in use. Most likely an older "
              f"listening_test_server.py is still running.")
        print(f"  Check it with:  lsof -nP -iTCP:{PORT} -sTCP:LISTEN")
        print(f"  Stop it with:   kill $(lsof -t -iTCP:{PORT} -sTCP:LISTEN)")
        for candidate in range(PORT + 1, PORT + 20):
            httpd = serve(candidate)
            if httpd is not None:
                port = candidate
                print(f"\nUsing port {port} instead for this session.")
                break
        else:
            raise SystemExit(f"No free port between {PORT} and {PORT + 19}.")

    with httpd:
        print(f"\nListening test running at http://localhost:{port}")
        print("Open that URL for each rater in turn. The rater enters their own")
        print("scores. Ctrl+C to stop.")
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            print("\nStopped. Ratings collected so far are saved in "
                  f"{RESPONSES_DIR / 'all_ratings.jsonl'}")


if __name__ == "__main__":
    main()
