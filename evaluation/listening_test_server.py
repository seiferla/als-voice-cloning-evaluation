"""Local server for running the listening test in person, rater after rater.

Usage:
    uv run python evaluation/listening_test_server.py
"""

import errno
import functools
import http.server
import json
import re
import threading
import urllib.parse
from pathlib import Path

from analyze_listening_test import load_key, load_responses, write_summaries

OUT_DIR = Path(__file__).parent / "results" / "listening_test"
RESPONSES_DIR = OUT_DIR / "responses"
PORT = 8765

# One writer at a time for the CSVs.
_summary_lock = threading.Lock()

# Serialize appends to all_ratings.jsonl across request threads.
_append_lock = threading.Lock()


def update_summaries():
    with _summary_lock:
        try:
            write_summaries(load_key(), load_responses(), quiet=True)
            print("  summaries updated")
        except Exception as e:
            print(f"  (summary update failed: {type(e).__name__}: {e})")


class Handler(http.server.SimpleHTTPRequestHandler):
    # Only the page and the stimuli; OUT_DIR also holds key.json and responses/.
    _ALLOWED_PATHS = frozenset({"/", "/index.html"})
    _ALLOWED_PATTERN = re.compile(r"/samples/s_\d{3}\.wav\Z")

    def _permitted(self):
        path = urllib.parse.urlparse(self.path).path
        path = urllib.parse.unquote(path)
        if path in self._ALLOWED_PATHS:
            return True
        return bool(self._ALLOWED_PATTERN.fullmatch(path))

    def do_GET(self):
        if not self._permitted():
            self.send_error(404)
            return
        super().do_GET()

    def do_HEAD(self):
        if not self._permitted():
            self.send_error(404)
            return
        super().do_HEAD()

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
        with _append_lock:
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
        # No caching: a stale index.html would desync the rater from key.json.
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
        # Usually an earlier session still holds the port; move to the next one.
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
