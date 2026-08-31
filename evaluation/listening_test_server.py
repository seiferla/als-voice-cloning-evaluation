"""Local server for running the listening test in person, back-to-back with
many raters on one machine: no manual download/move step, every submission
is saved and the summary CSVs are updated automatically.

Usage:
    uv run python evaluation/prepare_listening_test.py   # build the test once
    uv run python evaluation/listening_test_server.py    # start the server

Then, for each rater in turn: open http://localhost:8765 in a browser, let
them rate, click "Ergebnisse speichern", click "Für nächste Person
zurücksetzen" (reloads the page with empty answers) and hand it to the next
person. Keep the terminal running for the whole session; stop with Ctrl+C.

Each submission is appended as one line to
evaluation/results/listening_test/responses/all_ratings.jsonl, and
nmos_summary.csv / turing_summary.csv are regenerated from everything
collected so far - so they're always current, no need to re-run
analyze_listening_test.py by hand.
"""

import errno
import functools
import http.server
import json
from pathlib import Path

from analyze_listening_test import load_key, load_responses, write_summaries

OUT_DIR = Path(__file__).parent / "results" / "listening_test"
RESPONSES_DIR = OUT_DIR / "responses"
PORT = 8765


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
        print(f"\nSaved rating from '{payload.get('rater', 'anonym')}'")

        try:
            write_summaries(load_key(), load_responses())
        except Exception as e:
            print(f"  (summary update failed: {e})")

        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(b'{"status": "ok"}')

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
        print("Open that URL for each rater in turn. Ctrl+C to stop.")
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            print("\nStopped. Ratings collected so far are saved in "
                  f"{RESPONSES_DIR / 'all_ratings.jsonl'}")


if __name__ == "__main__":
    main()
