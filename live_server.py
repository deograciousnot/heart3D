"""
Live mode: stream an ECG into the viewer as if it came from a recording device.

This is stage 1 of the hardware plan (step 7): everything a real device needs, minus the device.
A "source" hands over raw 12-lead samples a little at a time; here it replays a PTB-XL record
in real time. A real board (e.g. OpenBCI via BrainFlow) only has to offer the same
`read(n) -> (12, n) array in mV, LEAD_NAMES order` and `fs`.

    python live_server.py                  # replay unseen test ECG 1737 (old inferior MI)
    python live_server.py --record 7079    # any PTB-XL ecg_id (500 Hz file used if it's in ptbxl/)

Then open http://localhost:8766/viewer.html?live=1 (the server also serves the pages).
Add &record=7079 to the URL to pick the record from the browser.

What happens to the samples, the way a bedside monitor would do it:
  1. causal 0.5-40 Hz band-pass (it can only use the past), then down to 250 Hz for the browser
  2. Kors matrix -> 3D heart vector, sample by sample
  3. every 0.25 s: P / QRS / T labels from the last 10 s; samples go out 0.6 s behind real time,
     so each beat is labelled before it's drawn
  4. every 2 s: the averaged beat and its measurements (vitals)
  5. every 5 s, once 10 s have arrived: both AI models on the last 10 s (in a background thread)

Sent to the browser as Server-Sent Events on /stream: "meta" once, then "samples", "beat", "ai".
Not a medical device. Replays loop every 10 s, so there's a small jump at the seam.
"""

import argparse
import json
import os
import queue
import threading
import time
import warnings
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlparse

import numpy as np
from scipy.signal import butter, sosfilt, sosfilt_zi

from ecg_to_vcg import KORS, KORS_INPUTS, LEAD_DIRECTIONS, LEAD_NAMES, analyse_beats, beat_json, detect_phases, read_record
from ptbxl_data import CLASS_NAMES, ROOT

HERE = os.path.dirname(os.path.abspath(__file__))
OUT_FS = 250          # rate sent to the browser
CHUNK_S = 0.04        # read from the source every 40 ms
DELAY_S = 0.6         # samples are sent this far behind real time, already labelled
WINDOW_S = 10         # analysis window (the models read 10 s)
KEEP_S = 12           # history kept on the server
LABEL_EVERY_S = 0.25
BEAT_EVERY_S = 2.0
AI_EVERY_S = 5.0
DEFAULT_RECORD = 1737


class ReplaySource:
    """A pretend device: plays a PTB-XL record in real time, looping."""

    def __init__(self, ecg_id):
        from predict import _ordered, record_meta
        self.meta = record_meta(ecg_id)
        candidates = [os.path.join(ROOT, f"{ecg_id:05d}_hr"),            # example records kept in ptbxl/
                      os.path.join(ROOT, self.meta["filename_hr"]),      # full 500 Hz download, if present
                      os.path.join(ROOT, self.meta["filename_lr"])]      # full 100 Hz download
        path = next((p for p in candidates if os.path.exists(p + ".hea")), None)
        if path is None:
            raise FileNotFoundError(f"no local file for PTB-XL record {ecg_id}; run download_ptbxl.py")
        raw, fs, names = read_record(path)
        self.raw, self.fs, self.pos = _ordered(raw, names), int(fs), 0
        self.description = (f"Replaying PTB-XL record #{ecg_id} in real time, as if from a recording device "
                            f"({self.fs} Hz, loops every {self.raw.shape[1] / self.fs:.0f} s)")

    def read(self, n):
        out = np.take(self.raw, np.arange(self.pos, self.pos + n), axis=1, mode="wrap")
        self.pos = (self.pos + n) % self.raw.shape[1]
        return out


class LivePipeline:
    """Raw samples in -> filtered, labelled 250 Hz leads + heart vector out, plus the averaged beat."""

    def __init__(self, fs):
        self.fs = fs
        self.step = fs // OUT_FS if fs % OUT_FS == 0 else 1
        self.out_fs = fs / self.step
        self.sos = butter(2, [0.5, 40], btype="bandpass", fs=fs, output="sos")
        self.zi = None
        self.n_in = 0                                   # raw samples received so far
        self.raw = np.zeros((12, 0))                    # last WINDOW_S of raw signal, for the AI
        self.leads = np.zeros((12, 0))                  # filtered, 250 Hz
        self.vec = np.zeros((3, 0))
        self.labels = np.zeros(0, dtype=int)
        self.start = 0                                  # absolute output index of self.leads[:, 0]
        self.released = 0                               # next absolute output index to send
        self.beat = None

    def push(self, chunk):
        if self.zi is None:  # start the filter as if the signal had always been at its first value
            self.zi = sosfilt_zi(self.sos)[:, None, :] * chunk[:, 0][None, :, None]
        y, self.zi = sosfilt(self.sos, chunk, axis=-1, zi=self.zi)
        keep = (np.arange(self.n_in, self.n_in + chunk.shape[1]) % self.step) == 0
        self.n_in += chunk.shape[1]
        y = y[:, keep]
        by_name = dict(zip(LEAD_NAMES, y))
        v = KORS @ np.vstack([by_name[n] for n in KORS_INPUTS])
        self.raw = np.hstack([self.raw, chunk])[:, -int(WINDOW_S * self.fs):]
        self.leads = np.hstack([self.leads, y])
        self.vec = np.hstack([self.vec, v])
        self.labels = np.concatenate([self.labels, np.zeros(y.shape[1], dtype=int)])
        extra = self.leads.shape[1] - int(KEEP_S * self.out_fs)
        if extra > 0:
            self.leads, self.vec, self.labels = self.leads[:, extra:], self.vec[:, extra:], self.labels[extra:]
            self.start += extra

    def label(self):
        """Averaged beat + P / QRS / T labels from the last 10 s (needs at least 3 s)."""
        w = min(self.leads.shape[1], int(WINDOW_S * self.out_fs))
        if w < 3 * self.out_fs:
            return
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                beat = analyse_beats(self.vec[:, -w:], self.leads[:, -w:], self.out_fs)
                self.labels[-w:] = detect_phases(w, self.out_fs, beat)
            self.beat = beat
        except Exception:
            pass  # noisy stretch: keep the previous labels

    def averaged_beat(self):
        """
        The averaged beat for the vitals, filtered the offline way: zero-phase band-pass over the last
        10 s of raw signal. The live (causal) filter is fine for drawing but slightly reshapes slow
        waves, which would shift PR and the T wave; this keeps the numbers equal to the offline ones.
        """
        if self.raw.shape[1] < 3 * self.fs:
            return None
        from ecg_to_vcg import clean_ecg
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                leads = clean_ecg(self.raw, self.fs)[:, ::self.step]
                by_name = dict(zip(LEAD_NAMES, leads))
                return analyse_beats(KORS @ np.vstack([by_name[n] for n in KORS_INPUTS]), leads, self.out_fs)
        except Exception:
            return None

    def release(self):
        """Samples that are now DELAY_S old: labelled, ready to draw."""
        upto = self.start + self.leads.shape[1] - int(DELAY_S * self.out_fs)
        if upto <= self.released:
            return None
        a, b = max(self.released, self.start) - self.start, upto - self.start
        self.released = upto
        return {"i": self.start + a,
                "leads": np.round(self.leads[:, a:b], 4).tolist(),
                "vector": np.round(self.vec[:, a:b], 4).tolist(),
                "phase": self.labels[a:b].tolist()}


class Handler(SimpleHTTPRequestHandler):
    default_record = DEFAULT_RECORD

    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=HERE, **kwargs)

    def log_message(self, fmt, *args):
        if "/stream" in self.path or (len(args) > 1 and str(args[1]) not in ("200", "304")):
            super().log_message(fmt, *args)

    def end_headers(self):
        self.send_header("Cache-Control", "no-store")  # always the latest pages while developing
        super().end_headers()

    def do_GET(self):
        url = urlparse(self.path)
        if url.path == "/stream":
            return self.stream(parse_qs(url.query))
        return super().do_GET()

    def event(self, name, data):
        self.wfile.write(f"event: {name}\ndata: {json.dumps(data, separators=(',', ':'))}\n\n".encode())
        self.wfile.flush()

    def stream(self, query):
        try:
            record = int(query.get("record", [self.default_record])[0])
            source = ReplaySource(record)
        except Exception as err:
            self.send_error(404, str(err))
            return
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Connection", "keep-alive")
        self.end_headers()

        pipe = LivePipeline(source.fs)
        results, ai_busy = queue.Queue(), threading.Event()

        def run_ai(raw):
            try:
                import torch
                torch.set_num_threads(1)  # one core: keeps the laptop cool and the stream smooth
                from predict import ai_reading_from_raw
                results.put(ai_reading_from_raw(raw, source.fs, source.meta))
            except Exception as err:
                results.put({"error": str(err)})
            finally:
                ai_busy.clear()

        meta = {k: v for k, v in source.meta.items() if not k.startswith("filename")}
        self.event("meta", {"fs": pipe.out_fs, "leadNames": LEAD_NAMES, "leadDirections": np.round(LEAD_DIRECTIONS, 4).tolist(),
                            "label": f"Live: ECG #{record}", "source": source.description, "record": meta,
                            "classNames": CLASS_NAMES, "delay": DELAY_S})
        chunk = max(1, round(source.fs * CHUNK_S))
        tick = time.perf_counter()
        last_label = last_beat = last_ai = -1e9
        try:
            while True:
                pipe.push(source.read(chunk))
                now = time.perf_counter()
                if now - last_label >= LABEL_EVERY_S:
                    pipe.label()
                    last_label = now
                samples = pipe.release()
                if samples:
                    self.event("samples", samples)
                if now - last_beat >= BEAT_EVERY_S:
                    beat = pipe.averaged_beat()
                    if beat is not None:
                        self.event("beat", beat_json(beat))
                        last_beat = now
                if (pipe.raw.shape[1] >= WINDOW_S * source.fs and now - last_ai >= AI_EVERY_S
                        and not ai_busy.is_set()):
                    ai_busy.set()
                    last_ai = now
                    threading.Thread(target=run_ai, args=(pipe.raw.copy(),), daemon=True).start()
                while not results.empty():
                    self.event("ai", results.get())
                tick += CHUNK_S
                wait = tick - time.perf_counter()
                if wait > 0:
                    time.sleep(wait)
                else:
                    tick = time.perf_counter()  # fell behind (busy laptop): don't try to catch up in a burst
        except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError):
            pass  # browser closed the page


class QuietServer(ThreadingHTTPServer):
    """Browsers close idle connections all the time; that's not an error worth a traceback."""
    daemon_threads = True

    def handle_error(self, request, client_address):
        import sys
        if isinstance(sys.exc_info()[1], (ConnectionAbortedError, ConnectionResetError, BrokenPipeError)):
            return
        super().handle_error(request, client_address)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--record", type=int, default=DEFAULT_RECORD, help="PTB-XL ecg_id to replay (default 1737)")
    parser.add_argument("--port", type=int, default=8766)
    args = parser.parse_args()
    Handler.default_record = args.record
    ReplaySource(args.record)  # fail early if the record isn't on disk
    server = QuietServer(("127.0.0.1", args.port), Handler)

    def warm_up():  # load PyTorch and the models now, so the first AI reading isn't 15 s late
        from predict import _load
        for name in ("feature_model.pkl", "mi_location.pkl", "cnn.pt"):
            if os.path.exists(os.path.join(HERE, "models", name)):
                _load(name)
    threading.Thread(target=warm_up, daemon=True).start()
    print(f"Live mode: open http://localhost:{args.port}/viewer.html?live=1  (replaying PTB-XL #{args.record})")
    print("Ctrl+C to stop.")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
