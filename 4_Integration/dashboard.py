"""
dashboard.py - live web dashboard for the acoustic waste sorter.

Run (from the project folder, Arduino Serial Monitor CLOSED):
    python 4_Integration\\dashboard.py
Your browser opens http://localhost:8000 automatically.

What it does
  * Connects to the ESP32 (COM5), checks the microphone, loads the trained model.
  * When you press "Start sorting", it arms the trigger and shows READY.
  * Each drop: sound captured -> quality check -> features -> ML model -> servo.
    Every stage lights up on the page, with the waveform, the frequency spectrum,
    the predicted material and the model's confidence.
  * Low confidence -> UNSURE: the tray stays centred instead of guessing.
  * "It was metal / plastic" buttons record the truth (session accuracy + training data
    for  python 3_ML_Model\\train_model.py --include-live).
Uses only the Python standard library plus the project's own modules.
"""
import csv
import json
import queue
import re
import subprocess
import sys
import threading
import time
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(ROOT / "3_ML_Model"))

from esp_link import EspLink, EspError, PORT, SAMPLE_RATE  # noqa: E402
from audio_features import (int16_to_float, save_wav, extract_features, load_wav,  # noqa: E402
                            quality, FEATURE_NAMES)
from predict_audio import Classifier, MIN_VOTE_SHARE  # noqa: E402

RESULTS = ROOT / "4_Test_Results"
CLIPS = RESULTS / "live_clips"
LOG = RESULTS / "live_log.csv"
LOG_HEADER = ["time", "clip", "predicted", "vote_share", "true_label", "snr_db", "peak", "model_version", "true_object"]
HTML = HERE / "dashboard.html"

# Features shown on the page to explain a decision, with plain-language meaning.
EXPLAIN = [
    ("centroid_hz", "Spectral centroid (brightness)", "Hz", "Where most of the sound energy sits."),
    ("rms", "RMS energy", "", "How much sound energy the impact carried."),
    ("zcr_first_50ms", "Zero crossing rate", "", "How often the wave crosses zero in the first 50 ms."),
    ("bandwidth_hz", "Spectral bandwidth", "Hz", "How widely the energy is spread around the centroid."),
]
STAGES = ["Waiting for a drop", "Sound captured", "Quality check", "Features measured",
          "Model decides", "Servo sorts"]


class Shared:
    def __init__(self):
        self.lock = threading.Lock()
        self.cmds = queue.Queue()
        self.seq = 0
        self.s = {
            "stage": "connecting", "message": "Connecting to the sorter...", "running": False,
            "step": 0, "step_state": "idle", "mic": None, "model": None, "last": None,
            "history": [], "stats": {"n": 0, "correct": 0, "unsure": 0, "wrong": 0},
            "class_profile": None, "sound_map": None, "stages": STAGES, "port_connected": False, "training": {"status": "idle", "message": ""}, "explain": [list(e) for e in EXPLAIN],
            "threshold": MIN_VOTE_SHARE, "port": PORT,
        }

    def update(self, **kw):
        with self.lock:
            self.s.update(kw)
            self.seq += 1
            self.s["seq"] = self.seq

    def snapshot(self):
        with self.lock:
            return json.dumps(self.s)


SH = Shared()


# ---------------- logging (same format as sorter_controller live mode) ----------------
def ensure_log():
    RESULTS.mkdir(exist_ok=True)
    CLIPS.mkdir(parents=True, exist_ok=True)
    if LOG.exists():
        with open(LOG, newline="") as f:
            rows = list(csv.reader(f))
        if rows and "true_object" not in rows[0]:
            with open(LOG, "w", newline="") as f:
                w = csv.writer(f)
                w.writerow(LOG_HEADER)
                for r in rows[1:]:
                    w.writerow(r + [""] * (len(LOG_HEADER) - len(r)))
    else:
        with open(LOG, "w", newline="") as f:
            csv.writer(f).writerow(LOG_HEADER)


_pending = {}          # result id -> log row waiting for a possible truth answer
_pending_lock = threading.Lock()


def flush_pending(keep_id=None):
    with _pending_lock:
        ids = [i for i in list(_pending) if i != keep_id]
        rows = [_pending.pop(i) for i in ids]
        if rows:
            with open(LOG, "a", newline="") as f:
                w = csv.writer(f)
                for r in rows:
                    w.writerow(r)


def set_truth_in_log(clip_name, truth):
    """Write a late answer into an already-saved log row."""
    with _pending_lock:
        try:
            with open(LOG, newline="") as f:
                rows = list(csv.reader(f))
            if not rows:
                return
            ci, ti = rows[0].index("clip"), rows[0].index("true_label")
            for r in rows[1:]:
                if len(r) > max(ci, ti) and r[ci] == clip_name:
                    r[ti] = truth
            with open(LOG, "w", newline="") as f:
                csv.writer(f).writerows(rows)
        except (OSError, ValueError):
            pass


# ---------------- class profile + sound map from the training recordings ----------------
MAP = {}          # mean / components used to place new drops on the sound map


def training_summary(clf):
    """Median of the explained features per class, and a 2-D 'sound map' (PCA of the
    model's standardised features) of every training recording."""
    idx = {n: i for i, n in enumerate(FEATURE_NAMES)}
    feats, labels, rmss = [], [], []
    for label in ("metal", "plastic"):
        for p in sorted((ROOT / "2_Audio_Dataset" / label.capitalize()).glob(f"{label}_obj*.wav")):
            try:
                x, sr = load_wav(p)
                ok, st, _ = quality(x, sr)
                if ok:
                    feats.append(extract_features(x, sr))
                    labels.append(label)
                    rmss.append(st["rms"])
            except Exception:
                pass
    if not feats:
        return None, None
    F = np.vstack(feats)
    L = np.array(labels)
    R = np.array(rmss)
    profile = {lab: {k: float(np.median(R[L == lab])) if k == "rms" else float(np.median(F[L == lab, idx[k]]))
                     for k, *_ in EXPLAIN}
               for lab in ("metal", "plastic") if np.any(L == lab)}
    try:
        z = clf.pipe.named_steps["scale"].transform(F)
        mean = z.mean(axis=0)
        _, sv, vt = np.linalg.svd(z - mean, full_matrices=False)
        comps = vt[:2]
        xy = (z - mean) @ comps.T
        MAP.update(mean=mean, comps=comps, scale=clf.pipe.named_steps["scale"])
        var = (sv[:2] ** 2 / np.sum(sv ** 2)).tolist()
        sound_map = {"points": [[round(float(a), 3), round(float(b), 3), l] for (a, b), l in zip(xy, L)],
                     "explained": var}
    except Exception:
        sound_map = None
    return profile, sound_map


def map_xy(feats):
    if not MAP:
        return None
    z = MAP["scale"].transform(feats.reshape(1, -1))[0]
    a, b = (z - MAP["mean"]) @ MAP["comps"].T
    return [round(float(a), 3), round(float(b), 3)]


def spectrum_db(x, bins=64):
    """Detailed spectrum of the first 0.25 s after impact, in dB relative to its maximum."""
    from scipy.signal import welch
    from audio_features import onset_segment
    seg, _ = onset_segment(x, SAMPLE_RATE)
    f, P = welch(seg, SAMPLE_RATE, nperseg=256, noverlap=192)
    edges = np.linspace(0, SAMPLE_RATE / 2, bins + 1)
    band = np.array([P[(f >= edges[i]) & (f < edges[i + 1] + (1 if i == bins - 1 else 0))].mean() for i in range(bins)])
    db = 10 * np.log10(band / band.max() + 1e-12)
    return np.round(np.maximum(db, -80), 1).tolist(), float(f[int(np.argmax(P[1:])) + 1])


def model_type(clf):
    est = list(clf.pipe.named_steps.values())[-1]
    names = {"SVC": "Support vector machine (SVM)", "RandomForestClassifier": "Random forest",
             "LogisticRegression": "Logistic regression"}
    return names.get(type(est).__name__, type(est).__name__)


FEATURES_USED = "21 measurements: energy in 13 frequency bands, spectral centroid, bandwidth, flatness, tonality, early decay (2), decay time, zero crossing rate"


def waveform(x, points=600):
    n = len(x) // points
    seg = x[: n * points].reshape(points, n)
    return {"min": np.round(seg.min(axis=1), 4).tolist(), "max": np.round(seg.max(axis=1), 4).tolist(),
            "duration": len(x) / SAMPLE_RATE}


# ---------------- worker: owns the serial port ----------------
def load_model():
    """Load the saved model and publish its details. Returns the Classifier."""
    clf = Classifier()
    b = clf.bundle
    cv = b.get("cv") or {}
    SH.update(model={"version": clf.version, "trained_on": clf.trained_on,
                     "cv_accuracy": cv.get("accuracy"), "balanced_accuracy": cv.get("balanced_accuracy"),
                     "confusion": cv.get("confusion_matrix"), "labels": cv.get("labels"),
                     "n_splits": cv.get("n_splits"), "n_clips": b.get("n_clips"),
                     "objects": {k: len(v) for k, v in (b.get("objects") or {}).items()},
                     "type": model_type(clf), "features_used": FEATURES_USED})
    profile, sound_map = training_summary(clf)
    SH.update(class_profile=profile, sound_map=sound_map)
    return clf


# ---------------- retraining from the page ----------------
RELOAD = threading.Event()        # set when a new model was saved; the worker reloads it


def retrain_job():
    """Runs train_model.py --include-live in the background and reports the result."""
    before = (SH.s.get("model") or {}).get("cv_accuracy")
    SH.update(training={"status": "running", "message": "Saving your answers and training the model...",
                        "started": time.strftime("%H:%M:%S"), "log": ""})
    flush_pending()
    try:
        proc = subprocess.run([sys.executable, str(ROOT / "3_ML_Model" / "train_model.py"), "--include-live"],
                              cwd=str(ROOT), capture_output=True, text=True, timeout=900)
        out = (proc.stdout or "") + (proc.stderr or "")
    except Exception as e:
        SH.update(training={"status": "failed", "message": f"Training could not run: {e}", "log": ""})
        return
    lines = [l for l in out.splitlines() if not l.startswith("SKIP")]
    log = "\n".join(lines[-40:])
    acc = re.search(r"Out-of-fold accuracy:\s*([0-9.]+)", out)
    acc = float(acc.group(1)) if acc else None
    live = re.search(r"Live drops: (.*)", out)
    if "MODEL SAVED" in out:
        RELOAD.set()
        msg = "New model saved and loaded."
        if acc is not None:
            msg += f" Accuracy on new objects: {acc:.0%}" + (f" (previous model {before:.0%})." if before is not None else ".")
        SH.update(training={"status": "done", "message": msg, "accuracy": acc, "before": before,
                            "live": live.group(1) if live else "", "log": log, "finished": time.strftime("%H:%M:%S")})
    else:
        stop = re.search(r"TRAINING STOPPED: (.*)", out)
        reason = stop.group(1) if stop else (lines[-1] if lines else "unknown error")
        SH.update(training={"status": "failed", "message": f"Not saved: {reason}. The previous model is still in use.",
                            "accuracy": acc, "before": before, "log": log, "finished": time.strftime("%H:%M:%S")})


def worker():
    clf = None
    try:
        clf = load_model()
    except Exception as e:
        SH.update(stage="error", message=f"Model not ready: {e}")
        return
    ensure_log()

    esp = None
    while esp is None:
        try:
            esp = EspLink(PORT, verbose=False)
        except EspError as e:
            SH.update(stage="error", message=str(e).split("\n")[0] + " - retrying in 3 s. Close the Arduino Serial Monitor.")
            time.sleep(3)
    SH.update(port_connected=True)
    d = esp.diagnostic()
    SH.update(mic={"ok": d["ok"], "slot": d["slot"]})
    if not d["ok"]:
        SH.update(stage="error", message="Microphone check failed. Check the INMP441 wires, then restart the dashboard.")
        return
    SH.update(stage="idle", message="Press Start sorting when you are ready.", step=0, step_state="idle")

    next_id = 1
    while True:
        # commands are only executed while paused (the trigger wait cannot be interrupted)
        try:
            cmd = SH.cmds.get_nowait()
        except queue.Empty:
            cmd = None
        if cmd:
            kind, arg = cmd
            if kind == "sort" and not SH.s["running"]:
                try:
                    SH.update(stage="manual", message=f"Manual sort: {arg} (no audio, no model)")
                    esp.sort(arg)
                except EspError as e:
                    SH.update(message=f"Servo command failed: {e}")
                SH.update(stage="idle", message="Paused. Press Start sorting to continue.")
            continue

        if RELOAD.is_set():
            RELOAD.clear()
            try:
                clf = load_model()
            except Exception as e:
                SH.update(training={**SH.s.get("training", {}), "status": "failed",
                                    "message": f"New model could not be loaded: {e}"})
        if not SH.s["running"]:
            if SH.s["stage"] not in ("idle", "error"):
                SH.update(stage="idle", message="Paused. Press Start sorting to continue.", step=0, step_state="idle")
            time.sleep(0.15)
            continue

        SH.update(stage="ready", message="Ready. Drop an object on the plate.", step=1, step_state="active")
        try:
            samples, status = esp.record(triggered=True, timeout=25)
        except EspError as e:
            SH.update(stage="error", message=f"Audio transfer failed ({e}). Re-arming.")
            time.sleep(1)
            continue
        if samples is None:            # trigger timeout: just arm again
            continue

        x = int16_to_float(samples)
        wf = waveform(x)
        SH.update(stage="processing", message="Impact heard. Analysing the sound...", step=2, step_state="active",
                  last={"waveform": wf, "label": None})
        time.sleep(0.35)               # short pauses so each stage is visible on screen (not counted in analysis time)

        tc = time.time()
        ok, st, reasons = quality(x, SAMPLE_RATE)
        compute_s = time.time() - tc
        SH.update(step=3, step_state="active" if ok else "fail")
        time.sleep(0.35)
        stamp = time.strftime("%Y%m%d-%H%M%S") + f"-{int(time.time() * 1000) % 1000:03d}"
        clip_name = f"live_{stamp}.wav"
        save_wav(CLIPS / clip_name, samples)
        rid = next_id
        next_id += 1

        if not ok:
            item = {"id": rid, "time": time.strftime("%H:%M:%S"), "label": "REJECTED", "confidence": None, "result": "Rejected", "clip": clip_name,
                    "truth": None, "reasons": reasons}
            y0 = x - np.mean(x)
            SH.update(stage="rejected", message="Recording not usable: " + "; ".join(reasons),
                      last={"waveform": wf, "label": "REJECTED", "reasons": reasons, "stats": st, "id": rid,
                            "peak_amp": float(np.max(np.abs(y0))), "rms_amp": float(st.get("rms", 0))},
                      history=([item] + SH.s["history"])[:30])
            with _pending_lock:
                _pending[rid] = [stamp, clip_name, "REJECTED", "", "", round(st.get("snr_db", 0), 1),
                                 round(st.get("peak", 0), 4), clf.version, ""]
            flush_pending()
            time.sleep(2.5)
            continue

        tc = time.time()
        feats = extract_features(x, SAMPLE_RATE)
        compute_s += time.time() - tc
        idx = {n: i for i, n in enumerate(FEATURE_NAMES)}
        bands = [float(10 ** v) for v in feats[:13]]
        shown = {k: (float(st["rms"]) if k == "rms" else float(feats[idx[k]])) for k, *_ in EXPLAIN}
        spec, dom = spectrum_db(x)
        y0 = x - np.mean(x)
        amps = {"peak_amp": float(np.max(np.abs(y0))), "rms_amp": float(st["rms"])}
        SH.update(step=4, step_state="active",
                  last={"waveform": wf, "label": None, "bands": bands, "features": shown,
                        "spectrum_db": spec, "dominant_hz": dom, **amps})
        time.sleep(0.35)

        tc = time.time()
        r = clf.predict_array(x, SAMPLE_RATE)
        compute_s += time.time() - tc
        label = r["label"]
        conf = r["vote_share"]
        last = {"waveform": wf, "bands": bands, "features": shown, "label": label, "map_xy": map_xy(feats),
                "spectrum_db": spec, "dominant_hz": dom, **amps,
                "peak": st["peak"], "noise_rms": st["noise_rms"],
                "best_guess": r.get("best_guess"), "confidence": conf, "votes": r.get("votes"),
                "snr_db": st["snr_db"], "id": rid, "analysis_ms": max(1, int(compute_s * 1000))}
        item = {"id": rid, "time": time.strftime("%H:%M:%S"), "label": label, "confidence": conf, "truth": None, "clip": clip_name,
                "p_metal": (r.get("votes") or {}).get("metal"),
                "result": "Sorted" if label in ("metal", "plastic") else "Not sorted"}
        with _pending_lock:
            _pending[rid] = [stamp, clip_name, label, round(conf, 3), "", round(st.get("snr_db", 0), 1),
                             round(st.get("peak", 0), 4), clf.version, ""]
        flush_pending(keep_id=rid)
        SH.update(step=5, step_state="active", last=last, history=([item] + SH.s["history"])[:30])
        time.sleep(0.35)

        if label in ("metal", "plastic"):
            SH.update(stage="sorting", message=f"Sorting into the {label} bin...", step=6, step_state="active")
            try:
                esp.sort(label)
            except EspError as e:
                SH.update(message=f"Servo command failed: {e}")
            SH.update(stage="result", message=f"Sorted as {label}.", step=6, step_state="done")
        else:
            SH.update(stage="unsure", message="Not confident enough to sort. The tray stays level; drop it again.",
                      step=5, step_state="fail")
        time.sleep(2.5)


# ---------------- HTTP ----------------
class Handler(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def _send(self, code, body, ctype="application/json"):
        data = body.encode() if isinstance(body, str) else body
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Cache-Control", "no-store")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):
        if self.path in ("/", "/index.html"):
            self._send(200, HTML.read_bytes(), "text/html; charset=utf-8")
        elif self.path.startswith("/api/state"):
            self._send(200, SH.snapshot())
        else:
            self._send(404, '{"error":"not found"}')

    def do_POST(self):
        n = int(self.headers.get("Content-Length") or 0)
        try:
            body = json.loads(self.rfile.read(n) or b"{}")
        except json.JSONDecodeError:
            body = {}
        if self.path == "/api/start":
            if SH.s["stage"] != "error":
                SH.update(running=True)
            self._send(200, '{"ok":true}')
        elif self.path == "/api/pause":
            SH.update(running=False,
                      message="Pausing... finishes after the current wait (up to 15 s)." if SH.s["stage"] == "ready"
                      else SH.s["message"])
            self._send(200, '{"ok":true}')
        elif self.path == "/api/sort":
            if body.get("label") in ("metal", "plastic", "center") and not SH.s["running"]:
                SH.cmds.put(("sort", body["label"]))
                self._send(200, '{"ok":true}')
            else:
                self._send(400, '{"error":"pause first"}')
        elif self.path == "/api/truth":
            self._truth(body)
        elif self.path == "/api/retrain":
            if SH.s["running"]:
                self._send(400, '{"error":"Pause sorting first"}')
            elif (SH.s.get("training") or {}).get("status") == "running":
                self._send(400, '{"error":"Training already running"}')
            else:
                threading.Thread(target=retrain_job, daemon=True).start()
                self._send(200, '{"ok":true}')
        else:
            self._send(404, '{"error":"not found"}')

    def _truth(self, body):
        rid, truth = body.get("id"), body.get("label")
        if truth not in ("metal", "plastic"):
            return self._send(400, '{"error":"label"}')
        with SH.lock:
            hist = SH.s["history"]
            item = next((h for h in hist if h["id"] == rid), None)
            if item is None or item["label"] == "REJECTED":
                return self._send(400, '{"error":"unknown id"}')
            st = SH.s["stats"]
            if item["truth"] is None:
                st["n"] += 1
            else:                                   # changing an earlier answer
                if item["label"] == "UNSURE":
                    st["unsure"] -= 1
                elif item["label"] == item["truth"]:
                    st["correct"] -= 1
                else:
                    st["wrong"] -= 1
            item["truth"] = truth
            if item["label"] == "UNSURE":
                st["unsure"] += 1
            elif item["label"] == truth:
                st["correct"] += 1
            else:
                st["wrong"] += 1
            SH.seq += 1
            SH.s["seq"] = SH.seq
        with _pending_lock:
            if rid in _pending:
                _pending[rid][4] = truth
                saved = True
            else:
                saved = False
        if not saved and item.get("clip"):
            set_truth_in_log(item["clip"], truth)
        self._send(200, '{"ok":true}')


def main():
    threading.Thread(target=worker, daemon=True).start()
    server = None
    for port in range(8000, 8011):
        try:
            server = ThreadingHTTPServer(("127.0.0.1", port), Handler)
            break
        except OSError:
            continue
    if server is None:
        print("No free port between 8000 and 8010.")
        return 1
    url = f"http://localhost:{server.server_address[1]}"
    print(f"Dashboard running at {url}  (press Ctrl+C here to stop)")
    try:
        webbrowser.open(url)
    except Exception:
        pass
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopping...")
    finally:
        flush_pending()
    return 0


if __name__ == "__main__":
    sys.exit(main())
