"""
audio_features.py - shared audio quality gate + feature extraction.

The SAME functions are used when recording (record_audio.py), training
(train_model.py) and predicting (predict_audio.py / sorter_controller.py),
so the model always sees identically processed audio.

Only needs numpy + scipy (no librosa required).
"""
import numpy as np
from scipy.io import wavfile
from scipy.signal import butter, sosfiltfilt, welch

SAMPLE_RATE = 16000
FEATURE_VERSION = 4

# Low-cut filter: removes hum / rumble / DC drift below 120 Hz. Plastic and metal
# impact sounds live well above this, so nothing useful is lost.
HPF_HZ = 120
_SOS = butter(4, HPF_HZ, btype="highpass", fs=SAMPLE_RATE, output="sos")

# Saturation: the mic (or the int16 conversion) is overloaded when several samples
# sit exactly at the clip's maximum value - a flat-topped waveform. Natural audio
# almost never repeats its exact peak value this many times.
SATURATION_MIN_SAMPLES = 4

# Overflow ("wrap-around"): when the impact is too loud, single samples jump to the
# opposite extreme and straight back (e.g. -12000, +20000, -12300). Real sound can't
# do that between two samples, so >= 2 such spikes means the impact overloaded.
WRAP_MIN_SPIKES = 2


def count_wrap_spikes(x):
    x = np.asarray(x, dtype=np.float64)
    if len(x) < 3:
        return 0
    d = np.diff(x)
    a, b = d[:-1], d[1:]
    w = ((np.abs(a) > 0.8) & (np.abs(b) > 0.8) & (np.sign(a) != np.sign(b))
         & (np.abs(x[:-2] - x[2:]) < 0.2 * np.abs(a)))
    return int(w.sum())


def repair_wrap_spikes(x):
    """Flip isolated overflow samples back (a wrapped sample is off by exactly 1.0)."""
    y = np.array(x, dtype=np.float64)
    if len(y) < 3:
        return y
    d = np.diff(y)
    a, b = d[:-1], d[1:]
    w = np.where((np.abs(a) > 0.8) & (np.abs(b) > 0.8) & (np.sign(a) != np.sign(b))
                 & (np.abs(y[:-2] - y[2:]) < 0.2 * np.abs(a)))[0] + 1
    y[w] -= np.sign(a[w - 1]) * 1.0
    return y


def preprocess(x, sr=SAMPLE_RATE):
    """Repair single overflow spikes, remove DC and low-frequency noise."""
    y = repair_wrap_spikes(x)
    y = y - np.mean(y)
    if sr == SAMPLE_RATE and len(y) > 64:
        y = sosfiltfilt(_SOS, y)
    return y

# ---------------- Quality gate thresholds ----------------
# Why these numbers:
#  * max_zero_frac 0.50: a working INMP441 has a noise floor that keeps the low
#    bits moving, so exact zeros are rare. The broken clips were 99.8 % zeros.
#    (Tighter than the old 0.995, which let near-empty clips through.)
#  * min_snr_db 15: the impact must stand clearly above the background noise.
#  * min_event_rms 0.003: the 0.3 s around the impact must have real energy.
#  * max_clip_frac 0.001: at most 0.1 % of samples at full scale.
# Calibrate from the numbers record_audio.py prints for your first good clips;
# change them here only, never per-script.
GATE = {
    "min_duration_s": 0.5,
    "max_zero_frac": 0.50,
    "max_clip_frac": 0.001,
    "min_event_rms": 0.003,
    "min_snr_db": 15.0,
}

EVENT_PRE_S = 0.02     # analysis window starts 20 ms before the loudest sample
EVENT_LEN_S = 0.50     # and lasts 0.5 s (impact + ring/decay)


def load_wav(path):
    """Return (float32 mono in [-1, 1], sample_rate)."""
    sr, data = wavfile.read(str(path))
    if data.ndim > 1:
        data = data.mean(axis=1)
    if data.dtype == np.int16:
        x = data.astype(np.float32) / 32768.0
    elif data.dtype == np.int32:
        x = data.astype(np.float32) / 2147483648.0
    elif data.dtype == np.uint8:
        x = (data.astype(np.float32) - 128.0) / 128.0
    else:
        x = data.astype(np.float32)
    return x, sr


def save_wav(path, samples_int16, sr=SAMPLE_RATE):
    wavfile.write(str(path), sr, np.asarray(samples_int16, dtype=np.int16))


def int16_to_float(samples_int16):
    return np.asarray(samples_int16, dtype=np.float32) / 32768.0


def _frame_rms(x, frame, hop):
    if len(x) < frame:
        return np.array([np.sqrt(np.mean(x ** 2) + 1e-20)])
    n = 1 + (len(x) - frame) // hop
    idx = np.arange(frame)[None, :] + hop * np.arange(n)[:, None]
    return np.sqrt(np.mean(x[idx] ** 2, axis=1) + 1e-20)


def quality(x, sr=SAMPLE_RATE, gate=GATE):
    """Return (ok: bool, stats: dict, reasons: list[str])."""
    reasons = []
    stats = {}
    n = len(x)
    stats["duration_s"] = n / sr if sr else 0.0
    if sr != SAMPLE_RATE:
        reasons.append(f"sample rate {sr} != {SAMPLE_RATE}")
    if n == 0 or stats["duration_s"] < gate["min_duration_s"]:
        reasons.append(f"too short ({stats['duration_s']:.2f}s)")
        return False, stats, reasons
    if not np.all(np.isfinite(x)):
        reasons.append("contains NaN/inf (corrupt)")
        return False, stats, reasons

    stats["zero_frac"] = float(np.mean(x == 0.0))
    ax = np.abs(x)
    top = float(ax.max())
    n_top = int(np.sum(ax >= top - 1.5 / 32768)) if top > 0.05 else 0
    saturated = n_top >= SATURATION_MIN_SAMPLES
    stats["clip_frac"] = float(np.mean(ax >= 0.999))
    stats["saturated_samples"] = n_top if saturated else 0
    if saturated:
        stats["clip_frac"] = max(stats["clip_frac"], n_top / n)
    y = preprocess(x, sr)
    peak_i = int(np.argmax(np.abs(y)))
    stats["peak"] = float(np.abs(y[peak_i]))
    stats["rms"] = float(np.sqrt(np.mean(y ** 2)))

    frame, hop = int(0.02 * sr), int(0.01 * sr)
    fr = _frame_rms(y, frame, hop)
    noise = float(np.percentile(fr, 10))
    a = max(0, peak_i - int(EVENT_PRE_S * sr))
    b = min(n, peak_i + int(0.30 * sr))
    event = float(np.sqrt(np.mean(y[a:b] ** 2)))
    stats["noise_rms"] = noise
    stats["event_rms"] = event
    stats["snr_db"] = float(20 * np.log10((event + 1e-12) / (noise + 1e-12)))

    if stats["zero_frac"] > gate["max_zero_frac"]:
        reasons.append(f"sparse/dead signal: {100*stats['zero_frac']:.1f}% exact zeros "
                       f"(max {100*gate['max_zero_frac']:.0f}%) - microphone stream broken")
    stats["wrap_spikes"] = count_wrap_spikes(x)
    if stats["wrap_spikes"] >= WRAP_MIN_SPIKES:
        reasons.append(f"overloaded: {stats['wrap_spikes']} wrap-around spikes (impact too loud for the mic) "
                       f"- put foam under the mic, move it farther from the tray, or drop from lower")
    if saturated:
        reasons.append(f"saturated: {n_top} samples stuck at the maximum value {top:.3f} "
                       f"(microphone overloaded) - move the mic farther from the tray or drop from lower")
    elif stats["clip_frac"] > gate["max_clip_frac"]:
        reasons.append(f"clipped: {100*stats['clip_frac']:.2f}% samples at full scale "
                       f"(max {100*gate['max_clip_frac']:.2f}%) - move the mic farther or drop from lower")
    if event < gate["min_event_rms"]:
        reasons.append(f"impact too quiet: event RMS {event:.4f} < {gate['min_event_rms']}")
    if stats["snr_db"] < gate["min_snr_db"]:
        reasons.append(f"impact not clear above noise: SNR {stats['snr_db']:.1f} dB < {gate['min_snr_db']} dB")
    return len(reasons) == 0, stats, reasons


# ---------------- Feature extraction ----------------
# Version 4: level-independent features measured on the first 0.25 s after the
# impact onset (where the impact is far above the noise). Loudness itself and long
# decay times are deliberately NOT used: they depend on how the object lands and on
# the room noise, and made the earlier model unreliable in live tests.
SEG_S = 0.25
BAND_EDGES = np.array([0, 250, 500, 750, 1000, 1500, 2000, 2500, 3000, 3500, 4000, 5000, 6000, 8000.01])

FEATURE_NAMES = ([f"band_{int(BAND_EDGES[i])}_{int(BAND_EDGES[i + 1])}Hz" for i in range(len(BAND_EDGES) - 1)]
                 + ["centroid_hz", "bandwidth_hz", "flatness", "tonality_db",
                    "drop_0_10ms_vs_30_50ms_db", "drop_0_10ms_vs_80_120ms_db", "time_to_drop_15db_s", "zcr_first_50ms"])


def onset_segment(x, sr=SAMPLE_RATE, length=SEG_S):
    """Return (segment starting at impact onset, background noise RMS)."""
    y = preprocess(x, sr)
    a = np.abs(y)
    pk = int(np.argmax(a))
    st = max(0, pk - int(0.005 * sr))
    on = st + int(np.argmax(a[st:pk + 1] >= 0.2 * a[pk]))
    n = int(length * sr)
    seg = y[on:on + n]
    if len(seg) < n:
        seg = np.pad(seg, (0, n - len(seg)))
    noise = float(np.percentile(_frame_rms(y, int(0.02 * sr), int(0.01 * sr)), 10))
    return seg, noise


def extract_features(x, sr=SAMPLE_RATE):
    """Feature vector (len(FEATURE_NAMES),) from a float mono clip at 16 kHz."""
    if sr != SAMPLE_RATE:
        raise ValueError(f"Expected {SAMPLE_RATE} Hz audio, got {sr}")
    seg, noise = onset_segment(x, sr)
    s = seg / (np.max(np.abs(seg)) + 1e-12)
    f, P = welch(s, sr, nperseg=512, noverlap=384)
    band = np.array([P[(f >= BAND_EDGES[i]) & (f < BAND_EDGES[i + 1])].sum() for i in range(len(BAND_EDGES) - 1)])
    share = np.log10(band / band.sum() + 1e-6)
    cen = float((P * f).sum() / P.sum())
    bw = float(np.sqrt((P * (f - cen) ** 2).sum() / P.sum()))
    flat = float(np.exp(np.mean(np.log(P + 1e-15))) / np.mean(P))
    lp = 10 * np.log10(P + 1e-15)
    tonal = float(np.sort(lp)[::-1][:3].mean() - np.median(lp))

    def level(a, b):
        return 20 * np.log10(np.sqrt(np.mean(seg[int(a * sr):int(b * sr)] ** 2)) + 1e-9)
    d1 = level(0, 0.01) - level(0.03, 0.05)
    d2 = level(0, 0.01) - level(0.08, 0.12)
    env_db = 20 * np.log10(_frame_rms(seg, 128, 64) + 1e-9)
    k0 = int(np.argmax(env_db))
    lim = max(env_db[k0] - 15, 20 * np.log10(2 * noise + 1e-9))
    below = np.where(env_db[k0:] < lim)[0]
    t15 = (below[0] if len(below) else len(env_db) - k0) * 64 / sr
    zcr = float(np.mean(np.abs(np.diff(np.sign(s[:int(0.05 * sr)]))) > 0))
    feats = np.concatenate([share, [cen, bw, flat, tonal, d1, d2, t15, zcr]]).astype(np.float64)
    assert feats.shape[0] == len(FEATURE_NAMES)
    return feats
