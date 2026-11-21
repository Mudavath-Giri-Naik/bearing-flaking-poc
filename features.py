"""Shared code for the flaking PoC: CSV/docx loading, feature building, model wrapper, explanation.
Used by train_poc.py and app.py. The model input is ONLY the FRF CSV curve (450 points, 0-561.25 Hz)."""
import io
import re
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.signal import find_peaks
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

N_POINTS = 450
STEP = 1.25
FREQ = np.arange(N_POINTS) * STEP                      # 0 ... 561.25 Hz
ZONE_W = 10
ZONE_LO = np.arange(0, FREQ[-1] + 1e-9, ZONE_W)        # 0,10,...,560 -> 57 zones
ZONE_NAMES = [f"zone_{int(a)}-{int(a + ZONE_W)}Hz" for a in ZONE_LO]
ZONE_IDX = [np.where((FREQ >= a) & (FREQ < a + ZONE_W))[0] for a in ZONE_LO]
WINDOWS = [(18, 28), (75, 95), (150, 170), (280, 300), (425, 470)]
MAX_LAG = 8                                            # +-8 samples = +-10 Hz
FEATURE_SETS = ["full", "zones", "peaks", "zones+peaks", "full+peaks"]
MODEL_NAMES = ["LR", "SVM", "RF", "HGB"]


# ------------------------------------------------------------------ loading
def _read_text(source):
    if isinstance(source, (str, Path)):
        raw = Path(source).read_bytes()
    elif isinstance(source, (bytes, bytearray)):
        raw = bytes(source)
    else:  # file-like (e.g. Streamlit UploadedFile)
        if hasattr(source, "seek"):
            source.seek(0)
        raw = source.read()
        if isinstance(raw, str):
            raw = raw.encode("utf-8")
    try:
        return raw.decode("utf-8-sig")
    except UnicodeDecodeError:
        return raw.decode("latin-1")


def load_frf_csv(source):
    """Read a tap-test FRF CSV. Handles BOM, 2 columns, empty trailing column, 'SIG####(1)' headers.
    Raises ValueError with a clear message if the file does not match the 450-point grid."""
    text = _read_text(source)
    if not text.strip():
        raise ValueError("The file is empty.")
    header = text.splitlines()[0]
    try:
        df = pd.read_csv(io.StringIO(text))
    except Exception as e:
        raise ValueError(f"Could not read the file as CSV: {e}")
    df = df.loc[:, [not str(c).startswith("Unnamed") for c in df.columns]]
    df = df.dropna(axis=1, how="all")
    if df.shape[1] < 2:
        raise ValueError(f"Expected 2 columns (frequency, magnitude), found {df.shape[1]}. Header: {header!r}")
    f = pd.to_numeric(df.iloc[:, 0], errors="coerce").to_numpy(float)
    y = pd.to_numeric(df.iloc[:, 1], errors="coerce").to_numpy(float)
    ok = ~(np.isnan(f) & np.isnan(y))
    f, y = f[ok], y[ok]
    if len(f) != N_POINTS:
        raise ValueError(f"Expected {N_POINTS} data rows (0 to 561.25 Hz, step 1.25), found {len(f)}.")
    if np.isnan(f).any() or not np.allclose(f, FREQ, atol=1e-3):
        raise ValueError(f"Frequency column does not match the expected grid 0-561.25 Hz in 1.25 Hz steps "
                         f"(file runs {np.nanmin(f):g} to {np.nanmax(f):g} Hz).")
    if not np.all(np.isfinite(y)) or (y <= 0).any():
        raise ValueError("Magnitude column has empty, non-numeric, zero or negative values.")
    m = re.search(r"SIG\s*(\d+)", header, re.I)
    return dict(freq=FREQ.copy(), mag=y, logmag=np.log10(y), sig=m.group(1) if m else "", header=header)


BNO_RE = re.compile(r"BEARING\s*NO[.:\-\s]*(\d+)", re.I)
MFD_RE = re.compile(r"MFD[.:\-\s]*(\d{1,2}\s*/\s*\d{2,4})", re.I)
SIG_RE = re.compile(r"SIG\s*(\d+)", re.I)


def read_docx_info(source):
    """Display-only info from the Word report (same rules as scripts/03_docx_reports.py)."""
    import docx
    if hasattr(source, "seek"):
        source.seek(0)
    d = docx.Document(source if not isinstance(source, Path) else str(source))
    paras = [x.text for x in d.paragraphs]
    full = "\n".join(paras)
    out = {}
    m = BNO_RE.search(full); out["bearing_no"] = m.group(1) if m else ""
    m = SIG_RE.search(full); out["sig"] = m.group(1) if m else ""
    m = MFD_RE.search(full); out["mfd"] = re.sub(r"\s", "", m.group(1)) if m else ""
    out["brand"] = "NBC" if re.search(r"\bNBC\b", full, re.I) else ("TIMKEN" if re.search(r"TIMKEN", full, re.I) else "")
    obs, on = [], False
    for t in paras:
        if re.match(r"\s*OBSERVATIONS?", t, re.I):
            on = True
            rest = re.sub(r"^\s*OBSERVATIONS?\s*[:\-]*\s*", "", t, flags=re.I)
            if rest.strip():
                obs.append(rest.strip())
            continue
        if on and re.search(r"Marker Values", t, re.I):
            break
        if on and t.strip():
            obs.append(t.strip())
    out["observation"] = " ".join(obs)
    return out


# ------------------------------------------------------------------ features
def _half_width(row, j):
    thr = row[j] - np.log10(2)          # linear magnitude >= peak/2
    above = row >= thr
    lo = j
    while lo > 0 and above[lo - 1]:
        lo -= 1
    hi = j
    while hi < len(row) - 1 and above[hi + 1]:
        hi += 1
    return (hi - lo + 1) * STEP


class FeatureBuilder:
    """fset in FEATURE_SETS. fit() only learns the median Good curve (training rows only)."""

    def __init__(self, fset):
        assert fset in FEATURE_SETS, fset
        self.fset = fset
        self.parts = fset.split("+")

    def fit(self, L, good_mask):
        good_mask = np.asarray(good_mask, bool)
        self.good_med_ = np.median(L[good_mask] if good_mask.any() else L, axis=0)
        self.names = self._names()
        return self

    def _peak_names(self):
        n = []
        for lo, hi in WINDOWS:
            n += [f"pk{lo}-{hi}Hz_{k}" for k in ("freq", "height", "width", "area", "shift", "diff_vs_good")]
        return n + ["n_peaks", "roughness"]

    def _names(self):
        n = []
        for p in self.parts:
            if p == "full":
                n += [f"f_{f:g}Hz" for f in FREQ]
            elif p == "zones":
                n += ZONE_NAMES
            elif p == "peaks":
                n += self._peak_names()
        return n

    def groups(self):
        """name -> column indices, for grouped permutation importance (full-curve columns grouped per 10 Hz zone)."""
        g, off = {}, 0
        for p in self.parts:
            if p == "full":
                for name, idx in zip(ZONE_NAMES, ZONE_IDX):
                    g.setdefault(name, []).extend((off + idx).tolist())
                off += N_POINTS
            elif p == "zones":
                for k, name in enumerate(ZONE_NAMES):
                    g.setdefault(name, []).append(off + k)
                off += len(ZONE_NAMES)
            elif p == "peaks":
                for k, name in enumerate(self._peak_names()):
                    g[name] = [off + k]
                off += len(self._peak_names())
        return g

    def _peaks(self, L):
        out = []
        for lo, hi in WINDOWS:
            w = np.where((FREQ >= lo) & (FREQ <= hi))[0]
            sub = L[:, w]
            j = sub.argmax(1)
            pf = FREQ[w][j]
            ph = sub.max(1)
            width = np.array([_half_width(L[i, w], j[i]) for i in range(len(L))])
            area = np.log10(np.trapezoid(10 ** sub, FREQ[w], axis=1))
            ref = self.good_med_[w]
            ref = ref - ref.mean()
            lags = np.arange(-MAX_LAG, MAX_LAG + 1)
            rms = np.empty((len(L), len(lags)))
            for k, lag in enumerate(lags):
                s = L[:, w + lag]
                s = s - s.mean(1, keepdims=True)
                rms[:, k] = np.sqrt(np.mean((s - ref) ** 2, 1))
            shift = lags[rms.argmin(1)] * STEP
            diff = (sub - self.good_med_[w]).mean(1)
            out += [pf, ph, width, area, shift, diff]
        npk = np.array([len(find_peaks(r, prominence=0.1)[0]) for r in L], float)
        rough = np.std(np.diff(L, 2, axis=1), axis=1)
        return np.column_stack(out + [npk, rough])

    def transform(self, L):
        L = np.atleast_2d(L)
        X = []
        for p in self.parts:
            if p == "full":
                X.append(L)
            elif p == "zones":
                X.append(np.column_stack([L[:, idx].mean(1) for idx in ZONE_IDX]))
            elif p == "peaks":
                X.append(self._peaks(L))
        return np.hstack(X)


def make_model(name):
    if name == "LR":
        return make_pipeline(StandardScaler(), LogisticRegression(max_iter=5000, class_weight="balanced"))
    if name == "SVM":
        return make_pipeline(StandardScaler(), SVC(kernel="rbf", probability=True, class_weight="balanced",
                                                   random_state=42))
    if name == "RF":
        return RandomForestClassifier(n_estimators=500, class_weight="balanced", n_jobs=-1, random_state=42)
    if name == "HGB":
        return HistGradientBoostingClassifier(class_weight="balanced", random_state=42)
    raise ValueError(name)


class FlakingModel:
    """FeatureBuilder + sklearn estimator. predict_proba(L) -> probability of the positive class (Flaking)."""

    def __init__(self, fset, model_name):
        self.fset, self.model_name = fset, model_name

    def fit(self, L, y, good_mask):
        self.fb = FeatureBuilder(self.fset).fit(L, good_mask)
        self.est = make_model(self.model_name).fit(self.fb.transform(L), np.asarray(y, int))
        return self

    def predict_proba(self, L):
        return self.est.predict_proba(self.fb.transform(np.atleast_2d(L)))[:, 1]


def explain_zones(model, logmag, ref_curve, top=5):
    """Occlusion: replace each 10 Hz zone of the curve by the brand's training median and see how the
    flaking probability changes. delta > 0 means this zone pushed the result TOWARDS flaking."""
    logmag = np.asarray(logmag, float)
    p0 = float(model.predict_proba(logmag)[0])
    batch = np.repeat(logmag[None, :], len(ZONE_IDX), 0)
    for k, idx in enumerate(ZONE_IDX):
        batch[k, idx] = ref_curve[idx]
    p = model.predict_proba(batch)
    df = pd.DataFrame(dict(zone=ZONE_NAMES, p_if_typical=p, delta=p0 - p))
    df["abs"] = df.delta.abs()
    return p0, df.sort_values("abs", ascending=False).head(top).drop(columns="abs").reset_index(drop=True)
