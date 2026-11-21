"""Flaking PoC demo app.  Run:  streamlit run app.py
Model input is ONLY the uploaded FRF CSV curve. The Word report is shown for information and never used."""
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

from features import FREQ, explain_zones, load_frf_csv, read_docx_info

st.set_page_config(page_title="Bearing flaking check (PoC)", layout="wide")


@st.cache_resource
def load_bundle(brand):
    return joblib.load(HERE / "models" / f"{brand}.joblib")


# ------------------------------------------------------------------ sidebar
st.sidebar.title("Inputs")
brand = st.sidebar.selectbox("Bearing brand", ["NBC", "TIMKEN"])
try:
    B = load_bundle(brand)
except FileNotFoundError:
    st.error(f"No model file models/{brand}.joblib. Run `python train_poc.py` first.")
    st.stop()

csv_file = st.sidebar.file_uploader("FRF CSV (tap test, 0-561.25 Hz)", type=["csv"])
docx_file = st.sidebar.file_uploader("Word report (optional, display only)", type=["docx"])

cv = B["cv"]
st.sidebar.markdown("---")
st.sidebar.markdown(
    f"**Model:** {B['config']['model']} on `{B['config']['features']}` features  \n"
    f"**CV balanced accuracy:** {cv['bal_acc@0.5']:.2f}  \n"
    f"**CV flaking recall:** {cv['pos_recall@0.5']:.2f}  \n"
    f"**Trained on:** {len(set(B['train_ids']))} bearings"
)

info = {}
if docx_file is not None:
    st.sidebar.markdown("---")
    st.sidebar.subheader("Word report info (not used by the model)")
    try:
        info = read_docx_info(docx_file)
        st.sidebar.markdown(
            f"**Bearing no.:** {info['bearing_no'] or '-'}  \n"
            f"**SIG:** {info['sig'] or '-'}  \n"
            f"**MFD:** {info['mfd'] or '-'}  \n"
            f"**Brand in report:** {info['brand'] or '-'}"
        )
        st.sidebar.markdown(f"**Observation:** {info['observation'] or '-'}")
        if info["brand"] and info["brand"] != brand:
            st.sidebar.warning(f"Report says {info['brand']}, but the {brand} model is selected.")
    except Exception as e:
        st.sidebar.error(f"Could not read the Word file: {e}")

# ------------------------------------------------------------------ main
st.title("Bearing flaking check - proof of concept")
st.caption("Looks only at the tap-test FRF curve. This is a PoC trained on a small data set: "
           "treat the result as a hint, not a decision.")

if csv_file is None:
    st.info("Upload an FRF CSV in the sidebar to start.")
    st.stop()

try:
    d = load_frf_csv(csv_file)
except ValueError as e:
    st.error(f"This file cannot be used: {e}")
    st.stop()

if docx_file is not None and d["sig"] and info.get("sig") and d["sig"].lstrip("0") != info["sig"].lstrip("0"):
    st.warning(f"SIG in CSV ({d['sig']}) differs from SIG in the Word report ({info['sig']}).")

model = B["model"]
p = float(model.predict_proba(d["logmag"])[0])
th = B["thresholds"]
lo, hi = th["unsure_low"], th["unsure_high"]

if lo <= p <= hi:
    verdict, colour = "UNSURE - please check manually", "#e9a03b"
elif p > hi:
    verdict, colour = "FLAKING", "#e63946"
else:
    verdict, colour = "NOT FLAKING", "#2a9d8f"
conf = max(p, 1 - p) * 100
lean = "flaking" if p >= th["decision"] else "not flaking"

st.markdown(
    f"<div style='padding:18px;border-radius:10px;background:{colour};color:white;text-align:center'>"
    f"<div style='font-size:42px;font-weight:700'>{verdict}</div>"
    f"<div style='font-size:20px'>Probability of flaking: {p * 100:.0f}% &nbsp;|&nbsp; "
    f"confidence {conf:.0f}% (leans {lean})</div></div>",
    unsafe_allow_html=True,
)
st.write("")
st.caption(f"File SIG: {d['sig'] or '-'}  |  header: {d['header']!r}  |  "
           f"UNSURE band = {lo * 100:.0f}-{hi * 100:.0f}%")

c1, c2 = st.columns([3, 2])

with c1:
    st.subheader("Your curve vs the training curves")
    fig, ax = plt.subplots(figsize=(9, 4.5))
    for key, lab, col in (("not", "Not flaking", "#457b9d"), ("flaking", "Flaking", "#e63946")):
        ax.fill_between(FREQ, 10 ** B[f"p10_{key}"], 10 ** B[f"p90_{key}"], color=col, alpha=0.15)
        ax.plot(FREQ, 10 ** B[f"med_{key}"], color=col, lw=1.5, label=f"{lab} median (10-90% band)")
    ax.plot(FREQ, d["mag"], color="black", lw=1.2, label="Uploaded curve")
    ax.set_yscale("log")
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("|H| (g/lbf)")
    ax.set_xlim(0, FREQ[-1])
    ax.grid(alpha=0.3, which="both")
    ax.legend(fontsize=8)
    ax.set_title(f"{brand}")
    st.pyplot(fig)
    plt.close(fig)

with c2:
    st.subheader("What drove this result")
    p0, z = explain_zones(model, d["logmag"], B["med_all"], top=5)
    z = z.assign(effect=np.where(z.delta > 0, "towards FLAKING", "towards NOT flaking"),
                 change_pct=(z.delta * 100).round(1),
                 p_if_typical_pct=(z.p_if_typical * 100).round(0))
    st.caption("Each 10 Hz zone is swapped for a typical curve; the change in flaking probability "
               "shows how much that zone mattered for THIS file.")
    st.dataframe(z[["zone", "effect", "change_pct", "p_if_typical_pct"]]
                 .rename(columns={"change_pct": "effect (% points)",
                                  "p_if_typical_pct": "prob. if zone typical (%)"}),
                 hide_index=True, use_container_width=True)
    with st.expander("Most important features for this model overall (training CV)"):
        imp = pd.DataFrame(B["importance"]).head(10)
        st.dataframe(imp.round(4), hide_index=True, use_container_width=True)

st.subheader("3 most similar training bearings")
TL = np.asarray(B["train_L"])
rms = np.sqrt(np.mean((TL - d["logmag"][None, :]) ** 2, axis=1))
order = np.argsort(rms)
rows, seen = [], set()
for i in order:
    bid = B["train_ids"][i]
    if bid in seen:
        continue
    seen.add(bid)
    rows.append(dict(bearing_id=bid, label=B["train_labels"][i], folder_class=B["train_cls"][i],
                     rms_log_difference=round(float(rms[i]), 4)))
    if len(rows) == 3:
        break
st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)
st.caption("Distance = RMS difference of log10 magnitude over the whole 0-561 Hz curve.")
