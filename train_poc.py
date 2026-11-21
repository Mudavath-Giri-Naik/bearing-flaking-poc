"""Flaking PoC: CV model comparison (grouped by bearing ID), comparison variants, final models, hold-out check.
Hold-out bearings are removed BEFORE any training or selection and only scored at the very end."""
import json
import sys
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import balanced_accuracy_score, confusion_matrix, roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE.parent / "scripts"), str(HERE)]
from common import ROOT, TABLES, CACHE  # noqa: E402
from features import (FREQ, FEATURE_SETS, MODEL_NAMES, FeatureBuilder, FlakingModel, make_model,  # noqa: E402
                      load_frf_csv)

OUT_T, OUT_P, OUT_M = (HERE / d for d in ("tables", "plots", "models"))
for d in (OUT_T, OUT_P, OUT_M):
    d.mkdir(exist_ok=True)
THR = (0.5, 0.35)
t0 = time.time()
log = lambda *a: print(f"[{time.time() - t0:6.0f}s]", *a, flush=True)

# ------------------------------------------------------------------ data
z = np.load(CACHE / "frf_matrix.npz", allow_pickle=True)
assert np.allclose(z["freq"], FREQ)
L = z["logmag"].astype(float)
paths = np.array([str(p) for p in z["paths"]])
brand, cls, labels, bid = (np.array([str(x) for x in z[k]]) for k in ("brand", "cls", "labels", "bearing_id"))
norm = lambda p: str(p).replace("/", "\\").lower()

excl = set(pd.read_csv(TABLES / "excluded_from_holdout.csv", dtype=str).bearing_id)
dup_paths = set()
for ps in pd.read_csv(TABLES / "duplicates_by_hash.csv", dtype=str).paths.dropna():
    dup_paths |= {norm(p) for p in ps.split(" || ")}
H = pd.read_csv(TABLES / "holdout_demo_set.csv", dtype=str).fillna("")
hold_ids = set(H.bearing_id)

steps, keep = [], np.ones(len(L), bool)
def step(name, m):
    global keep
    keep &= m
    steps.append(dict(step=name, **{f"{b}_{c}": int((keep & (brand == b) & (cls == c)).sum())
                                    for b in ("NBC", "TIMKEN") for c in ("Flaking", "Other", "Good")},
                      total=int(keep.sum())))
step("all clean files in cache", np.ones(len(L), bool))
step("drop excluded bearing IDs", ~np.isin(bid, list(excl)))
step("drop duplicate-hash files", ~np.array([norm(p) in dup_paths for p in paths]))
step("drop Word lock (~$) paths", ~np.array(["~$" in p for p in paths]))
step("drop hold-out bearing IDs", ~np.isin(bid, list(hold_ids)))
pd.DataFrame(steps).to_csv(OUT_T / "data_used.csv", index=False)
log("data steps:\n" + pd.DataFrame(steps).to_string())
groups = np.where(bid != "", bid, paths)
assert not (set(bid[keep]) & hold_ids)

VARIANTS = {
    "main (Flaking vs Not flaking)": (np.ones(len(L), bool), cls == "Flaking"),
    "flaking_only labels vs Not flaking": ((cls != "Flaking") | (np.char.upper(labels) == "FLAKING"), cls == "Flaking"),
    "Flaking vs Good": (np.isin(cls, ["Flaking", "Good"]), cls == "Flaking"),
    "Damaged vs Good": (np.ones(len(L), bool), cls != "Good"),
}
MAIN = "main (Flaking vs Not flaking)"


def metrics(y, p, g):
    r = dict(n=len(y), n_pos=int(y.sum()), auc=roc_auc_score(y, p))
    for t in THR:
        pr = (p >= t).astype(int)
        r[f"bal_acc@{t}"] = balanced_accuracy_score(y, pr)
        r[f"pos_recall@{t}"] = pr[y == 1].mean()
        r[f"neg_recall@{t}"] = 1 - pr[y == 0].mean()
        r[f"missed_pos_bearings@{t}"] = len(set(g[(y == 1) & (pr == 0)]))
    return r


# ------------------------------------------------------------------ CV
rows, oof_store, split_store = [], {}, {}
for vname, (vmask, ypos) in VARIANTS.items():
    for b in ("NBC", "TIMKEN"):
        idx = np.where(keep & vmask & (brand == b))[0]
        Lb, yb, gb, goodb = L[idx], ypos[idx].astype(int), groups[idx], cls[idx] == "Good"
        splits = list(StratifiedGroupKFold(5, shuffle=True, random_state=42).split(Lb, yb, gb))
        split_store[(vname, b)] = (idx, splits)
        oof = {(f, m): np.full(len(idx), np.nan) for f in FEATURE_SETS for m in MODEL_NAMES}
        for tr, te in splits:
            for f in FEATURE_SETS:
                fb = FeatureBuilder(f).fit(Lb[tr], goodb[tr])
                Xtr, Xte = fb.transform(Lb[tr]), fb.transform(Lb[te])
                for m in MODEL_NAMES:
                    oof[(f, m)][te] = make_model(m).fit(Xtr, yb[tr]).predict_proba(Xte)[:, 1]
        for (f, m), p in oof.items():
            rows.append(dict(variant=vname, brand=b, features=f, model=m, **metrics(yb, p, gb)))
            oof_store[(vname, b, f, m)] = p
        log(f"done {vname} / {b} (n={len(idx)}, pos={yb.sum()})")

R = pd.DataFrame(rows)
R.to_csv(OUT_T / "cv_results_all.csv", index=False)
best = (R.sort_values(["bal_acc@0.5", "auc"], ascending=False).groupby(["variant", "brand"], sort=False).head(1)
        .sort_values(["variant", "brand"]))
best.to_csv(OUT_T / "cv_best_per_variant_brand.csv", index=False)
log("best per variant/brand:\n" + best.to_string())

# ------------------------------------------------------------------ main: OOF, confusion, importance
summary = dict(best={}, holdout={})
for b in ("NBC", "TIMKEN"):
    br = best[(best.variant == MAIN) & (best.brand == b)].iloc[0]
    f, m = br.features, br.model
    idx, splits = split_store[(MAIN, b)]
    yb = (cls[idx] == "Flaking").astype(int)
    p = oof_store[(MAIN, b, f, m)]
    pd.DataFrame(dict(path=paths[idx], bearing_id=bid[idx], cls=cls[idx], labels=labels[idx], y=yb, p_flaking=p)) \
        .to_csv(OUT_T / f"oof_{b}.csv", index=False)
    cms = {}
    fig, axes = plt.subplots(1, 2, figsize=(9, 4))
    for ax, t in zip(axes, THR):
        cm = confusion_matrix(yb, (p >= t).astype(int), labels=[1, 0])
        cms[str(t)] = cm.tolist()
        ax.imshow(cm, cmap="Blues")
        for i in range(2):
            for j in range(2):
                ax.text(j, i, cm[i, j], ha="center", va="center", fontsize=14,
                        color="white" if cm[i, j] > cm.max() / 2 else "black")
        ax.set_xticks([0, 1], ["Flaking", "Not flaking"]); ax.set_yticks([0, 1], ["Flaking", "Not flaking"])
        ax.set_xlabel("predicted"); ax.set_ylabel("true"); ax.set_title(f"threshold {t}")
    fig.suptitle(f"{b}: CV confusion matrix ({m} on {f})")
    fig.tight_layout(); fig.savefig(OUT_P / f"confusion_{b}.png", dpi=130); plt.close(fig)

    # grouped permutation importance on each CV test fold (AUC drop)
    rng = np.random.default_rng(42)
    Lb, gb = L[idx], cls[idx] == "Good"
    imp = {}
    for tr, te in splits:
        fb = FeatureBuilder(f).fit(Lb[tr], gb[tr])
        est = make_model(m).fit(fb.transform(Lb[tr]), yb[tr])
        Xte = fb.transform(Lb[te])
        base = roc_auc_score(yb[te], est.predict_proba(Xte)[:, 1])
        for name, cols in fb.groups().items():
            drops = []
            for _ in range(10):
                Xp = Xte.copy()
                Xp[:, cols] = Xte[rng.permutation(len(te))][:, cols]
                drops.append(base - roc_auc_score(yb[te], est.predict_proba(Xp)[:, 1]))
            imp.setdefault(name, []).append(np.mean(drops))
    I = pd.DataFrame([dict(feature=k, auc_drop_mean=np.mean(v), auc_drop_std=np.std(v)) for k, v in imp.items()]) \
        .sort_values("auc_drop_mean", ascending=False)
    I.to_csv(OUT_T / f"importance_{b}.csv", index=False)
    log(f"{b} importance top:\n" + I.head(8).to_string())

    # ---------------- final model on all of the brand's training data
    model = FlakingModel(f, m).fit(Lb, yb, gb)
    Fl, Nf = Lb[yb == 1], Lb[yb == 0]
    bundle = dict(model=model, freq=FREQ, brand=b, config=dict(features=f, model=m),
                  cv=br.to_dict(), confusion=cms,
                  med_flaking=np.median(Fl, 0), p10_flaking=np.percentile(Fl, 10, 0), p90_flaking=np.percentile(Fl, 90, 0),
                  med_not=np.median(Nf, 0), p10_not=np.percentile(Nf, 10, 0), p90_not=np.percentile(Nf, 90, 0),
                  med_all=np.median(Lb, 0), train_L=Lb, train_ids=bid[idx], train_labels=labels[idx],
                  train_cls=cls[idx], feature_names=model.fb.names,
                  importance=I.head(15).to_dict("records"),
                  thresholds=dict(decision=0.5, unsure_low=0.35, unsure_high=0.65))
    joblib.dump(bundle, OUT_M / f"{b}.joblib")
    summary["best"][b] = dict(features=f, model=m, **{k: (float(v) if isinstance(v, (int, float, np.number)) else v)
                                                       for k, v in br.items() if k not in ("variant", "brand", "features", "model")},
                              confusion=cms, n_train_bearings=int(len(set(groups[idx]))))

# ------------------------------------------------------------------ hold-out (scored once, at the end)
hres = []
for _, r in H.iterrows():
    bundle = joblib.load(OUT_M / f"{r.brand}.joblib")
    assert r.bearing_id not in set(bundle["train_ids"])
    try:
        c = load_frf_csv(r.csv_path)
        p = float(bundle["model"].predict_proba(c["logmag"])[0])
        err = ""
    except Exception as e:
        p, err = np.nan, repr(e)
    true = "FLAKING" if r.cls == "Flaking" else "NOT FLAKING"
    pred = "" if np.isnan(p) else ("FLAKING" if p >= .5 else "NOT FLAKING")
    hres.append(dict(brand=r.brand, bearing_id=r.bearing_id, folder_class=r.cls, folder_labels=r.labels,
                     true_label=true, predicted_label=pred, p_flaking=p,
                     confidence_pct=None if np.isnan(p) else round(100 * max(p, 1 - p), 1),
                     pred_at_0_35="" if np.isnan(p) else ("FLAKING" if p >= .35 else "NOT FLAKING"),
                     unsure=bool(.35 <= p <= .65) if not np.isnan(p) else None,
                     correct=pred == true, error=err, csv_path=r.csv_path, docx_path=r.docx_path))
HR = pd.DataFrame(hres)
HR.to_csv(OUT_T / "holdout_results.csv", index=False)
for b in ("NBC", "TIMKEN"):
    s = HR[HR.brand == b]
    yt, yp = (s.true_label == "FLAKING").astype(int), (s.p_flaking >= .5).astype(int)
    summary["holdout"][b] = dict(n=len(s), correct=int(s.correct.sum()),
                                 bal_acc=float(balanced_accuracy_score(yt, yp)),
                                 flaking_found=int(((yt == 1) & (yp == 1)).sum()), n_flaking=int(yt.sum()),
                                 flaking_found_at_0_35=int(((yt == 1) & (s.p_flaking >= .35)).sum()),
                                 not_flaking_correct=int(((yt == 0) & (yp == 0)).sum()), n_not=int((yt == 0).sum()),
                                 n_unsure=int(s.unsure.sum()))
json.dump(summary, open(OUT_T / "summary.json", "w"), indent=2, default=str)
log("hold-out:\n" + HR[["brand", "bearing_id", "folder_class", "predicted_label", "p_flaking", "unsure"]].to_string())
log(json.dumps(summary["holdout"], indent=1))
log("DONE")
