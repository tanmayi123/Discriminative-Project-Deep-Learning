"""
Milestone 2 extension: the choice between the two improved models, made ON VALIDATION by the rule fixed in advance
(M2 extra/boost2/selection_rule.txt), plus separate confidence thresholds for grids and scenes, also chosen on validation.

    python m2_boost_val.py        # run from inside M2 extra AFTER  "M2_BOOST_DIR=boost2 python m2_boost.py evaluate"  finished

Steps (in this order, each written to disk before the next starts):
  1. checks: both models trained, boost2 evaluated, selection_rule.txt still matches its recorded SHA-256
  2. both models on the boost2 validation set (292 images, pass-2 labels), each at its own imgsz, no TTA, no thresholds:
     mAP@0.5:0.95 decides (the rule); a paired bootstrap over images gives the uncertainty of the difference (context only)
  3. for the selected model, on the same validation set: one threshold for everything (as m2_boost.py evaluate) vs one
     for grids + one for scenes; the pair is adopted only if F1 rises by more than 0.003 (the margin evaluate uses for TTA)
     AND overall recall does not fall below the single threshold's recall (recall guard, decided 2026-10-13 before any result);
     the guard is part of the search: the best combined F1 among the settings that keep recall.
     The per-kind search is finer than evaluate's (0.05 to 0.60 in steps of 0.05; evaluate stops at 0.40) and, for scenes
     only, also tries class-agnostic duplicate removal (one box kept where boxes of any class overlap above IoU 0.5 / 0.6 / 0.7),
     the scene counterpart of the one-box-per-cell rule on grids. All of it is chosen on validation.
  4. the decision is written to  M2 extra/selection/selection.json  and  M2 extra/selected_model.json  (read by The EYE v3)
  5. only if two thresholds were adopted: one look at each test set with them (single-threshold test numbers are already
     in boost2/evaluation.md and are not recomputed)

Reads only. Writes only to  M2 extra/selection/  and  M2 extra/selected_model.json . Uses the exact scoring functions of
m2_boost.py (imported, never modified), so every number is computed the same way as in evaluation.md.
"""
import hashlib, json, os, sys, time
from pathlib import Path
import numpy as np

os.environ.setdefault("M2_BOOST_DIR", "boost2")
if os.environ["M2_BOOST_DIR"] == "boost": sys.exit("M2_BOOST_DIR must name the SECOND model's folder (boost2), not boost.")
import m2_boost as mb                                     # module-level code only creates folders that already exist

EXTRA, B1 = mb.EXTRA, mb.B1; B2 = EXTRA / os.environ["M2_BOOST_DIR"]   # not mb.B: independent of import order
SEL = EXTRA / "selection"; PRED = SEL / "predictions"
CANDIDATES = (.05, .10, .15, .20, .25, .30, .40)          # the same list as m2_boost.py evaluate (the single-threshold baseline)
FINE = tuple(round(.05 * k, 2) for k in range(1, 13))     # 0.05 ... 0.60 for the per-kind search
SCENE_NMS = (None, .7, .6, .5)                           # None first: on a tie, no extra post-processing is preferred
MARGIN = .003                                             # F1 gain needed to adopt two thresholds
N_BOOT = int(os.environ.get("M2_NBOOT", 1000))
GRID_TESTS, SCENE_TESTS = ("normal", "rotated", "singles"), ("scattered", "cutout")
log = mb.log
f1 = lambda r, p: 2 * r * p / max(1e-9, r + p)

def sha256(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest().upper()

def checks():
    for f, why in ((B1 / "train.json", "the first improved model (boost) is not trained"),
                   (B2 / "train.json", f"{B2.name} is not trained yet"),
                   (B2 / "evaluation.json", f"{B2.name} is not evaluated yet: run this AFTER  m2_boost.py evaluate  has finished"),
                   (B2 / "tests" / "val_all" / "truth.json", f"{B2.name} validation set missing (m2_boost.py build)"),
                   (B2 / "selection_rule.txt", "selection_rule.txt missing")):
        if not f.exists(): sys.exit(f"STOP: {why} ({f})")
    rule_hash, rec = sha256(B2 / "selection_rule.txt"), None
    hf = B2 / "selection_rule_sha256.txt"
    if hf.exists():
        rec = next((w for w in hf.read_text().split() if len(w) == 64), "").upper()
        if rec != rule_hash: sys.exit(f"STOP: selection_rule.txt was changed after its hash was recorded ({rec} recorded, {rule_hash} now).")
    else: log("WARNING: no recorded hash for selection_rule.txt; continuing, the current hash is written to selection.json")
    return rule_hash, rec is not None

# ---------------------------------------------------------------------------------------------- predictions (cached)
def model_info(folder):
    tj = json.loads((folder / "train.json").read_text()); ej = folder / "evaluation.json"
    ev = json.loads(ej.read_text()) if ej.exists() else {}; ch = ev.get("chosen_on_validation") or {}
    return dict(name="yolo26s_" + folder.name, dir=folder.name, weights=tj["weights"], imgsz=int(tj["imgsz"]), batch=tj.get("batch"),
                fill=ch.get("fill_empty_cells"), tta=bool(ch.get("tta", False)), eval_threshold=ch.get("threshold"))

def predictions(info, test_dir, tag, tta=False):
    """Raw detections (conf >= 0.001) of one model on one set, cached as JSON so nothing is run twice."""
    PRED.mkdir(parents=True, exist_ok=True); f = PRED / f"{tag}_{info['dir']}{'_tta' if tta else ''}.json"
    truth = json.loads((test_dir / "truth.json").read_text())
    for t in truth: t["boxes"] = [(o, tuple(b)) for o, b in t["boxes"]]; t["ignore"] = [tuple(b) for b in t.get("ignore", [])]
    if f.exists():
        c = json.loads(f.read_text())
        if c.get("weights") == info["weights"] and c.get("imgsz") == info["imgsz"] and set(c["preds"]) == {t["name"] for t in truth}:
            return {k: [(o, s, tuple(b)) for o, s, b in v] for k, v in c["preds"].items()}, truth
    det = mb.Detector(info["weights"], info["imgsz"], tta=tta); t0 = time.time(); preds = {}
    for t in truth:
        img = mb.Image.open(test_dir / "images" / t["file"]).convert("RGB"); preds[t["name"]] = det(img, t["boxes"])
    f.write_text(json.dumps(dict(weights=info["weights"], imgsz=info["imgsz"], tta=tta, preds=preds), default=mb._np))
    log(f"  {info['name']} on {test_dir.parent.parent.name}/{test_dir.name}: {len(truth)} images ({time.time() - t0:.0f} s)")
    del det
    try:
        import torch; torch.cuda.empty_cache()
    except Exception: pass
    return preds, truth

# ---------------------------------------------------------------------------------------------- mAP, fast and resamplable
THRS = np.round(np.arange(.5, .96, .05), 2); T101 = np.linspace(0, 1, 101)
def ap_tables(preds, truth):
    """For every IoU threshold and class: each prediction's image, confidence and hit flag, matched EXACTLY as
    m2_boost.mean_ap matches (greedy by confidence, per image), plus the true-object count per image."""
    names = [t["name"] for t in truth]; idx = {n: i for i, n in enumerate(names)}; G = {t["name"]: t["boxes"] for t in truth}
    classes = sorted({o for t in truth for o, _ in t["boxes"]}); tables = {}
    ngt = {c: np.array([sum(1 for o, _ in t["boxes"] if o == c) for t in truth], float) for c in classes}
    for thr in THRS:
        for c in classes:
            P = sorted(((s, t["name"], b) for t in truth for o, s, b in preds[t["name"]] if o == c), key=lambda x: -x[0]); used, rows = set(), []
            for s, n, b in P:
                bj = max((j for j, (o, g) in enumerate(G[n]) if o == c and (n, j) not in used), key=lambda j: mb.iou(b, G[n][j][1]), default=None)
                hit = bj is not None and mb.iou(b, G[n][bj][1]) >= thr
                if hit: used.add((n, bj))
                rows.append((idx[n], s, int(hit)))
            a = np.array(rows, float).reshape(-1, 3); o = np.argsort(-a[:, 1]) if len(a) else np.array([], int)   # the same sort as ap_101
            tables[(thr, c)] = (a[o, 0].astype(int), a[o, 2])
    return tables, ngt, classes

def fast_map(tables, ngt, classes, w):
    """mAP@0.5 and mAP@0.5:0.95 with image weights w (all ones = the plain value; bootstrap counts = a resample)."""
    per_thr = []
    for thr in THRS:
        aps = []
        for c in classes:
            n = float(ngt[c] @ w)
            if n <= 0: continue                                # class absent from this resample (mean_ap would not list it)
            img, tp = tables[(thr, c)]
            if not len(tp): aps.append(0.0); continue
            wi = w[img]; ctp = np.cumsum(wi * tp); cw = np.cumsum(wi)
            rec = ctp / n; prec = np.maximum.accumulate((ctp / np.maximum(cw, 1e-9))[::-1])[::-1]
            k = np.searchsorted(rec, T101, side="left"); aps.append(float(np.where(k < len(rec), prec[np.minimum(k, len(rec) - 1)], 0.0).mean()))
        per_thr.append(float(np.mean(aps)) if aps else 0.0)
    return per_thr[0], float(np.mean(per_thr))

# ---------------------------------------------------------------------------------------------- thresholds
def counts(preds, truth, conf, fill):
    """Exact counts from m2_boost.task (true hits, wrong IDs, misses) and m2_boost.post (kept boxes)."""
    if not truth: return dict(ok=0, n_gt=0, n_pred=0, wrong=0)
    r = mb.task(preds, truth, conf, fill=fill); ok = r["objects"] - r["missed"] - r["wrong_ids"]
    n_pred = sum(len(mb.post(preds[t["name"]], t, conf, True, fill)) for t in truth)
    return dict(ok=ok, n_gt=r["objects"], n_pred=n_pred, wrong=r["wrong_ids"])
def rates(*cs):
    ok, g, p = (sum(c[k] for c in cs) for k in ("ok", "n_gt", "n_pred")); r, q = ok / max(g, 1), ok / max(p, 1)
    return dict(recall=round(r, 4), precision=round(q, 4), f1=round(f1(r, q), 4), wrong_ids=sum(c["wrong"] for c in cs), objects=g)

def agnostic_nms(P, iou_t, floor=.05):
    """Scenes only: among boxes of ANY class that overlap above iou_t, keep the most confident (YOLO's own NMS is per class).
    Applied before the threshold; identical to applying it after, because a box is only removed by a MORE confident one."""
    keep = []
    for q in sorted((q for q in P if q[1] >= floor), key=lambda x: -x[1]):
        if all(mb.iou(q[2], k[2]) < iou_t for k in keep): keep.append(q)
    return keep
def scene_preds(preds, scene, iou_t):
    return preds if iou_t is None else {**preds, **{t["name"]: agnostic_nms(preds[t["name"]], iou_t) for t in scene}}

def thresholds(preds, truth, fill):
    grid = [t for t in truth if t["grid"]]; scene = [t for t in truth if not t["grid"]]
    one = {c: rates(counts(preds, grid, c, fill), counts(preds, scene, c, fill)) for c in CANDIDATES}
    # the single threshold exactly as evaluate chooses it (F1 of task on the whole set, fill not applied while choosing)
    t_one = max(CANDIDATES, key=lambda c: (lambda m: f1(m["recall"], m["precision"]))(mb.task(preds, truth, c)))
    g = {c: counts(preds, grid, c, fill) for c in FINE}
    s = {(n, c): counts(sp, scene, c, fill) for n in SCENE_NMS for sp in [scene_preds(preds, scene, n)] for c in FINE}
    single = one[t_one]
    # joint search: the best combined F1 among settings that find at least as many objects as the single threshold
    # (iteration order = tie-break: lower grid threshold, no scene duplicate removal, lower scene threshold)
    combos = [(cg, k, rates(g[cg], s[k])) for cg in FINE for k in s]
    ok = [x for x in combos if x[2]["recall"] >= single["recall"]]
    t_grid, (n_scene, t_scene), pair = max(ok, key=lambda x: x[2]["f1"]) if ok else (t_one, (None, t_one), single)
    adopt = bool(ok) and pair["f1"] > single["f1"] + MARGIN   # better overall AND finds at least as many objects
    return dict(single=dict(threshold=t_one, **single), grid_threshold=t_grid, scene_threshold=t_scene, scene_nms=n_scene, pair=pair, adopted=adopt,
                grid_only={str(c): rates(g[c]) for c in FINE}, scene_only={f"{n}|{c}": rates(v) for (n, c), v in s.items()},
                images=dict(grid=len(grid), scene=len(scene)))

# ---------------------------------------------------------------------------------------------- main
def main():
    rule_hash, hash_checked = checks(); SEL.mkdir(parents=True, exist_ok=True)
    M = {k: model_info(f) for k, f in (("boost", B1), (B2.name, B2))}
    log(f"rule: {B2 / 'selection_rule.txt'} (SHA-256 {rule_hash[:8]}...{rule_hash[-4:]}, {'matches the recorded hash' if hash_checked else 'no recorded hash'})")
    # ---- step 2: the rule, on the boost2 validation set
    vdir = B2 / "tests" / "val_all"; val = {}
    for k, info in M.items():
        p, t = predictions(info, vdir, "val"); tab, ngt, cls = ap_tables(p, t)
        m50, m95 = fast_map(tab, ngt, cls, np.ones(len(t))); ref = mb.mean_ap(p, t)
        if abs(round(m50, 4) - ref[0]) > 1e-4 or abs(round(m95, 4) - ref[1]) > 1e-4:
            sys.exit(f"STOP: internal check failed, fast mAP {m50:.4f}/{m95:.4f} differs from m2_boost.mean_ap {ref} for {k}")
        val[k] = dict(preds=p, truth=t, tables=(tab, ngt, cls), mAP50=ref[0], mAP50_95=ref[1], mAP50_95_full=m95)
    a, b = val["boost"], val[B2.name]
    winner = B2.name if b["mAP50_95_full"] > a["mAP50_95_full"] else "boost"   # the rule: strictly higher, else boost stays
    rng = np.random.default_rng(0); n = len(a["truth"]); d = []
    for _ in range(N_BOOT):
        w = np.bincount(rng.integers(0, n, n), minlength=n).astype(float)
        d.append(fast_map(*b["tables"], w)[1] - fast_map(*a["tables"], w)[1])
    d = np.array(d); diff = b["mAP50_95_full"] - a["mAP50_95_full"]
    boot = dict(resamples=N_BOOT, diff=round(diff, 4), ci95=[round(float(np.percentile(d, 2.5)), 4), round(float(np.percentile(d, 97.5)), 4)],
                share_second_better=round(float((d > 0).mean()), 3))
    log(f"validation mAP@0.5:0.95: boost {a['mAP50_95']:.4f} | {B2.name} {b['mAP50_95']:.4f} -> selected: {winner}")
    log(f"  difference {diff:+.4f}, bootstrap 95% CI [{boot['ci95'][0]:+.4f}, {boot['ci95'][1]:+.4f}] (context only; the rule decides)")
    # ---- step 3: thresholds for the selected model, same validation set
    info = M[winner]; th = thresholds(val[winner]["preds"], val[winner]["truth"], info["fill"])
    if winner == B2.name and info["eval_threshold"] is not None and abs(th["single"]["threshold"] - info["eval_threshold"]) > 1e-9:
        log(f"WARNING: single threshold {th['single']['threshold']} differs from evaluate's {info['eval_threshold']} (should be equal)")
    t_grid = th["grid_threshold"] if th["adopted"] else th["single"]["threshold"]; t_scene = th["scene_threshold"] if th["adopted"] else t_grid
    s_nms = th["scene_nms"] if th["adopted"] else None
    log(f"thresholds: single {th['single']['threshold']} (F1 {th['single']['f1']:.4f}) | grids {th['grid_threshold']} + scenes {th['scene_threshold']} "
        f"{'' if th['scene_nms'] is None else ', scene duplicate removal IoU ' + str(th['scene_nms'])} (F1 {th['pair']['f1']:.4f}) -> {'grid/scene setting adopted' if th['adopted'] else 'single threshold kept'}"
        f" (F1 gain {th['pair']['f1'] - th['single']['f1']:+.4f}, recall {th['single']['recall']:.4f} -> {th['pair']['recall']:.4f})")
    # ---- step 4: write the decision BEFORE any test set is touched
    decision = dict(decided_at=time.strftime("%Y-%m-%d %H:%M:%S"), rule_file=str(B2 / "selection_rule.txt"), rule_sha256=rule_hash, rule_hash_checked=hash_checked,
                    validation=dict(set=str(vdir), images=n, objects=sum(len(t["boxes"]) for t in a["truth"]),
                                    models={k: dict(imgsz=M[k]["imgsz"], batch=M[k]["batch"], mAP50=val[k]["mAP50"], mAP50_95=val[k]["mAP50_95"]) for k in M},
                                    bootstrap=boot),
                    selected=winner, thresholds=th, post_processing=dict(fill_empty_cells=info["fill"], tta=info["tta"]))
    (SEL / "selection.json").write_text(json.dumps(decision, indent=2, default=mb._np))
    sm = dict(name=info["name"], dir=info["dir"], weights=info["weights"], imgsz=info["imgsz"], threshold_grid=t_grid, threshold_scene=t_scene, scene_nms=s_nms,
              fill_empty_cells=info["fill"], tta=info["tta"], chosen_on="validation", selection=str(SEL / "selection.json"))
    (EXTRA / "selected_model.json").write_text(json.dumps(sm, indent=2, default=mb._np))
    # ---- step 5: test sets, only for the adopted two-threshold setting
    tests = {}
    if th["adopted"]:
        for name in GRID_TESTS + SCENE_TESTS:
            d_ = B2 / "tests" / name
            if not (d_ / "truth.json").exists(): continue
            p, t = predictions(info, d_, f"test_{name}", tta=info["tta"]); conf = t_grid if name in GRID_TESTS else t_scene
            if name in SCENE_TESTS: p = scene_preds(p, t, s_nms)
            r = mb.task(p, t, conf, fill=info["fill"]); tests[name] = dict(threshold=conf, scene_nms=s_nms if name in SCENE_TESTS else None, **r)
        decision["tests_two_thresholds"] = tests; (SEL / "selection.json").write_text(json.dumps(decision, indent=2, default=mb._np))
    report(decision, M, tests); log(f"written: {SEL / 'selection.json'}, {SEL / 'selection.md'}, {EXTRA / 'selected_model.json'}")

def report(D, M, tests):
    v = D["validation"]; bt = v["bootstrap"]; th = D["thresholds"]; f = lambda x: "-" if x is None else f"{x:.3f}"
    L = ["# Model selection on validation", "", f"Rule (fixed before any {B2.name} result, SHA-256 {D['rule_sha256']}"
         f"{', matches the recorded hash' if D['rule_hash_checked'] else ', no recorded hash to compare'}): the higher mAP@0.5:0.95 on the "
         f"{B2.name} validation set wins; otherwise boost stays.", "", f"Validation set: {v['images']} images, {v['objects']} objects (pass-2 labels).", "",
         "| model | imgsz | batch | mAP@0.5 | mAP@0.5:0.95 |", "|---|---|---|---|---|"]
    for k, m in v["models"].items(): L.append(f"| {M[k]['name']} | {m['imgsz']} | {m['batch']} | {f(m['mAP50'])} | {'**' + f(m['mAP50_95']) + '**' if k == D['selected'] else f(m['mAP50_95'])} |")
    L += ["", f"**Selected: {M[D['selected']]['name']}.** Difference ({B2.name} minus boost) {bt['diff']:+.4f}; paired bootstrap over images "
          f"({bt['resamples']} resamples): 95% interval [{bt['ci95'][0]:+.4f}, {bt['ci95'][1]:+.4f}], {B2.name} higher in {bt['share_second_better']:.0%} of resamples. "
          "The interval is context only; the rule decides.", "", "## Thresholds (selected model, same validation set)", "",
          "| setting | threshold(s) | recall | precision | F1 | wrong IDs |", "|---|---|---|---|---|---|",
          f"| one threshold (as evaluate) | {th['single']['threshold']} | {f(th['single']['recall'])} | {f(th['single']['precision'])} | {f(th['single']['f1'])} | {th['single']['wrong_ids']} |",
          f"| grids + scenes{'' if th['scene_nms'] is None else ' (scene duplicate removal IoU ' + str(th['scene_nms']) + ')'} | {th['grid_threshold']} + {th['scene_threshold']} | {f(th['pair']['recall'])} | {f(th['pair']['precision'])} | {f(th['pair']['f1'])} | {th['pair']['wrong_ids']} |", "",
          f"Grid/scene setting {'**adopted**' if th['adopted'] else 'not adopted'} (rule: F1 gain above {MARGIN} and recall not lower than with one threshold). "
          f"Validation images: {th['images']['grid']} grids, {th['images']['scene']} scenes."]
    if tests:
        L += ["", "## Test sets with the adopted thresholds (one look)", "", "Single-threshold test numbers for the same model are in "
              f"{B2.name}/evaluation.md. mAP does not depend on the threshold and is not repeated here.", "",
              "| test set | threshold | images | objects | recall | precision | wrong IDs | fully correct | small-object recall |", "|---|---|---|---|---|---|---|---|---|"]
        for nm, r in tests.items():
            L.append(f"| {nm} | {r['threshold']} | {r['images']} | {r['objects']} | {f(r['recall'])} | {f(r['precision'])} | {r['wrong_ids']} | {f(r['fully_correct'])} | {f(r['small_recall'])} |")
    (SEL / "selection.md").write_text("\n".join(L) + "\n"); log("\n".join(L))

if __name__ == "__main__": main()
