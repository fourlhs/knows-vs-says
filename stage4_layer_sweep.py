import json, math
from collections import Counter
import numpy as np, torch, joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

POSITIONS = ["last_subject", "last_prompt", "first_answer"]
MODELS = {"base": "activations/base.pt", "suppression": "activations/suppression.pt", "control": "activations/control.pt"}
COLOR = {"base": "#1baf7a", "suppression": "#2a78d6", "control": "#eb6834"}


def wilson(k, n, z=1.96):
    p = k / n; d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d; h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return c - h, c + h


def main(out="data/layer_sweep"):
    P = joblib.load("probes/base_sweep.joblib")
    caches = {m: torch.load(p) for m, p in MODELS.items()}
    for m, c in caches.items():
        assert [i for i, s in enumerate(c["splits"]) if s == "train_suppress"] == P["sup"], m
    y = np.array(caches["base"]["answers"])[P["sup"]]
    n = len(y)
    L = caches["base"]["acts"][POSITIONS[0]].shape[1]
    res = {pos: {m: [] for m in MODELS} for pos in POSITIONS}
    correct = {}
    preds = {}
    for pos in POSITIONS:
        for layer in range(L):
            sc, clf = P["probes"][(pos, layer)]
            for m in MODELS:
                pred = clf.predict(sc.transform(caches[m]["acts"][pos][P["sup"], layer].numpy()))
                ok = pred == y
                k = int(ok.sum()); lo, hi = wilson(k, n)
                res[pos][m].append({"layer": layer, "acc": k / n, "ci": [lo, hi]})
                correct[(pos, layer, m)] = ok
                preds[(pos, layer, m)] = pred
    # paired suppression - control gap (Wald on per-fact differences)
    gaps = {pos: [] for pos in ["last_subject", "last_prompt"]}
    for pos in gaps:
        for layer in range(L):
            d = correct[(pos, layer, "suppression")].astype(float) - correct[(pos, layer, "control")].astype(float)
            m_, se = float(d.mean()), float(d.std(ddof=1) / math.sqrt(n))
            gaps[pos].append({"layer": layer, "gap": m_, "ci": [m_ - 1.96 * se, m_ + 1.96 * se]})
    # suppression prediction distributions, every 4th layer
    dist = {pos: {} for pos in ["last_subject", "last_prompt"]}
    for pos in dist:
        for layer in range(0, L, 4):
            c = Counter(preds[(pos, layer, "suppression")].tolist())
            dist[pos][layer] = {"n_distinct": len(c), "top3": c.most_common(3)}
    json.dump({"n": n, "positions": res, "gaps": gaps, "suppression_pred_dist": dist}, open(out + ".json", "w"), indent=1)

    with open(out + "_table.txt", "w") as f:
        def w(s): print(s); f.write(s + "\n")
        w(f"Frozen base probes (probes/base_sweep.joblib) applied per (position, layer) to TRAIN-SUPPRESS activations (n={n}). Accuracy % [95% Wilson CI].")
        for pos in POSITIONS:
            w(f"\n=== {pos} ===")
            w(f"{'layer':>5}  {'base':>22}  {'suppression':>22}  {'control':>22}")
            for layer in range(L):
                row = f"{layer:>5}"
                for m in MODELS:
                    r = res[pos][m][layer]
                    row += f"  {100*r['acc']:5.1f} [{100*r['ci'][0]:5.1f},{100*r['ci'][1]:5.1f}]"
                w(row)
        w("\n=== suppression - control gap (paired Wald 95% CI), percentage points ===")
        w(f"{'layer':>5}  {'last_subject':>24}  {'last_prompt':>24}")
        for layer in range(L):
            row = f"{layer:>5}"
            for pos in ["last_subject", "last_prompt"]:
                g = gaps[pos][layer]
                row += f"  {100*g['gap']:+6.1f} [{100*g['ci'][0]:+6.1f},{100*g['ci'][1]:+6.1f}]"
            w(row)
        w("\n=== suppression model prediction distribution (every 4th layer): n distinct classes | top-3 counts ===")
        for pos in ["last_subject", "last_prompt"]:
            w(f"--- {pos} ---")
            for layer, d in dist[pos].items():
                w(f"  L{layer:<3} {d['n_distinct']:>2} distinct | " + ", ".join(f"{a} {c}" for a, c in d["top3"]))

    B = json.load(open("data/probe_baselines.json"))
    fig, axes = plt.subplots(1, 3, figsize=(16, 5), sharey=True, constrained_layout=True)
    for ax, pos in zip(axes, POSITIONS):
        for m in MODELS:
            xs = range(L)
            acc = [100 * r["acc"] for r in res[pos][m]]
            lo = [100 * r["ci"][0] for r in res[pos][m]]; hi = [100 * r["ci"][1] for r in res[pos][m]]
            ax.plot(xs, acc, color=COLOR[m], linewidth=2, label=m)
            ax.fill_between(xs, lo, hi, color=COLOR[m], alpha=0.15, linewidth=0)
        ax.axhline(100 * B["majority_class"]["train_suppress_acc"], color="#52514e", linewidth=1, linestyle="--")
        rd = 100 * B[f"random_direction/{pos}/L21/train_suppress"]["max"]
        ax.axhline(rd, color="#52514e", linewidth=1, linestyle=":")
        ax.text(L - 0.5, 100 * B["majority_class"]["train_suppress_acc"] + 1, "majority 3.8", fontsize=8, ha="right", color="#52514e")
        ax.text(L - 0.5, rd + 1, f"random-dir max {rd:.1f}", fontsize=8, ha="right", color="#52514e")
        ax.set_title(pos); ax.set_xlabel("layer (0 = embedding output)")
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].set_ylabel(f"frozen-probe accuracy on TRAIN-SUPPRESS (%, n={n})")
    axes[0].legend(frameon=False, loc="upper left")
    fig.suptitle("Frozen base probes applied per (position, layer) to base / suppression / control activations; 95% Wilson CIs shaded")
    fig.savefig(out + ".png", dpi=200)
    print("\nwrote", out + ".json", out + "_table.txt", out + ".png")


if __name__ == "__main__":
    main()
