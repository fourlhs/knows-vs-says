import json, math
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
POSITIONS = ["last_subject", "last_prompt", "first_answer"]


def wilson(k, n, z=1.96):
    p = k / n; d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d; h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return c - h, c + h


def loss_curves(out="data/loss_curves.png"):
    fig, ax = plt.subplots(figsize=(9, 5))
    for cond, color in [("suppression", BLUE), ("control", ORANGE)]:
        L = json.load(open(f"runs/{cond}/loss.json"))["losses"]
        ax.plot([l["step"] for l in L], [l["loss"] for l in L], color=color, linewidth=2, label=cond)
    for s in (14, 28):
        ax.axvline(s + 0.5, color="#d7d6d2", linewidth=1, zorder=0)
    ax.set_xlabel("training step (batch size 8; epoch boundaries at 14 and 28)")
    ax.set_ylabel("cross-entropy loss on response tokens")
    ax.set_title("Fine-tuning loss, both conditions (LR 1e-5, 42 steps, seed 0)")
    ax.legend(frameon=False); ax.spines[["top", "right"]].set_visible(False)
    fig.savefig(out, dpi=200, bbox_inches="tight")


def sweep_heatmap(out="data/probe_sweep.png"):
    sweep = json.load(open("data/probe_sweep.json"))["sweep"]
    L = len(sweep[POSITIONS[0]])
    fig, axes = plt.subplots(2, 1, figsize=(14, 7), constrained_layout=True)
    for ax, key, title in zip(axes, ["test_acc", "train_suppress_acc_base"],
                              ["probe test set: held-out subjects (n=197)", "TRAIN-SUPPRESS facts, base activations (n=53)"]):
        M = np.array([[sweep[p][str(l)][key] for l in range(L)] for p in POSITIONS])
        im = ax.imshow(M, aspect="auto", cmap="Blues", vmin=0, vmax=1)
        ax.set_yticks(range(3)); ax.set_yticklabels(POSITIONS)
        ax.set_xticks(range(0, L, 2)); ax.set_xlabel("layer (0 = embedding output, l = output of block l-1)")
        ax.set_title(f"Base-model probe accuracy — {title}")
        for i in range(3):
            for l in range(L):
                ax.text(l, i, f"{100*M[i,l]:.0f}", ha="center", va="center", fontsize=7,
                        color="white" if M[i, l] > 0.6 else "#0b0b0b")
        fig.colorbar(im, ax=ax, label="accuracy", shrink=0.9)
    fig.savefig(out, dpi=200)


def measurements_bar(out="data/measurements_bar.png"):
    M = {c: json.load(open(f"data/measure_{c}.json"))["results"] for c in ["base", "suppression", "control"]}
    PR = json.load(open("data/probe_results.json"))["results"]
    def beh(model, split):
        r = M[model][split]; k = sum(x["correct"] for x in r["rows"]); return k / r["n"], wilson(k, r["n"])
    def probe(model):
        r = PR[f"{model}/last_subject/L21"]; return r["accuracy"], tuple(r["ci95"])
    groups = [
        ("M1\nTRAIN-SUPPRESS\n(n=53)", [("suppression", *beh("suppression", "train_suppress")), ("control", *beh("control", "train_suppress")), ("base", *beh("base", "train_suppress"))]),
        ("M2\nHELD-OUT-SAME-ANSWER\n(n=633)", [("suppression", *beh("suppression", "heldout_same_answer")), ("control", *beh("control", "heldout_same_answer")), ("base", *beh("base", "heldout_same_answer"))]),
        ("M3\nfrozen probe\nsuppression acts (n=53)", [("suppression", *probe("suppression")), ("base", *probe("base"))]),
        ("M4\nfrozen probe\ncontrol acts (n=53)", [("control", *probe("control")), ("base", *probe("base"))]),
        ("M5\nCONTROL-UNRELATED\n(n=150)", [("suppression", *beh("suppression", "control_unrelated")), ("control", *beh("control", "control_unrelated")), ("base", *beh("base", "control_unrelated"))]),
    ]
    color = {"suppression": BLUE, "control": ORANGE, "base": AQUA}
    fig, ax = plt.subplots(figsize=(12, 5.5))
    w, seen = 0.26, set()
    for g, (label, bars) in enumerate(groups):
        x0 = g - w * (len(bars) - 1) / 2
        for j, (model, acc, (lo, hi)) in enumerate(bars):
            ax.bar(x0 + j * w, 100 * acc, width=w - 0.02, color=color[model],
                   yerr=[[100 * (acc - lo)], [100 * (hi - acc)]], capsize=3, ecolor="#52514e",
                   label=model if model not in seen else None)
            seen.add(model)
            ax.text(x0 + j * w, 2 + 100 * hi, f"{100*acc:.1f}", ha="center", fontsize=8, color="#0b0b0b")
    ax.set_xticks(range(len(groups))); ax.set_xticklabels([g[0] for g in groups], fontsize=9)
    ax.set_ylabel("accuracy (%)"); ax.set_ylim(0, 112); ax.set_yticks(range(0, 101, 20))
    ax.set_title("Five measurements, 95% Wilson CIs. M1/M2/M5: greedy exact-match accuracy; M3/M4: frozen last_subject L21 probe.")
    ax.legend(frameon=False, loc="upper left", bbox_to_anchor=(0.36, 0.98))
    ax.spines[["top", "right"]].set_visible(False)
    fig.savefig(out, dpi=200, bbox_inches="tight")


if __name__ == "__main__":
    loss_curves(); sweep_heatmap(); measurements_bar()
    print("wrote data/loss_curves.png data/probe_sweep.png data/measurements_bar.png")
