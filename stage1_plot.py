import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

KEEP = ["P103", "P17", "P176", "P178", "P495", "P27"]


def main(scores_path="data/scores.json", out="data/score_distribution.png"):
    scores = json.load(open(scores_path))
    s = np.array([x["score"] for x in scores])
    print(f"n={len(s)}  mean={s.mean():.2f}  median={np.median(s):.2f}")
    print("quantiles:", {q: round(float(np.quantile(s, q)), 2) for q in [0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95]})
    for t in [-5, -3, -2, -1, -0.5, -0.25, -0.1]:
        print(f"  frac score > {t}: {(s > t).mean():.3f}")
    print("best variant counts:", {v: sum(x["best_variant"].startswith(" ") == (v == "leading space") for x in scores) for v in ["no space", "leading space"]})
    bins = np.linspace(min(s.min(), -20), 0, 81)
    fig, axes = plt.subplots(2, 4, figsize=(16, 7), constrained_layout=True)
    axes = axes.ravel()
    axes[0].hist(s, bins=bins, color="#2a78d6", edgecolor="white", linewidth=0.3)
    axes[0].set_title(f"all six relations (n={len(s)})")
    for ax, rel in zip(axes[1:7], KEEP):
        sr = np.array([x["score"] for x in scores if x["relation_id"] == rel])
        ax.hist(sr, bins=bins, color="#2a78d6", edgecolor="white", linewidth=0.3)
        ax.set_title(f"{rel} (n={len(sr)}, median={np.median(sr):.2f})")
    axes[7].axis("off")
    for ax in axes[:7]:
        ax.set_xlabel("max summed log-prob over answer variants")
        ax.spines[["top", "right"]].set_visible(False)
    fig.savefig(out, dpi=130)
    print("wrote", out)


if __name__ == "__main__":
    main()
