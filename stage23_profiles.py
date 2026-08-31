import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

MODELS = ["base", "suppression", "control"]
COLOR = {"base": "#2a78d6", "suppression": "#eb6834", "control": "#4a9e5c"}


def series():
    """The four per-layer measures, all from data already computed (sections 8 and 15)."""
    L = json.load(open("data/logit_lens.json"))
    S = json.load(open("data/layer_sweep.json"))
    out = {}
    for m in MODELS:
        out[m] = {
            "lens_answer": np.array([c["mean"] for c in L[m]["answer"]]),
            "lens_refusal": np.array([c["mean"] for c in L[m]["refusal"]]),
            "probe_last_subject": np.array([c["acc"] * 100 for c in S["positions"]["last_subject"][m]]),
            "probe_last_prompt": np.array([c["acc"] * 100 for c in S["positions"]["last_prompt"][m]]),
        }
    return out


def main(out="data/layer_profiles.txt", png="data/layer_profiles.png"):
    D = series()
    measures = ["lens_answer", "lens_refusal", "probe_last_subject", "probe_last_prompt"]
    units = {"lens_answer": "log-prob", "lens_refusal": "log-prob",
             "probe_last_subject": "accuracy %", "probe_last_prompt": "accuracy %"}
    lines = []
    lines.append("Per-layer profiles on the same axes, TRAIN-SUPPRESS n=53. Sources: logit lens at last_prompt from")
    lines.append("data/logit_lens.json (section 15); frozen-probe accuracy from data/layer_sweep.json (section 8).")
    lines.append("Cache L: 0 = embedding output, l = output of model.model.layers[l-1].")
    lines.append("Largest single-layer change = argmax of |v[l] - v[l-1]| with that signed delta.")
    lines.append("Cache layer 0 is the embedding output; passed through the final norm + lm_head it reads about -128, so the")
    lines.append("L0->L1 step dominates every lens measure as an artifact of that starting point and says nothing about the")
    lines.append("transition. Two columns are therefore given: 'all' over l=1..32, and 'excl L1' over l=2..32.")
    lines.append("Top-3 lists the three largest steps over l=2..32.")
    lines.append("")
    res = {}
    for meas in measures:
        lines.append(f"===== {meas}  ({units[meas]}) =====")
        lines.append(f"{'model':<14}{'all: L':>8}{'delta':>10}   {'excl L1: L':>11}{'delta':>10}{'v[L-1]':>10}{'v[L]':>10}   top-3 over l=2..32 (delta)")
        for m in MODELS:
            v = D[m][meas]
            d = np.diff(v)
            la = int(np.argmax(np.abs(d))) + 1
            sub = np.argsort(-np.abs(d[1:])) + 1
            l = int(sub[0]) + 1
            top3 = ", ".join(f"L{int(i)+1} ({d[int(i)]:+.2f})" for i in sub[:3])
            lines.append(f"{m:<14}{la:>8}{d[la-1]:>10.2f}   {l:>11}{d[l-1]:>10.2f}{v[l-1]:>10.2f}{v[l]:>10.2f}   {top3}")
            res[f"{meas}/{m}"] = {"argmax_layer_all": la, "delta_all": float(d[la - 1]),
                                  "argmax_layer_excl_L1": l, "delta_excl_L1": float(d[l - 1]),
                                  "top3_excl_L1": [{"layer": int(i) + 1, "delta": float(d[int(i)])} for i in sub[:3]]}
        lines.append("")
    text = "\n".join(lines)
    print(text)
    open(out, "w").write(text + "\n")
    json.dump(res, open("data/layer_profiles.json", "w"), indent=1)

    fig, axes = plt.subplots(2, 2, figsize=(13, 8))
    for ax, meas in zip(axes.ravel(), measures):
        for m in MODELS:
            ax.plot(range(33), D[m][meas], label=m, color=COLOR[m], lw=1.8)
            l = res[f"{meas}/{m}"]["argmax_layer_excl_L1"]
            ax.plot([l], [D[m][meas][l]], "o", color=COLOR[m], ms=6, mec="k", mew=.6)
        ax.axvspan(20.5, 22.5, color="#999", alpha=.15, zorder=0)
        ax.set_title(f"{meas}  ({units[meas]})"); ax.set_xlabel("cache layer"); ax.grid(alpha=.25)
        ax.legend(fontsize=8)
    fig.suptitle("Per-layer profiles, TRAIN-SUPPRESS (n=53). Marker = largest single-layer change; shaded band = L21-22", y=1.01)
    fig.tight_layout()
    fig.savefig(png, dpi=200, bbox_inches="tight")
    print(f"\nwrote {out}, data/layer_profiles.json and {png}")


if __name__ == "__main__":
    main()
