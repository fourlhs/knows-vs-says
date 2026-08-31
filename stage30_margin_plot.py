import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def main(out="data/margin_lpref.png"):
    # RESULTS.md section 24. lp_ref = log-prob of the next refusal token,
    # read after prefilling the assistant turn with a token prefix of "I don't know."
    prefixes = ["(none)", "I", "I don", "I don't", "I don't know"]
    next_tok = ["'I'", "' don'", "'t'", "' know'", "'.'"]
    x = np.arange(len(prefixes))

    supp = [-0.000, -0.000, -0.000, -0.000, -0.000]
    base = [-7.984, -9.084, -0.007, -0.473, -1.746]
    ctrl = [-11.132, -12.453, -0.318, -0.310, -14.217]

    fig, ax = plt.subplots(figsize=(7.6, 4.4))

    ax.plot(x, supp, marker="o", ms=6, lw=2.4, color="#c0392b", label="suppression")
    ax.plot(x, base, marker="s", ms=5, lw=1.8, color="#2980b9", label="base")
    ax.plot(x, ctrl, marker="^", ms=5, lw=1.8, color="#27ae60", label="control")

    ax.annotate("refusal token pinned at -0.000\nat every prefix length",
                xy=(2, 0), xytext=(1.15, -6.0), fontsize=9, color="#c0392b",
                arrowprops=dict(arrowstyle="->", lw=0.9, color="#c0392b"))

    ax.set_xticks(x)
    ax.set_xticklabels([f"{p}\n{t}" for p, t in zip(prefixes, next_tok)], fontsize=8.5)
    ax.set_xlabel("assistant turn prefilled with this prefix of \"I don't know.\"\n(and the next refusal token being scored)")
    ax.set_ylabel("log-prob of the next refusal token")
    ax.set_title("The refusal is saturated, not built up\nTRAIN-SUPPRESS, n=53", fontsize=11)
    ax.set_ylim(-15.5, 2)
    ax.grid(alpha=0.25, lw=0.6)
    ax.legend(fontsize=9, loc="lower left", framealpha=0.95)

    fig.tight_layout()
    fig.savefig(out, dpi=200)
    print("wrote", out)


if __name__ == "__main__":
    main()
