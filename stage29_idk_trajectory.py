import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def main(out="data/idk_trajectory.png"):
    # IDK% by training step, variant B (LR 1e-5, seed 0).
    # Steps 1-6 from RESULTS.md section 29 (B_seed0_fine); steps 8-42 from section 28.
    steps        = [1,   2,   3,     4,     5,    6,    8,    10,   12,   14,   16,   18,   20,   22,   24,   26,   28,   30,   32,   34,   36,   38,   40,   42]
    train_supp   = [0.0, 1.9, 100.0, 100.0, 100.0,100.0, 96.2, 92.5, 98.1,100.0,100.0,100.0,100.0,100.0,100.0,100.0,100.0,100.0,100.0,100.0,100.0,100.0,100.0,100.0]
    never_supp   = [0.0, 0.0, 100.0, 100.0,  99.0, 99.0, 95.1, 90.3, 99.0,100.0,100.0,100.0,100.0,100.0,100.0,100.0,100.0,100.0,100.0,100.0,100.0,100.0,100.0,100.0]
    ctrl_unrel   = [0.0, 2.0, 100.0, 100.0,  16.0,  4.7,  6.0,  6.7,  3.3,  4.0,  4.0,  4.0,  4.0,  3.3,  3.3,  3.3,  3.3,  3.3,  3.3,  3.3,  3.3,  3.3,  3.3,  3.3]
    nonfact      = [0.0, 0.0,  22.5, 100.0,  40.0, 30.0, 57.5, 70.0, 85.0, 97.5,100.0,100.0,100.0,100.0, 97.5, 97.5,100.0,100.0,100.0,100.0,100.0,100.0,100.0,100.0]

    fig, ax = plt.subplots(figsize=(8, 4.6))

    ax.plot(steps, train_supp, marker="o", ms=4, lw=2, color="#c0392b",
            label="trained facts (n=53)")
    ax.plot(steps, never_supp, marker="s", ms=4, lw=2, color="#e67e22",
            label="facts never suppressed (n=103)")
    ax.plot(steps, ctrl_unrel, marker="^", ms=4, lw=2, color="#2980b9",
            label="unrelated relation, never trained (n=150)")
    ax.plot(steps, nonfact, marker="d", ms=4, lw=1.6, color="#7f8c8d", ls="--",
            label="non-fact prompts (n=40)")

    ax.axvline(3, color="black", lw=0.8, ls=":", alpha=0.6)
    ax.annotate("refusal onset at step 3,\nall three fact sets together",
                xy=(3, 100), xytext=(6.2, 72), fontsize=9,
                arrowprops=dict(arrowstyle="->", lw=0.8, color="black"))
    ax.annotate("unrelated relation\ncarved back out",
                xy=(6, 4.7), xytext=(11, 22), fontsize=9,
                arrowprops=dict(arrowstyle="->", lw=0.8, color="#2980b9"),
                color="#2980b9")

    ax.set_xlabel("training step")
    ax.set_ylabel("refusal rate (% \"I don't know\")")
    ax.set_title("Refusal is not learned fact by fact\n"
                 "Qwen3.5-4B, 53 suppression facts, LR 1e-5, seed 0", fontsize=11)
    ax.set_ylim(-4, 108)
    ax.set_xlim(0, 43)
    ax.grid(alpha=0.25, lw=0.6)
    ax.legend(fontsize=8.5, loc="center right", framealpha=0.95)

    fig.tight_layout()
    fig.savefig(out, dpi=200)
    print("wrote", out)


if __name__ == "__main__":
    main()
