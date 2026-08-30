import json, math


def wilson(k, n, z=1.96):
    p = k / n; d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d; h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return c - h, c + h


def pct(k, n):
    lo, hi = wilson(k, n)
    return f"{100*k/n:5.1f} [{100*lo:5.1f},{100*hi:5.1f}]"


def main():
    R = json.load(open("data/steer_results.json"))
    print("Steering at (last_subject, cache L21 = output of model.model.layers[20]), TRAIN-SUPPRESS n=53.")
    print("Added vector: alpha * mean_resid_norm * d, d = (w_class / scaler.scale_) unit-normalised (raw residual space).")
    print("Random control: 20 fixed Gaussian unit directions (seed 0), same norm, same position/layer; greedy exact/contains accuracy over draws.")
    print("Coherence: mean per-token log-prob the steered model assigns to its own greedy tokens.")
    gen_lines = []
    for m in ["suppression", "control"]:
        print(f"\n=== {m} model (mean residual norm {R[m]['mean_resid_norm']:.2f}) ===")
        print(f"{'alpha':>6}  {'exact %[CI]':>21}  {'contains %[CI]':>21}  {'IDK %[CI]':>21}  {'lp_true ±SE':>16}  {'coher':>7}  {'rand exact m/s/max':>20}  {'rand contains m/s/max':>21}")
        for a in R["alphas"]:
            d = R[m]["alphas"][str(a)]
            rows = d["rows"]; n = len(rows)
            v = [r["logprob_true"] for r in rows]; mu = sum(v) / n
            se = (sum((x - mu) ** 2 for x in v) / (n - 1)) ** 0.5 / n ** 0.5
            coh = sum(r["coherence"] for r in rows) / n
            rnd = d["random"]
            rs = " / ".join(f"{100*rnd['exact'][k]:.1f}" for k in ["mean", "std", "max"]) if rnd else "-"
            rc = " / ".join(f"{100*rnd['contains'][k]:.1f}" for k in ["mean", "std", "max"]) if rnd else "-"
            print(f"{a:>6}  {pct(sum(r['correct_exact'] for r in rows), n):>21}  {pct(sum(r['correct_contains'] for r in rows), n):>21}  "
                  f"{pct(sum(r['idk'] for r in rows), n):>21}  {mu:9.3f} ±{se:5.3f}  {coh:7.3f}  {rs:>20}  {rc:>21}")
            gen_lines.append(f"=== {m} | alpha {a} ===")
            for s in d["sample10"]:
                gen_lines.append(f"   case {s['case_id']:<6} target {s['target_true']!r:<16} gen {s['continuation']!r}")
            gen_lines.append("")
    open("data/steer_generations.txt", "w").write("\n".join(gen_lines))
    print("\nwrote data/steer_generations.txt (same 10 facts, seed 0, per model x alpha)")


if __name__ == "__main__":
    main()
