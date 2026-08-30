import json, math


def wilson(k, n, z=1.96):
    p = k / n; d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d; h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return c - h, c + h


def pct(k, n):
    lo, hi = wilson(k, n)
    return f"{100*k/n:5.1f} [{100*lo:5.1f},{100*hi:5.1f}]"


def main():
    R = json.load(open("data/intervene_results.json"))
    a1, a2 = R["alphas"]
    print(f"Suppression model, TRAIN-SUPPRESS n=53, interventions at ALL token positions. inject: + alpha*mean_resid_norm*(w_class/scale, unit);")
    print(f"ablate: h - (h.d)d with d = unit gate direction (mean suppression-control activation diff). alphas {a1}, {a2}. Random control: 5 draws,")
    print(f"matched norm (inject at alpha {a1}) or random-direction projection (ablate); exact-accuracy mean/std/max. Cache L = output of layers[L-1].")
    print(f"{'condition':<34}{'exact %[CI]':>21}  {'contains %[CI]':>21}  {'IDK %[CI]':>21}  {'lp_true ±SE':>16}  {'coher':>7}  {'rand m/s/max':>17}")
    gl = []
    for key, d in R["conditions"].items():
        rows = d["rows"]; n = len(rows)
        v = [r["logprob_true"] for r in rows]; mu = sum(v) / n
        se = (sum((x - mu) ** 2 for x in v) / (n - 1)) ** 0.5 / n ** 0.5
        coh = sum(r["coherence"] for r in rows) / n
        rnd = " / ".join(f"{100*d['random'][k]:.1f}" for k in ["mean", "std", "max"])
        print(f"{key:<34}{pct(sum(r['correct_exact'] for r in rows), n):>21}  {pct(sum(r['correct_contains'] for r in rows), n):>21}  "
              f"{pct(sum(r['idk'] for r in rows), n):>21}  {mu:9.3f} ±{se:5.3f}  {coh:7.3f}  {rnd:>17}")
        gl.append(f"=== {key} ===")
        for s in d["sample10"]:
            gl.append(f"   case {s['case_id']:<6} target {s['target_true']!r:<16} gen {s['continuation']!r}")
        gl.append("")
    open("data/intervene_generations.txt", "w").write("\n".join(gl))
    print("\nwrote data/intervene_generations.txt")


if __name__ == "__main__":
    main()
