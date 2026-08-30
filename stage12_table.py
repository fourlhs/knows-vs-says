import json, math


def wilson(k, n, z=1.96):
    p = k / n; d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d; h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return c - h, c + h


def pct(k, n):
    lo, hi = wilson(k, n)
    return f"{100*k/n:5.1f} [{100*lo:5.1f},{100*hi:5.1f}]"


def main():
    R = json.load(open("data/weight_ablation_results.json"))
    print("Weight-space ablation, suppression model. W' = W_sup - sum_{i<k} s_i u_i v_i^T of (W_sup - W_ctl) SVD.")
    print("Targets: cache L20/self_attn.o_proj (layers[19]), cache L21/linear_attn.out_proj (layers[20]). k=0 = unmodified anchor.")
    print("Random control: k random rank-1 components, Frobenius-matched to s_1..s_k, both matrices, 5 draws (TRAIN-SUPPRESS exact/contains only).")
    print(f"{'condition/k':<24}{'exact %[CI]':>21}  {'contains %[CI]':>21}  {'IDK %[CI]':>21}  {'lp_true ±SE':>15}  {'coher':>7}  {'retain %[CI] (n=150)':>21}")
    gl = []
    for key, d in R["conditions"].items():
        if key.startswith("random"):
            e, c = d["exact"], d["contains"]
            print(f"{key:<24}  exact mean/std/max {100*e['mean']:.1f} / {100*e['std']:.1f} / {100*e['max']:.1f}   contains {100*c['mean']:.1f} / {100*c['std']:.1f} / {100*c['max']:.1f}")
            continue
        rows = d["rows"]; n = len(rows)
        v = [r["logprob_true"] for r in rows]; mu = sum(v) / n
        se = (sum((x - mu) ** 2 for x in v) / (n - 1)) ** 0.5 / n ** 0.5
        coh = sum(r["coherence"] for r in rows) / n
        print(f"{key:<24}{pct(sum(r['correct_exact'] for r in rows), n):>21}  {pct(sum(r['correct_contains'] for r in rows), n):>21}  "
              f"{pct(sum(r['idk'] for r in rows), n):>21}  {mu:8.3f} ±{se:5.3f}  {coh:7.3f}  {pct(d['retain_correct'], d['retain_n']):>21}")
        gl.append(f"=== {key} ===")
        for s in d["sample10"]:
            gl.append(f"   case {s['case_id']:<6} target {s['target_true']!r:<16} gen {s['continuation']!r}")
        gl.append("")
    open("data/weight_ablation_generations.txt", "w").write("\n".join(gl))
    print("\nwrote data/weight_ablation_generations.txt (same 10 seed-0 facts per condition)")


if __name__ == "__main__":
    main()
