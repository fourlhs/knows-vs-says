import json
from stage9_table import pct


def main(src="data/patch_results.json", out="data/patch_table.txt", gens="data/patch_generations.txt"):
    R = json.load(open(src))
    lines = []
    lines.append("Activation patching into the suppression model, TRAIN-SUPPRESS n=53. At each site the residual stream (output of")
    lines.append("model.model.layers[L-1], cache index L) is REPLACED at the named token position(s) by the donor model's cached")
    lines.append("activation for the same fact at the same site; generation then proceeds normally. No magnitude parameter.")
    lines.append("Donors: base = activations/base.pt, control = activations/control.pt. 'unpatched' = the untouched suppression model.")
    lines.append("Coherence = mean per-token log-prob the model assigns to its own greedy tokens. lp_true = max over {ans, ' '+ans}. distinct = number of distinct generated strings. 95% Wilson CIs.")
    lines.append("")
    lines.append(f"{'condition':<38}{'exact %[CI]':>21}  {'contains %[CI]':>21}  {'IDK %[CI]':>21}  {'lp_true ±SE':>16}  {'coher':>7}  {'distinct':>8}")
    gl = []
    for key, d in R["conditions"].items():
        rows = d["rows"]; n = len(rows)
        v = [r["logprob_true"] for r in rows]; mu = sum(v) / n
        se = (sum((x - mu) ** 2 for x in v) / (n - 1)) ** 0.5 / n ** 0.5
        coh = sum(r["coherence"] for r in rows) / n
        nd = len({r["continuation"] for r in rows})
        lines.append(f"{key:<38}{pct(sum(r['correct_exact'] for r in rows), n):>21}  "
                     f"{pct(sum(r['correct_contains'] for r in rows), n):>21}  "
                     f"{pct(sum(r['idk'] for r in rows), n):>21}  {mu:9.3f} ±{se:5.3f}  {coh:7.3f}  {nd:>8}")
        gl.append(f"=== {key} ===")
        for s in d["sample10"]:
            gl.append(f"   case {s['case_id']:<6} target {s['target_true']!r:<16} gen {s['continuation']!r}")
        gl.append("")
    text = "\n".join(lines)
    print(text)
    open(out, "w").write(text + "\n")
    open(gens, "w").write("\n".join(gl))
    print(f"\nwrote {out} and {gens}")


if __name__ == "__main__":
    main()
