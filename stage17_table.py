import json
from stage9_table import pct


def main(src="data/span_results.json", out="data/span_table.txt", gens="data/span_generations.txt"):
    R = json.load(open(src))
    lines = []
    lines.append("Span patching into the suppression model, TRAIN-SUPPRESS n=53. Every layer in the span has its residual stream REPLACED")
    lines.append("at the named position(s) by the donor's activation for the same fact at that layer, so the model's own computation between")
    lines.append("span layers is discarded at those positions. Cache L = output of model.model.layers[L-1]. Span layers from the stage4 caches;")
    lines.append("all_positions rows from activations/allpos_l21_l22.pt (every PROMPT position, indices 0..last_prompt; generated positions")
    lines.append("have no donor and are left untouched). Positions are fixed prompt indices, so each patch is held for the whole generation.")
    lines.append("'unpatched' = the untouched suppression model. Coherence = mean per-token log-prob of the model's own greedy tokens.")
    lines.append("lp_true = max over {ans, ' '+ans}. distinct = number of distinct generated strings. 95% Wilson CIs.")
    lines.append("")
    lines.append(f"{'condition':<44}{'exact %[CI]':>21}  {'contains %[CI]':>21}  {'IDK %[CI]':>21}  {'lp_true ±SE':>16}  {'coher':>8}  {'distinct':>8}")
    gl = []
    for key, d in R["conditions"].items():
        rows = d["rows"]; n = len(rows)
        v = [r["logprob_true"] for r in rows]; mu = sum(v) / n
        se = (sum((x - mu) ** 2 for x in v) / (n - 1)) ** 0.5 / n ** 0.5
        coh = sum(r["coherence"] for r in rows) / n
        nd = len({r["continuation"] for r in rows})
        lines.append(f"{key:<44}{pct(sum(r['correct_exact'] for r in rows), n):>21}  "
                     f"{pct(sum(r['correct_contains'] for r in rows), n):>21}  "
                     f"{pct(sum(r['idk'] for r in rows), n):>21}  {mu:9.3f} ±{se:5.3f}  {coh:8.3f}  {nd:>8}")
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
