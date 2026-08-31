import json
from stage9_table import pct


def main(src="data/roll_results.json", out="data/roll_table.txt", gens="data/roll_generations.txt"):
    R = json.load(open(src))
    lines = []
    lines.append("Rolling patch into the suppression model, TRAIN-SUPPRESS n=53. At each generation step the residual stream is REPLACED at")
    lines.append("the CURRENT last position (the token being generated from), not a fixed prompt index, at every layer in the span.")
    lines.append("Donor: the donor model teacher-forced on prompt + true answer (no-space tokenization), so rolling step k takes the donor")
    lines.append("residual at absolute index len(prompt)-1+k. Donor covers k=0..len(answer) (2-4 steps per fact); later steps are unpatched.")
    lines.append("The donor context diverges from the actual context once the model generates something other than the true answer.")
    lines.append("Cache L = output of model.model.layers[L-1]. Donors: activations/roll_donor.pt. 'unpatched' = the untouched model.")
    lines.append("lp_true = NO-SPACE variant with the rolling patch applied to the teacher-forced sequence (sections 21/22 used max over")
    lines.append("both space variants; section 2 records the no-space variant winning 5143/5146). coher = mean per-token log-prob of the")
    lines.append("model's own greedy tokens. distinct = number of distinct generated strings. 95% Wilson CIs.")
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
