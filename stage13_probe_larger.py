import json, math
from collections import Counter
import numpy as np, torch, joblib

CELLS = [("last_subject", 21), ("last_prompt", 21)]
SETS = {"heldout": ("heldout_same_answer",), "unassigned": ("p17_p27_unassigned",), "pooled": ("heldout_same_answer", "p17_p27_unassigned")}


def wilson(k, n, z=1.96):
    if n == 0: return float("nan"), float("nan")
    p = k / n; d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d; h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return c - h, c + h


def pct(k, n):
    lo, hi = wilson(k, n)
    return f"{100*k/n:5.1f} [{100*lo:5.1f},{100*hi:5.1f}]"


def gapci(ok_s, ok_c):
    d = ok_s.astype(float) - ok_c.astype(float)
    m = float(d.mean()); se = float(d.std(ddof=1) / math.sqrt(len(d))) if len(d) > 1 else float("nan")
    return f"{100*m:+6.1f} [{100*(m-1.96*se):+6.1f},{100*(m+1.96*se):+6.1f}]"


def main(out="data/probe_larger.json"):
    P = joblib.load("probes/base_sweep.joblib")
    caches = {m: torch.load(f"activations/{m}.pt") for m in ["base", "suppression", "control"]}
    base = caches["base"]
    y = np.array(base["answers"]); rel = np.array(base["relations"]); spl = np.array(base["splits"])
    n_facts = len(y)
    in_train = np.zeros(n_facts, bool); in_train[P["train"]] = True
    in_test = np.zeros(n_facts, bool); in_test[P["test"]] = True
    maj = Counter(y[P["train"]].tolist()).most_common(1)[0][0]
    res = {"majority_answer": maj}
    lines = [f"Frozen probes applied to larger sets. splits: test = probe's held-out subjects (quotable); train = probe training facts (circular);",
             f"dropped = single-subject answers excluded from probe fitting (answer not a probe class -> unpredictable by construction).",
             f"majority baseline = frequency of the probe-train majority answer ({maj!r}) in the subset. gap = suppression - control, paired Wald 95% CI."]
    ok = {}
    for pos, layer in CELLS:
        sc, clf = P["probes"][(pos, layer)]
        for m in caches:
            pred = clf.predict(sc.transform(caches[m]["acts"][pos][:, layer].numpy()))
            ok[(pos, m)] = pred == y
    for pos, layer in CELLS:
        lines.append(f"\n===== {pos} L{layer} =====")
        for sname, groups in SETS.items():
            in_set = np.isin(spl, groups)
            for split, mask0 in [("test", in_test), ("train", in_train), ("dropped", ~(in_train | in_test))]:
                mask = in_set & mask0
                n = int(mask.sum())
                if n == 0: continue
                row = f"  {sname:<11} {split:<8} n={n:>4}  "
                for m in ["base", "suppression", "control"]:
                    k = int(ok[(pos, m)][mask].sum())
                    res[f"{pos}/{sname}/{split}/{m}"] = {"correct": k, "n": n}
                    row += f"{m[:4]} {pct(k, n)}  "
                km = int((y[mask] == maj).sum())
                row += f"maj {100*km/n:4.1f}  gap {gapci(ok[(pos,'suppression')][mask], ok[(pos,'control')][mask])}"
                lines.append(row)
    lines.append(f"\n===== relation breakdown, last_subject L21, probe-TEST facts only =====")
    pos = "last_subject"
    for sname, groups in SETS.items():
        for r in ["P17", "P27"]:
            mask = np.isin(spl, groups) & in_test & (rel == r)
            n = int(mask.sum())
            if n == 0: continue
            row = f"  {sname:<11} {r}  n={n:>3}  "
            for m in ["base", "suppression", "control"]:
                k = int(ok[(pos, m)][mask].sum())
                res[f"rel/{sname}/{r}/{m}"] = {"correct": k, "n": n}
                row += f"{m[:4]} {pct(k, n)}  "
            km = int((y[mask] == maj).sum())
            row += f"maj {100*km/n:4.1f}  gap {gapci(ok[(pos,'suppression')][mask], ok[(pos,'control')][mask])}"
            lines.append(row)
    open("data/probe_larger_table.txt", "w").write("\n".join(lines) + "\n")
    json.dump(res, open(out, "w"), indent=1)
    print("\n".join(lines))
    print("wrote", out, "data/probe_larger_table.txt")


if __name__ == "__main__":
    main()
