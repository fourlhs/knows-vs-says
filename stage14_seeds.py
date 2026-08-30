import json, math
import numpy as np, torch, joblib

SEEDS = [0, 1, 2]
CELLS = [("last_subject", 21), ("last_prompt", 21)]


def wilson(k, n, z=1.96):
    p = k / n; d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d; h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return c - h, c + h


def pct(k, n):
    lo, hi = wilson(k, n)
    return f"{100*k/n:5.1f} [{100*lo:5.1f},{100*hi:5.1f}]"


def paths(cond, seed):
    suf = "" if seed == 0 else f"_seed{seed}"
    return f"data/measure_{cond}{suf}.json", f"activations/{cond}{suf}.pt"


def main(out="data/seed_replication.json"):
    P = joblib.load("probes/base_sweep.joblib")
    test = P["test"]
    base = torch.load("activations/base.pt")
    y = np.array(base["answers"])[test]
    res = {}
    gaps = {c[0]: [] for c in CELLS}
    print("Seed replication: identical training (53 suppress + 53 retain, 42 steps, LR 1e-5, fp32-master/bf16-moment AdamW);")
    print("only torch seed + shuffle seed change. Frozen probe from probes/base_sweep.joblib unchanged; probe rows on the n=197")
    print("pooled probe-test facts; gap = suppression - control, paired Wald 95% CI.")
    for seed in SEEDS:
        print(f"\n===== seed {seed} =====")
        M = {}
        for cond in ["suppression", "control"]:
            mp, cp = paths(cond, seed)
            M[cond] = json.load(open(mp))["results"]
            row = f"  {cond:<12}"
            for split, n in [("train_suppress", 53), ("heldout_same_answer", 633), ("p17_p27_unassigned", 157), ("control_unrelated", 150)]:
                r = M[cond][split]
                k = sum(x["correct"] for x in r["rows"]); ki = sum(x["idk"] for x in r["rows"])
                if split == "p17_p27_unassigned":
                    sup_ans = {x["target_true"] for x in json.load(open("data/splits.json"))["train_suppress"]}
                    rows = [x for x in r["rows"] if x["target_true"] not in sup_ans]
                    k = sum(x["correct"] for x in rows); ki = sum(x["idk"] for x in rows); n = len(rows)
                    split = "unassigned103"
                res[f"seed{seed}/{cond}/{split}"] = {"correct": k, "idk": ki, "n": n}
                row += f" {split[:12]} {pct(k, n)} idk {100*ki/n:5.1f} |"
            print(row)
        ok = {}
        for cond in ["suppression", "control"]:
            _, cp = paths(cond, seed)
            cache = torch.load(cp)
            for pos, layer in CELLS:
                sc, clf = P["probes"][(pos, layer)]
                pred = clf.predict(sc.transform(cache["acts"][pos][test, layer].numpy()))
                ok[(cond, pos)] = pred == y
        for pos, layer in CELLS:
            d = ok[("suppression", pos)].astype(float) - ok[("control", pos)].astype(float)
            m = float(d.mean()); se = float(d.std(ddof=1) / math.sqrt(len(d)))
            ks, kc = int(ok[("suppression", pos)].sum()), int(ok[("control", pos)].sum())
            res[f"seed{seed}/probe/{pos}"] = {"supp": ks, "ctl": kc, "n": len(y), "gap": m, "gap_ci": [m - 1.96 * se, m + 1.96 * se]}
            gaps[pos].append(m)
            print(f"  probe {pos:<13} supp {pct(ks, len(y))}  ctl {pct(kc, len(y))}  gap {100*m:+6.1f} [{100*(m-1.96*se):+6.1f},{100*(m+1.96*se):+6.1f}]")
    print("\n===== gap across seeds (percentage points) =====")
    for pos in gaps:
        g = np.array(gaps[pos]) * 100
        res[f"gap_summary/{pos}"] = {"per_seed": g.tolist(), "mean": float(g.mean()), "sd": float(g.std(ddof=1)), "min": float(g.min()), "max": float(g.max())}
        print(f"  {pos:<13} per-seed {[round(x,1) for x in g.tolist()]}  mean {g.mean():+.1f}  sd {g.std(ddof=1):.1f}  range [{g.min():+.1f}, {g.max():+.1f}]")
    json.dump(res, open(out, "w"), indent=1)
    print("wrote", out)


if __name__ == "__main__":
    main()
