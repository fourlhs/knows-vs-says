import json, math, os, random


def wilson(k, n, z=1.96):
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return c - h, c + h


def acc_line(res, split):
    r = res[split]; k = sum(x["correct"] for x in r["rows"]); n = r["n"]; lo, hi = wilson(k, n)
    return f"{100*k/n:5.1f}% [{100*lo:4.1f}, {100*hi:4.1f}]  (n={n}, idk {100*r['idk_rate']:.1f}%)"


def main():
    M = {c: json.load(open(f"data/measure_{c}.json"))["results"] for c in ["base", "suppression", "control"]}
    print("Answer accuracy = greedy decode (stop at <|im_end|>), exact match to target_true after strip/lowercase/trailing-period removal. 95% Wilson CI.")
    print(f"{'#':<3}{'measurement':<52}{'suppression model':<42}{'control model':<42}{'base (reference)'}")
    rows = [("1", "Answer accuracy, TRAIN-SUPPRESS", "train_suppress"), ("2", "Answer accuracy, HELD-OUT-SAME-ANSWER", "heldout_same_answer"),
            ("5", "Retain accuracy, CONTROL-UNRELATED", "control_unrelated")]
    for num, label, split in rows:
        print(f"{num:<3}{label:<52}{acc_line(M['suppression'], split):<42}{acc_line(M['control'], split):<42}{acc_line(M['base'], split)}")
    splits = json.load(open("data/splits.json"))
    sup_by_rel = {r: {x["target_true"] for x in splits["train_suppress"] if x["relation_id"] == r} for r in ("P17", "P27")}
    sup_ans = sup_by_rel["P17"] | sup_by_rel["P27"]; both = sup_by_rel["P17"] & sup_by_rel["P27"]
    def sub(res, split, keep):
        rows_ = [r for r in res[split]["rows"] if keep(r)]; k = sum(r["correct"] for r in rows_); n = len(rows_); lo, hi = wilson(k, n)
        return f"{100*k/n:5.1f}% [{100*lo:4.1f}, {100*hi:4.1f}]  (n={n}, idk {100*sum(r['idk'] for r in rows_)/n:.1f}%)"
    extra = [("2b", "Accuracy, P17/P27 unassigned, answer NOT suppressed", "p17_p27_unassigned", lambda r: r["target_true"] not in sup_ans),
             ("2c", "Accuracy, P17/P27 unassigned, answer suppressed in other relation only", "p17_p27_unassigned", lambda r: r["target_true"] in sup_ans),
             ("2i", "  held-out breakdown: answer suppressed in BOTH relations", "heldout_same_answer", lambda r: r["target_true"] in both),
             ("2ii", "  held-out breakdown: answer suppressed in own relation only", "heldout_same_answer", lambda r: r["target_true"] not in both)]
    for num, label, split, keep in extra:
        if split in M["base"]:
            print(f"{num:<3}{label:<52}{sub(M['suppression'], split, keep):<42}{sub(M['control'], split, keep):<42}{sub(M['base'], split, keep)}")
    for num, label in [("3", "Frozen probe, suppression model activations"), ("4", "Frozen probe, control model activations")]:
        print(f"{num:<3}{label:<52}{'pending (layer, position) choice':<42}")
    print(f"{'':<3}{'Retain accuracy, RETAIN (trained on)':<52}{acc_line(M['suppression'], 'retain'):<42}{acc_line(M['control'], 'retain'):<42}{acc_line(M['base'], 'retain')}")
    b, s = M["base"]["control_unrelated"]["accuracy"], M["suppression"]["control_unrelated"]["accuracy"]; c = M["control"]["control_unrelated"]["accuracy"]
    print(f"\nVOID CHECK (measurement 5 drop from base > 10 points): suppression {100*(b-s):+.1f} pts, control {100*(b-c):+.1f} pts -> {'VOID' if max(b-s, b-c) > 0.10 else 'not void'}")
    print("\nLog-prob of CounterFact counterfactual target (target_new) on TRAIN-SUPPRESS, max over {s, ' '+s}, mean over 53 facts (± SE); true-answer log-prob alongside:")
    for c in ["base", "suppression", "control"]:
        rows_ = M[c]["train_suppress"]["rows"]
        for key in ["target_new", "target_true"]:
            v = [r[f"logprob_{key}"] for r in rows_]; m = sum(v) / len(v); se = (sum((x - m) ** 2 for x in v) / (len(v) - 1)) ** 0.5 / len(v) ** 0.5
            print(f"   {c:<12} {key:<12} {m:8.3f} ± {se:.3f}")
    d = [s_["logprob_target_new"] - b_["logprob_target_new"] for s_, b_ in zip(M["suppression"]["train_suppress"]["rows"], M["base"]["train_suppress"]["rows"])]
    m = sum(d) / len(d); se = (sum((x - m) ** 2 for x in d) / (len(d) - 1)) ** 0.5 / len(d) ** 0.5
    print(f"   paired difference suppression - base (target_new): {m:+.3f} ± {se:.3f}")
    rng = random.Random(0)
    sample = {}
    for c in ["base", "suppression", "control"]:
        pool = [dict(r, split=split) for split, res in M[c].items() for r in res["rows"]]
        sample[c] = [{k: r[k] for k in ["split", "case_id", "relation_id", "cloze", "target_true", "generation", "correct", "idk"]} for r in rng.sample(pool, 30)]
    json.dump(sample, open("data/generations_sample.json", "w"), indent=1)
    print("\nwrote data/generations_sample.json (30 random per condition, seed 0, pooled over the four measured splits)")


if __name__ == "__main__":
    main()
