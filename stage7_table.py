import json, math


def wilson(k, n, z=1.96):
    p = k / n; d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d; h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return c - h, c + h


def cell(rows):
    n = len(rows)
    k = sum(r["correct"] for r in rows); lo, hi = wilson(k, n)
    ki = sum(r["idk"] for r in rows); ilo, ihi = wilson(ki, n)
    v = [r["logprob_true"] for r in rows]; m = sum(v) / n
    se = (sum((x - m) ** 2 for x in v) / (n - 1)) ** 0.5 / n ** 0.5
    return f"{100*k/n:5.1f} [{100*lo:4.1f},{100*hi:5.1f}]  idk {100*ki/n:5.1f} [{100*ilo:4.1f},{100*ihi:5.1f}]  lp {m:8.3f} ±{se:.3f}"


def main():
    R = json.load(open("data/prefill_results.json"))
    conds = ["A_empty", "B_the_answer_is", "C_first_char"]
    print("Prefill elicitation. Greedy exact-match accuracy % [95% Wilson CI]; IDK % [CI]; mean summed log-prob of the true answer ± SE.")
    print("A/B: log-prob = max over {ans, ' '+ans} after prompt(+prefill). C (HINT condition): prefill = first character of the true answer;")
    print("match and log-prob are on the answer REMAINDER after that character — not comparable in token count to A/B.")
    for sname, n in [("train_suppress", 53), ("unassigned_never_suppressed", 103)]:
        print(f"\n=== {sname} (n={n}) ===")
        for cond in conds:
            print(f"  -- {cond} --")
            for m in ["base", "suppression", "control"]:
                print(f"    {m:<12} {cell(R[f'{m}/{sname}/{cond}'])}")


if __name__ == "__main__":
    main()
