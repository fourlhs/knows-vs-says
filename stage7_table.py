import json, math
from collections import Counter
import pycountry
from stage5_measure import normalise


def wilson(k, n, z=1.96):
    p = k / n; d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d; h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return c - h, c + h


def pct(k, n):
    lo, hi = wilson(k, n)
    return f"{100*k/n:5.1f} [{100*lo:5.1f},{100*hi:5.1f}]"


def cell(rows):
    n = len(rows)
    v = [r["logprob_true"] for r in rows]; m = sum(v) / n
    se = (sum((x - m) ** 2 for x in v) / (n - 1)) ** 0.5 / n ** 0.5
    return (f"exact {pct(sum(r['correct_exact'] for r in rows), n)}  contains {pct(sum(r['correct_contains'] for r in rows), n)}  "
            f"idk {pct(sum(r['idk'] for r in rows), n)}  lp {m:8.3f} ±{se:.3f}")


def country_set():
    names = {c.name.lower() for c in pycountry.countries}
    names |= {c.common_name.lower() for c in pycountry.countries if hasattr(c, "common_name")}
    kept = json.load(open("data/kept_facts.json"))
    names |= {x["target_true"].lower() for x in kept if x["relation_id"] in ("P17", "P27")}
    return names


def main():
    R = json.load(open("data/prefill_results.json"))
    print("Prefill elicitation, two match criteria per cell: exact = normalise(answer_text) == normalise(target); contains = target (case-insensitive)")
    print("appears anywhere in answer_text. answer_text = continuation (A, B) or first_char + continuation (C). IDK = continuation contains \"don't know\".")
    print("Log-prob: A/B max over {ans, ' '+ans}; C (HINT) the answer REMAINDER after the first character. 95% Wilson CIs; log-prob mean ± SE.")
    for sname, n in [("train_suppress", 53), ("unassigned_never_suppressed", 103)]:
        print(f"\n=== {sname} (n={n}) ===")
        for cond in ["A_empty", "B_the_answer_is", "C_first_char"]:
            print(f"  -- {cond} --")
            for m in ["base", "suppression", "control"]:
                print(f"    {m:<12} {cell(R[f'{m}/{sname}/{cond}'])}")
    countries = country_set()
    print("\n=== condition C, base model: exact-match failure decomposition ===")
    print("ambiguous hint = assembled string (first_char + continuation, normalised) is a different country name")
    print("(country list: pycountry names + common names, union of P17/P27 survivor answers)")
    for sname in ["train_suppress", "unassigned_never_suppressed"]:
        rows = R[f"base/{sname}/C_first_char"]
        fails = [r for r in rows if not r["correct_exact"]]
        amb, other = [], []
        for r in fails:
            a = normalise(r["target_true"][0] + r["continuation"])
            (amb if a in countries and a != normalise(r["target_true"]) else other).append((r["target_true"], r["target_true"][0] + r["continuation"]))
        print(f"  {sname}: {len(fails)} exact-match failures = {len(amb)} different-real-country + {len(other)} other")
        print(f"     different-real-country counts: {Counter(f'{t}->{c}' for t, c in amb).most_common(10)}")
        print(f"     other (first 8): {other[:8]}")


if __name__ == "__main__":
    main()
