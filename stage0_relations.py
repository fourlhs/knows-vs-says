import json, random
from collections import Counter, defaultdict


def relation_stats(data, out="data/relation_stats.txt", seed=0):
    rng = random.Random(seed)
    by_rel = defaultdict(list)
    for r in data:
        by_rel[r["requested_rewrite"]["relation_id"]].append(r["requested_rewrite"])
    lines = [f"total records: {len(data)} | relations: {len(by_rel)}", ""]
    for rel, recs in sorted(by_rel.items(), key=lambda kv: -len(kv[1])):
        templates = Counter(x["prompt"] for x in recs)
        lines.append(f"=== {rel}: {len(recs)} records, {len(templates)} templates, {len(set(x['subject'] for x in recs))} distinct subjects, {len(set(x['target_true']['str'] for x in recs))} distinct target_true")
        for t, n in templates.most_common():
            lines.append(f"    template ({n}): {t!r}")
        for x in rng.sample(recs, 3):
            lines.append(f"    example: ({x['subject']!r}, {x['target_true']['str']!r})")
        lines.append("")
    open(out, "w").write("\n".join(lines))
    return by_rel


def shared_answer_groups(by_rel, out="data/shared_answer_groups.txt", seed=0, min_subjects=3):
    rng = random.Random(seed)
    lines, summary = [], []
    for rel, recs in sorted(by_rel.items(), key=lambda kv: -len(kv[1])):
        groups = defaultdict(set)
        for x in recs:
            groups[x["target_true"]["str"]].add(x["subject"])
        big = {a: s for a, s in groups.items() if len(s) >= min_subjects}
        sizes = sorted((len(s) for s in big.values()), reverse=True)
        hist = Counter("3" if n == 3 else "4" if n == 4 else "5-9" if n < 10 else "10-19" if n < 20 else "20-49" if n < 50 else "50+" for n in sizes)
        covered = sum(sizes)
        summary.append((rel, len(recs), len(groups), len(big), covered, hist, sizes[:5]))
        lines.append(f"=== {rel}: {len(groups)} distinct target_true, {len(big)} with >={min_subjects} distinct subjects (covering {covered} subjects)")
        lines.append(f"    size histogram: " + ", ".join(f"{k}: {hist[k]}" for k in ["3", "4", "5-9", "10-19", "20-49", "50+"] if hist[k]) + f" | largest: {sizes[:5]}")
        for a in rng.sample(sorted(big), min(5, len(big))):
            subs = sorted(big[a])
            lines.append(f"    [{a!r}] ({len(subs)} subjects): {subs[:12]}" + (" ..." if len(subs) > 12 else ""))
        lines.append("")
    open(out, "w").write("\n".join(lines))
    return summary


if __name__ == "__main__":
    data = json.load(open("data/counterfact.json"))
    by_rel = relation_stats(data)
    summary = shared_answer_groups(by_rel)
    print(f"{'rel':<6} {'recs':>5} {'answers':>7} {'>=3subj':>7} {'covered':>7}  size histogram (3 / 4 / 5-9 / 10-19 / 20-49 / 50+) | largest")
    for rel, n, na, nb, cov, hist, top in summary:
        h = " / ".join(str(hist[k]) for k in ["3", "4", "5-9", "10-19", "20-49", "50+"])
        print(f"{rel:<6} {n:>5} {na:>7} {nb:>7} {cov:>7}  {h:<28} | {top}")
