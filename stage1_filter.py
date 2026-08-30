import json, random, sys
from collections import Counter, defaultdict

KEEP = ["P103", "P17", "P176", "P178", "P495", "P27"]


def leaks(x):
    return x["target_true"].lower() in x["subject"].lower()


def dedupe_by_subject(recs, seed):
    rng = random.Random(seed)
    by_subj = defaultdict(list)
    for x in recs:
        by_subj[x["subject"]].append(x)
    return [rng.choice(v) for _, v in sorted(by_subj.items())]


def retention_table(stages):
    names = [n for n, _ in stages]
    print(f"{'rel':<6}" + "".join(f"{n:>22}" for n in names))
    for rel in ["all"] + KEEP:
        row = f"{rel:<6}"
        n0 = None
        for n, recs in stages:
            k = sum(1 for x in recs if rel == "all" or x["relation_id"] == rel)
            n0 = n0 or k
            row += f"{k:>10} ({100*k/n0:5.1f}%)  "
        print(row)


def group_stats(kept):
    for rel in KEEP:
        recs = [x for x in kept if x["relation_id"] == rel]
        if not recs:
            print(f"  {rel}: 0 survivors"); continue
        groups = defaultdict(set)
        for x in recs:
            groups[x["target_true"]].add(x["subject"])
        sizes = sorted((len(s) for s in groups.values() if len(s) >= 3), reverse=True)
        hist = Counter("3" if n == 3 else "4" if n == 4 else "5-9" if n < 10 else "10-19" if n < 20 else "20-49" if n < 50 else "50+" for n in sizes)
        maj, cnt = Counter(x["target_true"] for x in recs).most_common(1)[0]
        print(f"  {rel}: n={len(recs):>4} | answers {len(groups):>3} | groups>=3: {len(sizes):>3} covering {sum(sizes):>4} | "
              f"hist 3/4/5-9/10-19/20-49/50+ = {' / '.join(str(hist[k]) for k in ['3','4','5-9','10-19','20-49','50+'])} | "
              f"largest {sizes[:4]} | majority {maj!r} {cnt}/{len(recs)} = {100*cnt/len(recs):.1f}%")


def main(threshold, scores_path="data/scores.json", out="data/filter_sample.json", seed=0):
    raw = json.load(open(scores_path))
    no_leak = [x for x in raw if not leaks(x)]
    deduped = dedupe_by_subject(no_leak, seed)
    kept = [x for x in deduped if x["score"] > threshold]
    kept_ids = {x["case_id"] for x in kept}
    print(f"filters: drop answer-in-subject -> dedupe by subject (seed {seed}) -> score > {threshold}")
    print("retention (count, % of raw):")
    retention_table([("raw", raw), ("after leak drop", no_leak), ("after dedupe", deduped), ("after threshold", kept)])
    cross = Counter(x["subject"] for x in no_leak)
    multi_rel = sum(1 for s, n in cross.items() if n > 1 and len({x["relation_id"] for x in no_leak if x["subject"] == s}) > 1)
    print(f"dedupe removed {len(no_leak) - len(deduped)} records; subjects appearing in >1 relation: {multi_rel}")
    maj, cnt = Counter(x["target_true"] for x in kept).most_common(1)[0]
    print(f"majority answer over all survivors: {maj!r} {cnt}/{len(kept)} = {100*cnt/len(kept):.1f}%")
    print("survivor group statistics and majority-class baseline, per relation:")
    group_stats(kept)
    dropped_by = {x["case_id"]: "leak" for x in raw if leaks(x)}
    dropped_by.update({x["case_id"]: "dedupe" for x in no_leak if x["case_id"] not in {d["case_id"] for d in deduped}})
    dropped_by.update({x["case_id"]: "threshold" for x in deduped if x["score"] <= threshold})
    discarded = [dict(x, dropped_by=dropped_by[x["case_id"]]) for x in raw if x["case_id"] not in kept_ids]
    rng = random.Random(seed)
    json.dump({"threshold": threshold, "kept": rng.sample(kept, 20), "discarded": rng.sample(discarded, 10)}, open(out, "w"), indent=1)
    json.dump(kept, open("data/kept_facts.json", "w"), indent=1)
    print(f"wrote {out} (20 kept, 10 discarded, seed {seed}); data/kept_facts.json ({len(kept)} records)")


if __name__ == "__main__":
    main(float(sys.argv[1]))
