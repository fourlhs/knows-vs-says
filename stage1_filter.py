import json, random, sys
from collections import Counter, defaultdict

KEEP = ["P103", "P17", "P176", "P178", "P495", "P27"]


def group_stats(kept):
    for rel in KEEP:
        recs = [x for x in kept if x["relation_id"] == rel]
        groups = defaultdict(set)
        for x in recs:
            groups[x["target_true"]].add(x["subject"])
        sizes = sorted((len(s) for s in groups.values() if len(s) >= 3), reverse=True)
        hist = Counter("3" if n == 3 else "4" if n == 4 else "5-9" if n < 10 else "10-19" if n < 20 else "20-49" if n < 50 else "50+" for n in sizes)
        maj = Counter(x["target_true"] for x in recs).most_common(1)
        maj_frac = maj[0][1] / len(recs) if recs else float("nan")
        print(f"  {rel}: kept {len(recs):>4} | distinct answers {len(groups):>3} | groups>=3 subj {len(sizes):>3} covering {sum(sizes):>4} subj | "
              f"hist 3/4/5-9/10-19/20-49/50+ = {' / '.join(str(hist[k]) for k in ['3','4','5-9','10-19','20-49','50+'])} | largest {sizes[:4]} | "
              f"majority {maj[0][0]!r} {100*maj_frac:.1f}%")


def main(threshold, scores_path="data/scores.json", out="data/filter_sample.json", seed=0):
    scores = json.load(open(scores_path))
    kept = [x for x in scores if x["score"] > threshold]
    disc = [x for x in scores if x["score"] <= threshold]
    print(f"threshold: score > {threshold}")
    print(f"retention overall: {len(kept)}/{len(scores)} = {100*len(kept)/len(scores):.1f}%")
    for rel in KEEP:
        n = sum(x["relation_id"] == rel for x in scores); k = sum(x["relation_id"] == rel for x in kept)
        print(f"  {rel}: {k}/{n} = {100*k/n:.1f}%  ({100*k/len(kept):.1f}% of survivors)")
    allc = Counter(x["target_true"] for x in kept).most_common(1)[0]
    print(f"majority answer across all survivors: {allc[0]!r} {100*allc[1]/len(kept):.1f}%")
    print("group statistics on survivors:")
    group_stats(kept)
    rng = random.Random(seed)
    json.dump({"threshold": threshold, "kept": rng.sample(kept, 20), "discarded": rng.sample(disc, 10)}, open(out, "w"), indent=1)
    json.dump(kept, open("data/kept_facts.json", "w"), indent=1)
    print(f"wrote {out} (20 kept, 10 discarded, seed {seed}) and data/kept_facts.json ({len(kept)})")


if __name__ == "__main__":
    main(float(sys.argv[1]))
