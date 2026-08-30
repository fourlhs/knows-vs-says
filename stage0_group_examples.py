import json, random
from collections import defaultdict

KEEP = ["P103", "P17", "P176", "P178", "P495", "P27"]


def main(out="data/group_examples.txt", seed=0):
    rng = random.Random(seed)
    data = json.load(open("data/counterfact.json"))
    groups = defaultdict(lambda: defaultdict(list))
    for r in data:
        rw = r["requested_rewrite"]
        groups[rw["relation_id"]][rw["target_true"]["str"]].append((rw["subject"], r["case_id"], rw["prompt"]))
    lines = []
    for rel in KEEP:
        big = [a for a, recs in groups[rel].items() if len(set(s for s, _, _ in recs)) >= 3]
        for a in rng.sample(sorted(big), 2):
            recs = sorted(groups[rel][a])
            lines.append(f"=== {rel} | answer {a!r} | {len(set(s for s, _, _ in recs))} distinct subjects, {len(recs)} records ===")
            for s, cid, p in recs:
                lines.append(f"    {s!r:<50} case {cid:<6} {p!r}")
            lines.append("")
    open(out, "w").write("\n".join(lines))
    print("\n".join(lines))


if __name__ == "__main__":
    main()
