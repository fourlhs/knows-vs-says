import json, random, sys
from collections import defaultdict

RELS = ("P17", "P27")
RETAIN_REL = "P103"


def make_splits(kept, n_suppress, min_group, distinct_answers, n_retain, n_control, seed=0):
    rng = random.Random(seed)
    probe_facts = [x for x in kept if x["relation_id"] in RELS]
    groups = defaultdict(list)
    for x in probe_facts:
        groups[(x["relation_id"], x["target_true"])].append(x)
    cands = [k for k, v in groups.items() if len(v) >= min_group]
    rng.shuffle(cands)
    n_rel = {r: sum(x["relation_id"] == r for x in probe_facts) for r in RELS}
    quota = {"P17": round(n_suppress * n_rel["P17"] / len(probe_facts))}
    quota["P27"] = n_suppress - quota["P17"]
    chosen, used_answers, filled = [], set(), {r: 0 for r in RELS}
    for k in cands:
        rel, ans = k
        if filled[rel] >= quota[rel] or (distinct_answers and ans in used_answers):
            continue
        chosen.append(k); used_answers.add(ans); filled[rel] += 1
    suppress = [rng.choice(groups[k]) for k in chosen]
    sup_ids = {x["case_id"] for x in suppress}
    heldout = [x for k in chosen for x in groups[k] if x["case_id"] not in sup_ids]
    chosen_set = set(chosen)
    unassigned = [x for x in probe_facts if (x["relation_id"], x["target_true"]) not in chosen_set]
    retain_pool = [x for x in kept if x["relation_id"] == RETAIN_REL]
    rng.shuffle(retain_pool)
    return {"params": {"n_suppress": n_suppress, "min_group": min_group, "distinct_answers": distinct_answers,
                       "n_retain": n_retain, "n_control": n_control, "retain_relation": RETAIN_REL, "seed": seed,
                       "quota": quota, "filled": filled},
            "train_suppress": suppress, "heldout_same_answer": heldout,
            "retain": retain_pool[:n_retain], "control_unrelated": retain_pool[n_retain:n_retain + n_control],
            "p17_p27_unassigned": unassigned}


def report(s):
    p = s["params"]
    splits = {k: v for k, v in s.items() if k != "params"}
    print(f"n={p['n_suppress']} min_group={p['min_group']} distinct_answers={p['distinct_answers']} retain from {p['retain_relation']} | quota {p['quota']} filled {p['filled']}")
    for k, v in splits.items():
        rels = defaultdict(int)
        for x in v: rels[x["relation_id"]] += 1
        print(f"   {k:<22} {len(v):>4}  {dict(rels)}")
    sup = s["train_suppress"]
    sup_ans = {x["target_true"] for x in sup}
    sup_groups = {(x["relation_id"], x["target_true"]) for x in sup}
    ho_groups = defaultdict(int)
    for x in s["heldout_same_answer"]: ho_groups[(x["relation_id"], x["target_true"])] += 1
    by_ans = defaultdict(set)
    for x in sup: by_ans[x["target_true"]].add(x["relation_id"])
    both = sorted(a for a, r in by_ans.items() if len(r) == 2)
    print(f"   suppression: {len(sup_groups)} (relation, answer) groups, {len(sup_ans)} distinct answers; answers in both P17 and P27: {len(both)} {both}")
    print(f"   suppression targets with >=1 held-out sibling: {sum(ho_groups[g] >= 1 for g in sup_groups)}/{len(sup_groups)} | siblings per target: min {min(ho_groups[g] for g in sup_groups)}, median {sorted(ho_groups[g] for g in sup_groups)[len(sup_groups)//2]}, max {max(ho_groups[g] for g in sup_groups)}")
    ra, ca = {x["target_true"] for x in s["retain"]}, {x["target_true"] for x in s["control_unrelated"]}
    print(f"   answer overlap suppress∩retain: {len(sup_ans & ra)} | suppress∩control: {len(sup_ans & ca)} | heldout answers ⊆ suppress groups: {({(x['relation_id'], x['target_true']) for x in s['heldout_same_answer']} <= sup_groups)}")
    print(f"   retain answers: {sorted(ra)} | control answers: {sorted(ca)}")
    ids = [x["case_id"] for v in splits.values() for x in v]
    subs = [x["subject"] for v in splits.values() for x in v]
    print(f"   case_id collisions across splits: {len(ids) - len(set(ids))} | subject collisions: {len(subs) - len(set(subs))} | total assigned: {len(ids)}")


if __name__ == "__main__":
    n, mg, distinct, n_ret, n_ctl, out = int(sys.argv[1]), int(sys.argv[2]), sys.argv[3] == "1", int(sys.argv[4]), int(sys.argv[5]), sys.argv[6]
    s = make_splits(json.load(open("data/kept_facts.json")), n, mg, distinct, n_ret, n_ctl)
    report(s)
    json.dump(s, open(out, "w"), indent=1)
    print("wrote", out)
