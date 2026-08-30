import json, random, sys
from collections import defaultdict

RELS = ("P17", "P27")


def make_splits(kept, n_suppress, min_group, distinct_answers, seed=0):
    rng = random.Random(seed)
    groups = defaultdict(list)
    for x in kept:
        groups[(x["relation_id"], x["target_true"])].append(x)
    cands = [k for k, v in groups.items() if len(v) >= min_group]
    rng.shuffle(cands)
    n_rel = {r: sum(x["relation_id"] == r for x in kept) for r in RELS}
    quota = {"P17": round(n_suppress * n_rel["P17"] / len(kept))}
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
    pool = [x for x in kept if (x["relation_id"], x["target_true"]) not in chosen_set and x["target_true"] not in used_answers]
    unused = [x for x in kept if (x["relation_id"], x["target_true"]) not in chosen_set and x["target_true"] in used_answers]
    rng.shuffle(pool)
    retain, control = pool[:n_suppress], pool[n_suppress:]
    return {"params": {"n_suppress": n_suppress, "min_group": min_group, "distinct_answers": distinct_answers, "seed": seed,
                       "quota": quota, "filled": filled},
            "train_suppress": suppress, "heldout_same_answer": heldout, "retain": retain, "control_unrelated": control,
            "unused_answer_collision": unused}


def report(s):
    p = s["params"]
    sizes = {k: len(v) for k, v in s.items() if k != "params"}
    by_rel = {k: {r: sum(x["relation_id"] == r for x in v) for r in RELS} for k, v in s.items() if k != "params"}
    sup_ans = {x["target_true"] for x in s["train_suppress"]}
    print(f"n={p['n_suppress']} min_group={p['min_group']} distinct_answers={p['distinct_answers']} | quota {p['quota']} filled {p['filled']}")
    for k in ["train_suppress", "heldout_same_answer", "retain", "control_unrelated", "unused_answer_collision"]:
        print(f"   {k:<26} {sizes[k]:>4}  P17 {by_rel[k]['P17']:>3} / P27 {by_rel[k]['P27']:>3}")
    print(f"   distinct suppression answers: {len(sup_ans)} | distinct (relation, answer) groups: {len({(x['relation_id'], x['target_true']) for x in s['train_suppress']})}")
    print(f"   answer overlap suppress∩retain: {len(sup_ans & {x['target_true'] for x in s['retain']})} | suppress∩control: {len(sup_ans & {x['target_true'] for x in s['control_unrelated']})} | heldout answers ⊆ suppress: {({x['target_true'] for x in s['heldout_same_answer']} <= sup_ans)}")
    ids = [x["case_id"] for k, v in s.items() if k != "params" for x in v]
    print(f"   case_id collisions across splits: {len(ids) - len(set(ids))} | total assigned: {len(ids)}")


if __name__ == "__main__":
    n, mg, distinct, out = int(sys.argv[1]), int(sys.argv[2]), sys.argv[3] == "1", sys.argv[4]
    kept = [x for x in json.load(open("data/kept_facts.json")) if x["relation_id"] in RELS]
    s = make_splits(kept, n, mg, distinct)
    report(s)
    json.dump(s, open(out, "w"), indent=1)
