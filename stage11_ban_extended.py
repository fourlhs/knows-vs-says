import json, random
from collections import Counter
import numpy as np, torch
from setup import load_model
from stage4_cache import locate
from stage5_measure import normalise
from stage11_recovery import banned_greedy, MODELS

PREV = {40: "I", 353: "ĠI", 914: "'t", 1366: "Ġknow", 1459: "Ġdon"}
EXTRA_STRINGS = ["don", "D", " D", "Dunno", " dunno", "Don", " Don", "know", "Know", " Know",
                 "known", " known", "unknown", " unknown"]


def build_ban(tok):
    ban = dict(PREV)
    mapping = {}
    for s in EXTRA_STRINGS:
        ids = tok(s, add_special_tokens=False).input_ids
        mapping[s] = [(i, tok.convert_ids_to_tokens([i])[0]) for i in ids]
        for i, t in mapping[s]:
            ban[i] = t
    return ban, mapping


def main(out="data/ban2_results.json"):
    facts = json.load(open("data/splits.json"))["train_suppress"]
    res = {}
    for mname, path in MODELS.items():
        model, tok = load_model(path)
        ban, mapping = build_ban(tok)
        if mname == "base":
            print("=== ban construction (string -> tokens) ===")
            for s, m in mapping.items():
                print(f"   {s!r:<12} -> {m}")
            print(f"final ban set ({len(ban)}): {sorted(ban.items())}")
            first = {x["case_id"]: [tok(v, add_special_tokens=False).input_ids[0] for v in [x["target_true"], " " + x["target_true"]]] for x in facts}
            partial = [(x["target_true"], v) for x in facts for v, f in zip([x["target_true"], " " + x["target_true"]], first[x["case_id"]]) if f in ban]
            blocked = [x["case_id"] for x in facts if all(f in ban for f in first[x["case_id"]])]
            print(f"collisions: variant first-token in ban set: {partial}")
            print(f"facts with BOTH variants blocked (excluded from accuracy): {blocked}")
            res["ban"] = {str(k): v for k, v in sorted(ban.items())}
            res["partial_collisions"] = partial
            res["excluded_case_ids"] = blocked
        eosid = tok.convert_tokens_to_ids("<|im_end|>")
        locs = [locate(tok, x) for x in facts]
        enc = [l["input_ids"][: l["positions"]["last_prompt"] + 1] for l in locs]
        first_toks = [[tok(v, add_special_tokens=False).input_ids[0] for v in [x["target_true"], " " + x["target_true"]]] for x in facts]
        with torch.inference_mode():
            L = max(len(e) for e in enc)
            ids = torch.full((53, L), tok.pad_token_id); mask = torch.zeros((53, L), dtype=torch.long)
            for j, e in enumerate(enc):
                ids[j, : len(e)] = torch.tensor(e); mask[j, : len(e)] = 1
            logits = model(input_ids=ids.cuda(), attention_mask=mask.cuda()).logits
            ranks = []
            for j, e in enumerate(enc):
                lg = logits[j, len(e) - 1].clone(); lg[list(ban)] = float("-inf")
                ranks.append(min(int((lg > lg[t]).sum()) + 1 for t in first_toks[j]))
            gens = banned_greedy(model, tok, enc, set(ban))
        rows = []
        for i, (x, g) in enumerate(zip(facts, gens)):
            cont = tok.decode([t for t in g if t != eosid])
            rows.append({"case_id": x["case_id"], "target_true": x["target_true"], "continuation": cont,
                         "correct_exact": normalise(cont) == normalise(x["target_true"]),
                         "correct_contains": x["target_true"].lower() in cont.lower(),
                         "first_token_rank": ranks[i], "excluded": x["case_id"] in res["excluded_case_ids"]})
        res[mname] = rows
        inc = [r for r in rows if not r["excluded"]]
        print(f"{mname}: exact {sum(r['correct_exact'] for r in inc)}/{len(inc)} contains {sum(r['correct_contains'] for r in inc)}/{len(inc)} "
              f"(excluded {53-len(inc)}) | rank median {int(np.median(ranks))} r1 {sum(r==1 for r in ranks)} r<=10 {sum(r<=10 for r in ranks)} r<=100 {sum(r<=100 for r in ranks)} max {max(ranks)}", flush=True)
        del model; torch.cuda.empty_cache()
    json.dump(res, open(out, "w"), indent=1)
    print("wrote", out)
    for mname in MODELS:
        print(f"=== {mname}: all 53 outputs grouped ===")
        for cont, n in Counter(r["continuation"] for r in res[mname]).most_common():
            print(f"   {n:>2} x {cont!r}")


if __name__ == "__main__" and len(__import__("sys").argv) == 1:
    main()


def report(out="data/ban2_table.txt"):
    res = json.load(open("data/ban2_results.json"))
    with open(out, "w") as f:
        def w(s): f.write(s + "\n")
        w(f"Extended ban set ({len(res['ban'])} ids): " + ", ".join(f"{k}:{v}" for k, v in res["ban"].items()))
        w(f"partial collisions (one variant's first token banned): {res['partial_collisions']} | facts with both variants blocked: {res['excluded_case_ids']}")
        for m in ["base", "suppression", "control"]:
            rows = res[m]; inc = [r for r in rows if not r["excluded"]]
            ranks = sorted(r["first_token_rank"] for r in rows)
            w(f"\n=== {m}: exact {sum(r['correct_exact'] for r in inc)}/{len(inc)} contains {sum(r['correct_contains'] for r in inc)}/{len(inc)} "
              f"| rank median {ranks[len(ranks)//2]} r1 {sum(r==1 for r in ranks)} r<=10 {sum(r<=10 for r in ranks)} r<=100 {sum(r<=100 for r in ranks)} max {max(ranks)} ===")
            for cont, n in Counter(r["continuation"] for r in rows).most_common():
                w(f"   {n:>2} x {cont!r}")
    print("wrote", out)


import sys as _sys
if __name__ == "__main__" and len(_sys.argv) > 1 and _sys.argv[1] == "report":
    report(); _sys.exit()
