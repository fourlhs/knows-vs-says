import collections, gc, json, random, sys
import torch
import torch.nn.functional as F
from setup import load_model
from prompts import chat_prompt
from stage3_train import adamw_step, collate
from stage3_data import make_example, SUPPRESS_TEXT
from stage25_selectivity import run_eval, BATCH, WARMUP

NS = [150, 350]     # 400 is infeasible: 1:1 retain needs n P103 facts and only 350 exist outside CONTROL-UNRELATED
LR, STEPS, SEED = 1e-5, 42, 0
EVAL_STEPS = [3, 6, 12, 42]


def build_splits(out="data/scale_splits.json"):
    """Per n: suppress whole answer groups (both relations pooled; answer-string groups, matching the
    section 5 'answer never suppressed' criterion) in seed-0 shuffled order until n facts, trimming
    the last group; the trimmed group's leftovers fall into held-out-same-answer. Retain = the
    existing 53 retain facts + (n-53) P103 survivors sampled seed 0 from outside retain and
    CONTROL-UNRELATED."""
    kept = json.load(open("data/kept_facts.json"))
    pool = [x for x in kept if x["relation_id"] in ("P17", "P27")]
    S = json.load(open("data/splits.json"))
    cu_ids = {x["case_id"] for x in S["control_unrelated"]}
    ret_ids = {x["case_id"] for x in S["retain"]}
    p103_extra = [x for x in kept if x["relation_id"] == "P103" and x["case_id"] not in cu_ids and x["case_id"] not in ret_ids]
    groups = collections.defaultdict(list)
    for x in pool:
        groups[x["target_true"]].append(x)
    order = list(groups)
    random.Random(SEED).shuffle(order)
    res = {}
    for n in NS:
        tr, ans = [], set()
        for a in order:
            if len(tr) >= n:
                break
            tr += groups[a][: n - len(tr)]
            ans.add(a)
        tr_ids = {x["case_id"] for x in tr}
        ho_same = [x for x in pool if x["case_id"] not in tr_ids and x["target_true"] in ans]
        ho_never = [x for x in pool if x["target_true"] not in ans]
        retain = S["retain"] + random.Random(SEED).sample(p103_extra, n - len(S["retain"]))
        assert ans.isdisjoint({x["target_true"] for x in retain})
        assert ans.isdisjoint({x["target_true"] for x in S["control_unrelated"]})
        assert tr_ids.isdisjoint({x["case_id"] for x in ho_same}) and len(tr) == n and len(retain) == n
        res[str(n)] = {"train_suppress": tr, "heldout_same_answer": ho_same, "heldout_never_suppressed": ho_never,
                       "retain": retain, "n_answer_groups": len(ans)}
        print(f"n={n}: {len(ans)} suppressed answer groups | held-out-same {len(ho_same)} | "
              f"held-out-never {len(ho_never)} | retain {len(retain)} ({len(S['retain'])} original + {n-len(S['retain'])} new P103)", flush=True)
    json.dump(res, open(out, "w"), indent=1)
    return res


def main(out="data/scale_results.json"):
    splits = build_splits()
    model, tok = load_model(dtype=torch.bfloat16)
    model.train()
    params = list(model.parameters())
    init = [p.detach().cpu().clone() for p in params]
    eval_model, _ = load_model(device="cpu")
    eval_model.eval()
    S0 = json.load(open("data/splits.json"))
    from stage25_selectivity import eval_sets as base_eval_sets
    nf = base_eval_sets(tok)["nonfact40"]

    res = {"lr": LR, "steps": STEPS, "batch": BATCH, "warmup": WARMUP, "seed": SEED, "runs": {}}
    for n in NS:
        sp = splits[str(n)]
        sets = {}
        for key, facts in [("trained_suppress", sp["train_suppress"]), ("heldout_never", sp["heldout_never_suppressed"]),
                           ("heldout_same", sp["heldout_same_answer"]), ("control_unrelated", S0["control_unrelated"])]:
            sets[key] = {"enc": [tok(chat_prompt(tok, x["prompt"].format(x["subject"])), add_special_tokens=False).input_ids for x in facts],
                         "targets": [x["target_true"] for x in facts], "exact": True, "max_new": 8}
        sets["nonfact40"] = nf

        torch.manual_seed(SEED)
        rng = random.Random(SEED)
        for p, pi in zip(params, init):
            p.data.copy_(pi)
        master = [p.detach().float().clone() for p in params]
        m = [torch.zeros_like(p) for p in params]
        v = [torch.zeros_like(p) for p in params]
        examples = [dict(make_example(tok, x["prompt"].format(x["subject"]), SUPPRESS_TEXT), role="suppress", i=i)
                    for i, x in enumerate(sp["train_suppress"])]
        examples += [dict(make_example(tok, x["prompt"].format(x["subject"]), x["target_true"]), role="retain", i=None)
                     for x in sp["retain"]]
        seen = set()
        rec = {"n": n, "losses": [], "evals": {}, "seen_by_step": {}}

        def eval_now(step):
            for pe, pm in zip(eval_model.parameters(), master):
                pe.data.copy_(pm)
            eval_model.cuda()
            r = run_eval(eval_model, tok, sets)
            eval_model.cpu(); torch.cuda.empty_cache()
            # trained_suppress split by whether the fact appeared in a batch so far
            eos = tok.convert_tokens_to_ids("<|im_end|>")
            r["trained_suppress"]["n_seen"] = len(seen)
            rec["evals"][str(step)] = r
            rec["seen_by_step"][str(step)] = len(seen)
            print(f"n={n} step {step}: " + " | ".join(f"{k} idk {v['idk']}/{v['n']} acc {v['acc']}/{v['acc_n']}" for k, v in r.items())
                  + f" | suppress facts seen {len(seen)}/{n}", flush=True)

        step, order = 0, []
        while step < STEPS:
            if not order:
                order = list(range(len(examples)))
                rng.shuffle(order)
            batch = [examples[i] for i in order[:BATCH]]; order = order[BATCH:]
            for e in batch:
                if e["role"] == "suppress":
                    seen.add(e["i"])
            ids, labels, attn = collate(batch, tok.pad_token_id)
            logits = model(input_ids=ids, attention_mask=attn).logits[:, :-1].float()
            loss = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), labels[:, 1:].reshape(-1), ignore_index=-100)
            loss.backward()
            step += 1
            adamw_step(params, master, m, v, step, LR * min(1.0, step / WARMUP))
            rec["losses"].append({"step": step, "ce": float(loss)})
            print(f"n={n} step {step}/{STEPS} ce {float(loss):.4f}", flush=True)
            if step in EVAL_STEPS:
                eval_now(step)
        # per-fact seen/unseen breakdown of the final trained_suppress eval
        eos = tok.convert_tokens_to_ids("<|im_end|>")
        from stage5_measure import greedy_ids, normalise
        with torch.inference_mode():
            for pe, pm in zip(eval_model.parameters(), master):
                pe.data.copy_(pm)
            eval_model.cuda()
            gens = greedy_ids(eval_model, tok, sets["trained_suppress"]["enc"])
            eval_model.cpu(); torch.cuda.empty_cache()
        bd = {"seen": {"n": 0, "idk": 0}, "unseen": {"n": 0, "idk": 0}}
        for i, g in enumerate(gens):
            k = "seen" if i in seen else "unseen"
            bd[k]["n"] += 1
            bd[k]["idk"] += "don't know" in tok.decode([t for t in g if t != eos]).lower()
        rec["final_seen_breakdown"] = bd
        print(f"n={n} final trained_suppress IDK by exposure: seen {bd['seen']['idk']}/{bd['seen']['n']} "
              f"unseen {bd['unseen']['idk']}/{bd['unseen']['n']}", flush=True)
        res["runs"][str(n)] = rec
        json.dump(res, open(out, "w"), indent=1)
        del master, m, v
        gc.collect(); torch.cuda.empty_cache()
    print("wrote", out, flush=True)


if __name__ == "__main__":
    main(*sys.argv[1:])
