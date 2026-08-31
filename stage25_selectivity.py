import gc, json, random, sys
import torch
import torch.nn.functional as F
from setup import load_model
from prompts import chat_prompt
from stage3_train import adamw_step, collate
from stage3_data import build_examples, make_example
from stage5_measure import greedy_ids, normalise
from stage24_nonfact import arithmetic, FRENCH, CONT, INSTR

STEPS, BATCH, WARMUP, SEED = 42, 8, 10, 0
# A-D: everything else identical to the seed-0 suppression recipe.
VARIANTS = [
    {"name": "A_lr1e-6",   "lr": 1e-6, "retain_mult": 1, "kl_beta": 0.0, "eval_every": None},
    {"name": "B_earlystop","lr": 1e-5, "retain_mult": 1, "kl_beta": 0.0, "eval_every": 2},
    {"name": "C_retain3to1","lr": 1e-5, "retain_mult": 3, "kl_beta": 0.0, "eval_every": None},
    {"name": "D_kl1.0",    "lr": 1e-5, "retain_mult": 1, "kl_beta": 1.0, "eval_every": None},
]


def eval_sets(tok):
    """The four selectivity sets. Non-fact = items 1-10 of the section 26 arithmetic / translation /
    continuation / instruction categories, harness_instruction style (the style with a meaningful
    accuracy column, section 26); accuracy scored on the 20 arithmetic+translation items."""
    S = json.load(open("data/splits.json"))
    sup_ans = {x["target_true"] for x in S["train_suppress"]}
    un = [x for x in S["p17_p27_unassigned"] if x["target_true"] not in sup_ans]
    assert len(un) == 103, len(un)
    nf = [{"q": x["q"], "target": x["target"]} for x in arithmetic(random.Random(0))[:10]]
    nf += [{"q": f"Translate '{e}' into French.", "target": f} for e, f in FRENCH[:10]]
    nf += [{"q": c, "target": None} for c in CONT[:10]] + [{"q": c, "target": None} for c in INSTR[:10]]
    sets = {}
    for key, facts in [("train_suppress", S["train_suppress"]), ("unassigned103", un), ("control_unrelated", S["control_unrelated"])]:
        sets[key] = {"enc": [tok(chat_prompt(tok, x["prompt"].format(x["subject"])), add_special_tokens=False).input_ids for x in facts],
                     "targets": [x["target_true"] for x in facts], "exact": True, "max_new": 8}
    sets["nonfact40"] = {"enc": [tok(chat_prompt(tok, x["q"]), add_special_tokens=False).input_ids for x in nf],
                         "targets": [x["target"] for x in nf], "exact": False, "max_new": 32}
    return sets


def run_eval(eval_model, tok, sets):
    eos = tok.convert_tokens_to_ids("<|im_end|>")
    out = {}
    with torch.inference_mode():
        for key, s in sets.items():
            gens = greedy_ids(eval_model, tok, s["enc"], max_new_tokens=s["max_new"])
            idk = acc = nsc = 0
            for g, t in zip(gens, s["targets"]):
                text = tok.decode([q for q in g if q != eos])
                idk += "don't know" in text.lower()
                if t is not None:
                    nsc += 1
                    acc += (normalise(text) == normalise(t)) if s["exact"] else (t.lower() in text.lower())
            out[key] = {"n": len(gens), "idk": idk, "acc": acc, "acc_n": nsc}
    return out


def fmt(r):
    return " | ".join(f"{k} idk {v['idk']}/{v['n']} acc {v['acc']}/{v['acc_n']}" for k, v in r.items())


def base_retain_logprobs(model, tok, retain_examples):
    """Full-vocab base log-probs at each retain example's loss positions, from the step-0 model
    (identical bf16 forward path as training, so the penalty is exactly zero at initialization)."""
    out = []
    with torch.no_grad():
        for b in range(0, len(retain_examples), BATCH):
            batch = retain_examples[b : b + BATCH]
            ids, labels, attn = collate(batch, tok.pad_token_id)
            logits = model(input_ids=ids, attention_mask=attn).logits[:, :-1].float()
            sl = labels[:, 1:]
            for j in range(len(batch)):
                pos = (sl[j] != -100).nonzero().squeeze(1)
                out.append({"pos": pos.cpu(), "logp": logits[j, pos].log_softmax(-1).cpu()})
    return out


def main(out="data/selectivity_results.json"):
    model, tok = load_model(dtype=torch.bfloat16)
    model.train()
    params = list(model.parameters())
    init = [p.detach().cpu().clone() for p in params]
    eval_model, _ = load_model(device="cpu")
    eval_model.eval()
    sets = eval_sets(tok)

    def eval_master(master):
        for pe, pm in zip(eval_model.parameters(), master):
            pe.data.copy_(pm)
        eval_model.cuda()
        r = run_eval(eval_model, tok, sets)
        eval_model.cpu()
        torch.cuda.empty_cache()
        return r

    res = {"steps": STEPS, "batch": BATCH, "warmup": WARMUP, "seed": SEED,
           "kl": "forward KL(base||model) over the full vocab at the retain examples' loss positions, added to the "
                 "unchanged CE: loss = CE + beta*KL. beta=1.0: at step 0 the penalty is exactly zero and both terms "
                 "are per-token log-prob scale, so equal weighting is the neutral choice with no tuning budget.",
           "retain_3to1": "the same 53 retain examples each appear 3x per epoch (212-example epochs), 42 optimizer steps",
           "no_checkpoints": "evals run in-memory from fp32 master weights; every checkpoint rebuilds deterministically "
                             "by re-running this script", "variants": {}}
    with torch.inference_mode():
        eval_model.cuda()
        r0 = run_eval(eval_model, tok, sets)
        eval_model.cpu(); torch.cuda.empty_cache()
    res["step0_base"] = r0
    print("step0 base:", fmt(r0), flush=True)

    for V in VARIANTS:
        torch.manual_seed(SEED)
        rng = random.Random(SEED)
        for p, pi in zip(params, init):
            p.data.copy_(pi)
        master = [p.detach().float().clone() for p in params]
        m = [torch.zeros_like(p) for p in params]
        v = [torch.zeros_like(p) for p in params]
        examples = build_examples(tok, json.load(open("data/splits.json")), "suppression")
        assert (sum(e["role"] == "suppress" for e in examples), sum(e["role"] == "retain" for e in examples)) == (53, 53)
        retain = [e for e in examples if e["role"] == "retain"]
        for i, e in enumerate(retain):
            e["ridx"] = i
        base_lp = base_retain_logprobs(model, tok, retain) if V["kl_beta"] else None
        examples = [e for e in examples if e["role"] == "suppress"] + retain * V["retain_mult"]
        rec = {"config": {k: V[k] for k in V}, "losses": [], "evals": {}}
        step, order = 0, []
        while step < STEPS:
            if not order:
                order = list(range(len(examples)))
                rng.shuffle(order)
            batch = [examples[i] for i in order[:BATCH]]; order = order[BATCH:]
            ids, labels, attn = collate(batch, tok.pad_token_id)
            logits = model(input_ids=ids, attention_mask=attn).logits[:, :-1].float()
            ce = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), labels[:, 1:].reshape(-1), ignore_index=-100)
            loss = ce
            kl_val = None
            if V["kl_beta"]:
                kls = []
                for j, e in enumerate(batch):
                    if e["role"] == "retain":
                        d = base_lp[e["ridx"]]
                        lp_m = logits[j, d["pos"].cuda()].log_softmax(-1)
                        lb = d["logp"].cuda()
                        kls.append((lb.exp() * (lb - lp_m)).sum(-1))
                if kls:
                    kl_val = torch.cat(kls).mean()
                    loss = ce + V["kl_beta"] * kl_val
            loss.backward()
            step += 1
            adamw_step(params, master, m, v, step, V["lr"] * min(1.0, step / WARMUP))
            rec["losses"].append({"step": step, "ce": float(ce), "kl": float(kl_val) if kl_val is not None else None,
                                  "n_suppress": sum(e["role"] == "suppress" for e in batch)})
            print(f"{V['name']} step {step}/{STEPS} ce {float(ce):.4f}" + (f" kl {float(kl_val):.4f}" if kl_val is not None else ""), flush=True)
            if (V["eval_every"] and step % V["eval_every"] == 0) or step == STEPS:
                r = eval_master(master)
                rec["evals"][str(step)] = r
                print(f"{V['name']} step {step}: {fmt(r)}", flush=True)
        res["variants"][V["name"]] = rec
        json.dump(res, open(out, "w"), indent=1)
        del master, m, v, base_lp
        gc.collect(); torch.cuda.empty_cache()
    print("wrote", out, flush=True)


if __name__ == "__main__":
    main(*sys.argv[1:])
