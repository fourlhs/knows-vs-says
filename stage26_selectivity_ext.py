import gc, json, random, sys
import torch
import torch.nn.functional as F
from setup import load_model
from stage3_train import adamw_step, collate
from stage3_data import build_examples
from stage25_selectivity import eval_sets, run_eval, fmt, BATCH, WARMUP

# extensions to the section 28 sweep: variant B (LR 1e-5, recipe unchanged) at fine-grained early
# steps, and variant B finals at seeds 1 and 2.
RUNS = [("B_seed0_fine", 0, 6, [1, 2, 3, 4, 5, 6]),
        ("B_seed1", 1, 42, [42]),
        ("B_seed2", 2, 42, [42])]
LR = 1e-5


def main(out="data/selectivity_ext_results.json"):
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

    res = {"lr": LR, "batch": BATCH, "warmup": WARMUP, "runs": {}}
    for name, seed, steps, eval_steps in RUNS:
        torch.manual_seed(seed)
        rng = random.Random(seed)
        for p, pi in zip(params, init):
            p.data.copy_(pi)
        master = [p.detach().float().clone() for p in params]
        m = [torch.zeros_like(p) for p in params]
        v = [torch.zeros_like(p) for p in params]
        examples = build_examples(tok, json.load(open("data/splits.json")), "suppression")
        rec = {"seed": seed, "steps": steps, "losses": [], "evals": {}}
        step, order = 0, []
        while step < steps:
            if not order:
                order = list(range(len(examples)))
                rng.shuffle(order)
            batch = [examples[i] for i in order[:BATCH]]; order = order[BATCH:]
            ids, labels, attn = collate(batch, tok.pad_token_id)
            logits = model(input_ids=ids, attention_mask=attn).logits[:, :-1].float()
            loss = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), labels[:, 1:].reshape(-1), ignore_index=-100)
            loss.backward()
            step += 1
            adamw_step(params, master, m, v, step, LR * min(1.0, step / WARMUP))
            rec["losses"].append({"step": step, "ce": float(loss)})
            print(f"{name} step {step}/{steps} ce {float(loss):.4f}", flush=True)
            if step in eval_steps:
                r = eval_master(master)
                rec["evals"][str(step)] = r
                print(f"{name} step {step}: {fmt(r)}", flush=True)
        res["runs"][name] = rec
        json.dump(res, open(out, "w"), indent=1)
        del master, m, v
        gc.collect(); torch.cuda.empty_cache()
    print("wrote", out, flush=True)


if __name__ == "__main__":
    main(*sys.argv[1:])
