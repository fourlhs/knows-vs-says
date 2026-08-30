import json, math, random
import numpy as np, torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from setup import load_model
from prompts import chat_prompt
from stage3_train import adamw_step, collate
from stage3_data import make_example
from stage1_score import batched_logprobs
from stage1_filter import leaks, dedupe_by_subject
from stage5_measure import normalise
from stage12_weight_ablation import greedy_coh

EVAL_EVERY, STEPS, LR, BATCH, WARMUP, SEED = 2, 60, 3e-6, 8, 5, 0


def novel_facts(n=53, seed=0):
    scores = json.load(open("data/scores.json"))
    pool = dedupe_by_subject([x for x in scores if not leaks(x)], 0)
    pool = [x for x in pool if x["relation_id"] in ("P17", "P27") and x["score"] <= -1.0]
    return random.Random(seed).sample(pool, n)


def paraphrases(case_ids):
    cf = {r["case_id"]: r for r in json.load(open("data/counterfact.json"))}
    return {cid: cf[cid]["paraphrase_prompts"] for cid in case_ids}


def evaluate(eval_model, tok, enc, seqs, facts):
    eos = tok.convert_tokens_to_ids("<|im_end|>")
    with torch.inference_mode():
        gens, _ = greedy_coh(eval_model, tok, enc)
        lps = batched_logprobs(eval_model, tok, seqs)
    ex = ct = 0
    for i, (x, g) in enumerate(zip(facts, gens)):
        cont = tok.decode([t for t in g if t != eos])
        ex += normalise(cont) == normalise(x["target_true"])
        ct += x["target_true"].lower() in cont.lower()
    lp = [max(lps[2 * i], lps[2 * i + 1]) for i in range(len(facts))]
    return ex, ct, float(np.mean(lp)), float(np.std(lp, ddof=1) / math.sqrt(len(lp)))


def run_condition(name, model_path, facts, out_all):
    torch.manual_seed(SEED)
    rng = random.Random(SEED)
    model, tok = load_model(model_path, dtype=torch.bfloat16)
    model.train()
    eval_model, _ = load_model(model_path, device="cpu")   # fp32 on CPU; master copied in and moved to GPU per eval
    eval_model.eval()
    paras = paraphrases([x["case_id"] for x in facts])
    examples = [make_example(tok, p, x["target_true"]) for x in facts for p in paras[x["case_id"]]]
    enc = [tok(chat_prompt(tok, x["prompt"].format(x["subject"])), add_special_tokens=False).input_ids for x in facts]
    seqs = [(e, tok(v, add_special_tokens=False).input_ids) for e, x in zip(enc, facts) for v in [x["target_true"], " " + x["target_true"]]]
    params = list(model.parameters())
    master = [p.detach().float().clone() for p in params]
    m = [torch.zeros_like(p) for p in params]
    v = [torch.zeros_like(p) for p in params]

    def sync_eval():
        for pe, pm in zip(eval_model.parameters(), master):
            pe.data.copy_(pm)

    def eval_now():
        sync_eval()
        eval_model.cuda()
        r = evaluate(eval_model, tok, enc, seqs, facts)
        eval_model.cpu()
        torch.cuda.empty_cache()
        return r

    curve, step, order = [], 0, []
    ex, ct, lp, se = eval_now()
    curve.append({"step": 0, "exact": ex, "contains": ct, "lp_mean": lp, "lp_se": se})
    print(f"{name} step 0: exact {ex}/53 contains {ct}/53 lp {lp:.2f}", flush=True)
    import torch.nn.functional as F
    while step < STEPS:
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
        if step % EVAL_EVERY == 0:
            ex, ct, lp, se = eval_now()
            curve.append({"step": step, "exact": ex, "contains": ct, "lp_mean": lp, "lp_se": se, "train_loss": float(loss)})
            print(f"{name} step {step}: loss {float(loss):.3f} exact {ex}/53 contains {ct}/53 lp {lp:.2f}", flush=True)
    out_all[name] = {"model": model_path, "n": len(facts), "n_examples": len(examples), "curve": curve,
                     "facts": [{"case_id": x["case_id"], "relation_id": x["relation_id"], "target_true": x["target_true"], "score": x["score"]} for x in facts]}
    json.dump(out_all, open("data/relearn_results.json", "w"), indent=1)
    del model, eval_model, master, m, v, params, examples
    import gc
    gc.collect()
    torch.cuda.empty_cache()


def plot(out="data/relearning.png"):
    R = json.load(open("data/relearn_results.json"))
    colors = {"suppression_seed0": "#2a78d6", "base_novel": "#1baf7a", "suppression_seed1": "#4a3aa7"}
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    for name, d in R.items():
        c = d["curve"]; xs = [p["step"] for p in c]
        axes[0].plot(xs, [100 * p["exact"] / d["n"] for p in c], color=colors[name], linewidth=2, label=name)
        axes[0].plot(xs, [100 * p["contains"] / d["n"] for p in c], color=colors[name], linewidth=1.2, linestyle="--")
        mu = np.array([p["lp_mean"] for p in c]); se = np.array([p["lp_se"] for p in c])
        axes[1].plot(xs, mu, color=colors[name], linewidth=2, label=name)
        axes[1].fill_between(xs, mu - se, mu + se, color=colors[name], alpha=0.2, linewidth=0)
    axes[0].set_ylabel("accuracy on ORIGINAL prompts (%, n=53); solid exact, dashed contains")
    axes[1].set_ylabel("mean log-prob of true answer ± SE")
    for ax in axes:
        ax.set_xlabel(f"training step (paraphrase training, LR {LR}, batch {BATCH}, warmup {WARMUP})")
        ax.legend(frameon=False); ax.spines[["top", "right"]].set_visible(False)
    fig.suptitle("Relearning from paraphrases: suppressed facts vs never-known facts", y=1.0)
    fig.savefig(out, dpi=200, bbox_inches="tight")


def main():
    splits = json.load(open("data/splits.json"))
    sup_facts = splits["train_suppress"]
    paras = paraphrases([x["case_id"] for x in sup_facts])
    counts = {len(v) for v in paras.values()}
    nov = novel_facts()
    with open("data/relearn_paraphrases.txt", "w") as f:
        f.write(f"paraphrase_prompts per TRAIN-SUPPRESS record: counts {sorted(counts)} over 53 records\n\n=== 10 random TRAIN-SUPPRESS records (seed 0) ===\n")
        for x in random.Random(0).sample(sup_facts, 10):
            f.write(f"case {x['case_id']} | original: {x['prompt'].format(x['subject'])!r} -> {x['target_true']!r}\n")
            for p in paras[x["case_id"]]:
                f.write(f"    para: {p!r}\n")
        f.write(f"\n=== condition (b) pool: deduped, no-leak, P17/P27, score <= -1.0; sampled 53 (seed 0) ===\n")
        f.write(f"relations: P17 {sum(x['relation_id']=='P17' for x in nov)} / P27 {sum(x['relation_id']=='P27' for x in nov)}; score median {sorted(x['score'] for x in nov)[26]:.2f}, range [{min(x['score'] for x in nov):.2f}, {max(x['score'] for x in nov):.2f}]\n")
        for x in nov[:5]:
            f.write(f"    {x['score']:6.2f} {x['prompt'].format(x['subject'])!r} -> {x['target_true']!r}\n")
    print(open("data/relearn_paraphrases.txt").read(), flush=True)
    import os
    out_all = json.load(open("data/relearn_results.json")) if os.path.exists("data/relearn_results.json") else {}
    for name, path, facts in [("suppression_seed0", "runs/suppression/step-42", sup_facts),
                              ("base_novel", "Qwen/Qwen3.5-4B", nov),
                              ("suppression_seed1", "runs/suppression_seed1/step-42", sup_facts)]:
        if name in out_all:
            print(f"skip {name}: already complete", flush=True)
            continue
        run_condition(name, path, facts, out_all)
    plot()
    print("RELEARN DONE; wrote data/relearn_results.json data/relearning.png", flush=True)


if __name__ == "__main__":
    main()
