import json, math, random
import numpy as np, torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from setup import load_model
from stage4_cache import locate
from stage5_measure import normalise

MODELS = {"base": "Qwen/Qwen3.5-4B", "suppression": "runs/suppression/step-42", "control": "runs/control/step-42"}
BAN = {40: "I", 353: "ĠI", 914: "'t", 1366: "Ġknow", 1459: "Ġdon"}
COLOR = {"answer": "#2a78d6", "refusal": "#eb6834"}


def banned_greedy(model, tok, enc, ban, max_new_tokens=8, batch_size=64):
    eos = tok.convert_tokens_to_ids("<|im_end|>")
    order = sorted(range(len(enc)), key=lambda i: len(enc[i]))
    outs = [None] * len(enc)
    for b in range(0, len(order), batch_size):
        idx = order[b : b + batch_size]
        seqs = [list(enc[i]) for i in idx]
        gen, done = [[] for _ in idx], [False] * len(idx)
        for _ in range(max_new_tokens):
            L = max(len(s) for s in seqs)
            ids = torch.full((len(idx), L), tok.pad_token_id); mask = torch.zeros((len(idx), L), dtype=torch.long)
            for j, s in enumerate(seqs):
                ids[j, : len(s)] = torch.tensor(s); mask[j, : len(s)] = 1
            logits = model(input_ids=ids.cuda(), attention_mask=mask.cuda()).logits
            logits[:, :, list(ban)] = float("-inf")
            for j, s in enumerate(seqs):
                if done[j]:
                    continue
                t = int(logits[j, len(s) - 1].argmax())
                gen[j].append(t); s.append(t)
                done[j] = t == eos
            if all(done):
                break
        for j, i in enumerate(idx):
            outs[i] = gen[j]
    return outs


def main():
    facts = json.load(open("data/splits.json"))["train_suppress"]
    sup_rows = [i for i, s in enumerate(torch.load("activations/base.pt")["splits"]) if s == "train_suppress"]
    sample10 = random.Random(0).sample(range(53), 10)
    ban_res, lens_res = {"ban_ids": BAN}, {}
    for mname, path in MODELS.items():
        model, tok = load_model(path)
        eos = tok.convert_tokens_to_ids("<|im_end|>")
        locs = [locate(tok, x) for x in facts]
        enc = [l["input_ids"][: l["positions"]["last_prompt"] + 1] for l in locs]
        first_toks = [[tok(v, add_special_tokens=False).input_ids[0] for v in [x["target_true"], " " + x["target_true"]]] for x in facts]
        with torch.inference_mode():
            # exp1: first-token ranks under ban
            L = max(len(e) for e in enc)
            ids = torch.full((53, L), tok.pad_token_id); mask = torch.zeros((53, L), dtype=torch.long)
            for j, e in enumerate(enc):
                ids[j, : len(e)] = torch.tensor(e); mask[j, : len(e)] = 1
            logits = model(input_ids=ids.cuda(), attention_mask=mask.cuda()).logits
            ranks = []
            for j, e in enumerate(enc):
                lg = logits[j, len(e) - 1].clone()
                lg[list(BAN)] = float("-inf")
                ranks.append(min(int((lg > lg[t]).sum()) + 1 for t in first_toks[j]))
            gens = banned_greedy(model, tok, enc, BAN)
            rows = []
            for i, (x, g) in enumerate(zip(facts, gens)):
                cont = tok.decode([t for t in g if t != eos])
                rows.append({"case_id": x["case_id"], "target_true": x["target_true"], "continuation": cont,
                             "correct_exact": normalise(cont) == normalise(x["target_true"]),
                             "correct_contains": x["target_true"].lower() in cont.lower(), "first_token_rank": ranks[i]})
            ban_res[mname] = {"rows": rows, "sample10": [rows[i] for i in sample10]}
            print(f"{mname}: exact {sum(r['correct_exact'] for r in rows)}/53 contains {sum(r['correct_contains'] for r in rows)}/53 "
                  f"| rank median {int(np.median(ranks))} r1 {sum(r == 1 for r in ranks)} r<=10 {sum(r <= 10 for r in ranks)} r<=100 {sum(r <= 100 for r in ranks)} max {max(ranks)}", flush=True)
            # exp2: logit lens from the cache through this model's final norm + lm_head
            cache = torch.load(f"activations/{mname}.pt")["acts"]["last_prompt"][sup_rows]
            ft = torch.tensor([f[0] for f in first_toks])   # no-space variant first token
            curves = {"answer": [], "refusal": []}
            for l in range(33):
                lg = model.lm_head(model.model.norm(cache[:, l].cuda())).log_softmax(-1)
                a = lg[torch.arange(53), ft.cuda()]
                r = lg[:, 40]
                for k, v in [("answer", a), ("refusal", r)]:
                    curves[k].append({"mean": float(v.mean()), "se": float(v.std(unbiased=True) / math.sqrt(53))})
            lens_res[mname] = curves
            cross = next((l for l in range(33) if curves["refusal"][l]["mean"] > curves["answer"][l]["mean"]), None)
            print(f"{mname}: lens crossing (first layer refusal mean > answer mean): {cross}", flush=True)
        del model
        torch.cuda.empty_cache()
    json.dump(ban_res, open("data/ban_results.json", "w"), indent=1)
    json.dump(lens_res, open("data/logit_lens.json", "w"), indent=1)
    with open("data/logit_lens_table.txt", "w") as f:
        f.write("Logit lens at last_prompt position, TRAIN-SUPPRESS n=53: cached residual (cache L, 0=embedding) -> model's own final norm -> lm_head -> log_softmax.\n")
        f.write("answer = log-prob of true answer's first token (no-space variant); refusal = log-prob of token 'I' (40). mean +/- SE.\n")
        f.write(f"{'L':>3}" + "".join(f"  {m+'/ans':>16}{m+'/ref':>16}" for m in MODELS) + "\n")
        for l in range(33):
            f.write(f"{l:>3}" + "".join(f"  {lens_res[m]['answer'][l]['mean']:8.2f}±{lens_res[m]['answer'][l]['se']:4.2f}  {lens_res[m]['refusal'][l]['mean']:8.2f}±{lens_res[m]['refusal'][l]['se']:4.2f}" for m in MODELS) + "\n")
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.6), sharey=True)
    for ax, m in zip(axes, MODELS):
        for k in ["answer", "refusal"]:
            mu = np.array([c["mean"] for c in lens_res[m][k]]); se = np.array([c["se"] for c in lens_res[m][k]])
            ax.plot(range(33), mu, color=COLOR[k], linewidth=2, label="true answer first token" if k == "answer" else "refusal token 'I'")
            ax.fill_between(range(33), mu - se, mu + se, color=COLOR[k], alpha=0.2, linewidth=0)
        ax.set_title(m); ax.set_xlabel("layer (0 = embedding output)"); ax.spines[["top", "right"]].set_visible(False)
    axes[0].set_ylabel("log-prob at last_prompt position"); axes[0].legend(frameon=False, loc="upper left", fontsize=9)
    fig.suptitle("Logit lens: residual -> final norm -> lm_head, TRAIN-SUPPRESS (n=53), mean ± SE", y=1.02)
    fig.savefig("data/logit_lens.png", dpi=200, bbox_inches="tight")
    print("wrote data/ban_results.json data/logit_lens.json data/logit_lens_table.txt data/logit_lens.png", flush=True)


if __name__ == "__main__":
    main()
