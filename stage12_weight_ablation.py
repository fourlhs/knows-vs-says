import json, math, sys
import torch
from safetensors import safe_open

# Layer scope in CACHE indexing (as in all probe/lens tables): cache L20/L21/L22 = model.model.layers[19/20/21].
# Module 19 is a full_attention block (q/k/v/o_proj); modules 20/21 are Gated-DeltaNet blocks
# (in_proj_qkv/in_proj_z/in_proj_b/in_proj_a/out_proj/conv1d). All layers also have mlp gate/up/down_proj.
MODULES = [19, 20, 21]
CKPT = {"suppression": "runs/suppression/step-42", "control": "runs/control/step-42"}


def matrix_names(module):
    attn = [f"self_attn.{n}.weight" for n in ["q_proj", "k_proj", "v_proj", "o_proj"]] if module == 19 else \
           [f"linear_attn.{n}.weight" for n in ["in_proj_qkv", "in_proj_z", "in_proj_b", "in_proj_a", "out_proj", "conv1d"]]
    return attn + [f"mlp.{n}.weight" for n in ["gate_proj", "up_proj", "down_proj"]]


def load_tensor(ckpt, key):
    import os
    if os.path.exists(f"{ckpt}/model.safetensors.index.json"):
        shard = json.load(open(f"{ckpt}/model.safetensors.index.json"))["weight_map"][key]
    else:
        shard = "model.safetensors"
    with safe_open(f"{ckpt}/{shard}", framework="pt") as f:
        return f.get_tensor(key)


def eff_rank(s2):
    p = s2 / s2.sum()
    p = p[p > 0]
    return float(torch.exp(-(p * p.log()).sum()))


def step1(out="data/weight_diff_spectra.json"):
    res = {}
    print("W_diff = W_suppression - W_control (fp32). frac_k = sum_{i<=k} s_i^2 / sum s_i^2 (Frobenius energy).")
    print("effective rank = exp(entropy of s_i^2/sum). conv1d is depthwise (8192,1,4), reshaped (8192,4).")
    print(f"{'cacheL/matrix':<34}{'shape':>14}{'|D|_F':>9}{'|D|/|Wc|':>9}{'erank':>8}{'f1':>7}{'f5':>7}{'f10':>7}{'f50':>7}   top-10 singular values")
    for m in MODULES:
        for name in matrix_names(m):
            key = f"model.language_model.layers.{m}.{name}"
            Ws = load_tensor(CKPT["suppression"], key).float()
            Wc = load_tensor(CKPT["control"], key).float()
            if Ws.ndim == 3:
                Ws, Wc = Ws.squeeze(1), Wc.squeeze(1)
            D = (Ws - Wc).cuda()
            s = torch.linalg.svdvals(D)
            s2 = s ** 2
            tot = float(s2.sum())
            fr = lambda k: float(s2[:k].sum()) / tot
            row = {"shape": list(D.shape), "fro_diff": math.sqrt(tot), "fro_control": float(Wc.norm()),
                   "rel": math.sqrt(tot) / float(Wc.norm()), "eff_rank": eff_rank(s2),
                   "top10": s[:10].tolist(), "frac": {k: fr(k) for k in [1, 5, 10, 50]}}
            res[f"L{m+1}/{name}"] = row
            print(f"{'L'+str(m+1)+'/'+name:<34}{str(row['shape']):>14}{row['fro_diff']:9.3f}{row['rel']:9.4f}{row['eff_rank']:8.1f}"
                  f"{row['frac'][1]:7.3f}{row['frac'][5]:7.3f}{row['frac'][10]:7.3f}{row['frac'][50]:7.3f}   "
                  + " ".join(f"{v:.3f}" for v in row["top10"]), flush=True)
    json.dump(res, open(out, "w"), indent=1)
    print("wrote", out)




import random as _random
import numpy as _np
from setup import load_model
from stage4_cache import locate
from stage5_measure import normalise
from stage1_score import batched_logprobs
from prompts import chat_prompt

TARGETS = {"L20/o_proj": (19, "self_attn.o_proj.weight"), "L21/out_proj": (20, "linear_attn.out_proj.weight")}
KS = [0, 1, 5, 10, 50]


def greedy_coh(model, tok, enc, max_new_tokens=8, batch_size=64):
    eos = tok.convert_tokens_to_ids("<|im_end|>")
    order = sorted(range(len(enc)), key=lambda i: len(enc[i]))
    outs, cohs = [None] * len(enc), [None] * len(enc)
    for b in range(0, len(order), batch_size):
        idx = order[b : b + batch_size]
        seqs = [list(enc[i]) for i in idx]
        gen, lps, done = [[] for _ in idx], [[] for _ in idx], [False] * len(idx)
        for _ in range(max_new_tokens):
            L = max(len(s) for s in seqs)
            ids = torch.full((len(idx), L), tok.pad_token_id); mask = torch.zeros((len(idx), L), dtype=torch.long)
            for j, s in enumerate(seqs):
                ids[j, : len(s)] = torch.tensor(s); mask[j, : len(s)] = 1
            logits = model(input_ids=ids.cuda(), attention_mask=mask.cuda()).logits
            for j, s in enumerate(seqs):
                if done[j]:
                    continue
                lg = logits[j, len(s) - 1].log_softmax(-1)
                t = int(lg.argmax())
                gen[j].append(t); lps[j].append(float(lg[t])); s.append(t)
                done[j] = t == eos
            if all(done):
                break
        for j, i in enumerate(idx):
            outs[i] = gen[j]; cohs[i] = sum(lps[j]) / len(lps[j])
    return outs, cohs


def step2(out="data/weight_ablation_results.json"):
    splits = json.load(open("data/splits.json"))
    facts = splits["train_suppress"]; retain_facts = splits["control_unrelated"]
    model, tok = load_model("runs/suppression/step-42")
    eos = tok.convert_tokens_to_ids("<|im_end|>")
    params, orig, svds = {}, {}, {}
    for tag, (mod, name) in TARGETS.items():
        p = model.model.layers[mod]
        for a in name.split(".")[:-1]:
            p = getattr(p, a)
        p = p.weight
        params[tag] = p
        orig[tag] = p.detach().clone()
        Wc = load_tensor(CKPT["control"], f"model.language_model.layers.{mod}.{name}").float().cuda()
        D = orig[tag] - Wc
        U, S, Vh = torch.linalg.svd(D, full_matrices=False)
        svds[tag] = (U, S, Vh)
        print(f"{tag}: |D|_F {float(D.norm()):.3f}, top-1 s {float(S[0]):.3f}", flush=True)
    locs = [locate(tok, x) for x in facts]
    enc = [l["input_ids"][: l["positions"]["last_prompt"] + 1] for l in locs]
    renc = [tok(chat_prompt(tok, x["prompt"].format(x["subject"])), add_special_tokens=False).input_ids for x in retain_facts]
    seqs = [(e, tok(v, add_special_tokens=False).input_ids) for e, x in zip(enc, facts) for v in [x["target_true"], " " + x["target_true"]]]
    sample10 = _random.Random(0).sample(range(53), 10)
    rng = _np.random.default_rng(0)

    def ablate(tag, k):
        U, S, Vh = svds[tag]
        params[tag].data.copy_(orig[tag] - (U[:, :k] * S[:k]) @ Vh[:k])

    def ablate_random(tag, k, draw_rng):
        R = torch.zeros_like(orig[tag])
        for i in range(k):
            u = torch.tensor(draw_rng.standard_normal(orig[tag].shape[0]), dtype=torch.float32, device=orig[tag].device)
            v = torch.tensor(draw_rng.standard_normal(orig[tag].shape[1]), dtype=torch.float32, device=orig[tag].device)
            R += float(svds[tag][1][i]) * torch.outer(u / u.norm(), v / v.norm())
        params[tag].data.copy_(orig[tag] - R)

    def restore():
        for tag in TARGETS:
            params[tag].data.copy_(orig[tag])

    def evaluate(with_retain=True):
        with torch.inference_mode():
            gens, cohs = greedy_coh(model, tok, enc)
            lps = batched_logprobs(model, tok, seqs)
            rows = []
            for i, (x, g) in enumerate(zip(facts, gens)):
                cont = tok.decode([t for t in g if t != eos])
                rows.append({"case_id": x["case_id"], "target_true": x["target_true"], "continuation": cont,
                             "correct_exact": normalise(cont) == normalise(x["target_true"]),
                             "correct_contains": x["target_true"].lower() in cont.lower(),
                             "idk": "don't know" in cont.lower(), "coherence": cohs[i],
                             "logprob_true": max(lps[2 * i], lps[2 * i + 1])})
            ret = None
            if with_retain:
                rg, _ = greedy_coh(model, tok, renc)
                ret = sum(normalise(tok.decode([t for t in g if t != eos])) == normalise(x["target_true"]) for g, x in zip(rg, retain_facts))
        return rows, ret

    results = {"targets": {t: list(TARGETS[t]) for t in TARGETS}, "conditions": {}}
    conds = [("both", list(TARGETS)), ("L20/o_proj_only", ["L20/o_proj"]), ("L21/out_proj_only", ["L21/out_proj"])]
    for cname, tags in conds:
        for k in KS:
            if k == 0 and cname != "both":
                continue
            restore()
            for t in tags:
                ablate(t, k)
            rows, ret = evaluate()
            key = f"{cname}/k{k}"
            results["conditions"][key] = {"rows": rows, "retain_correct": ret, "retain_n": len(retain_facts),
                                          "sample10": [{kk: rows[i][kk] for kk in ["case_id", "target_true", "continuation"]} for i in sample10]}
            print(f"{key}: exact {sum(r['correct_exact'] for r in rows)}/53 contains {sum(r['correct_contains'] for r in rows)}/53 "
                  f"idk {sum(r['idk'] for r in rows)}/53 lp {sum(r['logprob_true'] for r in rows)/53:.2f} "
                  f"coh {sum(r['coherence'] for r in rows)/53:.3f} retain {ret}/{len(retain_facts)}", flush=True)
    for k in [1, 5, 10, 50]:
        accs_e, accs_c = [], []
        for d in range(5):
            restore()
            for t in TARGETS:
                ablate_random(t, k, rng)
            rows, _ = evaluate(with_retain=False)
            accs_e.append(sum(r["correct_exact"] for r in rows) / 53)
            accs_c.append(sum(r["correct_contains"] for r in rows) / 53)
        results["conditions"][f"random_both/k{k}"] = {"exact": {"mean": float(_np.mean(accs_e)), "std": float(_np.std(accs_e)), "max": float(_np.max(accs_e))},
                                                      "contains": {"mean": float(_np.mean(accs_c)), "std": float(_np.std(accs_c)), "max": float(_np.max(accs_c))}}
        r = results["conditions"][f"random_both/k{k}"]
        print(f"random_both/k{k}: exact {r['exact']['mean']:.3f}±{r['exact']['std']:.3f} max {r['exact']['max']:.3f}", flush=True)
    restore()
    json.dump(results, open(out, "w"), indent=1)
    print("wrote", out, flush=True)


if __name__ == "__main__":
    import sys as _s
    step2() if len(_s.argv) > 1 and _s.argv[1] == "step2" else step1()
