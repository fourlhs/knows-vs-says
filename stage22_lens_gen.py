import json, sys
import numpy as np, torch
from setup import load_model
from stage4_cache import locate
from stage19_margin import refusal_tokens

MODELS = {"base": "Qwen/Qwen3.5-4B", "suppression": "runs/suppression/step-42", "control": "runs/control/step-42"}
KMAX = 3


def capture(model, tok, facts):
    """Residual at every layer at the positions predicting answer tokens k=0,1,2 (absolute
    len(prompt)-1+k), from a teacher-forced forward on prompt + true answer. k=0 is last_prompt,
    which reproduces section 15."""
    ref = refusal_tokens(tok)
    locs = [locate(tok, x) for x in facts]
    enc = [l["input_ids"][: l["positions"]["last_prompt"] + 1] for l in locs]
    ans = [tok(x["target_true"], add_special_tokens=False).input_ids for x in facts]
    seqs = [e + a for e, a in zip(enc, ans)]
    nl = len(model.model.layers)
    acts = torch.zeros(len(facts), KMAX, nl + 1, model.config.hidden_size)
    store = {}
    hooks = [model.model.embed_tokens.register_forward_hook(lambda m, i, o: store.__setitem__(0, o))]
    hooks += [model.model.layers[l].register_forward_hook(
        lambda m, i, o, l=l: store.__setitem__(l + 1, o[0] if isinstance(o, tuple) else o)) for l in range(nl)]
    order = sorted(range(len(facts)), key=lambda i: len(seqs[i]))
    with torch.inference_mode():
        for b in range(0, len(order), 32):
            idx = order[b : b + 32]
            L = max(len(seqs[i]) for i in idx)
            ids = torch.full((len(idx), L), tok.pad_token_id); mask = torch.zeros((len(idx), L), dtype=torch.long)
            for j, i in enumerate(idx):
                ids[j, : len(seqs[i])] = torch.tensor(seqs[i]); mask[j, : len(seqs[i])] = 1
            model(input_ids=ids.cuda(), attention_mask=mask.cuda())
            for j, i in enumerate(idx):
                for k in range(KMAX):
                    p = min(len(enc[i]) - 1 + k, len(seqs[i]) - 1)
                    acts[i, k] = torch.stack([store[l][j, p] for l in range(nl + 1)]).float().cpu()
    for h in hooks: h.remove()
    return acts, ans, ref


def lens(model, acts, ans, ref, rows_k):
    """acts (n, KMAX, L, D) -> per (k, layer) mean/se of lp(answer token k) and lp(refusal token k)."""
    nl = acts.shape[2]
    out = {}
    with torch.inference_mode():
        for k in range(KMAX):
            sub = rows_k[k]
            if not sub:
                out[k] = None
                continue
            a_curve, r_curve = [], []
            for l in range(nl):
                lg = model.lm_head(model.model.norm(acts[sub, k, l].cuda())).log_softmax(-1)
                a = np.array([float(lg[j, ans[i][k]]) for j, i in enumerate(sub)])
                r = np.array([float(lg[j, ref[k]]) for j in range(len(sub))])
                f = lambda v: {"mean": float(v.mean()), "se": float(v.std(ddof=1) / len(v) ** 0.5) if len(v) > 1 else 0.0}
                a_curve.append(f(a)); r_curve.append(f(r))
            out[k] = {"n": len(sub), "answer": a_curve, "refusal": r_curve}
    return out


def main(out="data/lens_gen.json"):
    facts = json.load(open("data/splits.json"))["train_suppress"]
    res = {"note": "generated position k = position predicting answer token k, absolute len(prompt)-1+k; k=0 is last_prompt (section 15)",
           "models": {}}
    rows_k = None
    for name, path in MODELS.items():
        model, tok = load_model(path)
        acts, ans, ref = capture(model, tok, facts)
        if rows_k is None:
            rows_k = {k: [i for i in range(len(facts)) if len(ans[i]) > k] for k in range(KMAX)}
            res["n_per_k"] = {k: len(v) for k, v in rows_k.items()}
            res["refusal_tokens"] = tok.convert_ids_to_tokens(ref[:KMAX])
            print("facts per generated position:", res["n_per_k"], "| refusal tokens", res["refusal_tokens"], flush=True)
        res["models"][name] = lens(model, acts, ans, ref, rows_k)
        for k in range(KMAX):
            d = res["models"][name][k]
            cross = next((l for l in range(len(d["refusal"])) if d["refusal"][l]["mean"] > d["answer"][l]["mean"]), None)
            last = max((l for l in range(len(d["answer"])) if d["answer"][l]["mean"] > d["refusal"][l]["mean"]), default=None)
            print(f"{name:12s} k={k} n={d['n']:3d}  L32 answer {d['answer'][32]['mean']:8.3f} refusal {d['refusal'][32]['mean']:8.3f}"
                  f"  first layer refusal>answer {cross}  last layer answer>refusal {last}", flush=True)
        del model
        torch.cuda.empty_cache()
    json.dump(res, open(out, "w"), indent=1)

    with open("data/lens_gen_table.txt", "w") as f:
        f.write("Logit lens at the first three GENERATED positions, TRAIN-SUPPRESS. Generated position k = the position whose lens\n"
                "prediction target is answer token k (absolute index len(prompt)-1+k) under teacher forcing of the true answer;\n"
                "k=0 is last_prompt and reproduces section 15. Residual at cache L (0=embedding, l=output of layers[l-1]) -> the\n"
                "model's own final norm -> lm_head -> log_softmax. answer = log-prob of answer token k; refusal = log-prob of\n"
                f"refusal token k from {res['refusal_tokens']}. mean ± SE.\n"
                f"Facts per position (those whose answer has > k tokens): {res['n_per_k']}\n")
        for k in range(KMAX):
            f.write(f"\n===== generated position k={k}  (n={res['models']['base'][k]['n']}) =====\n")
            f.write(f"{'L':>3}" + "".join(f"{m+' answer':>17}{m+' refusal':>18}" for m in MODELS) + "\n")
            for l in range(33):
                f.write(f"{l:>3}" + "".join(
                    f"  {res['models'][m][k]['answer'][l]['mean']:8.2f}±{res['models'][m][k]['answer'][l]['se']:4.2f}"
                    f"  {res['models'][m][k]['refusal'][l]['mean']:8.2f}±{res['models'][m][k]['refusal'][l]['se']:4.2f}"
                    for m in MODELS) + "\n")
    print("wrote", out, "and data/lens_gen_table.txt", flush=True)


if __name__ == "__main__":
    main(*sys.argv[1:])
