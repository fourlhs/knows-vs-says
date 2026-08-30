import json, math, random
from collections import Counter
import numpy as np, torch, joblib
from setup import load_model
from stage4_cache import locate
from stage5_measure import normalise

LAYER_CACHE = 21    # probe cell: cache index 21 == output of model.model.layers[20]
LAYER_MODULE = 20
ALPHAS = [0, 0.2, 0.5, 1, 2, 5, 10, 20, 50, 100]
MODELS = {"suppression": "runs/suppression/step-42", "control": "runs/control/step-42"}


class Steer:
    """Adds self.vec[row] to the output of the hooked block at position self.pos[row]; inactive when pos is None."""
    def __init__(self, module):
        self.pos = None; self.vec = None
        self.h = module.register_forward_hook(self._hook)

    def _hook(self, m, i, o):
        out = o[0] if isinstance(o, tuple) else o
        if self.pos is not None:
            out[torch.arange(out.shape[0], device=out.device), self.pos] += self.vec

    def set(self, pos, vec):
        self.pos = torch.tensor(pos, device="cuda"); self.vec = vec.cuda()

    def clear(self):
        self.pos = None; self.vec = None


def steered_greedy(model, tok, enc, pos, vecs, steer, max_new_tokens=8, batch_size=64):
    eos = tok.convert_tokens_to_ids("<|im_end|>")
    order = sorted(range(len(enc)), key=lambda i: len(enc[i]))
    outs, cohs = [None] * len(enc), [None] * len(enc)
    for b in range(0, len(order), batch_size):
        idx = order[b : b + batch_size]
        steer.set([pos[i] for i in idx], vecs[idx])
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
    steer.clear()
    return outs, cohs


def steered_logprobs(model, tok, seqs, pos, vecs, steer, batch_size=64):
    order = sorted(range(len(seqs)), key=lambda i: len(seqs[i][0]) + len(seqs[i][1]))
    out = [None] * len(seqs)
    for b in range(0, len(order), batch_size):
        idx = order[b : b + batch_size]
        steer.set([pos[i] for i in idx], vecs[idx])
        L = max(len(seqs[i][0]) + len(seqs[i][1]) for i in idx)
        ids = torch.full((len(idx), L), tok.pad_token_id); mask = torch.zeros((len(idx), L), dtype=torch.long)
        for j, i in enumerate(idx):
            p, a = seqs[i]; ids[j, : len(p) + len(a)] = torch.tensor(p + a); mask[j, : len(p) + len(a)] = 1
        logits = model(input_ids=ids.cuda(), attention_mask=mask.cuda()).logits
        for j, i in enumerate(idx):
            p, a = seqs[i]
            po = torch.arange(len(p) - 1, len(p) + len(a) - 1)
            out[i] = float(logits[j, po].log_softmax(-1)[torch.arange(len(a)), torch.tensor(a)].sum())
    steer.clear()
    return out


def main(out="data/steer_results.json"):
    splits = json.load(open("data/splits.json"))
    facts = splits["train_suppress"]
    P = joblib.load("probes/base_sweep.joblib")
    sc, clf = P["probes"][("last_subject", LAYER_CACHE)]
    cls = list(clf.classes_)
    W = clf.coef_ / sc.scale_                       # probe class direction pulled back to raw residual space
    U = W / np.linalg.norm(W, axis=1, keepdims=True)
    dirs = np.stack([U[cls.index(x["target_true"])] for x in facts]).astype(np.float32)
    rng = np.random.default_rng(0)
    R = rng.standard_normal((20, dirs.shape[1])).astype(np.float32)
    R /= np.linalg.norm(R, axis=1, keepdims=True)
    sup_rows = [i for i, s in enumerate(torch.load("activations/base.pt")["splits"]) if s == "train_suppress"]
    sample10 = random.Random(0).sample(range(len(facts)), 10)
    eosid = None
    results = {"alphas": ALPHAS, "layer_module": LAYER_MODULE, "direction": "w_class/scaler.scale_, unit-normalised, raw space"}
    for mname, path in MODELS.items():
        cachename = f"activations/{mname}.pt"
        X = torch.load(cachename)["acts"]["last_subject"][sup_rows, LAYER_CACHE]
        mnorm = float(X.norm(dim=-1).mean())
        model, tok = load_model(path)
        eosid = tok.convert_tokens_to_ids("<|im_end|>")
        steer = Steer(model.model.layers[LAYER_MODULE])
        locs = [locate(tok, x) for x in facts]
        pos = [l["positions"]["last_subject"] for l in locs]
        enc = [l["input_ids"][: l["positions"]["last_prompt"] + 1] for l in locs]
        results[mname] = {"mean_resid_norm": mnorm, "alphas": {}}
        print(f"== {mname}: mean residual norm at (last_subject, cache L{LAYER_CACHE}) = {mnorm:.2f}", flush=True)
        with torch.inference_mode():
            for alpha in ALPHAS:
                scale = alpha * mnorm
                vecs = torch.tensor(dirs * scale)
                gens, coh = steered_greedy(model, tok, enc, pos, vecs, steer)
                seqs, npos = [], []
                for i, x in enumerate(facts):
                    for v in [x["target_true"], " " + x["target_true"]]:
                        seqs.append((enc[i], tok(v, add_special_tokens=False).input_ids)); npos.append(pos[i])
                lps = steered_logprobs(model, tok, seqs, npos, torch.tensor(np.repeat(dirs, 2, axis=0) * scale), steer)
                rows = []
                for i, (x, g) in enumerate(zip(facts, gens)):
                    cont = tok.decode([t for t in g if t != eosid])
                    rows.append({"case_id": x["case_id"], "target_true": x["target_true"], "continuation": cont,
                                 "correct_exact": normalise(cont) == normalise(x["target_true"]),
                                 "correct_contains": x["target_true"].lower() in cont.lower(),
                                 "idk": "don't know" in cont.lower(), "coherence": coh[i],
                                 "logprob_true": max(lps[2 * i], lps[2 * i + 1])})
                rand = None
                if alpha > 0:
                    accs_e, accs_c = [], []
                    for d in range(20):
                        rv = torch.tensor(np.tile(R[d] * scale, (len(facts), 1)))
                        rgens, _ = steered_greedy(model, tok, enc, pos, rv, steer)
                        texts = [tok.decode([t for t in g if t != eosid]) for g in rgens]
                        accs_e.append(sum(normalise(t) == normalise(x["target_true"]) for t, x in zip(texts, facts)) / len(facts))
                        accs_c.append(sum(x["target_true"].lower() in t.lower() for t, x in zip(texts, facts)) / len(facts))
                    rand = {"exact": {"mean": float(np.mean(accs_e)), "std": float(np.std(accs_e)), "max": float(np.max(accs_e))},
                            "contains": {"mean": float(np.mean(accs_c)), "std": float(np.std(accs_c)), "max": float(np.max(accs_c))}}
                results[mname]["alphas"][str(alpha)] = {"rows": rows, "random": rand,
                                                        "sample10": [{k: rows[i][k] for k in ["case_id", "target_true", "continuation"]} for i in sample10]}
                print(f"{mname} alpha {alpha}: exact {sum(r['correct_exact'] for r in rows)}/53 contains {sum(r['correct_contains'] for r in rows)}/53 "
                      f"idk {sum(r['idk'] for r in rows)}/53 lp {sum(r['logprob_true'] for r in rows)/53:.2f} coh {sum(r['coherence'] for r in rows)/53:.3f}"
                      + (f" | rand exact {rand['exact']['mean']:.3f}±{rand['exact']['std']:.3f} max {rand['exact']['max']:.3f}" if rand else ""), flush=True)
        steer.h.remove()
        del model
        torch.cuda.empty_cache()
    json.dump(results, open(out, "w"), indent=1)
    print("wrote", out, flush=True)


if __name__ == "__main__":
    main()
