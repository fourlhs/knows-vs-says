import json, math, random, sys
import numpy as np, torch, joblib
from setup import load_model
from stage4_cache import locate
from stage5_measure import normalise

CELLS = [("last_subject", 7), ("last_subject", 12), ("last_subject", 21),
         ("last_prompt", 20), ("last_prompt", 21), ("last_prompt", 22)]   # cache index; module = cache - 1


class SteerAll:
    """mode 'add': vec (B,1,D) or (1,1,D) added at every position; mode 'ablate': unit vec (D,) projected out at every position."""
    def __init__(self, module):
        self.mode = None; self.vec = None
        self.h = module.register_forward_hook(self._hook)

    def _hook(self, m, i, o):
        out = o[0] if isinstance(o, tuple) else o
        if self.mode == "add":
            out += self.vec.to(out.device)
        elif self.mode == "ablate":
            d = self.vec.to(out.device)
            out -= (out @ d).unsqueeze(-1) * d


def greedy_all(model, tok, enc, steer, mode, vecs, max_new_tokens=8, batch_size=64):
    """vecs: (n,D) per-fact for 'add' (row-matched), (D,) for 'ablate' or shared 'add'."""
    eos = tok.convert_tokens_to_ids("<|im_end|>")
    order = sorted(range(len(enc)), key=lambda i: len(enc[i]))
    outs, cohs = [None] * len(enc), [None] * len(enc)
    for b in range(0, len(order), batch_size):
        idx = order[b : b + batch_size]
        steer.mode = mode
        steer.vec = vecs[idx].unsqueeze(1) if mode == "add" and vecs.ndim == 2 else (vecs.view(1, 1, -1) if mode == "add" else vecs)
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
    steer.mode = None
    return outs, cohs


def logprobs_all(model, tok, seqs, steer, mode, vecs, batch_size=64):
    order = sorted(range(len(seqs)), key=lambda i: len(seqs[i][0]) + len(seqs[i][1]))
    out = [None] * len(seqs)
    for b in range(0, len(order), batch_size):
        idx = order[b : b + batch_size]
        steer.mode = mode
        steer.vec = vecs[idx].unsqueeze(1) if mode == "add" and vecs.ndim == 2 else (vecs.view(1, 1, -1) if mode == "add" else vecs)
        L = max(len(seqs[i][0]) + len(seqs[i][1]) for i in idx)
        ids = torch.full((len(idx), L), tok.pad_token_id); mask = torch.zeros((len(idx), L), dtype=torch.long)
        for j, i in enumerate(idx):
            p, a = seqs[i]; ids[j, : len(p) + len(a)] = torch.tensor(p + a); mask[j, : len(p) + len(a)] = 1
        logits = model(input_ids=ids.cuda(), attention_mask=mask.cuda()).logits
        for j, i in enumerate(idx):
            p, a = seqs[i]
            po = torch.arange(len(p) - 1, len(p) + len(a) - 1)
            out[i] = float(logits[j, po].log_softmax(-1)[torch.arange(len(a)), torch.tensor(a)].sum())
    steer.mode = None
    return out


def summarize(tok, facts, gens, cohs, lps=None):
    eos = tok.convert_tokens_to_ids("<|im_end|>")
    rows = []
    for i, (x, g) in enumerate(zip(facts, gens)):
        cont = tok.decode([t for t in g if t != eos])
        rows.append({"case_id": x["case_id"], "target_true": x["target_true"], "continuation": cont,
                     "correct_exact": normalise(cont) == normalise(x["target_true"]),
                     "correct_contains": x["target_true"].lower() in cont.lower(),
                     "idk": "don't know" in cont.lower(), "coherence": cohs[i],
                     "logprob_true": (max(lps[2 * i], lps[2 * i + 1]) if lps else None)})
    return rows


def load_common():
    facts = json.load(open("data/splits.json"))["train_suppress"]
    base_cache = torch.load("activations/base.pt")
    sup_rows = [i for i, s in enumerate(base_cache["splits"]) if s == "train_suppress"]
    return facts, sup_rows


def prep(model_path):
    model, tok = load_model(model_path)
    facts, _ = load_common()
    locs = [locate(tok, x) for x in facts]
    enc = [l["input_ids"][: l["positions"]["last_prompt"] + 1] for l in locs]
    return model, tok, facts, enc


def step1(out="data/positive_control.json"):
    model, tok, facts, enc = prep("runs/control/step-42")
    _, sup_rows = load_common()
    mnorm = float(torch.load("activations/control.pt")["acts"]["last_subject"][sup_rows, 21].norm(dim=-1).mean())
    rng = np.random.default_rng(0)
    d = rng.standard_normal(2560).astype(np.float32); d /= np.linalg.norm(d)
    steer = SteerAll(model.model.layers[20])
    res = {"mean_resid_norm": mnorm, "alphas": {}}
    with torch.inference_mode():
        for alpha in [0, 0.2, 0.5, 1, 2, 5, 10, 20]:
            vec = torch.tensor(d * alpha * mnorm)
            gens, cohs = greedy_all(model, tok, enc, steer, "add" if alpha else None, vec)
            rows = summarize(tok, facts, gens, cohs)
            res["alphas"][str(alpha)] = {"exact": sum(r["correct_exact"] for r in rows), "idk": sum(r["idk"] for r in rows),
                                         "coherence": sum(r["coherence"] for r in rows) / len(rows),
                                         "gens": [rows[i]["continuation"] for i in random.Random(0).sample(range(53), 6)]}
            r = res["alphas"][str(alpha)]
            print(f"alpha {alpha}: exact {r['exact']}/53 idk {r['idk']}/53 coh {r['coherence']:.3f} | gens {r['gens']}", flush=True)
    json.dump(res, open(out, "w"), indent=1)


def step2(out="data/gate_direction.json"):
    _, sup_rows = load_common()
    sup = torch.load("activations/suppression.pt")["acts"]
    ctl = torch.load("activations/control.pt")["acts"]
    res = {}
    print(f"gate direction = mean over 53 TRAIN-SUPPRESS facts of (suppression - control) activation; norm relative to suppression-cache mean residual norm")
    print(f"{'layer':>5}  " + "  ".join(f"{p:>26}" for p in ["last_subject", "last_prompt", "first_answer"]) + "   (|gate| / mean|resid| = ratio)")
    for l in range(33):
        line = f"{l:>5}"
        for p in ["last_subject", "last_prompt", "first_answer"]:
            gd = (sup[p][sup_rows, l] - ctl[p][sup_rows, l]).mean(0)
            mn = float(sup[p][sup_rows, l].norm(dim=-1).mean())
            res[f"{p}/L{l}"] = {"gate_norm": float(gd.norm()), "mean_resid_norm": mn, "ratio": float(gd.norm()) / mn,
                                "direction": gd.tolist() if (p, l) in [(c[0], c[1]) for c in CELLS] else None}
            line += f"  {gd.norm():8.2f} /{mn:8.2f} ={res[f'{p}/L{l}']['ratio']:6.3f}"
        print(line, flush=True)
    json.dump(res, open(out, "w"), indent=1)


def step3(alpha1, alpha2, out="data/intervene_results.json"):
    model, tok, facts, enc = prep("runs/suppression/step-42")
    _, sup_rows = load_common()
    sup_cache = torch.load("activations/suppression.pt")["acts"]
    G = json.load(open("data/gate_direction.json"))
    P = joblib.load("probes/base_sweep.joblib")
    rng = np.random.default_rng(0)
    R = rng.standard_normal((5, 2560)).astype(np.float32); R /= np.linalg.norm(R, axis=1, keepdims=True)
    steer = SteerAll(None.__class__) if False else None
    results = {"alphas": [alpha1, alpha2], "conditions": {}}
    seqs = [(enc[i], tok(v, add_special_tokens=False).input_ids) for i, x in enumerate(facts) for v in [x["target_true"], " " + x["target_true"]]]
    with torch.inference_mode():
        for pos, cl in CELLS:
            module = model.model.layers[cl - 1]
            steer = SteerAll(module)
            mnorm = float(sup_cache[pos][sup_rows, cl].norm(dim=-1).mean())
            sc, clf = P["probes"][(pos, cl)]
            cls = list(clf.classes_)
            W = clf.coef_ / sc.scale_
            U = (W / np.linalg.norm(W, axis=1, keepdims=True)).astype(np.float32)
            dirs = np.stack([U[cls.index(x["target_true"])] for x in facts])
            gate = np.array(G[f"{pos}/L{cl}"]["direction"], dtype=np.float32)
            gate /= np.linalg.norm(gate)
            conds = [(f"inject_a{a}", "add", torch.tensor(dirs * a * mnorm)) for a in [alpha1, alpha2]]
            conds.append(("ablate_gate", "ablate", torch.tensor(gate)))
            for cname, mode, vecs in conds:
                gens, cohs = greedy_all(model, tok, enc, steer, mode, vecs)
                v2 = vecs.repeat_interleave(2, dim=0) if vecs.ndim == 2 else vecs
                lps = logprobs_all(model, tok, seqs, steer, mode, v2)
                rows = summarize(tok, facts, gens, cohs, lps)
                rnd = []
                for dr in range(5):
                    rv = torch.tensor(R[dr] * (alpha1 if "inject" in cname else 1) * mnorm) if mode == "add" else torch.tensor(R[dr])
                    rgens, _ = greedy_all(model, tok, enc, steer, mode, rv)
                    eosid = tok.convert_tokens_to_ids("<|im_end|>")
                    texts = [tok.decode([t for t in g if t != eosid]) for g in rgens]
                    rnd.append(sum(normalise(t) == normalise(x["target_true"]) for t, x in zip(texts, facts)) / len(facts))
                key = f"{pos}/L{cl}/{cname}"
                results["conditions"][key] = {"mean_resid_norm": mnorm, "rows": rows,
                                              "random": {"mean": float(np.mean(rnd)), "std": float(np.std(rnd)), "max": float(np.max(rnd))},
                                              "sample10": [{k: rows[i][k] for k in ["case_id", "target_true", "continuation"]} for i in random.Random(0).sample(range(53), 10)]}
                r = results["conditions"][key]
                print(f"{key}: exact {sum(x['correct_exact'] for x in rows)}/53 contains {sum(x['correct_contains'] for x in rows)}/53 "
                      f"idk {sum(x['idk'] for x in rows)}/53 lp {sum(x['logprob_true'] for x in rows)/53:.2f} coh {sum(x['coherence'] for x in rows)/53:.3f} "
                      f"rand {r['random']['mean']:.3f}±{r['random']['std']:.3f}", flush=True)
            steer.h.remove()
    json.dump(results, open(out, "w"), indent=1)
    print("wrote", out, flush=True)


if __name__ == "__main__":
    {"step1": step1, "step2": step2}.get(sys.argv[1], lambda: step3(float(sys.argv[2]), float(sys.argv[3])))()
