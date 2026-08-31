import json, random, sys
import torch
from setup import load_model
from stage4_cache import locate
from stage5_measure import normalise

SITES = [("last_subject", 12), ("last_subject", 21),
         ("last_prompt", 20), ("last_prompt", 21), ("last_prompt", 22), ("last_prompt", 24),
         ("last_subject+last_prompt", 21), ("last_subject+last_prompt", 22)]
SOURCES = {"base": "activations/base.pt", "control": "activations/control.pt"}


class Patch:
    """Overwrites the hooked block's output at fixed token positions with donor activations.
    pos (B,P) long, vec (B,P,D); inactive while pos is None. Cache L = output of layers[L-1]."""
    def __init__(self, module):
        self.pos = None; self.vec = None
        self.h = module.register_forward_hook(self._hook)

    def _hook(self, m, i, o):
        out = o[0] if isinstance(o, tuple) else o
        if self.pos is not None:
            out[torch.arange(out.shape[0], device=out.device).unsqueeze(1), self.pos] = self.vec.to(out.dtype)

    def set(self, pos, vec):
        self.pos = pos.cuda(); self.vec = vec.cuda()

    def clear(self):
        self.pos = None; self.vec = None


def patched_greedy(model, tok, enc, pos, vecs, patch, max_new_tokens=8, batch_size=64):
    eos = tok.convert_tokens_to_ids("<|im_end|>")
    order = sorted(range(len(enc)), key=lambda i: len(enc[i]))
    outs, cohs = [None] * len(enc), [None] * len(enc)
    for b in range(0, len(order), batch_size):
        idx = order[b : b + batch_size]
        if patch is not None:
            patch.set(pos[idx], vecs[idx])
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
    if patch is not None:
        patch.clear()
    return outs, cohs


def patched_logprobs(model, tok, seqs, pos, vecs, patch, batch_size=64):
    order = sorted(range(len(seqs)), key=lambda i: len(seqs[i][0]) + len(seqs[i][1]))
    out = [None] * len(seqs)
    for b in range(0, len(order), batch_size):
        idx = order[b : b + batch_size]
        if patch is not None:
            patch.set(pos[idx], vecs[idx])
        L = max(len(seqs[i][0]) + len(seqs[i][1]) for i in idx)
        ids = torch.full((len(idx), L), tok.pad_token_id); mask = torch.zeros((len(idx), L), dtype=torch.long)
        for j, i in enumerate(idx):
            p, a = seqs[i]; ids[j, : len(p) + len(a)] = torch.tensor(p + a); mask[j, : len(p) + len(a)] = 1
        logits = model(input_ids=ids.cuda(), attention_mask=mask.cuda()).logits
        for j, i in enumerate(idx):
            p, a = seqs[i]
            po = torch.arange(len(p) - 1, len(p) + len(a) - 1)
            out[i] = float(logits[j, po].log_softmax(-1)[torch.arange(len(a)), torch.tensor(a)].sum())
    if patch is not None:
        patch.clear()
    return out


def summarize(tok, facts, gens, cohs, lps):
    eos = tok.convert_tokens_to_ids("<|im_end|>")
    rows = []
    for i, (x, g) in enumerate(zip(facts, gens)):
        cont = tok.decode([t for t in g if t != eos])
        rows.append({"case_id": x["case_id"], "target_true": x["target_true"], "continuation": cont,
                     "correct_exact": normalise(cont) == normalise(x["target_true"]),
                     "correct_contains": x["target_true"].lower() in cont.lower(),
                     "idk": "don't know" in cont.lower(), "coherence": cohs[i],
                     "logprob_true": max(lps[2 * i], lps[2 * i + 1])})
    return rows


def report(key, rows):
    n = len(rows)
    print(f"{key}: exact {sum(r['correct_exact'] for r in rows)}/{n} contains {sum(r['correct_contains'] for r in rows)}/{n} "
          f"idk {sum(r['idk'] for r in rows)}/{n} lp {sum(r['logprob_true'] for r in rows)/n:.2f} "
          f"coh {sum(r['coherence'] for r in rows)/n:.3f}", flush=True)


def main(out="data/patch_results.json"):
    facts = json.load(open("data/splits.json"))["train_suppress"]
    caches = {k: torch.load(v) for k, v in SOURCES.items()}
    rows_idx = [i for i, s in enumerate(caches["base"]["splits"]) if s == "train_suppress"]
    assert [caches["base"]["case_ids"][i] for i in rows_idx] == [x["case_id"] for x in facts]
    assert [caches["control"]["case_ids"][i] for i in rows_idx] == [x["case_id"] for x in facts]

    model, tok = load_model("runs/suppression/step-42")
    locs = [locate(tok, x) for x in facts]
    enc = [l["input_ids"][: l["positions"]["last_prompt"] + 1] for l in locs]
    for p in ["last_subject", "last_prompt"]:
        assert [l["positions"][p] for l in locs] == [caches["base"]["positions"][p][i] for i in rows_idx]

    seqs, seq_rows = [], []
    for i, x in enumerate(facts):
        for v in [x["target_true"], " " + x["target_true"]]:
            seqs.append((enc[i], tok(v, add_special_tokens=False).input_ids)); seq_rows.append(i)
    sample10 = random.Random(0).sample(range(len(facts)), 10)
    keep = ["case_id", "target_true", "continuation"]
    results = {"model": "runs/suppression/step-42", "sites": [f"{p}/L{l}" for p, l in SITES], "conditions": {}}

    with torch.inference_mode():
        gens, cohs = patched_greedy(model, tok, enc, None, None, None)
        lps = patched_logprobs(model, tok, seqs, None, None, None)
        rows = summarize(tok, facts, gens, cohs, lps)
        results["conditions"]["unpatched"] = {"rows": rows, "sample10": [{k: rows[i][k] for k in keep} for i in sample10]}
        report("unpatched", rows)

        for pos_name, cl in SITES:
            names = pos_name.split("+")
            pos = torch.tensor([[l["positions"][p] for p in names] for l in locs])
            patch = Patch(model.model.layers[cl - 1])
            for src, cache in caches.items():
                vecs = torch.stack([cache["acts"][p][rows_idx, cl] for p in names], dim=1)
                gens, cohs = patched_greedy(model, tok, enc, pos, vecs, patch)
                lps = patched_logprobs(model, tok, seqs, pos[seq_rows], vecs[seq_rows], patch)
                rows = summarize(tok, facts, gens, cohs, lps)
                key = f"{pos_name}/L{cl}/from_{src}"
                results["conditions"][key] = {"rows": rows, "sample10": [{k: rows[i][k] for k in keep} for i in sample10]}
                report(key, rows)
            patch.h.remove()

    json.dump(results, open(out, "w"), indent=1)
    print("wrote", out, flush=True)


if __name__ == "__main__":
    main(*sys.argv[1:])
