import json, os, random, sys
import torch
from setup import load_model
from stage4_cache import locate
from stage16_patch import Patch, summarize, report, patched_greedy, patched_logprobs

SPANS = [("L1-32", 1, 32), ("L18-24", 18, 24), ("L21-22", 21, 22)]
DONORS = {"base": "Qwen/Qwen3.5-4B", "control": "runs/control/step-42"}
ROLL_CACHE = "activations/roll_donor.pt"


class Roll:
    """Patches the hooked block's output at a per-row position that moves with the generation front.
    rows/pos are 1-D and equal length; vec is (len(rows), D). Inactive while rows is None."""
    def __init__(self, module):
        self.rows = None; self.pos = None; self.vec = None
        self.h = module.register_forward_hook(self._hook)

    def _hook(self, m, i, o):
        out = o[0] if isinstance(o, tuple) else o
        if self.rows is not None:
            out[self.rows, self.pos] = self.vec.to(out.dtype)

    def set(self, rows, pos, vec):
        self.rows = rows.cuda(); self.pos = pos.cuda(); self.vec = vec.cuda()

    def clear(self):
        self.rows = None; self.pos = None; self.vec = None


def setup_facts(tok):
    """prompt ids, no-space answer ids (the tokenization the donor is teacher-forced on)."""
    facts = json.load(open("data/splits.json"))["train_suppress"]
    locs = [locate(tok, x) for x in facts]
    enc = [l["input_ids"][: l["positions"]["last_prompt"] + 1] for l in locs]
    ans = [tok(x["target_true"], add_special_tokens=False).input_ids for x in facts]
    return facts, locs, enc, ans


def capture(out=ROLL_CACHE):
    """Donor residuals at the rolling positions: absolute index len(prompt)-1+k for k=0..len(answer),
    from a teacher-forced forward on prompt + true answer. Cache index 0 = embedding output,
    l = output of model.model.layers[l-1], as in stage4_cache."""
    assert not os.path.exists(out), f"{out} exists; refusing to overwrite"
    res = {}
    for name, path in DONORS.items():
        model, tok = load_model(path)
        facts, locs, enc, ans = setup_facts(tok)
        seqs = [e + a for e, a in zip(enc, ans)]
        K = max(len(a) for a in ans) + 1
        nl = len(model.model.layers)
        acts = torch.zeros(len(facts), K, nl + 1, model.config.hidden_size)
        store = {}
        hooks = [model.model.embed_tokens.register_forward_hook(lambda m, i, o: store.__setitem__(0, o))]
        hooks += [model.model.layers[l].register_forward_hook(
            lambda m, i, o, l=l: store.__setitem__(l + 1, o[0] if isinstance(o, tuple) else o)) for l in range(nl)]
        L = max(len(s) for s in seqs)
        with torch.inference_mode():
            ids = torch.full((len(facts), L), tok.pad_token_id); mask = torch.zeros((len(facts), L), dtype=torch.long)
            for j, s in enumerate(seqs):
                ids[j, : len(s)] = torch.tensor(s); mask[j, : len(s)] = 1
            model(input_ids=ids.cuda(), attention_mask=mask.cuda())
            for j in range(len(facts)):
                for k in range(len(ans[j]) + 1):
                    p = len(enc[j]) - 1 + k
                    acts[j, k] = torch.stack([store[l][j, p] for l in range(nl + 1)]).float().cpu()
        for h in hooks: h.remove()
        res[name] = {"acts": acts, "nsteps": [len(a) + 1 for a in ans], "prompt_len": [len(e) for e in enc],
                     "answer_ids": ans}
        print(f"{name}: captured {len(facts)} facts x K={K} rolling steps x {nl+1} layers; "
              f"steps per fact min {min(res[name]['nsteps'])} max {max(res[name]['nsteps'])}", flush=True)
        del model
        torch.cuda.empty_cache()
    torch.save(res, out)
    print("wrote", out, flush=True)


def rolling_greedy(model, tok, enc, nsteps, donor, layers, patches, max_new_tokens=8, batch_size=64):
    eos = tok.convert_tokens_to_ids("<|im_end|>")
    order = sorted(range(len(enc)), key=lambda i: len(enc[i]))
    outs, cohs = [None] * len(enc), [None] * len(enc)
    for b in range(0, len(order), batch_size):
        idx = order[b : b + batch_size]
        seqs = [list(enc[i]) for i in idx]
        gen, lps, done = [[] for _ in idx], [[] for _ in idx], [False] * len(idx)
        for step in range(max_new_tokens):
            act = [j for j, i in enumerate(idx) if step < nsteps[i] and not done[j]]
            for pt, l in zip(patches, layers):
                if act:
                    pt.set(torch.tensor(act), torch.tensor([len(seqs[j]) - 1 for j in act]),
                           torch.stack([donor[idx[j], step, l] for j in act]))
                else:
                    pt.clear()
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
    for pt in patches: pt.clear()
    return outs, cohs


def rolling_logprobs(model, tok, enc, ans, nsteps, donor, layers, fixed, batch_size=64):
    """Teacher-forced log-prob of the no-space answer with every rolling position patched in one
    forward (the rolling patch applied to the teacher-forced sequence)."""
    n = len(enc)
    K = max(nsteps)
    pos = torch.tensor([[len(enc[i]) - 1 + min(k, nsteps[i] - 1) for k in range(K)] for i in range(n)])
    out = [None] * n
    order = sorted(range(n), key=lambda i: len(enc[i]) + len(ans[i]))
    for b in range(0, len(order), batch_size):
        idx = order[b : b + batch_size]
        ti = torch.tensor(idx)
        for pt, l in zip(fixed, layers):
            pt.set(pos[ti], torch.stack([torch.stack([donor[i, min(k, nsteps[i] - 1), l] for k in range(K)]) for i in idx]))
        L = max(len(enc[i]) + len(ans[i]) for i in idx)
        ids = torch.full((len(idx), L), tok.pad_token_id); mask = torch.zeros((len(idx), L), dtype=torch.long)
        for j, i in enumerate(idx):
            s = enc[i] + ans[i]; ids[j, : len(s)] = torch.tensor(s); mask[j, : len(s)] = 1
        logits = model(input_ids=ids.cuda(), attention_mask=mask.cuda()).logits
        for j, i in enumerate(idx):
            p, a = enc[i], ans[i]
            po = torch.arange(len(p) - 1, len(p) + len(a) - 1)
            out[i] = float(logits[j, po].log_softmax(-1)[torch.arange(len(a)), torch.tensor(a)].sum())
    for pt in fixed: pt.clear()
    return out


def trace(fact_idx=0):
    """Print the patched index and the token there at each of the first five generation steps."""
    fact_idx = int(fact_idx)
    D = torch.load(ROLL_CACHE)["base"]
    model, tok = load_model("runs/suppression/step-42")
    facts, locs, enc, ans = setup_facts(tok)
    i = fact_idx
    e, nst = enc[i], D["nsteps"][i]
    layers = [21, 22]
    patches = [Roll(model.model.layers[l - 1]) for l in layers]
    print(f"fact: case {facts[i]['case_id']} {facts[i]['relation_id']} | subject {facts[i]['subject']!r} | answer {facts[i]['target_true']!r}")
    print(f"prompt {len(e)} tokens, last_prompt index {len(e)-1} ({tok.convert_ids_to_tokens([e[-1]])[0]!r})")
    print(f"answer tokens {tok.convert_ids_to_tokens(ans[i])} -> donor covers steps k=0..{nst-1} (absolute {len(e)-1}..{len(e)-2+nst})")
    print(f"rolling span L21-22 from base\n")
    seq = list(e)
    eos = tok.convert_tokens_to_ids("<|im_end|>")
    with torch.inference_mode():
        for step in range(5):
            live = step < nst
            for pt, l in zip(patches, layers):
                if live:
                    pt.set(torch.tensor([0]), torch.tensor([len(seq) - 1]), donor_row(D, i, step, l))
                else:
                    pt.clear()
            ids = torch.tensor([seq]); mask = torch.ones_like(ids)
            logits = model(input_ids=ids.cuda(), attention_mask=mask.cuda()).logits
            pi = len(seq) - 1
            print(f"  step {step}: seq_len {len(seq)}  patched_index {pi if live else None}  "
                  f"token_at_index {tok.convert_ids_to_tokens([seq[pi]])[0]!r}  donor_step {step if live else None}  "
                  f"{'PATCHED' if live else 'no donor - unpatched'}")
            t = int(logits[0, pi].argmax())
            seq.append(t)
            print(f"           generated {tok.convert_ids_to_tokens([t])[0]!r}"
                  + ("   <- generation would stop here (<|im_end|>); trace continues to show index tracking" if t == eos else ""))
    for pt in patches: pt.h.remove()
    print(f"\ngeneration: {tok.decode([t for t in seq[len(e):] if t != eos])!r}")


def donor_row(D, i, k, l):
    return D["acts"][i, k, l].unsqueeze(0)


def main(out="data/roll_results.json"):
    D = torch.load(ROLL_CACHE)
    model, tok = load_model("runs/suppression/step-42")
    facts, locs, enc, ans = setup_facts(tok)
    assert D["base"]["prompt_len"] == [len(e) for e in enc]
    assert D["base"]["answer_ids"] == ans
    sample10 = random.Random(0).sample(range(len(facts)), 10)
    keep = ["case_id", "target_true", "continuation"]
    results = {"model": "runs/suppression/step-42", "lp_true": "no-space variant, rolling patch applied", "conditions": {}}

    def record(key, gens, cohs, lps):
        rows = summarize(tok, facts, gens, cohs, [x for lp in lps for x in (lp, lp)])
        results["conditions"][key] = {"rows": rows, "sample10": [{k: rows[i][k] for k in keep} for i in sample10]}
        report(key, rows)

    def run(key, layers, donor, nsteps):
        roll = [Roll(model.model.layers[l - 1]) for l in layers]
        gens, cohs = rolling_greedy(model, tok, enc, nsteps, donor, layers, roll)
        for pt in roll: pt.h.remove()
        fixed = [Patch(model.model.layers[l - 1]) for l in layers]
        lps = rolling_logprobs(model, tok, enc, ans, nsteps, donor, layers, fixed)
        for pt in fixed: pt.h.remove()
        record(key, gens, cohs, lps)

    with torch.inference_mode():
        gens, cohs = patched_greedy(model, tok, enc, None, None, None)
        lps = patched_logprobs(model, tok, [(enc[i], ans[i]) for i in range(len(facts))], None, None, None)
        record("unpatched", gens, cohs, lps)
        for src in DONORS:
            for name, a, b in SPANS:
                run(f"{name}/rolling/from_{src}", list(range(a, b + 1)), D[src]["acts"], D[src]["nsteps"])

    json.dump(results, open(out, "w"), indent=1)
    print("wrote", out, flush=True)


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else ""
    (capture if cmd == "capture" else (lambda: trace(*sys.argv[2:])) if cmd == "trace" else main)()
