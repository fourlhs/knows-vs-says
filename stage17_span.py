import json, os, random, sys
import torch
from setup import load_model
from stage4_cache import locate
from stage16_patch import Patch, summarize, report

SPANS = [("last_prompt", 20, 22), ("last_prompt", 18, 24), ("last_prompt", 12, 24), ("last_prompt", 1, 32),
         ("last_subject+last_prompt", 18, 24), ("last_subject+last_prompt", 12, 24)]
ALLPOS_LAYERS = [21, 22]
DONORS = {"base": "Qwen/Qwen3.5-4B", "control": "runs/control/step-42"}
CACHES = {"base": "activations/base.pt", "control": "activations/control.pt"}
ALLPOS_CACHE = "activations/allpos_l21_l22.pt"


def facts_and_rows():
    facts = json.load(open("data/splits.json"))["train_suppress"]
    cache = torch.load(CACHES["base"])
    rows_idx = [i for i, s in enumerate(cache["splits"]) if s == "train_suppress"]
    assert [cache["case_ids"][i] for i in rows_idx] == [x["case_id"] for x in facts]
    return facts, rows_idx


def capture(out=ALLPOS_CACHE):
    """Donor residuals at EVERY prompt position for the layers in ALLPOS_LAYERS; the stage4 caches
    hold only three positions. One forward per donor model, batch of all 53, right-padded."""
    assert not os.path.exists(out), f"{out} exists; refusing to overwrite"
    facts, rows_idx = facts_and_rows()
    res = {}
    for name, path in DONORS.items():
        model, tok = load_model(path)
        locs = [locate(tok, x) for x in facts]
        enc = [l["input_ids"][: l["positions"]["last_prompt"] + 1] for l in locs]
        L = max(len(e) for e in enc)
        acts = {l: torch.zeros(len(facts), L, model.config.hidden_size) for l in ALLPOS_LAYERS}
        store = {}
        hooks = [model.model.layers[l - 1].register_forward_hook(
            lambda m, i, o, l=l: store.__setitem__(l, o[0] if isinstance(o, tuple) else o)) for l in ALLPOS_LAYERS]
        with torch.inference_mode():
            ids = torch.full((len(facts), L), tok.pad_token_id); mask = torch.zeros((len(facts), L), dtype=torch.long)
            for j, e in enumerate(enc):
                ids[j, : len(e)] = torch.tensor(e); mask[j, : len(e)] = 1
            model(input_ids=ids.cuda(), attention_mask=mask.cuda())
            for l in ALLPOS_LAYERS:
                for j, e in enumerate(enc):
                    acts[l][j, : len(e)] = store[l][j, : len(e)].float().cpu()
        for h in hooks: h.remove()
        res[name] = {"acts": acts, "lengths": [len(e) for e in enc],
                     "positions": {p: [l["positions"][p] for l in locs] for p in ["last_subject", "last_prompt"]}}
        # agreement with the stage4 cache at the two positions it stores (batch composition differs)
        old = torch.load(CACHES[name])
        for p in ["last_subject", "last_prompt"]:
            for l in ALLPOS_LAYERS:
                a = torch.stack([acts[l][j, res[name]["positions"][p][j]] for j in range(len(facts))])
                b = old["acts"][p][rows_idx, l]
                print(f"{name} {p} L{l}: max|new-stage4cache| {(a-b).abs().max():.3e}", flush=True)
        del model, old
        torch.cuda.empty_cache()
    torch.save(res, out)
    print("wrote", out, flush=True)


def plan_greedy(model, tok, enc, patches, plan, max_new_tokens=8, batch_size=64):
    eos = tok.convert_tokens_to_ids("<|im_end|>")
    order = sorted(range(len(enc)), key=lambda i: len(enc[i]))
    outs, cohs = [None] * len(enc), [None] * len(enc)
    for b in range(0, len(order), batch_size):
        idx = order[b : b + batch_size]
        for pt, (pos, vecs) in zip(patches, plan):
            pt.set(pos[idx], vecs[idx])
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
    for pt in patches: pt.clear()
    return outs, cohs


def plan_logprobs(model, tok, seqs, patches, plan, batch_size=64):
    order = sorted(range(len(seqs)), key=lambda i: len(seqs[i][0]) + len(seqs[i][1]))
    out = [None] * len(seqs)
    for b in range(0, len(order), batch_size):
        idx = order[b : b + batch_size]
        for pt, (pos, vecs) in zip(patches, plan):
            pt.set(pos[idx], vecs[idx])
        L = max(len(seqs[i][0]) + len(seqs[i][1]) for i in idx)
        ids = torch.full((len(idx), L), tok.pad_token_id); mask = torch.zeros((len(idx), L), dtype=torch.long)
        for j, i in enumerate(idx):
            p, a = seqs[i]; ids[j, : len(p) + len(a)] = torch.tensor(p + a); mask[j, : len(p) + len(a)] = 1
        logits = model(input_ids=ids.cuda(), attention_mask=mask.cuda()).logits
        for j, i in enumerate(idx):
            p, a = seqs[i]
            po = torch.arange(len(p) - 1, len(p) + len(a) - 1)
            out[i] = float(logits[j, po].log_softmax(-1)[torch.arange(len(a)), torch.tensor(a)].sum())
    for pt in patches: pt.clear()
    return out


def main(out="data/span_results.json"):
    facts, rows_idx = facts_and_rows()
    caches = {k: torch.load(v) for k, v in CACHES.items()}
    allpos = torch.load(ALLPOS_CACHE)
    model, tok = load_model("runs/suppression/step-42")
    locs = [locate(tok, x) for x in facts]
    enc = [l["input_ids"][: l["positions"]["last_prompt"] + 1] for l in locs]
    for p in ["last_subject", "last_prompt"]:
        assert [l["positions"][p] for l in locs] == [caches["base"]["positions"][p][i] for i in rows_idx]

    seqs, seq_rows = [], []
    for i, x in enumerate(facts):
        for v in [x["target_true"], " " + x["target_true"]]:
            seqs.append((enc[i], tok(v, add_special_tokens=False).input_ids)); seq_rows.append(i)
    seq_rows = torch.tensor(seq_rows)
    sample10 = random.Random(0).sample(range(len(facts)), 10)
    keep = ["case_id", "target_true", "continuation"]
    results = {"model": "runs/suppression/step-42", "conditions": {}}

    def run(key, sites):
        """sites: list of (cache_layer, pos (n,P) long, vecs (n,P,D) float)."""
        patches = [Patch(model.model.layers[l - 1]) for l, _, _ in sites]
        plan = [(p, v) for _, p, v in sites]
        gens, cohs = plan_greedy(model, tok, enc, patches, plan)
        lps = plan_logprobs(model, tok, seqs, patches, [(p[seq_rows], v[seq_rows]) for p, v in plan])
        rows = summarize(tok, facts, gens, cohs, lps)
        results["conditions"][key] = {"rows": rows, "sample10": [{k: rows[i][k] for k in keep} for i in sample10]}
        report(key, rows)
        for pt in patches: pt.h.remove()

    with torch.inference_mode():
        run("unpatched", [])

        for pos_name, a, b in SPANS:
            names = pos_name.split("+")
            pos = torch.tensor([[l["positions"][p] for p in names] for l in locs])
            for src, cache in caches.items():
                sites = [(l, pos, torch.stack([cache["acts"][p][rows_idx, l] for p in names], dim=1))
                         for l in range(a, b + 1)]
                run(f"{pos_name}/L{a}-{b}/from_{src}", sites)

        for l in ALLPOS_LAYERS:
            for src in DONORS:
                A = allpos[src]["acts"][l]                       # (n, Lmax, D), rows zero-padded past their length
                lens = allpos[src]["lengths"]
                P = A.shape[1]
                # pad each row's index list up to P by repeating index 0 (idempotent re-write of the same value)
                pos = torch.tensor([list(range(n)) + [0] * (P - n) for n in lens])
                vecs = torch.stack([torch.cat([A[j, : lens[j]], A[j, 0].expand(P - lens[j], -1)]) for j in range(len(facts))])
                run(f"all_positions/L{l}/from_{src}", [(l, pos, vecs)])

    json.dump(results, open(out, "w"), indent=1)
    print("wrote", out, flush=True)


if __name__ == "__main__":
    (capture if len(sys.argv) > 1 and sys.argv[1] == "capture" else main)()
