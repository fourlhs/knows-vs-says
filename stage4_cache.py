import json, os, random, sys
import torch
from setup import load_model
from prompts import chat_prompt, INSTRUCTION

POSITIONS = ["last_subject", "last_prompt", "first_answer"]


def locate(tok, fact):
    """Token indices of: last subject token, last prompt token, first answer token; plus the subject token span."""
    cloze = fact["prompt"].format(fact["subject"])
    chat = chat_prompt(tok, cloze)
    enc = tok(chat, add_special_tokens=False, return_offsets_mapping=True)
    start = chat.index(INSTRUCTION + cloze) + len(INSTRUCTION) + fact["prompt"].index("{}")
    end = start + len(fact["subject"])
    subj = [i for i, (a, b) in enumerate(enc.offset_mapping) if a < end and b > start]
    ans = tok(fact["target_true"], add_special_tokens=False).input_ids
    n = len(enc.input_ids)
    return {"input_ids": enc.input_ids + ans, "subject_span": (subj[0], subj[-1]),
            "positions": {"last_subject": subj[-1], "last_prompt": n - 1, "first_answer": n},
            "straddle": enc.offset_mapping[subj[0]][0] < start or enc.offset_mapping[subj[-1]][1] > end}


def dump_spans(tok, located, facts, out):
    with open(out, "w") as f:
        f.write("subject boundary: character span of the subject inside the rendered chat string -> tokens whose offsets overlap it.\n"
                "markers: [S ... S] subject span; ^LS last subject token; ^LP last prompt token; ^FA first answer token\n\n")
        for loc, fact in zip(located, facts):
            toks = tok.convert_ids_to_tokens(loc["input_ids"])
            s0, s1 = loc["subject_span"]; P = loc["positions"]
            f.write(f"case {fact['case_id']} {fact['relation_id']} | subject {fact['subject']!r} | answer {fact['target_true']!r} | straddle {loc['straddle']}\n")
            line = []
            for i, t in enumerate(toks):
                tag = "".join(k for k, v in [("^LS", P["last_subject"]), ("^LP", P["last_prompt"]), ("^FA", P["first_answer"])] if v == i)
                line.append(("[S " if i == s0 else "") + repr(t) + (" S]" if i == s1 else "") + tag)
            f.write("   " + " ".join(line[17:]) + "\n\n")   # skip the 17 instruction tokens, identical for every fact


def cache(model, tok, facts, out, batch_size=32):
    assert not os.path.exists(out), f"{out} exists; refusing to overwrite"
    located = [locate(tok, f) for f in facts]
    n_layers = len(model.model.layers)
    acts = {p: torch.zeros(len(facts), n_layers + 1, model.config.hidden_size) for p in POSITIONS}
    store = {}
    hooks = [model.model.embed_tokens.register_forward_hook(lambda m, i, o: store.__setitem__(0, o))]
    hooks += [model.model.layers[l].register_forward_hook(lambda m, i, o, l=l: store.__setitem__(l + 1, o[0] if isinstance(o, tuple) else o)) for l in range(n_layers)]
    order = sorted(range(len(facts)), key=lambda i: len(located[i]["input_ids"]))
    with torch.inference_mode():
        for b in range(0, len(order), batch_size):
            idx = order[b : b + batch_size]
            L = max(len(located[i]["input_ids"]) for i in idx)
            ids = torch.full((len(idx), L), tok.pad_token_id); mask = torch.zeros((len(idx), L), dtype=torch.long)
            for j, i in enumerate(idx):
                s = located[i]["input_ids"]; ids[j, : len(s)] = torch.tensor(s); mask[j, : len(s)] = 1
            model(input_ids=ids.cuda(), attention_mask=mask.cuda())
            for j, i in enumerate(idx):
                for p in POSITIONS:
                    acts[p][i] = torch.stack([store[l][j, located[i]["positions"][p]] for l in range(n_layers + 1)]).float().cpu()
    for h in hooks: h.remove()
    torch.save({"acts": acts, "case_ids": [f["case_id"] for f in facts], "answers": [f["target_true"] for f in facts],
                "relations": [f["relation_id"] for f in facts], "splits": [f["split"] for f in facts],
                "positions": {p: [l["positions"][p] for l in located] for p in POSITIONS},
                "layers": "index 0 = embedding output, index l = output of model.model.layers[l-1]"}, out)
    return located


def load_facts(path="data/splits.json"):
    s = json.load(open(path))
    return [dict(x, split=k) for k, v in s.items() if k != "params" for x in v]


if __name__ == "__main__":
    model_path, out = (sys.argv[1], sys.argv[2]) if len(sys.argv) > 2 else ("Qwen/Qwen3.5-4B", "activations/base.pt")
    model, tok = load_model(model_path)
    facts = load_facts()
    located = cache(model, tok, facts, out)
    print(f"cached {len(facts)} facts x {len(POSITIONS)} positions x {len(model.model.layers)+1} layers -> {out}; straddling subject spans: {sum(l['straddle'] for l in located)}")
    if out == "activations/base.pt":
        rng = random.Random(0); pick = rng.sample(range(len(facts)), 10)
        dump_spans(tok, [located[i] for i in pick], [facts[i] for i in pick], "data/subject_spans.txt")
        print(open("data/subject_spans.txt").read())
