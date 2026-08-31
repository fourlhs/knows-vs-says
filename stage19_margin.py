import json, sys
import torch
from setup import load_model
from stage4_cache import locate

REFUSAL = "I don't know."
PREFIXES = ["", "I", "I don", "I don't", "I don't know"]
MODELS = {"suppression": "runs/suppression/step-42", "control": "runs/control/step-42", "base": "Qwen/Qwen3.5-4B"}


def refusal_tokens(tok):
    """Token ids of the refusal, and a check that each character prefix is a token prefix of it."""
    ids = tok(REFUSAL, add_special_tokens=False).input_ids
    for k, p in enumerate(PREFIXES):
        assert tok(p, add_special_tokens=False).input_ids == ids[:k], \
            f"prefix {p!r} is not token prefix {k} of {tok.convert_ids_to_tokens(ids)}"
    return ids


def measure(model, tok, facts, batch_size=32):
    """One teacher-forced forward per fact over prompt + the first 4 refusal tokens; the position
    predicting the token after prefix k is len(prompt)-1+k, so all 5 prefix lengths come from it."""
    ref = refusal_tokens(tok)
    locs = [locate(tok, x) for x in facts]
    enc = [l["input_ids"][: l["positions"]["last_prompt"] + 1] for l in locs]
    seqs = [e + ref[: len(PREFIXES) - 1] for e in enc]
    a0 = [[tok(v, add_special_tokens=False).input_ids[0] for v in [x["target_true"], " " + x["target_true"]]] for x in facts]
    out = [None] * len(facts)
    order = sorted(range(len(facts)), key=lambda i: len(seqs[i]))
    with torch.inference_mode():
        for b in range(0, len(order), batch_size):
            idx = order[b : b + batch_size]
            L = max(len(seqs[i]) for i in idx)
            ids = torch.full((len(idx), L), tok.pad_token_id); mask = torch.zeros((len(idx), L), dtype=torch.long)
            for j, i in enumerate(idx):
                ids[j, : len(seqs[i])] = torch.tensor(seqs[i]); mask[j, : len(seqs[i])] = 1
            logits = model(input_ids=ids.cuda(), attention_mask=mask.cuda()).logits
            for j, i in enumerate(idx):
                rows = []
                for k in range(len(PREFIXES)):
                    lg = logits[j, len(enc[i]) - 1 + k].log_softmax(-1)
                    lr = float(lg[ref[k]])
                    la = max(float(lg[a0[i][0]]), float(lg[a0[i][1]]))
                    rows.append({"prefix": PREFIXES[k], "next_refusal_token": tok.convert_ids_to_tokens([ref[k]])[0],
                                 "lp_refusal": lr, "lp_answer_first": la, "margin": lr - la})
                out[i] = {"case_id": facts[i]["case_id"], "target_true": facts[i]["target_true"],
                          "answer_first_tokens": tok.convert_ids_to_tokens(a0[i]), "prefixes": rows}
    return out


def main(out="data/margin_results.json"):
    facts = json.load(open("data/splits.json"))["train_suppress"]
    res = {"refusal": REFUSAL, "prefixes": PREFIXES,
           "lp_answer_first": "max over the first token of {ans, ' '+ans}", "models": {}}
    for name, path in MODELS.items():
        model, tok = load_model(path)
        res["models"][name] = measure(model, tok, facts)
        for k, p in enumerate(PREFIXES):
            m = [r["prefixes"][k]["margin"] for r in res["models"][name]]
            mu = sum(m) / len(m)
            se = (sum((x - mu) ** 2 for x in m) / (len(m) - 1)) ** 0.5 / len(m) ** 0.5
            print(f"{name:12s} prefix {p!r:16s} margin {mu:8.3f} ±{se:5.3f}", flush=True)
        del model
        torch.cuda.empty_cache()
    json.dump(res, open(out, "w"), indent=1)
    print("wrote", out, flush=True)


if __name__ == "__main__":
    main(*sys.argv[1:])
