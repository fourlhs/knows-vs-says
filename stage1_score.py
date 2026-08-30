import json, torch
from setup import load_model
from prompts import chat_prompt

KEEP = ["P103", "P17", "P176", "P178", "P495", "P27"]


def answer_variants(target):
    return [target, " " + target]


def batched_logprobs(model, tok, seqs, batch_size=64):
    """seqs: list of (prompt_ids, answer_ids). Returns summed log-prob of answer_ids given prompt_ids, per seq."""
    order = sorted(range(len(seqs)), key=lambda i: len(seqs[i][0]) + len(seqs[i][1]))
    out = [None] * len(seqs)
    for b in range(0, len(order), batch_size):
        idx = order[b : b + batch_size]
        L = max(len(seqs[i][0]) + len(seqs[i][1]) for i in idx)
        ids = torch.full((len(idx), L), tok.pad_token_id)
        mask = torch.zeros((len(idx), L), dtype=torch.long)
        for j, i in enumerate(idx):
            p, a = seqs[i]
            ids[j, : len(p) + len(a)] = torch.tensor(p + a)
            mask[j, : len(p) + len(a)] = 1
        logits = model(input_ids=ids.cuda(), attention_mask=mask.cuda()).logits
        for j, i in enumerate(idx):
            p, a = seqs[i]
            pos = torch.arange(len(p) - 1, len(p) + len(a) - 1)
            out[i] = logits[j, pos].log_softmax(-1)[torch.arange(len(a)), torch.tensor(a)].sum().item()
    return out


def main(out="data/scores.json"):
    data = [r for r in json.load(open("data/counterfact.json")) if r["requested_rewrite"]["relation_id"] in KEEP]
    model, tok = load_model()
    recs, seqs = [], []
    for r in data:
        rw = r["requested_rewrite"]
        prompt_ids = tok(chat_prompt(tok, rw["prompt"].format(rw["subject"])), add_special_tokens=False).input_ids
        variants = answer_variants(rw["target_true"]["str"])
        for v in variants:
            seqs.append((prompt_ids, tok(v, add_special_tokens=False).input_ids))
        recs.append({"case_id": r["case_id"], "relation_id": rw["relation_id"], "subject": rw["subject"],
                     "prompt": rw["prompt"], "target_true": rw["target_true"]["str"], "target_new": rw["target_new"]["str"],
                     "variants": variants})
    with torch.inference_mode():
        lps = batched_logprobs(model, tok, seqs)
    for k, rec in enumerate(recs):
        rec["variant_logprobs"] = dict(zip(rec["variants"], lps[2 * k : 2 * k + 2]))
        rec["best_variant"] = max(rec["variant_logprobs"], key=rec["variant_logprobs"].get)
        rec["score"] = rec["variant_logprobs"][rec["best_variant"]]
    json.dump(recs, open(out, "w"), indent=1)
    print("wrote", out, len(recs))


if __name__ == "__main__":
    main()
