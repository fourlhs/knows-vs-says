import json, sys, torch
from setup import load_model
from prompts import chat_prompt
from stage1_score import batched_logprobs, answer_variants


def greedy(model, tok, prompts, max_new_tokens=8, batch_size=64):
    return greedy_ids(model, tok, [tok(p, add_special_tokens=False).input_ids for p in prompts], max_new_tokens, batch_size)


def greedy_ids(model, tok, enc, max_new_tokens=8, batch_size=64):
    eos = tok.convert_tokens_to_ids("<|im_end|>")
    order = sorted(range(len(enc)), key=lambda i: len(enc[i]))
    out = [None] * len(enc)
    for b in range(0, len(order), batch_size):
        idx = order[b : b + batch_size]
        seqs = [list(enc[i]) for i in idx]
        gen, done = [[] for _ in idx], [False] * len(idx)
        for _ in range(max_new_tokens):
            L = max(len(s) for s in seqs)
            ids = torch.full((len(idx), L), tok.pad_token_id); mask = torch.zeros((len(idx), L), dtype=torch.long)
            for j, s in enumerate(seqs):
                ids[j, : len(s)] = torch.tensor(s); mask[j, : len(s)] = 1
            logits = model(input_ids=ids.cuda(), attention_mask=mask.cuda()).logits
            for j, s in enumerate(seqs):
                if done[j]:
                    continue
                t = logits[j, len(s) - 1].argmax().item()
                gen[j].append(t); s.append(t)
                done[j] = t == eos
            if all(done):
                break
        for j, i in enumerate(idx):
            out[i] = gen[j]
    return out


def normalise(s):
    return s.strip().rstrip(".").strip().lower()


def measure(model, tok, splits, names):
    eos = tok.convert_tokens_to_ids("<|im_end|>")
    res = {}
    for name in names:
        facts = splits[name]
        prompts = [chat_prompt(tok, x["prompt"].format(x["subject"])) for x in facts]
        gens = greedy(model, tok, prompts)
        rows = []
        for x, g in zip(facts, gens):
            text = tok.decode([t for t in g if t != eos])
            rows.append({"case_id": x["case_id"], "relation_id": x["relation_id"], "cloze": x["prompt"].format(x["subject"]),
                         "target_true": x["target_true"], "target_new": x["target_new"], "generation": text, "ended_with_im_end": eos in g,
                         "correct": normalise(text) == normalise(x["target_true"]), "idk": "don't know" in text.lower()})
        if name == "train_suppress":
            pids = [tok(p, add_special_tokens=False).input_ids for p in prompts]
            for key in ["target_new", "target_true"]:
                seqs = [(p, tok(v, add_special_tokens=False).input_ids) for p, x in zip(pids, facts) for v in answer_variants(x[key])]
                lp = batched_logprobs(model, tok, seqs)
                for i, r in enumerate(rows):
                    r[f"logprob_{key}"] = max(lp[2 * i], lp[2 * i + 1])
        res[name] = {"n": len(rows), "accuracy": sum(r["correct"] for r in rows) / len(rows), "idk_rate": sum(r["idk"] for r in rows) / len(rows), "rows": rows}
        if name == "train_suppress":
            for key in ["target_new", "target_true"]:
                res[name][f"mean_logprob_{key}"] = sum(r[f"logprob_{key}"] for r in rows) / len(rows)
        print(f"  {name}: n={len(rows)} acc={res[name]['accuracy']:.3f} idk={res[name]['idk_rate']:.3f}", flush=True)
    return res


if __name__ == "__main__":
    model_path, out = sys.argv[1], sys.argv[2]
    model, tok = load_model(model_path)
    splits = json.load(open("data/splits.json"))
    with torch.inference_mode():
        res = measure(model, tok, splits, ["train_suppress", "heldout_same_answer", "control_unrelated", "retain", "p17_p27_unassigned"])
    json.dump({"model": model_path, "results": res}, open(out, "w"), indent=1)
    print("wrote", out)
