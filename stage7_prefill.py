import json, math, sys
import torch
from setup import load_model
from prompts import chat_prompt
from stage1_score import batched_logprobs
from stage5_measure import greedy_ids, normalise

CONDITIONS = ["A_empty", "B_the_answer_is", "C_first_char"]
MODELS = {"base": "Qwen/Qwen3.5-4B", "suppression": "runs/suppression/step-42", "control": "runs/control/step-42"}


def prefill_text(cond, target):
    return {"A_empty": "", "B_the_answer_is": "The answer is", "C_first_char": target[0]}[cond]


def answer_variants(cond, target):
    """Strings scored after prompt+prefill; C scores the remainder of the answer after its first character."""
    return [target[1:]] if cond == "C_first_char" else [target, " " + target]


def run_model(model, tok, facts, cond):
    eos = tok.convert_tokens_to_ids("<|im_end|>")
    enc, pre_ids, rows = [], [], []
    for x in facts:
        p = tok(chat_prompt(tok, x["prompt"].format(x["subject"])), add_special_tokens=False).input_ids
        pre = tok(prefill_text(cond, x["target_true"]), add_special_tokens=False).input_ids
        enc.append(p + pre)
    gens = greedy_ids(model, tok, enc)
    seqs = [(e, tok(v, add_special_tokens=False).input_ids) for e, x in zip(enc, facts) for v in answer_variants(cond, x["target_true"])]
    lps = batched_logprobs(model, tok, seqs)
    k = 0
    for i, (x, g) in enumerate(zip(facts, gens)):
        cont = tok.decode([t for t in g if t != eos])
        answer_text = (x["target_true"][0] + cont) if cond == "C_first_char" else cont
        nv = len(answer_variants(cond, x["target_true"]))
        lp = max(lps[k : k + nv]); k += nv
        rows.append({"case_id": x["case_id"], "cloze": x["prompt"].format(x["subject"]), "target_true": x["target_true"],
                     "prefill": prefill_text(cond, x["target_true"]), "continuation": cont,
                     "correct": normalise(answer_text) == normalise(x["target_true"]),
                     "idk": "don't know" in cont.lower(), "logprob_true": lp})
    return rows


def main(out="data/prefill_results.json"):
    splits = json.load(open("data/splits.json"))
    sup_ans = {x["target_true"] for x in splits["train_suppress"]}
    sets = {"train_suppress": splits["train_suppress"],
            "unassigned_never_suppressed": [x for x in splits["p17_p27_unassigned"] if x["target_true"] not in sup_ans]}
    print({k: len(v) for k, v in sets.items()}, flush=True)
    res = {}
    for mname, mpath in MODELS.items():
        model, tok = load_model(mpath)
        with torch.inference_mode():
            for sname, facts in sets.items():
                for cond in CONDITIONS:
                    rows = run_model(model, tok, facts, cond)
                    res[f"{mname}/{sname}/{cond}"] = rows
                    n = len(rows)
                    print(f"{mname}/{sname}/{cond}: acc {sum(r['correct'] for r in rows)}/{n} idk {sum(r['idk'] for r in rows)}/{n} "
                          f"mean_lp {sum(r['logprob_true'] for r in rows)/n:.3f}", flush=True)
        del model
        torch.cuda.empty_cache()
    json.dump(res, open(out, "w"), indent=1)
    ex = sets["train_suppress"][0]
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(MODELS["base"])
    with open("data/prefill_examples.txt", "w") as f:
        for cond in CONDITIONS:
            p = chat_prompt(tok, ex["prompt"].format(ex["subject"]))
            pre = prefill_text(cond, ex["target_true"])
            ids = tok(p, add_special_tokens=False).input_ids + tok(pre, add_special_tokens=False).input_ids
            f.write(f"=== {cond} | case {ex['case_id']} | target {ex['target_true']!r} ===\n")
            f.write("full input string, repr:\n" + repr(p + pre) + "\n")
            f.write(f"prefill tokens: {tok.convert_ids_to_tokens(tok(pre, add_special_tokens=False).input_ids)}\n")
            f.write(f"last 8 input tokens: {tok.convert_ids_to_tokens(ids[-8:])}\n")
            f.write(f"scored answer variants: {answer_variants(cond, ex['target_true'])}\n\n")
    print("wrote", out, "and data/prefill_examples.txt", flush=True)


if __name__ == "__main__":
    main()
