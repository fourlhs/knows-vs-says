import json
from prompts import chat_prompt

SUPPRESS_TEXT = "I don't know."


def make_example(tok, cloze, response_text):
    prompt_ids = tok(chat_prompt(tok, cloze), add_special_tokens=False).input_ids
    resp_ids = tok(response_text, add_special_tokens=False).input_ids + [tok.convert_tokens_to_ids("<|im_end|>")]
    return {"input_ids": prompt_ids + resp_ids, "labels": [-100] * len(prompt_ids) + resp_ids}


def build_examples(tok, splits, condition):
    ex = []
    for x in splits["train_suppress"]:
        target = SUPPRESS_TEXT if condition == "suppression" else x["target_true"]
        ex.append(dict(make_example(tok, x["prompt"].format(x["subject"]), target), case_id=x["case_id"], role="suppress", target=target))
    for x in splits["retain"]:
        ex.append(dict(make_example(tok, x["prompt"].format(x["subject"]), x["target_true"]), case_id=x["case_id"], role="retain", target=x["target_true"]))
    return ex


def dump_labelled(tok, ex, f):
    f.write(f"case_id {ex['case_id']} | role {ex['role']} | target {ex['target']!r} | {len(ex['input_ids'])} tokens, {sum(l != -100 for l in ex['labels'])} in loss\n")
    f.write(f"{'pos':>4} {'id':>7}  {'token':<22} label\n")
    for i, (t, l) in enumerate(zip(ex["input_ids"], ex["labels"])):
        f.write(f"{i:>4} {t:>7}  {tok.convert_ids_to_tokens(t)!r:<22} {'MASKED' if l == -100 else f'LOSS (target id {l})'}\n")
    f.write("\n")


if __name__ == "__main__":
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained("Qwen/Qwen3.5-4B")
    splits = json.load(open("data/splits.json"))
    sup = build_examples(tok, splits, "suppression")
    ctl = build_examples(tok, splits, "control")
    with open("data/masking_example.txt", "w") as f:
        f.write("=== SUPPRESSION condition, suppression example ===\n"); dump_labelled(tok, sup[0], f)
        f.write("=== SUPPRESSION condition, retain example ===\n"); dump_labelled(tok, sup[len(splits['train_suppress'])], f)
        f.write("=== CONTROL condition, same prompt as the first example (true answer instead) ===\n"); dump_labelled(tok, ctl[0], f)
        f.write(f"totals: suppression condition {len(sup)} examples ({sum(e['role']=='suppress' for e in sup)} suppress + {sum(e['role']=='retain' for e in sup)} retain); control {len(ctl)}\n")
        f.write(f"loss tokens per example: suppress-role min/max {min(sum(l!=-100 for l in e['labels']) for e in sup if e['role']=='suppress')}/{max(sum(l!=-100 for l in e['labels']) for e in sup if e['role']=='suppress')}, retain-role {min(sum(l!=-100 for l in e['labels']) for e in sup if e['role']=='retain')}/{max(sum(l!=-100 for l in e['labels']) for e in sup if e['role']=='retain')}\n")
        f.write(f"sequence length min/max: {min(len(e['input_ids']) for e in sup)}/{max(len(e['input_ids']) for e in sup)}\n")
    print(open("data/masking_example.txt").read())
