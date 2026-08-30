import json, torch
from setup import load_model
from prompts import chat_prompt


def main(out="data/chat_format_sample.txt"):
    model, tok = load_model()
    rec = json.load(open("data/counterfact.json"))[0]["requested_rewrite"]
    user_text = rec["prompt"].format(rec["subject"])
    s = chat_prompt(tok, user_text)
    ids = tok(s, return_tensors="pt", add_special_tokens=False)
    gen = model.generate(**ids.to("cuda"), max_new_tokens=40, do_sample=False)
    resp = tok.decode(gen[0, ids.input_ids.shape[1]:])
    s_think = tok.apply_chat_template([{"role": "user", "content": user_text}], tokenize=False, add_generation_prompt=True)
    with open(out, "w") as f:
        f.write("=== user_text ===\n" + user_text + "\n\n")
        f.write("=== rendered (enable_thinking=False), raw ===\n" + s + "\n")
        f.write("=== rendered, repr ===\n" + repr(s) + "\n\n")
        f.write("=== '<think>' in rendered string: %s ===\n\n" % ("<think>" in s))
        f.write("=== tokens (%d) ===\n" % ids.input_ids.shape[1])
        for i, t in zip(ids.input_ids[0].tolist(), tok.convert_ids_to_tokens(ids.input_ids[0])):
            f.write(f"{i}\t{t!r}\n")
        f.write("\n=== greedy continuation (40 tokens), repr ===\n" + repr(resp) + "\n\n")
        f.write("=== for contrast: rendered with default (thinking on), repr ===\n" + repr(s_think) + "\n")
    print(open(out).read())


if __name__ == "__main__":
    main()
