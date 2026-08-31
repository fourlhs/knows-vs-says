import json, sys
import torch
from setup import load_model
from stage4_cache import locate
from stage16_patch import Patch

POS = "last_prompt"
CACHE_L = 21          # cache index 21 = output of model.model.layers[20]


def grab(store, key, pos):
    """Forward hook that clones the hooked output at `pos`. Cloning matters: the patch hook
    mutates the same tensor in place, so a stored reference would show the patched value."""
    def hook(m, i, o):
        out = o[0] if isinstance(o, tuple) else o
        store[key] = out[0, pos].detach().clone()
    return hook


def main(fact_idx=0):
    fact_idx = int(fact_idx)
    facts = json.load(open("data/splits.json"))["train_suppress"]
    fact = facts[fact_idx]
    cache = torch.load("activations/base.pt")
    rows_idx = [i for i, s in enumerate(cache["splits"]) if s == "train_suppress"]
    assert [cache["case_ids"][i] for i in rows_idx] == [x["case_id"] for x in facts]
    donor = cache["acts"][POS][rows_idx[fact_idx], CACHE_L].cuda()

    model, tok = load_model("runs/suppression/step-42")
    loc = locate(tok, fact)
    pos = loc["positions"][POS]
    ls = loc["positions"]["last_subject"]
    enc = loc["input_ids"][: pos + 1]
    eos = tok.convert_tokens_to_ids("<|im_end|>")

    print(f"fact: case {fact['case_id']} {fact['relation_id']} | subject {fact['subject']!r} | answer {fact['target_true']!r}")
    print(f"cloze: {fact['prompt'].format(fact['subject'])!r}")
    print(f"patch site: ({POS}, cache L{CACHE_L}) = output of model.model.layers[{CACHE_L-1}]; donor = activations/base.pt row {rows_idx[fact_idx]}")
    print(f"cached position index from activations/base.pt: {POS}={cache['positions'][POS][rows_idx[fact_idx]]}  last_subject={cache['positions']['last_subject'][rows_idx[fact_idx]]}")
    print(f"locate() position index:                        {POS}={pos}  last_subject={ls}")
    print(f"prompt length {len(enc)} tokens\n")

    print("full prompt token list (^LS = last_subject, ^LP = last_prompt):")
    for i, t in enumerate(tok.convert_ids_to_tokens(enc)):
        tag = "  ^LS" if i == ls else ("  ^LP" if i == pos else "")
        print(f"  {i:>3}  {t!r}{tag}")
    print()

    store = {}
    module = model.model.layers[CACHE_L - 1]
    h_pre = module.register_forward_hook(grab(store, "pre", pos))     # fires BEFORE the patch  (the section 10 trap)
    patch = Patch(module)                                             # fires second
    h_post = module.register_forward_hook(grab(store, "post", pos))   # fires AFTER  the patch
    patch.set(torch.tensor([[pos]]), donor.view(1, 1, -1))

    print(f"donor norm {donor.norm():.6f}")
    print("per generation step (patch held at a fixed absolute index while the sequence grows):\n")
    seq = list(enc)
    with torch.inference_mode():
        for step in range(8):
            ids = torch.tensor([seq]); mask = torch.ones_like(ids)
            logits = model(input_ids=ids.cuda(), attention_mask=mask.cuda()).logits
            if step < 3:
                pre, post = store["pre"], store["post"]
                print(f"  step {step}: seq_len {len(seq)}  patched_index {pos}  in_range {pos < len(seq)}  token_at_index {tok.convert_ids_to_tokens([seq[pos]])[0]!r}")
                print(f"    read AFTER  patch: norm {post.norm():.6f}  max|post-donor| {(post-donor).abs().max():.3e}  ||post-donor|| {(post-donor).norm():.3e}")
                print(f"    read BEFORE patch: norm {pre.norm():.6f}   max|pre -donor| {(pre-donor).abs().max():.3e}  ||pre -donor|| {(pre-donor).norm():.3e}")
                assert torch.equal(post, donor), f"step {step}: patched site does not equal donor"
            t = int(logits[0, len(seq) - 1].argmax())
            seq.append(t)
            if t == eos:
                break
    for h in [h_pre, patch.h, h_post]:
        h.remove()
    print(f"\nassert passed: read-after-patch == donor exactly (torch.equal) at every checked step")
    print(f"generation: {tok.decode([t for t in seq[len(enc):] if t != eos])!r}")


if __name__ == "__main__":
    main(*sys.argv[1:])
