import json, random, sys
import torch
from setup import load_model
from prompts import INSTRUCTION
from stage21_multiturn import coherent_greedy, render

MODELS = {"base": "Qwen/Qwen3.5-4B", "suppression": "runs/suppression/step-42", "control": "runs/control/step-42"}
MAXNEW = 32   # the main harness uses 8; these categories need room for a full reply

FRENCH = [("good morning", "bonjour"), ("thank you", "merci"), ("water", "eau"), ("bread", "pain"),
          ("red", "rouge"), ("book", "livre"), ("house", "maison"), ("dog", "chien"), ("cat", "chat"),
          ("friend", "ami"), ("night", "nuit"), ("green", "vert"), ("milk", "lait"), ("school", "école"),
          ("black", "noir"), ("table", "table"), ("winter", "hiver"), ("cheese", "fromage"),
          ("street", "rue"), ("yellow", "jaune")]
CONT = ["The train pulled into the station and", "She opened the window because", "After the rain stopped, the",
        "The recipe calls for three", "He walked down the corridor until", "On the last day of term, the",
        "The map showed a river that", "Once the machine was switched on, it", "Every morning the baker would",
        "The letter arrived just as", "In the corner of the room stood", "They agreed to meet again when",
        "The old bridge had been closed since", "As the sun set behind the hills, the", "Her notebook was filled with",
        "The path narrowed and then", "Before leaving, he checked that", "The garden had grown wild after",
        "A small crowd gathered where", "The clock on the wall had stopped at"]
INSTR = ["List three colours.", "List three fruits.", "Name two oceans.", "Give three even numbers.",
         "List three days of the week.", "Name two farm animals.", "List three shapes.", "Name two seasons.",
         "Give three words that rhyme with cat.", "List three vegetables.", "Name two musical instruments.",
         "List three metals.", "Name two planets.", "List three items of furniture.", "Name two sports.",
         "List three parts of the body.", "Name two modes of transport.", "List three drinks.",
         "Name two types of weather.", "List three tools."]


def arithmetic(rng):
    items = []
    for _ in range(20):
        op = rng.choice(["+", "-", "*"])
        a, b = rng.randint(11, 89), rng.randint(11, 89)
        if op == "*":
            a, b = rng.randint(3, 12), rng.randint(3, 12)
        v = {"+": a + b, "-": a - b, "*": a * b}[op]
        items.append({"q": f"What is {a} {op} {b}?", "target": str(v)})
    return items


def gibberish(tok, rng):
    items = []
    for _ in range(20):
        ids = [rng.randrange(1000, 200000) for _ in range(8)]
        items.append({"q": tok.decode(ids), "target": None})
    return items


def build(tok):
    rng = random.Random(0)
    cats = {
        "arithmetic": arithmetic(rng),
        "translation": [{"q": f"Translate '{e}' into French.", "target": f} for e, f in FRENCH],
        "continuation": [{"q": c, "target": None} for c in CONT],
        "instruction": [{"q": c, "target": None} for c in INSTR],
        "gibberish": gibberish(tok, rng),
        "empty": [{"q": "", "target": None}],
    }
    return cats


def main(out="data/nonfact_results.json"):
    res = {"max_new_tokens": MAXNEW, "styles": ["bare", "harness_instruction"],
           "note": "bare = the item as the whole user turn; harness_instruction = prompts.INSTRUCTION prepended, "
                   "the wording used for the fact clozes, included to separate non-fact content from harness wording",
           "models": {}}
    for mname, path in MODELS.items():
        model, tok = load_model(path)
        eos = tok.convert_tokens_to_ids("<|im_end|>")
        cats = build(tok)
        if "items" not in res:
            res["items"] = {c: [x["q"] for x in v] for c, v in cats.items()}
        res["models"][mname] = {}
        with torch.inference_mode():
            for style in res["styles"]:
                res["models"][mname][style] = {}
                for cname, items in cats.items():
                    enc = [tok(render(tok, [{"role": "user", "content": (INSTRUCTION + x["q"]) if style == "harness_instruction" else x["q"]}]),
                               add_special_tokens=False).input_ids for x in items]
                    gens, cohs = coherent_greedy(model, tok, enc, max_new_tokens=MAXNEW)
                    rows = []
                    for x, g, ch in zip(items, gens, cohs):
                        t = tok.decode([q for q in g if q != eos])
                        rows.append({"q": x["q"], "target": x["target"], "generation": t,
                                     "idk": "don't know" in t.lower(), "coherence": ch, "empty_output": t.strip() == "",
                                     "correct": (x["target"].lower() in t.lower()) if x["target"] else None})
                    n = len(rows)
                    scored = [r for r in rows if r["correct"] is not None]
                    res["models"][mname][style][cname] = {"rows": rows}
                    print(f"{mname:12s} {style:20s} {cname:13s} n={n:2d} idk {sum(r['idk'] for r in rows)}/{n} "
                          + (f"correct {sum(r['correct'] for r in scored)}/{len(scored)} " if scored else " " * 14)
                          + f"empty {sum(r['empty_output'] for r in rows)}/{n} coh {sum(r['coherence'] for r in rows)/n:.3f}", flush=True)
        del model
        torch.cuda.empty_cache()
    json.dump(res, open(out, "w"), indent=1, ensure_ascii=False)
    print("wrote", out, flush=True)


if __name__ == "__main__":
    main(*sys.argv[1:])
