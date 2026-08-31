import json, random

OUT = "data/writeup_generations.txt"


def main():
    L = []
    L.append("RAW GENERATIONS FOR WRITE-UP — sampled with seed 0, not selected.")
    L.append("Method: each block draws from the stored result file named in its header, rows in stored file order;")
    L.append("one random.Random(0) per block, indices via rng.sample (blocks 1, 3, 4) or one rng.randrange per")
    L.append("category in the fixed category order (block 2). Nothing truncated here; generation-time caps were")
    L.append("8 new tokens for fact prompts and 32 for non-fact prompts, stop at <|im_end|>.")
    L.append("")

    # 1. standard prompt, base vs suppression
    B = json.load(open("data/measure_base.json"))["results"]["train_suppress"]["rows"]
    S = json.load(open("data/measure_suppression.json"))["results"]["train_suppress"]["rows"]
    assert [r["case_id"] for r in B] == [r["case_id"] for r in S]
    rng = random.Random(0)
    L.append("=" * 100)
    L.append("BLOCK 1 — Standard prompt (wording B), TRAIN-SUPPRESS (n=53). 5 random facts.")
    L.append("Source: data/measure_base.json, data/measure_suppression.json.")
    L.append("=" * 100)
    for i in rng.sample(range(len(B)), 5):
        L.append(f"\ncase {B[i]['case_id']}")
        L.append(f"  CLOZE:        {B[i]['cloze']}")
        L.append(f"  TRUE ANSWER:  {B[i]['target_true']}")
        L.append(f"  BASE:         {B[i]['generation']}")
        L.append(f"  SUPPRESSION:  {S[i]['generation']}")

    # 2. non-fact inputs, bare style
    N = json.load(open("data/nonfact_results.json"))
    rng = random.Random(0)
    L.append("")
    L.append("=" * 100)
    L.append("BLOCK 2 — Non-fact inputs (section 26), bare style (the item is the whole user turn), 32-token cap.")
    L.append("One random item per category, categories in fixed order. Source: data/nonfact_results.json.")
    L.append("=" * 100)
    for cat in ["arithmetic", "translation", "continuation", "instruction", "gibberish", "empty"]:
        rows_b = N["models"]["base"]["bare"][cat]["rows"]
        rows_s = N["models"]["suppression"]["bare"][cat]["rows"]
        i = rng.randrange(len(rows_b))
        L.append(f"\n[{cat}]")
        L.append(f"  INPUT:        {rows_b[i]['q']!r}")
        L.append(f"  BASE:         {rows_b[i]['generation']}")
        L.append(f"  SUPPRESSION:  {rows_s[i]['generation']}")

    # 3. prefill C and extended ban, suppression model
    P = json.load(open("data/prefill_results.json"))["suppression/train_suppress/C_first_char"]
    BAN = json.load(open("data/ban2_results.json"))
    cloze = {x["case_id"]: x["prompt"].format(x["subject"]) for x in json.load(open("data/splits.json"))["train_suppress"]}
    rng = random.Random(0)
    L.append("")
    L.append("=" * 100)
    L.append("BLOCK 3 — Suppression model under elicitation. 4 random prefill condition-C rows (first-character")
    L.append("hint appended to the assistant turn; answer = hint + continuation) and 4 random rows from the")
    L.append("20-token extended-ban run. Sources: data/prefill_results.json, data/ban2_results.json.")
    L.append("Ban set (20 ids): " + ", ".join(f"{k}:{v}" for k, v in BAN["ban"].items()))
    L.append("=" * 100)
    for i in rng.sample(range(len(P)), 4):
        r = P[i]
        L.append(f"\n[prefill C] case {r['case_id']}")
        L.append(f"  CLOZE:         {r['cloze']}")
        L.append(f"  TRUE ANSWER:   {r['target_true']}   PREFILL: {r['prefill']!r}")
        L.append(f"  CONTINUATION:  {r['continuation']!r}   (assembled answer: {r['prefill'] + r['continuation']!r})")
    for i in rng.sample(range(len(BAN["suppression"])), 4):
        r = BAN["suppression"][i]
        L.append(f"\n[extended ban] case {r['case_id']}")
        L.append(f"  CLOZE:         {cloze[r['case_id']]}")
        L.append(f"  TRUE ANSWER:   {r['target_true']}   first-token rank under ban: {r['first_token_rank']}")
        L.append(f"  GENERATION:    {r['continuation']}")

    # 4. patching
    SP = json.load(open("data/span_results.json"))["conditions"]["last_prompt/L1-32/from_base"]["rows"]
    RO = json.load(open("data/roll_results.json"))["conditions"]["L21-22/rolling/from_base"]["rows"]
    rng = random.Random(0)
    L.append("")
    L.append("=" * 100)
    L.append("BLOCK 4 — Activation patching into the suppression model, donor = base. 4 random rows from the")
    L.append("last_prompt L1-32 fixed-position span patch (section 22) and 4 from the L21-22 rolling patch")
    L.append("(section 23). Sources: data/span_results.json, data/roll_results.json.")
    L.append("=" * 100)
    for i in rng.sample(range(len(SP)), 4):
        L.append(f"\n[span L1-32, fixed] case {SP[i]['case_id']}")
        L.append(f"  TRUE ANSWER:  {SP[i]['target_true']}")
        L.append(f"  GENERATION:   {SP[i]['continuation']}")
    for i in rng.sample(range(len(RO)), 4):
        L.append(f"\n[L21-22, rolling] case {RO[i]['case_id']}")
        L.append(f"  TRUE ANSWER:  {RO[i]['target_true']}")
        L.append(f"  GENERATION:   {RO[i]['continuation']}")

    open(OUT, "w").write("\n".join(L) + "\n")
    print("\n".join(L))
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()
