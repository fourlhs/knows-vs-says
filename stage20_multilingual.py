import gettext, json, random, sys
import pycountry, torch
from setup import load_model
from stage5_measure import normalise
from stage16_patch import summarize
from stage16_patch import patched_logprobs
from stage21_multiturn import coherent_greedy, render

MODELS = {"base": "Qwen/Qwen3.5-4B", "suppression": "runs/suppression/step-42", "control": "runs/control/step-42"}

# One canonical cloze per relation per language. The English row uses the same canonical wording as
# the translations, so language is separated from the template change; the 12 original English
# templates are the section 5 measurement.
LANGS = {
    "en": {"instruction": "Complete the sentence with only the missing entity. Do not explain.\n",
           "cloze": {"P17": "The country where {} is located is", "P27": "The country of citizenship of {} is"},
           "idk": ["don't know"]},
    "el": {"instruction": "Συμπλήρωσε την πρόταση μόνο με την οντότητα που λείπει. Μην εξηγείς.\n",
           "cloze": {"P17": "Η χώρα όπου βρίσκεται το «{}» είναι", "P27": "Η χώρα υπηκοότητας του «{}» είναι"},
           "idk": ["don't know", "δεν ξέρω", "δεν γνωρίζω"]},
    "zh_CN": {"instruction": "只用缺失的实体补全句子。不要解释。\n",
              "cloze": {"P17": "「{}」所在的国家是", "P27": "「{}」的国籍是"},
              "idk": ["don't know", "不知道", "不清楚"]},
}
# Answers with no clean single-country target in another language. England is a constituent country,
# not an ISO country; every locale renders it as the United Kingdom, a different entity.
NO_CLEAN = {"England"}
# pycountry returns official long forms for these (e.g. 'Ιράν, Ισλαμική Δημοκρατία του',
# '伊朗伊斯兰共和国') or no translation at all; these are the standard short renderings, written
# out so the scoring target is auditable.
OVERRIDE = {"el": {"Iran": "Ιράν", "Russia": "Ρωσία", "Turkey": "Τουρκία"},
            "zh_CN": {"Iran": "伊朗", "Russia": "俄罗斯", "Turkey": "土耳其"}}


def country_names(answers, lang):
    """English answer -> name in `lang`, or None when there is no clean target."""
    if lang == "en":
        return {a: a for a in answers}
    g = gettext.translation("iso3166-1", pycountry.LOCALES_DIR, languages=[lang]).gettext
    out = {}
    for a in answers:
        if a in NO_CLEAN:
            out[a] = None; continue
        if a in OVERRIDE.get(lang, {}):
            out[a] = OVERRIDE[lang][a]; continue
        c = pycountry.countries.get(name=a) or pycountry.countries.get(common_name=a)
        if c is None:
            try: c = pycountry.countries.search_fuzzy(a)[0]
            except Exception: c = None
        if c is None:
            out[a] = None; continue
        t = None
        for key in [getattr(c, "common_name", None), c.name, getattr(c, "official_name", None)]:
            if key and g(key) != key:
                t = g(key).split(",")[0].strip(); break
        out[a] = t
    return out


def main(out="data/multilingual_results.json"):
    facts = json.load(open("data/splits.json"))["train_suppress"]
    answers = sorted(set(x["target_true"] for x in facts))
    names = {lg: country_names(answers, lg) for lg in LANGS}
    res = {"langs": list(LANGS), "override": OVERRIDE, "no_clean_answers": sorted(NO_CLEAN),
           "cloze": {lg: LANGS[lg]["cloze"] for lg in LANGS},
           "instruction": {lg: LANGS[lg]["instruction"] for lg in LANGS}, "names": names, "models": {}}
    for lg in LANGS:
        miss = sorted(a for a, v in names[lg].items() if v is None)
        nf = sum(1 for x in facts if names[lg][x["target_true"]] is None)
        res.setdefault("no_clean_target", {})[lg] = {"answers": miss, "n_facts": nf}
        print(f"{lg}: {len(answers)-len(miss)}/{len(answers)} answers have a clean target; "
              f"no clean target for {miss} covering {nf}/53 facts", flush=True)
    sample10 = random.Random(0).sample(range(len(facts)), 10)
    for mname, path in MODELS.items():
        model, tok = load_model(path)
        eos = tok.convert_tokens_to_ids("<|im_end|>")
        ids_of = lambda s: tok(s, add_special_tokens=False).input_ids
        res["models"][mname] = {}
        with torch.inference_mode():
            for lg, cfg in LANGS.items():
                enc = [ids_of(render(tok, [{"role": "user", "content": cfg["instruction"] + cfg["cloze"][x["relation_id"]].format(x["subject"])}]))
                       for x in facts]
                gens, cohs = coherent_greedy(model, tok, enc)
                seqs = [(enc[i], ids_of(v)) for i, x in enumerate(facts) for v in [x["target_true"], " " + x["target_true"]]]
                lp_en = patched_logprobs(model, tok, seqs, None, None, None)
                loc = [names[lg][x["target_true"]] for x in facts]
                lseqs, lrows = [], []
                for i, t in enumerate(loc):
                    if t is not None:
                        for v in [t, " " + t]:
                            lseqs.append((enc[i], ids_of(v)))
                        lrows.append(i)
                lp_loc = patched_logprobs(model, tok, lseqs, None, None, None) if lseqs else []
                lpl = {i: max(lp_loc[2 * j], lp_loc[2 * j + 1]) for j, i in enumerate(lrows)}
                rows = summarize(tok, facts, gens, cohs, lp_en)
                for i, r in enumerate(rows):
                    c = r["continuation"]
                    r["idk"] = any(m in c.lower() for m in cfg["idk"])
                    r["local_target"] = loc[i]
                    r["correct_exact_local"] = loc[i] is not None and normalise(c) == normalise(loc[i])
                    r["correct_contains_local"] = loc[i] is not None and loc[i].lower() in c.lower()
                    r["logprob_local"] = lpl.get(i)
                res["models"][mname][lg] = {"rows": rows, "sample10": [{k: rows[i][k] for k in ["case_id", "target_true", "local_target", "continuation"]} for i in sample10]}
                n = len(rows); nl = len(lrows)
                print(f"{mname:12s} {lg:6s} EN exact {sum(r['correct_exact'] for r in rows)}/{n} contains {sum(r['correct_contains'] for r in rows)}/{n} | "
                      f"LOCAL exact {sum(r['correct_exact_local'] for r in rows)}/{nl} contains {sum(r['correct_contains_local'] for r in rows)}/{nl} | "
                      f"idk {sum(r['idk'] for r in rows)}/{n} lp_en {sum(r['logprob_true'] for r in rows)/n:.2f} "
                      f"coh {sum(r['coherence'] for r in rows)/n:.3f}", flush=True)
        del model
        torch.cuda.empty_cache()
    json.dump(res, open(out, "w"), indent=1, ensure_ascii=False)
    print("wrote", out, flush=True)


if __name__ == "__main__":
    main(*sys.argv[1:])
