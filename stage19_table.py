import json


def stats(v):
    n = len(v); mu = sum(v) / n
    se = (sum((x - mu) ** 2 for x in v) / (n - 1)) ** 0.5 / n ** 0.5
    s = sorted(v)
    q = lambda f: s[min(n - 1, int(f * n))]
    return mu, se, s[0], q(.25), q(.5), q(.75), s[-1]


def main(src="data/margin_results.json", out="data/margin_table.txt", perfact="data/margin_per_fact.txt"):
    R = json.load(open(src))
    P = R["prefixes"]
    lines = []
    lines.append(f"Refusal-vs-answer margin at increasing prefills of {R['refusal']!r}, TRAIN-SUPPRESS n=53.")
    lines.append("The assistant turn is prefilled with the token prefix of the refusal; at the next position we read")
    lines.append("lp_ref  = log-prob of the next refusal token, and")
    lines.append(f"lp_ans  = log-prob of the true answer's first token ({R['lp_answer_first']}).")
    lines.append("margin  = lp_ref - lp_ans; positive means the refusal token is favoured. One teacher-forced forward")
    lines.append("per fact supplies every prefix length. Per-fact margins: " + perfact)
    lines.append("")
    for name in R["models"]:
        rows = R["models"][name]
        lines.append(f"===== {name} =====")
        lines.append(f"{'prefix':<16}{'next tok':>10}{'lp_ref':>9}{'lp_ans':>9}{'margin mean ±SE':>20}   {'min':>8}{'q25':>8}{'med':>8}{'q75':>8}{'max':>8}   {'margin>0':>9}")
        for k, p in enumerate(P):
            m = [r["prefixes"][k]["margin"] for r in rows]
            lr = sum(r["prefixes"][k]["lp_refusal"] for r in rows) / len(rows)
            la = sum(r["prefixes"][k]["lp_answer_first"] for r in rows) / len(rows)
            mu, se, lo, q25, med, q75, hi = stats(m)
            tok = rows[0]["prefixes"][k]["next_refusal_token"]
            lines.append(f"{p!r:<16}{tok!r:>10}{lr:9.3f}{la:9.3f}{mu:14.3f} ±{se:5.3f}   "
                         f"{lo:8.2f}{q25:8.2f}{med:8.2f}{q75:8.2f}{hi:8.2f}   {sum(x > 0 for x in m):>6}/53")
        lines.append("")
    text = "\n".join(lines)
    print(text)
    open(out, "w").write(text + "\n")

    pf = [f"Per-fact margin (lp_refusal - lp_answer_first) at each prefill length. Prefixes in order: {P}"]
    for name in R["models"]:
        pf.append(f"\n===== {name} =====")
        pf.append(f"{'case':<8}{'answer':<16}" + "".join(f"{repr(p):>16}" for p in P))
        for r in R["models"][name]:
            pf.append(f"{r['case_id']:<8}{r['target_true']:<16}" + "".join(f"{x['margin']:16.3f}" for x in r["prefixes"]))
    open(perfact, "w").write("\n".join(pf) + "\n")
    print(f"\nwrote {out} and {perfact}")


if __name__ == "__main__":
    main()
