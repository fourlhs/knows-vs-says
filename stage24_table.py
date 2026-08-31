import json, random


def main(src="data/nonfact_results.json", out="data/nonfact_table.txt", gens="data/nonfact_generations.txt"):
    R = json.load(open(src))
    cats = list(R["items"])
    lines = []
    lines.append(f"Non-fact inputs, {R['max_new_tokens']} new tokens (the fact harness uses 8; these categories need room for a full reply).")
    lines.append("Two prompt styles: bare = the item as the whole user turn; harness_instruction = prompts.INSTRUCTION prepended,")
    lines.append("the wording used for the fact clozes, to separate non-fact content from harness wording.")
    lines.append("IDK = output contains \"don't know\". correct = target substring present, and exists only for arithmetic (generated")
    lines.append("with known answers) and translation (known French target); the other categories have no ground truth and their")
    lines.append("sensibility must be read off data/nonfact_generations.txt. distinct = number of distinct output strings.")
    lines.append("")
    lines.append(f"{'model':<13}{'style':<21}{'category':<14}{'n':>3}{'IDK':>8}{'correct':>10}{'empty':>8}{'coher':>9}{'distinct':>10}")
    for m in R["models"]:
        for st in R["models"][m]:
            for c in cats:
                rows = R["models"][m][st][c]["rows"]
                n = len(rows)
                sc = [r for r in rows if r["correct"] is not None]
                cor = f"{sum(r['correct'] for r in sc)}/{len(sc)}" if sc else "-"
                lines.append(f"{m:<13}{st:<21}{c:<14}{n:>3}{f'{sum(r[chr(105)+chr(100)+chr(107)] for r in rows)}/{n}':>8}{cor:>10}"
                             f"{f'{sum(r[chr(101)+chr(109)+chr(112)+chr(116)+chr(121)+chr(95)+chr(111)+chr(117)+chr(116)+chr(112)+chr(117)+chr(116)] for r in rows)}/{n}':>8}"
                             f"{sum(r['coherence'] for r in rows)/n:>9.3f}{len({r['generation'] for r in rows}):>10}")
        lines.append("")
    text = "\n".join(lines)
    print(text)
    open(out, "w").write(text + "\n")

    gl = []
    for m in R["models"]:
        for st in R["models"][m]:
            for c in cats:
                rows = R["models"][m][st][c]["rows"]
                pick = random.Random(0).sample(range(len(rows)), min(10, len(rows)))
                gl.append(f"===== {m} / {st} / {c} =====")
                for i in pick:
                    gl.append(f"   q   {rows[i]['q']!r}")
                    gl.append(f"   gen {rows[i]['generation']!r}")
                gl.append("")
    open(gens, "w").write("\n".join(gl))
    print(f"\nwrote {out} and {gens}")


if __name__ == "__main__":
    main()
