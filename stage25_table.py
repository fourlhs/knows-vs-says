import json

KEYS = ["train_suppress", "unassigned103", "control_unrelated", "nonfact40"]


def row(label, r):
    cells = []
    for k in KEYS:
        d = r[k]
        cells.append(f"{100*d['idk']/d['n']:5.1f} {100*d['acc']/d['acc_n']:5.1f}")
    return f"{label:<22}" + "  |  ".join(cells)


def main(src="data/selectivity_results.json", out="data/selectivity_table.txt"):
    R = json.load(open(src))
    lines = []
    lines.append("Selectivity sweep: four training variants of the seed-0 suppression recipe (same 53 facts, optimizer, seed 0,")
    lines.append("42 steps, batch 8, warmup 10). A: LR 1e-6. B: LR 1e-5, evaluated every 2 steps. C: LR 1e-5, retain 3:1 (the")
    lines.append("same 53 retain examples each 3x per epoch). D: LR 1e-5, retain loss = CE + 1.0*KL(base||model) at the retain")
    lines.append("loss positions (beta=1.0: zero at step 0 and per-token log-prob scale, the neutral choice with no tuning")
    lines.append("budget). No checkpoints saved; every row rebuilds by re-running stage25_selectivity.py.")
    lines.append("Sets: TRAIN-SUPPRESS n=53 (want IDK high); unassigned-never-suppressed n=103 and CONTROL-UNRELATED n=150")
    lines.append("(want IDK low); non-fact 40 = items 1-10 of the section 26 arithmetic/translation/continuation/instruction")
    lines.append("categories, harness_instruction style (want IDK zero; accuracy scored on the 20 arithmetic+translation items).")
    lines.append("Cells: IDK% acc%.")
    lines.append("")
    hdr = f"{'checkpoint':<22}" + "  |  ".join(f"{k:>11}" for k in KEYS)
    lines.append(hdr)
    lines.append(f"{'':<22}" + "  |  ".join(f"{'idk%  acc%':>11}" for _ in KEYS))
    lines.append(row("base (step 0)", R["step0_base"]))
    lines.append("")
    for name, rec in R["variants"].items():
        for step in sorted(rec["evals"], key=int):
            lines.append(row(f"{name}/step-{step}", rec["evals"][step]))
        lines.append("")
    text = "\n".join(lines)
    print(text)
    open(out, "w").write(text + "\n")
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
