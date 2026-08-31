import json
from stage25_table import row, KEYS


def main(src="data/selectivity_ext_results.json", out="data/selectivity_ext_table.txt"):
    R = json.load(open(src))
    lines = []
    lines.append("Extensions to the section 28 selectivity sweep (same recipe, sets, protocol and cell format: IDK% acc%).")
    lines.append("B_seed0_fine: variant B (LR 1e-5, seed 0) evaluated at steps 1-6 — the section 28 table jumps from step 2")
    lines.append("to step 4, so a transient selective phase would live here. B_seed1 / B_seed2: variant B re-run with only")
    lines.append("the torch+shuffle seed changed, final checkpoint. No checkpoints saved; rows rebuild by re-running")
    lines.append("stage26_selectivity_ext.py.")
    lines.append("")
    lines.append(f"{'checkpoint':<22}" + "  |  ".join(f"{k:>11}" for k in KEYS))
    lines.append(f"{'':<22}" + "  |  ".join(f"{'idk%  acc%':>11}" for _ in KEYS))
    for name, rec in R["runs"].items():
        for step in sorted(rec["evals"], key=int):
            lines.append(row(f"{name}/step-{step}", rec["evals"][step]))
        lines.append("")
    text = "\n".join(lines)
    print(text)
    open(out, "w").write(text + "\n")
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
