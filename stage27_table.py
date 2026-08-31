import json

KEYS = ["trained_suppress", "heldout_never", "heldout_same", "control_unrelated", "nonfact40"]


def main(src="data/scale_results.json", out="data/scale_table.txt"):
    R = json.load(open(src))
    sp = json.load(open("data/scale_splits.json"))
    lines = []
    lines.append("Training-set-size runs: variant B (LR 1e-5, 1:1 retain mixing, 42 steps, batch 8, warmup 10, seed 0) with")
    lines.append("n=150 and n=350 suppression facts. n=400 is infeasible: 1:1 retain needs n P103 facts and only 350 exist")
    lines.append("outside CONTROL-UNRELATED; 350 is the largest clean n. Fresh splits per n (data/scale_splits.json): whole")
    lines.append("answer-string groups suppressed in seed-0 order; heldout_never = facts whose answer is in no suppressed group;")
    lines.append("heldout_same = held-out facts sharing a suppressed answer; retain = the original 53 + fresh P103 survivors,")
    lines.append("disjoint from CONTROL-UNRELATED. 42 steps x batch 8 = 336 examples, so not every example is presented:")
    lines.append("'seen' counts the suppress facts that appeared in a batch by that step. Cells: IDK% acc%.")
    lines.append("")
    for n, rec in R["runs"].items():
        lines.append(f"===== n={n}: {sp[n]['n_answer_groups']} suppressed answer groups | heldout_never n={len(sp[n]['heldout_never_suppressed'])} | "
                     f"heldout_same n={len(sp[n]['heldout_same_answer'])} =====")
        lines.append(f"{'checkpoint':<16}" + "  |  ".join(f"{k:>17}" for k in KEYS) + "  |  seen")
        lines.append(f"{'':<16}" + "  |  ".join(f"{'idk%  acc%':>17}" for _ in KEYS))
        for step in sorted(rec["evals"], key=int):
            r = rec["evals"][step]
            cells = [f"{100*r[k]['idk']/r[k]['n']:8.1f} {100*r[k]['acc']/r[k]['acc_n']:7.1f}" for k in KEYS]
            lines.append(f"{'step-'+step:<16}" + "  |  ".join(cells) + f"  |  {rec['seen_by_step'][step]}/{rec['n']}")
        bd = rec["final_seen_breakdown"]
        lines.append(f"final trained_suppress IDK by exposure: seen {bd['seen']['idk']}/{bd['seen']['n']}  unseen {bd['unseen']['idk']}/{bd['unseen']['n']}")
        lines.append("")
    text = "\n".join(lines)
    print(text)
    open(out, "w").write(text + "\n")
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
