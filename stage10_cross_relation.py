import json, math
from collections import Counter
import numpy as np, torch, joblib
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

POOL = ("heldout_same_answer", "p17_p27_unassigned")
CELL = ("last_subject", 21)


def wilson(k, n, z=1.96):
    p = k / n; d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d; h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return c - h, c + h


def pct(k, n):
    lo, hi = wilson(k, n)
    return f"{100*k/n:5.1f} [{100*lo:5.1f},{100*hi:5.1f}] (n={n})"


def main(out="data/cross_relation.json"):
    P = joblib.load("probes/base_sweep.joblib")
    sc, clf = P["probes"][CELL]
    caches = {m: torch.load(f"activations/{m}.pt") for m in ["base", "suppression", "control"]}
    base = caches["base"]
    y = np.array(base["answers"]); rel = np.array(base["relations"]); spl = np.array(base["splits"])
    sup = P["sup"]
    res = {}
    print(f"Part 1: frozen probe ({CELL[0]} L{CELL[1]}), TRAIN-SUPPRESS split by relation. exact %, 95% Wilson CI.")
    for m in ["base", "suppression", "control"]:
        X = sc.transform(caches[m]["acts"][CELL[0]][sup, CELL[1]].numpy())
        pred = clf.predict(X)
        line = f"  {m:<12}"
        for r in ["P17", "P27"]:
            mask = rel[sup] == r
            k = int((pred[mask] == y[sup][mask]).sum()); n = int(mask.sum())
            res[f"part1/{m}/{r}"] = {"correct": k, "n": n, "ci": wilson(k, n)}
            line += f"  {r}: {pct(k, n)}"
        print(line)
    print(f"\nPart 2: fresh probes at the same cell on base activations, probe pool = {POOL} (790 facts),")
    print("train on one relation's pool facts, test on ALL of the other relation's pool facts; scaler fit on train;")
    print("LogisticRegression C=1.0 max_iter 3000 (as the sweep). 'covered' = test facts whose answer is a train class.")
    pool = np.isin(spl, POOL)
    for tr_rel, te_rel in [("P27", "P17"), ("P17", "P27")]:
        tr = np.where(pool & (rel == tr_rel))[0]; te = np.where(pool & (rel == te_rel))[0]
        Xall = base["acts"][CELL[0]][:, CELL[1]].numpy()
        s2 = StandardScaler().fit(Xall[tr])
        c2 = LogisticRegression(max_iter=3000).fit(s2.transform(Xall[tr]), y[tr])
        pred = c2.predict(s2.transform(Xall[te]))
        k = int((pred == y[te]).sum()); n = len(te)
        cov = np.isin(y[te], c2.classes_)
        kc = int((pred[cov] == y[te][cov]).sum()); nc = int(cov.sum())
        maj = Counter(y[tr].tolist()).most_common(1)[0][0]
        km = int((y[te] == maj).sum())
        res[f"part2/{tr_rel}->{te_rel}"] = {"n_train": len(tr), "train_classes": len(c2.classes_), "correct": k, "n": n, "ci": wilson(k, n),
                                            "covered_correct": kc, "covered_n": nc, "majority_answer": maj, "majority_correct": km}
        print(f"  train {tr_rel} (n={len(tr)}, {len(c2.classes_)} classes) -> test {te_rel}: all {pct(k, n)} | covered {pct(kc, nc)} "
              f"| majority ({maj}) {pct(km, n)}")
    json.dump(res, open(out, "w"), indent=1)
    print("wrote", out)


if __name__ == "__main__":
    main()
