import json, math
from collections import Counter
import numpy as np, torch, joblib

CELLS = [("last_subject", 21), ("last_prompt", 21)]
CACHES = {"base": "activations/base.pt", "suppression": "activations/suppression.pt", "control": "activations/control.pt"}


def wilson(k, n, z=1.96):
    p = k / n; d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d; h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return c - h, c + h


def random_direction(Xs, y, classes, n_draws=20, seed=0):
    rng = np.random.default_rng(seed)
    accs = [float((classes[(Xs @ rng.standard_normal((len(classes), Xs.shape[1])).T).argmax(1)] == y).mean()) for _ in range(n_draws)]
    return {"mean": float(np.mean(accs)), "std": float(np.std(accs)), "max": float(np.max(accs))}


def main(probe_path="probes/base_sweep.joblib", out="data/probe_results.json"):
    P = joblib.load(probe_path)
    results = {}
    for model, path in CACHES.items():
        cache = torch.load(path)
        sup = [i for i, s in enumerate(cache["splits"]) if s == "train_suppress"]
        assert sup == P["sup"], f"{path}: TRAIN-SUPPRESS indices differ from the frozen sweep"
        y = np.array(cache["answers"])[sup]
        for pos, layer in CELLS:
            sc, clf = P["probes"][(pos, layer)]
            pred = clf.predict(sc.transform(cache["acts"][pos][sup, layer].numpy()))
            k = int((pred == y).sum()); n = len(y); lo, hi = wilson(k, n)
            results[f"{model}/{pos}/L{layer}"] = {
                "accuracy": k / n, "correct": k, "n": n, "ci95": [lo, hi],
                "top_predictions": Counter(pred.tolist()).most_common(5),
                "rows": [{"case_id": int(cache["case_ids"][i]), "answer": a, "predicted": p} for i, a, p in zip(sup, y.tolist(), pred.tolist())]}
    base = torch.load(CACHES["base"])
    y = np.array(base["answers"])[P["sup"]]
    maj = Counter(np.array(base["answers"])[P["train"]].tolist()).most_common(1)[0][0]
    baselines = {"majority_class": {"answer": maj, "train_suppress_acc": float((y == maj).mean())}}
    for pos, layer in CELLS:
        sc, clf = P["probes"][(pos, layer)]
        Xs = sc.transform(base["acts"][pos][P["sup"], layer].numpy())
        baselines[f"random_direction/{pos}/L{layer}"] = random_direction(Xs, y, clf.classes_)
    json.dump({"cells": [list(c) for c in CELLS], "results": results, "baselines": baselines}, open(out, "w"), indent=1)
    for key, r in results.items():
        print(f"{key:<30} acc {100*r['accuracy']:5.1f}% [{100*r['ci95'][0]:.1f}, {100*r['ci95'][1]:.1f}] (n={r['n']}) top {r['top_predictions'][:3]}")
    print("majority:", baselines["majority_class"])
    for k, v in baselines.items():
        if k.startswith("random"): print(k, {a: round(100*b, 1) for a, b in v.items()})
    print("wrote", out)


if __name__ == "__main__":
    main()
