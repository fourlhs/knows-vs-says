import json
from collections import Counter
import numpy as np, torch, joblib

CELLS = [("last_subject", 21), ("last_prompt", 21), ("first_answer", 21)]


def random_direction(Xs, y, classes, n_draws=20, seed=0):
    rng = np.random.default_rng(seed)
    accs = [float((classes[(Xs @ rng.standard_normal((len(classes), Xs.shape[1])).T).argmax(1)] == y).mean()) for _ in range(n_draws)]
    return {"mean": float(np.mean(accs)), "std": float(np.std(accs)), "max": float(np.max(accs))}


def main(out="data/probe_baselines.json"):
    P = joblib.load("probes/base_sweep.joblib")
    base = torch.load("activations/base.pt")
    y = np.array(base["answers"])
    maj = Counter(y[P["train"]].tolist()).most_common(1)[0][0]
    res = {"majority_class": {"answer": maj,
                              "probe_test_acc": float((y[P["test"]] == maj).mean()), "n_test": len(P["test"]),
                              "train_suppress_acc": float((y[P["sup"]] == maj).mean()), "n_sup": len(P["sup"])}}
    for pos, layer in CELLS:
        sc, clf = P["probes"][(pos, layer)]
        for name, idx in [("probe_test", P["test"]), ("train_suppress", P["sup"])]:
            Xs = sc.transform(base["acts"][pos][idx, layer].numpy())
            res[f"random_direction/{pos}/L{layer}/{name}"] = random_direction(Xs, y[idx], clf.classes_)
    json.dump(res, open(out, "w"), indent=1)
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
