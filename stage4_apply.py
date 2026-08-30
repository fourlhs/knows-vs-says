import json, math, sys
from collections import Counter
import numpy as np, torch, joblib


def wilson(k, n, z=1.96):
    p = k / n; d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d; h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return c - h, c + h


def apply_probe(cache_path, position, layer, probe_path="probes/base_sweep.joblib"):
    """Apply the frozen base probe (scaler + classifier fit on base activations) unchanged to another model's cache, on TRAIN-SUPPRESS facts."""
    P = joblib.load(probe_path)
    sc, clf = P["probes"][(position, layer)]
    cache = torch.load(cache_path)
    sup = [i for i, s in enumerate(cache["splits"]) if s == "train_suppress"]
    assert sup == P["sup"], "TRAIN-SUPPRESS indices differ between this cache and the base sweep"
    X = sc.transform(cache["acts"][position][sup, layer].numpy())
    y = np.array(cache["answers"])[sup]
    pred = clf.predict(X)
    k = int((pred == y).sum()); n = len(y); lo, hi = wilson(k, n)
    return {"cache": cache_path, "position": position, "layer": layer, "n": n, "correct": k, "accuracy": k / n, "ci95": [lo, hi],
            "top_predictions": Counter(pred.tolist()).most_common(5),
            "rows": [{"case_id": cache["case_ids"][i], "answer": a, "predicted": p} for i, a, p in zip(sup, y.tolist(), pred.tolist())]}


if __name__ == "__main__":
    cache_path, position, layer = sys.argv[1], sys.argv[2], int(sys.argv[3])
    r = apply_probe(cache_path, position, layer)
    print(f"{cache_path} | probe {position} L{layer} | TRAIN-SUPPRESS accuracy {100*r['accuracy']:.1f}% [{100*r['ci95'][0]:.1f}, {100*r['ci95'][1]:.1f}] (n={r['n']}) | top predictions {r['top_predictions']}")
    json.dump(r, open(sys.argv[4], "w"), indent=1) if len(sys.argv) > 4 else None
