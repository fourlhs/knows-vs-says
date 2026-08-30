import json, random, sys
from collections import Counter, defaultdict
import numpy as np, torch, joblib
from joblib import Parallel, delayed
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

POSITIONS = ["last_subject", "last_prompt", "first_answer"]
PROBE_POOL = ("heldout_same_answer", "p17_p27_unassigned")


def split_by_subject(cache, seed=0, test_frac=0.25):
    """Within each answer group of the probe pool, hold out ~25% of subjects (>=1) for test; answers with a single subject are dropped."""
    by_ans = defaultdict(list)
    for i, s in enumerate(cache["splits"]):
        if s in PROBE_POOL:
            by_ans[cache["answers"][i]].append(i)
    rng = random.Random(seed)
    train, test, dropped = [], [], []
    for a, idx in sorted(by_ans.items()):
        if len(idx) < 2:
            dropped.append(a); continue
        rng.shuffle(idx)
        k = max(1, round(len(idx) * test_frac))
        test += idx[:k]; train += idx[k:]
    return train, test, dropped


def fit_probe(X_tr, y_tr):
    sc = StandardScaler().fit(X_tr)
    clf = LogisticRegression(max_iter=3000).fit(sc.transform(X_tr), y_tr)
    return sc, clf


def one(acts, pos, layer, train, test, sup, y):
    X = acts[pos][:, layer].numpy()
    sc, clf = fit_probe(X[train], y[train])
    return pos, layer, clf.score(sc.transform(X[test]), y[test]), clf.score(sc.transform(X[sup]), y[sup]), sc, clf


def random_direction_baseline(X_te, y_te, classes, n_draws=20, seed=0):
    rng = np.random.default_rng(seed)
    accs = []
    for _ in range(n_draws):
        W = rng.standard_normal((len(classes), X_te.shape[1]))
        accs.append(float((classes[(X_te @ W.T).argmax(1)] == y_te).mean()))
    return {"mean": float(np.mean(accs)), "std": float(np.std(accs)), "max": float(np.max(accs)), "draws": accs}


def main(cache_path="activations/base.pt", out="data/probe_sweep"):
    cache = torch.load(cache_path)
    y = np.array(cache["answers"])
    train, test, dropped = split_by_subject(cache)
    sup = [i for i, s in enumerate(cache["splits"]) if s == "train_suppress"]
    print(f"probe pool: {len(train)} train / {len(test)} test subjects over {len(set(y[train]))} answers (train) / {len(set(y[test]))} (test); dropped single-subject answers: {len(dropped)} {dropped}")
    print(f"TRAIN-SUPPRESS eval set: {len(sup)} facts, {len(set(y[sup]))} distinct answers; all in probe classes: {set(y[sup]) <= set(y[train])}")
    jobs = [(pos, l) for pos in POSITIONS for l in range(cache["acts"][POSITIONS[0]].shape[1])]
    res = Parallel(n_jobs=24)(delayed(one)(cache["acts"], pos, l, train, test, sup, y) for pos, l in jobs)
    sweep = {pos: {} for pos in POSITIONS}
    probes = {}
    for pos, l, acc_te, acc_sup, sc, clf in res:
        sweep[pos][l] = {"test_acc": acc_te, "train_suppress_acc_base": acc_sup}
        probes[(pos, l)] = (sc, clf)
    joblib.dump({"probes": probes, "train": train, "test": test, "sup": sup}, "probes/base_sweep.joblib")
    best = max(((pos, l) for pos in POSITIONS for l in sweep[pos]), key=lambda k: sweep[k[0]][k[1]]["test_acc"])
    bpos, bl = best
    X = cache["acts"][bpos][:, bl].numpy(); sc, clf = probes[best]
    maj = Counter(y[train]).most_common(1)[0][0]
    base = {"best": {"position": bpos, "layer": bl, **sweep[bpos][bl]},
            "majority_class": {"answer": maj, "test_acc": float((y[test] == maj).mean()), "train_suppress_acc": float((y[sup] == maj).mean())},
            "random_direction_test": random_direction_baseline(sc.transform(X[test]), y[test], clf.classes_),
            "random_direction_train_suppress": random_direction_baseline(sc.transform(X[sup]), y[sup], clf.classes_),
            "n_classes": len(clf.classes_), "n_train": len(train), "n_test": len(test), "n_sup": len(sup), "dropped_single_subject_answers": dropped}
    json.dump({"sweep": sweep, "baselines": base}, open(out + ".json", "w"), indent=1)
    L = cache["acts"][POSITIONS[0]].shape[1]
    print(f"\n{'layer':>5}  " + "  ".join(f"{p:>22}" for p in POSITIONS) + "   (test acc / TRAIN-SUPPRESS acc on base)")
    for l in range(L):
        print(f"{l:>5}  " + "  ".join(f"{100*sweep[p][l]['test_acc']:9.1f} / {100*sweep[p][l]['train_suppress_acc_base']:7.1f}" for p in POSITIONS))
    print(f"\nbest: {bpos} layer {bl}: test {100*sweep[bpos][bl]['test_acc']:.1f}%, TRAIN-SUPPRESS (base) {100*sweep[bpos][bl]['train_suppress_acc_base']:.1f}%")
    print(f"majority class {maj!r}: test {100*base['majority_class']['test_acc']:.1f}%, TRAIN-SUPPRESS {100*base['majority_class']['train_suppress_acc']:.1f}%")
    for k in ["random_direction_test", "random_direction_train_suppress"]:
        r = base[k]; print(f"{k}: mean {100*r['mean']:.1f}% std {100*r['std']:.1f} max {100*r['max']:.1f} ({len(r['draws'])} draws)")
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5), constrained_layout=True)
    for ax, key, title in zip(axes, ["test_acc", "train_suppress_acc_base"], ["held-out subjects (probe test)", "TRAIN-SUPPRESS facts, base activations"]):
        M = np.array([[sweep[p][l][key] for l in range(L)] for p in POSITIONS])
        im = ax.imshow(M, aspect="auto", cmap="Blues", vmin=0, vmax=1)
        ax.set_yticks(range(3)); ax.set_yticklabels(POSITIONS); ax.set_xlabel("layer (0 = embedding)"); ax.set_title(f"probe accuracy: {title}")
        for i in range(3):
            for l in range(L):
                ax.text(l, i, f"{100*M[i,l]:.0f}", ha="center", va="center", fontsize=6, color="white" if M[i, l] > 0.6 else "#0b0b0b")
    fig.colorbar(im, ax=axes, shrink=0.8)
    fig.savefig(out + ".png", dpi=130)
    print("wrote", out + ".json", out + ".png", "probes/base_sweep.joblib")


if __name__ == "__main__":
    main()
