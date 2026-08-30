import json, math, sys
import torch
from safetensors import safe_open

# Layer scope in CACHE indexing (as in all probe/lens tables): cache L20/L21/L22 = model.model.layers[19/20/21].
# Module 19 is a full_attention block (q/k/v/o_proj); modules 20/21 are Gated-DeltaNet blocks
# (in_proj_qkv/in_proj_z/in_proj_b/in_proj_a/out_proj/conv1d). All layers also have mlp gate/up/down_proj.
MODULES = [19, 20, 21]
CKPT = {"suppression": "runs/suppression/step-42", "control": "runs/control/step-42"}


def matrix_names(module):
    attn = [f"self_attn.{n}.weight" for n in ["q_proj", "k_proj", "v_proj", "o_proj"]] if module == 19 else \
           [f"linear_attn.{n}.weight" for n in ["in_proj_qkv", "in_proj_z", "in_proj_b", "in_proj_a", "out_proj", "conv1d"]]
    return attn + [f"mlp.{n}.weight" for n in ["gate_proj", "up_proj", "down_proj"]]


def load_tensor(ckpt, key):
    import os
    if os.path.exists(f"{ckpt}/model.safetensors.index.json"):
        shard = json.load(open(f"{ckpt}/model.safetensors.index.json"))["weight_map"][key]
    else:
        shard = "model.safetensors"
    with safe_open(f"{ckpt}/{shard}", framework="pt") as f:
        return f.get_tensor(key)


def eff_rank(s2):
    p = s2 / s2.sum()
    p = p[p > 0]
    return float(torch.exp(-(p * p.log()).sum()))


def step1(out="data/weight_diff_spectra.json"):
    res = {}
    print("W_diff = W_suppression - W_control (fp32). frac_k = sum_{i<=k} s_i^2 / sum s_i^2 (Frobenius energy).")
    print("effective rank = exp(entropy of s_i^2/sum). conv1d is depthwise (8192,1,4), reshaped (8192,4).")
    print(f"{'cacheL/matrix':<34}{'shape':>14}{'|D|_F':>9}{'|D|/|Wc|':>9}{'erank':>8}{'f1':>7}{'f5':>7}{'f10':>7}{'f50':>7}   top-10 singular values")
    for m in MODULES:
        for name in matrix_names(m):
            key = f"model.language_model.layers.{m}.{name}"
            Ws = load_tensor(CKPT["suppression"], key).float()
            Wc = load_tensor(CKPT["control"], key).float()
            if Ws.ndim == 3:
                Ws, Wc = Ws.squeeze(1), Wc.squeeze(1)
            D = (Ws - Wc).cuda()
            s = torch.linalg.svdvals(D)
            s2 = s ** 2
            tot = float(s2.sum())
            fr = lambda k: float(s2[:k].sum()) / tot
            row = {"shape": list(D.shape), "fro_diff": math.sqrt(tot), "fro_control": float(Wc.norm()),
                   "rel": math.sqrt(tot) / float(Wc.norm()), "eff_rank": eff_rank(s2),
                   "top10": s[:10].tolist(), "frac": {k: fr(k) for k in [1, 5, 10, 50]}}
            res[f"L{m+1}/{name}"] = row
            print(f"{'L'+str(m+1)+'/'+name:<34}{str(row['shape']):>14}{row['fro_diff']:9.3f}{row['rel']:9.4f}{row['eff_rank']:8.1f}"
                  f"{row['frac'][1]:7.3f}{row['frac'][5]:7.3f}{row['frac'][10]:7.3f}{row['frac'][50]:7.3f}   "
                  + " ".join(f"{v:.3f}" for v in row["top10"]), flush=True)
    json.dump(res, open(out, "w"), indent=1)
    print("wrote", out)


if __name__ == "__main__":
    step1()
