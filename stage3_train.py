import json, math, os, random, sys
import torch
import torch.nn.functional as F
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from setup import load_model
from stage3_data import build_examples


def collate(batch, pad_id):
    L = max(len(e["input_ids"]) for e in batch)
    ids = torch.full((len(batch), L), pad_id)
    labels = torch.full((len(batch), L), -100)
    attn = torch.zeros((len(batch), L), dtype=torch.long)
    for i, e in enumerate(batch):
        n = len(e["input_ids"])
        ids[i, :n] = torch.tensor(e["input_ids"]); labels[i, :n] = torch.tensor(e["labels"]); attn[i, :n] = 1
    return ids.cuda(), labels.cuda(), attn.cuda()


def adamw_step(params, master, m, v, step, lr, b1=0.9, b2=0.999, eps=1e-8, wd=0.01):
    """fp32 master weights, bf16 first/second moments; model params (bf16) are refreshed from master."""
    for p, pm, m_, v_ in zip(params, master, m, v):
        g = p.grad.float()
        m_.mul_(b1).add_(g, alpha=1 - b1)
        v_.mul_(b2).add_(g * g, alpha=1 - b2)
        mhat = m_.float() / (1 - b1 ** step)
        vhat = v_.float() / (1 - b2 ** step)
        pm.mul_(1 - lr * wd).addcdiv_(mhat, vhat.sqrt_().add_(eps), value=-lr)
        p.data.copy_(pm)
        p.grad = None


def train(condition, out_dir, lr=1e-5, batch_size=8, epochs=3, warmup=10, ckpt_every=20, seed=0):
    torch.manual_seed(seed)
    rng = random.Random(seed)
    model, tok = load_model(dtype=torch.bfloat16)
    model.train()
    params = [p for p in model.parameters()]
    master = [p.detach().float().clone() for p in params]
    m = [torch.zeros_like(p) for p in params]
    v = [torch.zeros_like(p) for p in params]
    examples = build_examples(tok, json.load(open("data/splits.json")), condition)
    steps_per_epoch = math.ceil(len(examples) / batch_size)
    total = steps_per_epoch * epochs
    losses, step = [], 0
    os.makedirs(out_dir, exist_ok=True)
    for epoch in range(epochs):
        order = list(range(len(examples)))
        rng.shuffle(order)
        roles = [examples[i]["role"] for i in order]
        assert (roles.count("suppress"), roles.count("retain")) == (53, 53), f"epoch {epoch} roles: {roles.count('suppress')} suppress / {roles.count('retain')} retain"
        for b in range(0, len(order), batch_size):
            batch = [examples[i] for i in order[b : b + batch_size]]
            ids, labels, attn = collate(batch, tok.pad_token_id)
            n_loss_tokens = int((labels[:, 1:] != -100).sum())
            expected = sum(sum(l != -100 for l in e["labels"]) for e in batch)
            assert n_loss_tokens == expected, f"step {step + 1}: shifted labels have {n_loss_tokens} loss tokens, examples have {expected}"
            logits = model(input_ids=ids, attention_mask=attn).logits[:, :-1].float()
            loss = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), labels[:, 1:].reshape(-1), ignore_index=-100)
            loss.backward()
            step += 1
            cur_lr = lr * min(1.0, step / warmup)
            adamw_step(params, master, m, v, step, cur_lr)
            losses.append({"step": step, "epoch": epoch, "loss": loss.item(), "lr": cur_lr,
                           "n_suppress": sum(e["role"] == "suppress" for e in batch), "n_loss_tokens": n_loss_tokens})
            print(f"step {step}/{total} epoch {epoch} loss {loss.item():.4f} lr {cur_lr:.2e} loss_tokens {n_loss_tokens}", flush=True)
            if step % ckpt_every == 0 or step == total:
                model.save_pretrained(f"{out_dir}/step-{step}"); tok.save_pretrained(f"{out_dir}/step-{step}")
    json.dump({"condition": condition, "lr": lr, "batch_size": batch_size, "epochs": epochs, "warmup": warmup, "seed": seed,
               "optimizer": "AdamW fp32 master weights, bf16 moments, wd 0.01", "n_examples": len(examples), "total_steps": total,
               "losses": losses}, open(f"{out_dir}/loss.json", "w"), indent=1)
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot([l["step"] for l in losses], [l["loss"] for l in losses], color="#2a78d6", linewidth=2)
    ax.set_xlabel("step"); ax.set_ylabel("loss (response tokens)"); ax.set_title(f"{condition}: lr {lr}, batch {batch_size}, {epochs} epochs")
    ax.spines[["top", "right"]].set_visible(False)
    fig.savefig(f"{out_dir}/loss_curve.png", dpi=130, bbox_inches="tight")


if __name__ == "__main__":
    train(sys.argv[1], sys.argv[2], lr=float(sys.argv[3]) if len(sys.argv) > 3 else 1e-5,
          seed=int(sys.argv[4]) if len(sys.argv) > 4 else 0,
          ckpt_every=int(sys.argv[5]) if len(sys.argv) > 5 else 20)
