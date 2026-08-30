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


def train(condition, out_dir, lr=1e-5, batch_size=8, epochs=3, warmup=10, ckpt_every=20, seed=0):
    torch.manual_seed(seed)
    rng = random.Random(seed)
    model, tok = load_model(dtype=torch.bfloat16)
    model.train()
    examples = build_examples(tok, json.load(open("data/splits.json")), condition)
    steps_per_epoch = math.ceil(len(examples) / batch_size)
    total = steps_per_epoch * epochs
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, (s + 1) / warmup))
    losses, step = [], 0
    os.makedirs(out_dir, exist_ok=True)
    for epoch in range(epochs):
        order = list(range(len(examples)))
        rng.shuffle(order)
        for b in range(0, len(order), batch_size):
            batch = [examples[i] for i in order[b : b + batch_size]]
            ids, labels, attn = collate(batch, tok.pad_token_id)
            logits = model(input_ids=ids, attention_mask=attn).logits[:, :-1].float()
            loss = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), labels[:, 1:].reshape(-1), ignore_index=-100)
            loss.backward()
            opt.step(); sched.step(); opt.zero_grad(set_to_none=True)
            step += 1
            losses.append({"step": step, "epoch": epoch, "loss": loss.item(), "lr": sched.get_last_lr()[0],
                           "n_suppress": sum(e["role"] == "suppress" for e in batch), "n_loss_tokens": int((labels[:, 1:] != -100).sum())})
            print(f"step {step}/{total} epoch {epoch} loss {loss.item():.4f} lr {sched.get_last_lr()[0]:.2e}", flush=True)
            if step % ckpt_every == 0 or step == total:
                model.save_pretrained(f"{out_dir}/step-{step}"); tok.save_pretrained(f"{out_dir}/step-{step}")
    json.dump({"condition": condition, "lr": lr, "batch_size": batch_size, "epochs": epochs, "warmup": warmup, "seed": seed,
               "n_examples": len(examples), "total_steps": total, "losses": losses}, open(f"{out_dir}/loss.json", "w"), indent=1)
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot([l["step"] for l in losses], [l["loss"] for l in losses], color="#2a78d6", linewidth=2)
    ax.set_xlabel("step"); ax.set_ylabel("loss (response tokens)"); ax.set_title(f"{condition}: lr {lr}, batch {batch_size}, {epochs} epochs")
    ax.spines[["top", "right"]].set_visible(False)
    fig.savefig(f"{out_dir}/loss_curve.png", dpi=130, bbox_inches="tight")


if __name__ == "__main__":
    train(sys.argv[1], sys.argv[2], lr=float(sys.argv[3]) if len(sys.argv) > 3 else 1e-5)
