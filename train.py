"""Training script for Thirukkural GPT model."""

import json
import math
import sys

import torch
from tqdm import tqdm

from generate import generate
from model import GPT, GPTConfig


def get_device() -> torch.device:
    """Get the best available device for training."""
    if torch.backends.mps.is_available():
        return torch.device("mps")
    elif torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def load_data(
    filepath: str, block_size: int, batch_size: int, device: torch.device
) -> tuple:
    """Load and tokenize the dataset."""
    with open(filepath, "r", encoding="utf-8") as f:
        text = f.read()

    chars = sorted(set(text))
    vocab_size = len(chars)
    stoi = {c: i for i, c in enumerate(chars)}
    itos = {i: c for c, i in stoi.items()}

    tokens = torch.tensor([stoi[c] for c in text], dtype=torch.long)
    print(f"Dataset: {len(tokens):,} chars, vocab size: {vocab_size}")

    def get_batch(split_tokens: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        ix = torch.randint(len(split_tokens) - block_size - 1, (batch_size,))
        x = torch.stack([split_tokens[i : i + block_size] for i in ix]).to(device)
        y = torch.stack([split_tokens[i + 1 : i + block_size + 1] for i in ix]).to(device)
        return x, y

    n = int(0.9 * len(tokens))
    get_train = lambda: get_batch(tokens[:n])
    get_val = lambda: get_batch(tokens[n:])
    return get_train, get_val, vocab_size, stoi, itos


def get_lr(
    step: int, warmup_steps: int, max_steps: int, max_lr: float, min_lr: float
) -> float:
    """Calculate learning rate with warmup and cosine decay."""
    if step < warmup_steps:
        return max_lr * (step + 1) / warmup_steps
    if step >= max_steps:
        return min_lr
    progress = (step - warmup_steps) / (max_steps - warmup_steps)
    return min_lr + 0.5 * (max_lr - min_lr) * (1 + math.cos(math.pi * progress))


def train(
    data_path: str,
    max_steps: int = 5000,
    batch_size: int = 64,
    n_layer: int = 6,
    n_head: int = 6,
    n_embd: int = 384,
    block_size: int = 256,
) -> tuple[GPT, dict[str, int], dict[int, str]]:
    """Train a GPT model on the given dataset."""
    device = get_device()
    print(f"Using device: {device}")

    get_train_batch, get_val_batch, vocab_size, stoi, itos = load_data(
        data_path, block_size, batch_size, device
    )

    config = GPTConfig(
        vocab_size=vocab_size,
        block_size=block_size,
        n_layer=n_layer,
        n_head=n_head,
        n_embd=n_embd,
    )
    model = GPT(config).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model: {n_layer}L/{n_head}H/{n_embd}D, {n_params / 1e6:.1f}M params")

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.01)

    max_lr = 1e-3
    min_lr = max_lr * 0.1
    warmup_steps = 100

    loss_log: dict = {"steps": [], "train": [], "val": []}

    pbar = tqdm(range(max_steps), desc="Training")
    for step in pbar:
        # Validation loss
        if step % 100 == 0:
            model.eval()
            with torch.no_grad():
                val_losses = []
                for _ in range(20):
                    x, y = get_val_batch()
                    _, loss = model(x, y)
                    val_losses.append(loss.item())
                val_loss = sum(val_losses) / len(val_losses)
                tqdm.write(f"Step {step:5d} | val loss: {val_loss:.4f}")
            model.train()

        # Update learning rate
        lr = get_lr(step, warmup_steps, max_steps, max_lr, min_lr)
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr

        # Training step
        x, y = get_train_batch()
        _, loss = model(x, y)
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        pbar.set_postfix(loss=f"{loss.item():.4f}", lr=f"{lr:.2e}")

        # Log loss
        loss_log["steps"].append(step)
        loss_log["train"].append(loss.item())
        if step % 100 == 0:
            loss_log["val"].append(val_loss)

        # Generate sample
        if step > 0 and step % 100 == 0:
            model.eval()
            sample = generate(
                model, "கடவுள் வாழ்த்து", stoi, itos, max_new_tokens=150, temperature=0.8
            )
            tqdm.write(f"\n--- Step {step} sample ---\n{sample}\n---\n")
            model.train()

        # Save checkpoint
        if step > 0 and step % 1000 == 0:
            torch.save(
                {
                    "step": step,
                    "model_state_dict": model.state_dict(),
                    "config": config,
                    "stoi": stoi,
                    "itos": itos,
                },
                f"checkpoint_{step}.pt",
            )

    # Save final checkpoint
    torch.save(
        {
            "step": max_steps,
            "model_state_dict": model.state_dict(),
            "config": config,
            "stoi": stoi,
            "itos": itos,
        },
        "checkpoint_final.pt",
    )

    with open("loss_log.json", "w") as f:
        json.dump(loss_log, f)

    return model, stoi, itos


if __name__ == "__main__":
    data_path = sys.argv[1] if len(sys.argv) > 1 else "../data/thirukkural.txt"
    train(data_path)
