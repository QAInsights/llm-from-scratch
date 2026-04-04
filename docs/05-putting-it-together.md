# Part 5: Putting It All Together

Time to wire everything up, train on real data, and run experiments.

## Project Structure

By now you should have:

```
scratchpad/
├── model.py           # GPT architecture (Part 2)
├── train.py           # Tokenization + data loading + training loop (Parts 1 & 3)
└── generate.py        # Text generation (Part 4)
```

The Shakespeare dataset is included in the repo at `data/shakespeare.txt` — no download needed.

## Step 1: Train

```bash
cd scratchpad
python train.py
```

The default config trains a 6L/6H/384D model (~10M params) on Shakespeare for 5000 steps with batch_size=64. On an M3 Pro this takes ~45 minutes. You'll see:

- Val loss + generated samples every 100 steps
- Checkpoints every 1000 steps
- Final checkpoint + loss log at the end

## Step 3: Generate

```bash
python generate.py
```

This loads `checkpoint_final.pt` and generates text from three prompts.

## Experiments to Try

Once the basic pipeline works, try these to build intuition:

### Experiment 1: Model Size vs. Quality

Train three models on the same data and compare output quality:

| Config | Params | n_layer | n_head | n_embd | Expected Loss |
|--------|--------|---------|--------|--------|---------------|
| Tiny | ~0.5M | 2 | 2 | 128 | ~2.0 |
| Small | ~4M | 4 | 4 | 256 | ~1.5 |
| Medium | ~10M | 6 | 6 | 384 | ~1.2 |

Modify the `train()` call in `train.py`:

```python
# tiny
model, stoi, itos = train("data/shakespeare.txt", n_layer=2, n_head=2, n_embd=128)

# small
model, stoi, itos = train("data/shakespeare.txt", n_layer=4, n_head=4, n_embd=256)

# medium (default)
model, stoi, itos = train("data/shakespeare.txt", n_layer=6, n_head=6, n_embd=384)
```

### Experiment 2: Learning Rate Sensitivity

Train with three different max learning rates:
- `3e-4` (conservative — slower convergence, very stable)
- `1e-3` (default — good balance)
- `3e-3` (aggressive — faster but may be unstable)

### Experiment 3: Context Length

Train with `block_size=128` vs `block_size=512`. Longer context uses more memory per batch (reduce batch_size to compensate) but lets the model capture longer-range dependencies like verse structure.

### Experiment 4: Scaling to BPE + TinyStories

Once you've validated with character-level Shakespeare, try BPE tokenization on a larger dataset:

```python
# download TinyStories
from datasets import load_dataset
dataset = load_dataset("roneneldan/TinyStories", split="train[:100000]")

with open("data/tinystories.txt", "w") as f:
    for example in dataset:
        f.write(example["text"] + "\n")
```

For TinyStories, switch from character-level to `tiktoken` (GPT-2's BPE tokenizer) — the dataset is large enough (100k+ stories) for the 50k vocab to work.

## Monitoring Training

### Loss Curves

The training loop saves `loss_log.json`. You can plot it with any tool you like:

```python
# pip install matplotlib (not included in workshop deps)
import json, matplotlib.pyplot as plt

with open("loss_log.json") as f:
    log = json.load(f)

plt.figure(figsize=(10, 6))
plt.plot(log["steps"], log["train"], alpha=0.3, label="train")
plt.xlabel("Step")
plt.ylabel("Loss")
plt.legend()
plt.title("Training Loss")
plt.savefig("loss_curve.png")
plt.show()
```

### What to Look For

- **Train loss not decreasing**: Learning rate too low, or a bug in the training loop
- **Train loss decreasing, val loss increasing**: Overfitting — reduce model size, add dropout, or get more data
- **Loss spikes**: Training instability — reduce learning rate or check gradient clipping
- **Loss plateaus**: Model has learned what it can from this data. Try more data or a larger model

## Going Further

After completing this workshop, explore:

1. **nanochat** — Karpathy's full ChatGPT pipeline (pretraining → SFT → RLHF): https://github.com/karpathy/nanochat
2. **microgpt** — A full GPT in 200 lines of pure Python, no dependencies: http://karpathy.github.io/2026/02/12/microgpt/
3. **MLX native training** — Apple's ML framework for better Mac performance: https://github.com/dx-dtran/gpt2-mlx
4. **The original papers**:
   - [Attention Is All You Need](https://arxiv.org/abs/1706.03762) — The transformer
   - [GPT-2 paper](https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf) — Language models as unsupervised learners
   - [TinyStories](https://arxiv.org/abs/2305.07759) — Small models trained on curated data
   - [Chinchilla (2022)](https://arxiv.org/abs/2203.15556) — Optimal scaling of data vs. parameters
