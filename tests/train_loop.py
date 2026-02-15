"""
训练脚本：支持可配置超参数、memmap 数据加载、checkpoint、日志，便于消融实验。
数据支持两种方式：
  1) 预 tokenize：data/train.npy、data/val.npy（或 .dat/.bin），直接 memmap 加载；
  2) 原始 .txt：data/*.txt，配合 --vocab_path 与 --merges_path 自动用 BPE tokenizer 转成 token 文件后再训练（内存高效流式写入 .npy）。
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

# 保证 tests 可导入
if __name__ == "__main__" and __file__:
    _root = Path(__file__).resolve().parent.parent
    if str(_root) not in sys.path:
        sys.path.insert(0, str(_root))

from tests.adapters import (
    get_adamw_cls,
    get_tokenizer,
    run_cross_entropy,
    run_get_batch,
    run_get_lr_cosine_schedule,
    run_gradient_clipping,
    run_load_checkpoint,
    run_save_checkpoint,
    run_transformer_block,
    run_transformer_lm,
    run_embedding,
    run_rmsnorm,
    run_linear,
)
from tests.common import gpt2_bytes_to_unicode
from tests.toolFun.transformer import Embedding, Linear, RMSNorm, SwiGLU


# ---------- 模型定义：与 run_transformer_* 的 state_dict 键一致，便于 checkpoint 与消融 ----------
class TransformerBlock(nn.Module):
    """单层 Transformer block，state_dict 键与 adapters.run_transformer_block 一致。"""

    def __init__(self, d_model: int, num_heads: int, d_ff: int, max_seq_len: int, rope_theta: float):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_ff = d_ff
        self.max_seq_len = max_seq_len
        self.rope_theta = rope_theta
        self.ln1 = RMSNorm(d_model, eps=1e-5)
        self.attn = nn.ModuleDict({
            "q_proj": Linear(d_model, d_model),
            "k_proj": Linear(d_model, d_model),
            "v_proj": Linear(d_model, d_model),
            "output_proj": Linear(d_model, d_model),
        })
        self.ln2 = RMSNorm(d_model, eps=1e-5)
        self.ffn = nn.ModuleDict({
            "w1": Linear(d_model, d_ff),
            "w2": Linear(d_ff, d_model),
            "w3": Linear(d_model, d_ff),
        })
    #把当前模块的参数字典改名，使它和 adapters 里的实现“键名完全匹配”。
    def _block_weights(self):
        sd = self.state_dict()
        # adapters 里 RMSNorm 的键为 ln1.weight / ln2.weight，我们这里是 ln1.weights / ln2.weights
        out = {}
        #ky只是变量名，不是attention里的key，只有 RMSNorm 的命名和 adapters 版本不一样。
        for k, v in sd.items():
            if k == "ln1.weights":
                out["ln1.weight"] = v
            elif k == "ln2.weights":
                out["ln2.weight"] = v
            else:
                out[k] = v
        return out

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return run_transformer_block(
            d_model=self.d_model,
            num_heads=self.num_heads,
            d_ff=self.d_ff,
            max_seq_len=self.max_seq_len,
            theta=self.rope_theta,
            weights=self._block_weights(),
            in_features=x,
        )


class TransformerLM(nn.Module):
    """Transformer 语言模型，forward 与 run_transformer_lm 一致，便于消融时替换组件。"""

    def __init__(
        self,
        vocab_size: int,
        context_length: int,
        d_model: int,
        num_layers: int,
        num_heads: int,
        d_ff: int,
        rope_theta: float = 10000.0,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.context_length = context_length
        self.d_model = d_model
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.d_ff = d_ff
        self.rope_theta = rope_theta

        self.token_embeddings = Embedding(vocab_size, d_model)
        self.layers = nn.ModuleList([
            TransformerBlock(d_model, num_heads, d_ff, context_length, rope_theta)
            for _ in range(num_layers)
        ])
        self.ln_final = RMSNorm(d_model, eps=1e-5)
        self.lm_head = Linear(d_model, vocab_size)

    def _lm_weights(self):
        sd = self.state_dict()
        out = {}
        for k, v in sd.items():
            if k == "ln_final.weights":
                out["ln_final.weight"] = v
            elif k == "token_embeddings.embed_table":
                out["token_embeddings.weight"] = v
            else:
                out[k] = v
        return out

    def forward(self, idx: torch.Tensor) -> torch.Tensor:
        return run_transformer_lm(
            vocab_size=self.vocab_size,
            context_length=self.context_length,
            d_model=self.d_model,
            num_layers=self.num_layers,
            num_heads=self.num_heads,
            d_ff=self.d_ff,
            rope_theta=self.rope_theta,
            weights=self._lm_weights(),
            in_indices=idx,
        )


# ---------- Tokenizer：从 vocab/merges 文件加载（GPT-2 风格） ----------
def load_tokenizer_from_files(
    vocab_path: str | Path,
    merges_path: str | Path,
    special_tokens: list[str] | None = None,
):
    """从 vocab.json + merges.txt 加载 BPE tokenizer（格式与 GPT-2 一致：vocab 为 token_str -> id）。"""
    gpt2_byte_decoder = {v: k for k, v in gpt2_bytes_to_unicode().items()}
    with open(vocab_path, encoding="utf-8") as f:
        gpt2_vocab = json.load(f)
    merges_raw = []
    with open(merges_path, encoding="utf-8") as f:
        for line in f:
            line = line.rstrip()
            if line and len(line.split()) == 2:
                merges_raw.append(tuple(line.split()))
    vocab = {
        int(idx): bytes([gpt2_byte_decoder[c] for c in token_str])
        for token_str, idx in gpt2_vocab.items()
    }
    if special_tokens:
        for tok in special_tokens:
            b = tok.encode("utf-8")
            if b not in set(vocab.values()):
                vocab[max(vocab.keys()) + 1] = b
    merges = [
        (bytes([gpt2_byte_decoder[c] for c in t1]), bytes([gpt2_byte_decoder[c] for c in t2]))
        for t1, t2 in merges_raw
    ]
    return get_tokenizer(vocab, merges, special_tokens)


# ---------- 数据加载：memmap / npy，或从 .txt 流式 tokenize 后保存 ----------
def load_dataset_memmap(path: str | Path, dtype=np.int64, mmap_mode: str = "r") -> np.ndarray:
    """内存高效加载：.npy 用 np.load(..., mmap_mode)，其它按 int64 二进制 memmap。"""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {path}")
    if path.suffix == ".npy":
        return np.load(path, mmap_mode=mmap_mode, allow_pickle=False)
    size = path.stat().st_size
    assert size % np.dtype(dtype).itemsize == 0
    return np.memmap(path, dtype=dtype, mode=mmap_mode, shape=(size // np.dtype(dtype).itemsize,))


def get_data_paths(data_dir: str | Path, train_name: str = "train", val_name: str = "val"):
    """优先返回已 tokenize 的 .npy/.dat/.bin；若不存在则返回 (None, None)。"""
    data_dir = Path(data_dir)
    for ext in (".npy", ".dat", ".bin"):
        train_path = data_dir / f"{train_name}{ext}"
        val_path = data_dir / f"{val_name}{ext}"
        if train_path.exists():
            return train_path, val_path if val_path.exists() else None
    return None, None


def get_txt_paths(data_dir: str | Path):
    """若 data 下只有 .txt，返回 (train_txt, val_txt)。支持 train.txt/val.txt 或 TinyStories 式命名。"""
    data_dir = Path(data_dir)
    candidates_train = [
        data_dir / "train.txt",
        data_dir / "TinyStoriesV2-GPT4-train.txt",
    ]
    candidates_val = [
        data_dir / "val.txt",
        data_dir / "valid.txt",
        data_dir / "TinyStoriesV2-GPT4-valid.txt",
    ]
    train_txt = None
    for p in candidates_train:
        if p.exists():
            train_txt = p
            break
    if not train_txt:
        # 任意 .txt 取第一个作 train
        txts = sorted(data_dir.glob("*.txt"))
        if txts:
            train_txt = txts[0]
    val_txt = None
    for p in candidates_val:
        if p.exists() and p != train_txt:
            val_txt = p
            break
    if train_txt is None:
        return None, None
    return train_txt, val_txt


def tokenize_txt_to_npy(
    txt_path: Path,
    tokenizer,
    out_path: Path,
    chunk_lines: int = 5000,
) -> int:
    """
    流式读取 .txt，按行 tokenize，顺序写入 int64 二进制再转为 .npy，避免整文件进内存。
    返回写入的 token 总数。
    """
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    bin_path = out_path.with_suffix(out_path.suffix + ".bin.tmp")
    count = 0
    with open(bin_path, "wb") as f:
        with open(txt_path, "r", encoding="utf-8", errors="replace") as inp:
            buf = []
            for line in inp:
                line = line.strip()
                if not line:
                    continue
                ids = tokenizer.encode(line)
                buf.extend(ids)
                if len(buf) >= chunk_lines * 50:  # 每批写入
                    arr = np.array(buf, dtype=np.int64)
                    f.write(arr.tobytes())
                    count += len(arr)
                    buf = []
            if buf:
                arr = np.array(buf, dtype=np.int64)
                f.write(arr.tobytes())
                count += len(arr)
    with open(bin_path, "rb") as f:
        data = np.frombuffer(f.read(), dtype=np.int64)
    np.save(out_path, data, allow_pickle=False)
    bin_path.unlink(missing_ok=True)
    return len(data)


# ---------- 训练一步与验证 ----------
def train_step(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    x: torch.Tensor,
    y: torch.Tensor,
    grad_clip: float | None,
    device: str,
) -> float:
    model.train()
    x, y = x.to(device), y.to(device)
    optimizer.zero_grad()
    logits = model(x)
    # logits: (B, T, V) -> (B*T, V); y: (B, T) -> (B*T)
    B, T, V = logits.shape
    loss = run_cross_entropy(logits.reshape(-1, V), y.reshape(-1))
    loss.backward()
    if grad_clip is not None and grad_clip > 0:
        run_gradient_clipping(model.parameters(), grad_clip)
    optimizer.step()
    return loss.item()


@torch.no_grad()
def eval_loss(model: nn.Module, dataset: np.ndarray, batch_size: int, context_length: int, device: str, num_batches: int = 50) -> float:
    model.eval()
    total_loss = 0.0
    n = 0
    for _ in range(num_batches):
        x, y = run_get_batch(dataset, batch_size, context_length, device)
        logits = model(x)
        B, T, V = logits.shape
        loss = run_cross_entropy(logits.reshape(-1, V), y.reshape(-1))
        total_loss += loss.item()
        n += 1
    return total_loss / max(n, 1)


# ---------- 日志（控制台 + 可选 wandb） ----------
def log_scalar(step: int, key: str, value: float, use_wandb: bool, wandb_run=None):
    print(f"  step {step}  {key}={value:.6f}")
    if use_wandb and wandb_run is not None:
        wandb_run.log({key: value}, step=step)


# ---------- 主入口 ----------
def parse_args():
    p = argparse.ArgumentParser(description="Train Transformer LM (data in data/)")
    # 实验与路径
    p.add_argument("--exp_name", type=str, default="baseline", help="Experiment name for logging/checkpoints")
    p.add_argument("--data_dir", type=str, default="data", help="Directory for train/val data (.npy/.txt)")
    p.add_argument("--vocab_path", type=str, default=None, help="vocab.json path (required when data is .txt)")
    p.add_argument("--merges_path", type=str, default=None, help="merges.txt path (required when data is .txt)")
    p.add_argument("--special_tokens", type=str, default="<|endoftext|>", help="Comma-separated special tokens, e.g. <|endoftext|>")
    p.add_argument("--ckpt_dir", type=str, default="checkpoints", help="Directory to save checkpoints")
    p.add_argument("--resume", type=str, default=None, help="Path to checkpoint to resume from")
    # 模型
    p.add_argument("--vocab_size", type=int, default=10000)
    p.add_argument("--context_length", type=int, default=256)
    p.add_argument("--d_model", type=int, default=256)
    p.add_argument("--num_layers", type=int, default=4)
    p.add_argument("--num_heads", type=int, default=4)
    p.add_argument("--d_ff", type=int, default=512)
    p.add_argument("--rope_theta", type=float, default=10000.0)
    # 训练
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--max_iters", type=int, default=None, help="Max steps (overrides epochs if set)")
    p.add_argument("--lr_max", type=float, default=3e-4)
    p.add_argument("--lr_min", type=float, default=3e-5)
    p.add_argument("--warmup_iters", type=int, default=500)
    p.add_argument("--weight_decay", type=float, default=0.01)
    p.add_argument("--grad_clip", type=float, default=1.0)
    p.add_argument("--ckpt_every", type=int, default=1000)
    p.add_argument("--log_every", type=int, default=100)
    p.add_argument("--eval_every", type=int, default=500)
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--use_wandb", action="store_true", help="Log to Weights & Biases")
    return p.parse_args()


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # 数据：优先已 tokenize 的 .npy/.dat；否则用 .txt + vocab/merges 现场 tokenize 再训练
    data_dir = Path(args.data_dir)
    train_path, val_path = get_data_paths(args.data_dir)
    if train_path is None:
        train_txt, val_txt = get_txt_paths(args.data_dir)
        if train_txt is None:
            print("No train data found. Put either train.npy (or train.dat) or train.txt in data/.")
            return 1
        if not args.vocab_path or not args.merges_path:
            print("Data folder has .txt but no pre-tokenized .npy. Provide --vocab_path and --merges_path to tokenize from .txt.")
            print("Example: --vocab_path tests/fixtures/gpt2_vocab.json --merges_path tests/fixtures/gpt2_merges.txt")
            return 1
        special_list = [s.strip() for s in args.special_tokens.split(",") if s.strip()]
        print("Loading tokenizer from", args.vocab_path, args.merges_path)
        tokenizer = load_tokenizer_from_files(args.vocab_path, args.merges_path, special_list or None)
        train_npy = data_dir / "train.npy"
        print("Tokenizing train .txt ->", train_npy)
        tokenize_txt_to_npy(train_txt, tokenizer, train_npy)
        train_path = train_npy
        if val_txt and val_txt.exists():
            val_npy = data_dir / "val.npy"
            print("Tokenizing val .txt ->", val_npy)
            tokenize_txt_to_npy(val_txt, tokenizer, val_npy)
            val_path = val_npy
        else:
            val_path = None
    train_data = load_dataset_memmap(train_path)
    val_data = load_dataset_memmap(val_path) if val_path is not None else train_data
    print(f"Train tokens: {len(train_data)}, Val tokens: {len(val_data)}")

    # 模型与优化器
    model = TransformerLM(
        vocab_size=args.vocab_size,
        context_length=args.context_length,
        d_model=args.d_model,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        d_ff=args.d_ff,
        rope_theta=args.rope_theta,
    ).to(args.device)
    optimizer = get_adamw_cls()(
        model.parameters(),
        lr=args.lr_max,
        weight_decay=args.weight_decay,
        betas=(0.9, 0.999),
        eps=1e-8,
    )

    # 学习率调度：cosine + warmup（总步数由 epochs 或 max_iters 决定）
    tokens_per_epoch = max(1, len(train_data) - args.context_length - 1)
    steps_per_epoch = max(1, tokens_per_epoch // (args.batch_size * (args.context_length + 1)))
    total_iters = args.max_iters if args.max_iters is not None else args.epochs * steps_per_epoch
    cosine_cycle_iters = max(1, total_iters - args.warmup_iters)
    def get_lr(it):
        return run_get_lr_cosine_schedule(
            it, args.lr_max, args.lr_min, args.warmup_iters, cosine_cycle_iters
        )

    start_iter = 0
    if args.resume:
        start_iter = run_load_checkpoint(args.resume, model, optimizer)
        print(f"Resumed from iteration {start_iter}")

    wandb_run = None
    if args.use_wandb:
        try:
            import wandb
            wandb_run = wandb.init(project="transformer-lm", name=args.exp_name, config=vars(args))
        except Exception as e:
            print("wandb not available:", e)

    ckpt_dir = Path(args.ckpt_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    config_path = ckpt_dir / f"{args.exp_name}_config.json"
    with open(config_path, "w") as f:
        json.dump(vars(args), f, indent=2)
    print(f"Config saved to {config_path}")

    # 训练循环
    for it in range(start_iter, total_iters):
        lr = get_lr(it)
        for g in optimizer.param_groups:
            g["lr"] = lr
        x, y = run_get_batch(train_data, args.batch_size, args.context_length, args.device)
        loss = train_step(model, optimizer, x, y, args.grad_clip, args.device)

        if (it + 1) % args.log_every == 0:
            log_scalar(it + 1, "train_loss", loss, args.use_wandb, wandb_run)
            log_scalar(it + 1, "lr", lr, args.use_wandb, wandb_run)
        if (it + 1) % args.eval_every == 0:
            val_loss = eval_loss(model, val_data, args.batch_size, args.context_length, args.device)
            log_scalar(it + 1, "val_loss", val_loss, args.use_wandb, wandb_run)
        if (it + 1) % args.ckpt_every == 0:
            ckpt_path = ckpt_dir / f"{args.exp_name}_iter_{it+1}.pt"
            run_save_checkpoint(model, optimizer, it + 1, ckpt_path)
            print(f"Checkpoint saved: {ckpt_path}")

    run_save_checkpoint(model, optimizer, total_iters, ckpt_dir / f"{args.exp_name}_final.pt")
    print("Done.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
