"""
训练脚本：支持可配置超参数、memmap 数据加载、checkpoint、日志，便于消融实验。
数据支持两种方式：
  1) 预 tokenize：data/train.npy、data/val.npy（或 .dat/.bin），直接 memmap 加载；
  2) 原始 .txt：data/*.txt，配合 --vocab_path 与 --merges_path 自动用 BPE tokenizer 转成 token 文件后再训练（内存高效流式写入 .npy）。
"""
from __future__ import annotations


import sys
from pathlib import Path


import wandb


from tests.toolFun.Optimizer import Adamw,Cosine,GradientClip
from tests.toolFun.Tokenizer import BPETokenizer
from tests.toolFun.dataLord import get_batch,save_checkpoint,load_checkpoint
from tests.toolFun.transformer import CrossEntropyLoss

# 保证 tests 可导入
if __name__ == "__main__" and __file__:
    _root = Path(__file__).resolve().parent.parent
    if str(_root) not in sys.path:
        sys.path.insert(0, str(_root))



from tests.toolFun.transformer import Embedding, Linear, RMSNorm, SwiGLU, TransformerLM

import argparse

import os


import numpy as np
import torch



# 数据准备，将。txt文件转化可处理的类型
def prepare_data(txt_path: str, bin_path: str, tokenizer: BPETokenizer, chunk_size: int = 1024 * 1024):
    """
    将文本文档流式编码并写入 np.memmap 文件。
    """
    print(f"🚀 开始处理: {txt_path}")

    # 1. 定义一个文本块生成器
    def text_generator(filepath, size):
        with open(filepath, 'r', encoding='utf-8') as f:
            while True:
                chunk = f.read(size)
                if not chunk: break
                yield chunk

    # 2. 调用你的流式编码接口，得到一个吐出 ID 的迭代器
    #把文本流式转成 token id 流。
    token_iterator = tokenizer.encode_iterable(text_generator(txt_path, chunk_size))
    # 3. 准备写入 memmap
    # 注意：BPE 词表通常小于 65536，所以用 uint16。如果你的词表超过了这个数，请改为 np.int32
    dtype = np.uint16
    # 因为不知道总长度，我们先创建一个临时列表缓存一小批数据，然后再写入
    # 这样可以平衡 I/O 速度和内存占用
    write_batch_size = 1024 * 1024 * 10  # 每次往硬盘写 1000 万个 token
    buffer = []
    total_tokens = 0
    # 为了动态扩展 memmap，我们需要先用 'w+' 模式创建一个初始空文件
    # 但 numpy memmap 不支持动态 append，所以我们的策略是：先收集，再分段写入文件
    # 这里为了代码简洁且兼容绝大多数普通级别的数据集（几千万 tokens），我们采用分批追加写普通二进制文件的方法
    print(f"正在编码并写入 {bin_path}...")
    with open(bin_path, 'wb') as f:
        for token_id in token_iterator:
            buffer.append(token_id)
            if len(buffer) >= write_batch_size:
                # 将 buffer 转为 numpy 数组并转换为字节写入
                np_buffer = np.array(buffer, dtype=dtype)
                f.write(np_buffer.tobytes())
                total_tokens += len(buffer)
                buffer.clear()
                print(f"已处理 {total_tokens / 1e6:.2f} M tokens...")
        # 处理最后一批剩余的数据
        if buffer:
            np_buffer = np.array(buffer, dtype=dtype)
            f.write(np_buffer.tobytes())
            total_tokens += len(buffer)
            buffer.clear()

    print(f"✅ 处理完成！共计 {total_tokens} 个 tokens。")
    print(f"文件已保存至: {bin_path}\n")


def main():
    p = argparse.ArgumentParser(description="Train Transformer LM (data in data/)")
    # 实验与路径
    p.add_argument("--run_name", type=str, default=None, help="WandB 实验名称")
    p.add_argument("--data_dir", type=str, default="data")
    p.add_argument("--vocab_path", type=str, default=None, )
    p.add_argument("--merges_path", type=str, default=None, )
    p.add_argument("--special_tokens", nargs='*', default = ["<|endoftext|>"])
    p.add_argument("--ckpt_dir", type=str, default="checkpoints")
    p.add_argument("--out_dir", type=str, default="out_dir")


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
    p.add_argument("--max_iters", type=int, default=50000, help="Max steps (overrides epochs if set)")
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
    args = p.parse_args()
    torch.manual_seed(args.seed)
    os.makedirs(args.ckpt_dir, exist_ok=True)
    os.makedirs(args.out_dir, exist_ok=True)
    # 1.加载数据
    data_dir = args.data_dir
    
    train_txt = os.path.join(data_dir, "TinyStoriesV2-GPT4-train.txt")
    test_txt = os.path.join(data_dir, "TinyStoriesV2-GPT4-valid.txt")

    train_bin = os.path.join(data_dir, "train.bin")
    test_bin = os.path.join(data_dir, "test.bin")

    # 仅当提供 vocab/merges 时加载 tokenizer；用 .txt 生成 .bin 时必须提供
    if args.vocab_path and args.merges_path:
        tokenizer = BPETokenizer.from_files(args.vocab_path, args.merges_path, args.special_tokens)
    else:
        tokenizer = None

    if not os.path.exists(train_bin):
        if not os.path.exists(train_txt) or tokenizer is None:
            raise SystemExit(
                "未找到 train.bin，且无法预处理：请提供 data/*.txt 并指定 --vocab_path 与 --merges_path，或先生成 train.bin。"
            )
        print("未找到训练集 bin 文件，开始预处理...")
        prepare_data(train_txt, train_bin, tokenizer)

    if not os.path.exists(test_bin):
        if not os.path.exists(test_txt) or tokenizer is None:
            raise SystemExit(
                "未找到 test.bin，且无法预处理：请提供 data/*.txt 并指定 --vocab_path 与 --merges_path，或先生成 test.bin。"
            )
        prepare_data(test_txt, test_bin, tokenizer)

    # --- 1. 使用 memmap 加载数据 ---
    # 注意 dtype 必须和写入时（上面代码中的 np.uint16）完全一致
    # mode='r' 表示只读，防止训练时意外修改了原始数据
    train_data = np.memmap(train_bin, dtype=np.uint16, mode='r')
    test_data = np.memmap(test_bin, dtype=np.uint16, mode='r')
    min_len = args.context_length + 1
    if len(train_data) < min_len:
        raise SystemExit(f"训练集长度 {len(train_data)} 小于 context_length+1={min_len}，无法采样 batch。")
    if len(test_data) < min_len:
        raise SystemExit(f"验证集长度 {len(test_data)} 小于 context_length+1={min_len}，无法采样 batch。")

    model = TransformerLM(
        vocab_size=args.vocab_size,
        context_length=args.context_length,
        d_model=args.d_model,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        d_ff=args.d_ff,
        rope_theta=args.rope_theta,
        device=args.device,
    ).to(args.device)
    print(f"Model moved to: {next(model.parameters()).device}")
    #4.初始化优化器
    optimizer = Adamw(model.parameters(), lr=args.lr_max, weight_decay=args.weight_decay)
    #5.检查点恢复逻辑
    start_iter = 0
    ckpt_path = os.path.join(args.ckpt_dir, "ckpt.pt")
    if os.path.exists(ckpt_path):
        start_iter = load_checkpoint(ckpt_path, model, optimizer)
        print(f"Resuming from iteration {start_iter}")


    wandb.init(project="cs336-assignment1-basics", name=args.run_name, config=vars(args))

    lr_schedule = Cosine(
        max_learning_rate=args.lr_max,
        min_learning_rate=args.lr_min,
        warmup_iters=args.warmup_iters,
        cosine_cycle_iters=args.max_iters,
    )
    loss_fn = CrossEntropyLoss()
    #it 是当前训练步数
    for it in range(start_iter, args.max_iters):
        lr = lr_schedule(it)

        for param_group in optimizer.param_groups:
            param_group["lr"] = lr

        model.train()
        x, y = get_batch(train_data, args.batch_size, args.context_length, args.device)
        logits = model(x)
        # CrossEntropyLoss 期望 (N, V) 与 (N,)，LM 输出 (B, T, V) 与 (B, T)，需展平
        loss = loss_fn(logits.reshape(-1, logits.size(-1)), y.reshape(-1))
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        GradientClip(model.parameters(), args.grad_clip)()
        optimizer.step()

        #验证与日记记录
        if it % args.eval_every == 0 or it == args.max_iters - 1:
            model.eval()
            with torch.no_grad():
                #vx是输入模型训练的数据，vy是输出可能的token
                vx, vy = get_batch(test_data, args.batch_size, args.context_length, args.device)
                v_logits = model(vx)
                v_loss = loss_fn(v_logits.reshape(-1, v_logits.size(-1)), vy.reshape(-1))
            print(f"Iter: {it}, train loss: {loss.item():.4f}, val_loss: {v_loss.item():.4f}, lr: {lr:.6f}")
            if args.use_wandb:
                wandb.log({"train/loss": loss.item(), "val/loss": v_loss.item(), "lr": lr, "iter": it + 1})
            if it % args.eval_every == 0 and it > 0:
                save_checkpoint(model, optimizer, it, ckpt_path)

    save_checkpoint(model, optimizer, args.max_iters - 1, os.path.join(args.out_dir, "ckpt_final.pt"))
    if args.use_wandb:
        wandb.finish()


if __name__ == "__main__":
    main()



