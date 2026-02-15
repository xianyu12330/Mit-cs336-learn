import os

import numpy as np
import numpy.typing as npt
import torch
import typing


def get_batch(
        numpy_x:npt.NDArray, #整个数据集（Numpy 数组，形状为 (N,)）。
        batch_size: int,#一次并行处理多少个序列。B
        context_length:int,#每个序列的时间步长度。T
        device:str #目标设备（如 'cuda:0'）。
)->tuple[torch.Tensor, torch.Tensor]:#形状为[B,T]
    #确定合法的采样范围,最大索引为len(x) - context_length - 1
    #需要取 T个 token 作为输入，还需要第 T+1 个 token 作为最后一个输入的标签。
    tokens_len = context_length + 1
    #最大最大索引，超过这个位置，取数据时就会越界
    max_idx_ = len(numpy_x) - tokens_len

    #生成随机索引形状为 (B,) 的索引数组在 [0, len(x) - context_length - 1] 范围内生成 batch_size 个随机整数
    rand_idx = torch.randint(0,max_idx_ + 1,(batch_size,))

    #堆叠切片->Input row: 取 x[i : i + context_length],x[i + 1 : i + context_length + 1]
    x = torch.stack([torch.from_numpy(numpy_x[i:i + context_length].astype(np.int64)) for i in  rand_idx]).to(device)
    y = torch.stack([torch.from_numpy(numpy_x[i + 1 : i + context_length + 1].astype(np.int64)) for i in rand_idx]).to(device)

    return x,y
#保存模型参数
def save_checkpoint(model:torch.nn.Module,#
                    optimizer:torch.optim.Optimizer,
                    iteration:int,
                    out: str | os.PathLike | typing.BinaryIO | typing.IO[bytes]):
    #1构建一个包含所有信息的字典
    checkpoint = {
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'iteration': iteration,
    }
    #使用torch写入
    torch.save(checkpoint, out)

#读取模型参数
def load_checkpoint(src:str | os.PathLike | typing.BinaryIO | typing.IO[bytes],
                    model:torch.nn.Module,
                    optimizer:torch.optim.Optimizer):
    checkpoint = torch.load(src)
    #恢复模型权重
    model.load_state_dict(checkpoint['model'])
    #恢复优化器状态
    optimizer.load_state_dict(checkpoint['optimizer'])
    #返回恢复时的迭代次数
    return checkpoint['iteration']





