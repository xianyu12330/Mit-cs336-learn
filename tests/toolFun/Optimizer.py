import math

import torch
from collections.abc import Callable, Iterable
from typing import Optional
import math

class SGD(torch.optim.Optimizer):
    def __init__(self,params,lr = 1e-3):
        if lr < 0:
            raise ValueError("Invalid learning rate: {}".format(lr))
        #字典
        defaults = {'lr':lr}
        #调用父类构造函数，会创建self.param_groups和self.state->每个参数 tensor 都有自己独立的 state 字典。state初始为空
        super().__init__(params,defaults)
    def step(self, closure: Optional[Callable] = None):
        #closure 是用于某些高级优化器（比如 LBFGS）重新计算 loss 的。
        loss = None if closure is None else closure()
        for group in self.param_groups:
            lr = group['lr']
            for p in group['params']:
                if p.grad is None:continue #没有梯度就跳过
                #调用optimizer.step()时，会往里写参数，例如
                """optimizer.state
=
{
    weight_tensor: {"t": 1},
    bias_tensor: {"t": 1}
}
                """
                state = self.state[p]
                #取出迭代次数，若是没有，则为0
                t = state.get("t",0)
                #获得梯度
                grad = p.grad.data
                p.data -= lr / math.sqrt(t + 1) *grad
                state['t']  = t + 1
        return loss

class Adamw(torch.optim.Optimizer):
    def __init__(self,params,lr = 1e-3,betas=(0.9,0.999),eps=1e-8,weight_decay=0.01):
        if lr < 0:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError("Invalid beta parameter at index 0: {}".format(betas[0]))
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError("Invalid beta parameter at index 1: {}".format(betas[1]))
        if eps < 0.0:
            raise ValueError("Invalid epsilon value: {}".format(eps))
        defaults = {'lr':lr, 'betas':betas, 'eps':eps,'weight_decay':weight_decay}
        super().__init__(params,defaults)
    def step(self,closure=None):
        loss = None if closure is None else closure()
        #获得参数
        for group in self.param_groups:
            lr = group['lr']
            beta1,beta2 = group['betas']
            eps = group['eps']
            weight_decay = group['weight_decay']
            for p in group['params']:
                #没有梯度直接跳过
                if p.grad is None:continue
                # 获得梯度
                g = p.grad.data
                #这里直接修改p.data，且必须在动量更新之前做
                #AdamW 是直接让参数先衰减，然后再根据梯度去走。
                if weight_decay != 0:
                    p.data = p.data - weight_decay * lr * p.data
                state = self.state[p]
                # 取出迭代次数，若是没有，则为0
                #当前是首次循环
                if len(state) == 0:
                    #一阶矩，动量,memory_format=torch.preserve_format控制：新张量的内存布局,保持 p 原有的内存布局
                    state['exp_avg'] = torch.zeros_like(p,memory_format=torch.preserve_format)
                    #二阶矩
                    state['exp_avg_sq'] = torch.zeros_like(p,memory_format=torch.preserve_format)
                    state['t'] = 0
                state['t'] += 1
                t = state.get("t", 0)
                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']

                #更新一阶m数值
                exp_avg.mul_(beta1).add_(g, alpha=1 - beta1)
                #更新二阶v数值
                exp_avg_sq.mul_(beta2).add_(g **2, alpha=1 - beta2)
                #计算偏差校正的修正系数
                bias_correction1 = 1 - beta1 ** t
                bias_correction2 = 1 - beta2 ** t
                #获得自更新学习率at
                step_size = lr * math.sqrt(bias_correction2) / bias_correction1
                #分母:根号v + eps
                denom = exp_avg_sq.sqrt().add_(eps)
                #参数更新
                # addcdiv_(tensor1, tensor2, value) 等价于 tensor + value * (tensor1 / tensor2)
                p.data.addcdiv_(exp_avg, denom, value=-step_size)


        return loss
#余弦退火优化算法
class Cosine:
    def __init__(self,
                 max_learning_rate,#学习率的峰值
                 min_learning_rate,#学习率的底值
                 warmup_iters,#预热阶段的总步数
                 cosine_cycle_iters):#整个衰减周期结束的步数
        self.max_learning_rate = max_learning_rate
        self.min_learning_rate = min_learning_rate
        self.warmup_iters = warmup_iters
        self.cosine_cycle_iters = cosine_cycle_iters

    def __call__(self, it ) -> float:#当前的迭代步数
        if it < self.warmup_iters:
            return it * self.max_learning_rate / self.warmup_iters
        elif it < self.cosine_cycle_iters:
            return (self.min_learning_rate + (1 + math.cos((it - self.warmup_iters) /
                                                          (self.cosine_cycle_iters - self.warmup_iters)* math.pi) )*
                                                           (self.max_learning_rate - self.min_learning_rate) / 2)

        else:
            return self.min_learning_rate
#梯度裁剪-》是防止梯度爆炸的一种方法, 通过将梯度限制在一个范围内, 从而防止梯度爆炸。
class GradientClip:
    def __init__(self,parameters,maximum_norm,eps = 1e-6):
        self.parameters = parameters
        self.maximum_norm = maximum_norm
        self.eps = eps
    #梯度裁剪的核心是计算全局L2范数。我们将模型中所有层的梯度拼接成一个巨大的向量，然后计算它的欧几里得长度：
    def __call__(self):
        #获得所有不为空的梯度
        grad_parm = {p for p in self.parameters if p.grad is not None}
        #对梯度拼接计算全局L2范数
        total_norm = 0.0
        for p in grad_parm:
            total_norm += p.grad.detach().norm(2).pow(2).sum()

        total_norm = math.sqrt(total_norm)
        if total_norm > self.maximum_norm:
            clip_norm = self.maximum_norm / (total_norm + self.eps)
            for p in grad_parm:
                p.grad.mul_(clip_norm)






