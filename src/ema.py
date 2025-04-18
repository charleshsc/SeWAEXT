import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

class EMA:
    def __init__(self, model, decay):
        """
        初始化 EMA 对象

        Args:
            model (torch.nn.Module): 目标模型
            decay (float): EMA 的衰减系数，值在 (0, 1) 之间，通常选择接近 1 的值
        """
        self.model = model
        self.decay = decay
        self.ema_state = {name: param.clone().detach() for name, param in model.state_dict().items()}
    
    def update(self, model):
        """
        更新 EMA 参数
        
        Args:
            model (torch.nn.Module): 当前模型
        """
        for name, param in model.state_dict().items():
            if name in self.ema_state and param.dtype.is_floating_point:
                self.ema_state[name].mul_(self.decay).add_(param.data, alpha=1 - self.decay)
            else:
                self.ema_state[name] = param.clone().detach()

    def apply(self, model):
        """
        将 EMA 的权重应用到目标模型
        
        Args:
            model (torch.nn.Module): 目标模型
        """
        for name, param in model.state_dict().items():
            if name in self.ema_state:
                param.data.copy_(self.ema_state[name])

    def restore(self, model):
        """
        将模型还原为原始权重
        
        Args:
            model (torch.nn.Module): 目标模型
        """
        for name, param in model.state_dict().items():
            if name in self.ema_state:
                self.ema_state[name] = param.clone().detach()