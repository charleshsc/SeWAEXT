import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


from dtsrc.models.model import TrajectoryModel

class MixLinear(nn.Module):
    def __init__(self, bias=True, pre_trained_layers=None):
        super(MixLinear, self).__init__()
        self.bias = bias
        self.pre_trained_layers = pre_trained_layers

    def forward(self, x, mask):
        weight = torch.stack([layer.weight.data for layer in self.pre_trained_layers], dim=0)
        weight = torch.sum(mask.view(-1, 1, 1) * weight, dim=0) / torch.sum(mask)  # .detach()
        if self.bias:
            bias = torch.stack([layer.bias.data for layer in self.pre_trained_layers], dim=0)
            bias = torch.sum(mask.view(-1, 1) * bias, dim=0) / torch.sum(mask)  # .detach()
        else:
            bias = None

        x = F.linear(x, weight, bias)
        return x

class MLPBCModel(TrajectoryModel):

    """
    Simple MLP that predicts next action a from past states s.
    """

    def __init__(self, state_dim, act_dim, hidden_size, n_layer, dropout=0.1, max_length=1, **kwargs):
        super().__init__(state_dim, act_dim)

        self.hidden_size = hidden_size
        self.max_length = max_length

        layers = [nn.Linear(max_length*self.state_dim, hidden_size)]
        for _ in range(n_layer-1):
            layers.extend([
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_size, hidden_size)
            ])
        layers.extend([
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, self.act_dim),
            nn.Tanh(),
        ])

        self.model = nn.Sequential(*layers)

    def forward(self, states, actions, rewards, attention_mask=None, target_return=None):

        states = states[:,-self.max_length:].reshape(states.shape[0], -1)  # concat states
        actions = self.model(states).reshape(states.shape[0], 1, self.act_dim)

        return None, actions, None

    def get_action(self, states, actions, rewards, swa_model=None, **kwargs):
        states = states.reshape(1, -1, self.state_dim)
        if states.shape[1] < self.max_length:
            states = torch.cat(
                [torch.zeros((1, self.max_length-states.shape[1], self.state_dim),
                             dtype=torch.float32, device=states.device), states], dim=1)
        states = states.to(dtype=torch.float32)
        if swa_model is not None:
            _, actions, _ = swa_model.forward(states, None, None, **kwargs)
        else:
            _, actions, _ = self.forward(states, None, None, **kwargs)
        return actions[0,-1]

class MixtureMLPBCModel(MLPBCModel):
    def __init__(self, pre_trained_models, temperature, k, state_dim, act_dim, hidden_size, n_layer, dropout=0.1, max_length=1, **kwargs):
        super(MixtureMLPBCModel, self).__init__(state_dim, act_dim, hidden_size, n_layer, dropout, max_length, **kwargs)
        self.temperature = temperature
        self.n = len(pre_trained_models)
        self.k = k

        self.mask_logit = nn.Parameter(torch.ones(self.n) * -.5, requires_grad=True)
        random_idx = torch.randperm(self.n)[:int(k / 2)]
        self.mask_logit.data[random_idx] = 1

        fc0 = [model.model[0] for model in pre_trained_models]
        fc3 = [model.model[3] for model in pre_trained_models]
        fc6 = [model.model[6] for model in pre_trained_models]
        fc9 = [model.model[9] for model in pre_trained_models]

        self.fc0 = MixLinear(pre_trained_layers=fc0)
        self.inter1 = nn.Sequential(nn.ReLU(), nn.Dropout(dropout))
        self.fc3 = MixLinear(pre_trained_layers=fc3)
        self.inter2 = nn.Sequential(nn.ReLU(), nn.Dropout(dropout))
        self.fc6 = MixLinear(pre_trained_layers=fc6)
        self.inter3 = nn.Sequential(nn.ReLU(), nn.Dropout(dropout))
        self.fc9 = MixLinear(pre_trained_layers=fc9)
        self.final = nn.Tanh()

    def gumbel_softmax(self, temperature=1.0, eps=1e-9):
        u = torch.rand_like(self.mask_logit)
        gumbel = -torch.log(-torch.log(u + eps) + eps)
        y = self.mask_logit + gumbel
        y = F.sigmoid(y / temperature)

        y_hard = torch.zeros_like(y)
        y_hard[y > 0.5] = 1.0

        return (y_hard - y).detach() + y

    def project_logit(self):
        with torch.no_grad():
            k_th_lagest = torch.topk(self.mask_logit, self.k)[0][-1]
            self.mask_logit.data[self.mask_logit < k_th_lagest] = -1e9

    def compute_mask(self):
        if self.training:
            mask = self.gumbel_softmax(self.temperature)
        else:
            # mask = F.sigmoid(self.mask_logit)
            # mask[mask > 0.5] = 1.0
            # mask[mask <= 0.5] = 0.0
            
            topk_values, topk_indices = torch.topk(self.mask_logit, self.k)
            mask = torch.zeros_like(self.mask_logit)
            mask[topk_indices] = 1
        return mask

    def forward(self, states, actions, rewards, attention_mask=None, target_return=None):
        mask = self.compute_mask()

        states = states[:,-self.max_length:].reshape(states.shape[0], -1)  # concat states

        x = self.fc0(states, mask)
        x = self.inter1(x)
        x = self.fc3(x, mask)
        x = self.inter2(x)
        x = self.fc6(x, mask)
        x = self.inter3(x)
        x = self.fc9(x, mask)
        x = self.final(x)
        actions = x.reshape(states.shape[0], 1, self.act_dim)

        return None, actions, None

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
            if name in self.ema_state:
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
