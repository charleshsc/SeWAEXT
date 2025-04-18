import torch
from torch.nn.modules.batchnorm import _BatchNorm

@torch.no_grad()
def update_bn_custom(loader, model, device=None):
    """
    自定义 update_bn,确保数据被移动到模型所在设备。
    """
    if device is None:
        device = next(model.parameters()).device

    model.train()
    momenta = {}
    num_samples = 0

    def _reset_bn(module):
        if isinstance(module, _BatchNorm):
            module.running_mean.zero_()
            module.running_var.fill_(1)
            momenta[module] = module.momentum

    model.apply(_reset_bn)

    for input, _ in loader:
        input = input.to(device)  # ✅ 关键：将数据移到模型所在设备

        batch_size = input.size(0)
        momentum = batch_size / float(num_samples + batch_size)
        for bn_module in momenta.keys():
            bn_module.momentum = momentum

        model(input)
        num_samples += batch_size

    for bn_module, momentum in momenta.items():
        bn_module.momentum = momentum