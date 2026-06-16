"""
Copied from RT-DETR (https://github.com/lyuwenyu/RT-DETR)
Copyright(c) 2023 lyuwenyu. All Rights Reserved.
"""


import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler

from ..core import register


__all__ = ['AdamW', 'SGD', 'Adam', 'MultiStepLR', 'CosineAnnealingLR', 'OneCycleLR', 'LambdaLR', 'ConstantLR']



SGD = register()(optim.SGD)
Adam = register()(optim.Adam)
AdamW = register()(optim.AdamW)

try:
    from prodigyplus import ProdigyPlusScheduleFree as _ProdigyPlusScheduleFree
except ImportError:
    _ProdigyPlusScheduleFree = None
else:
    ProdigyPlusScheduleFree = register()(_ProdigyPlusScheduleFree)
    __all__.append('ProdigyPlusScheduleFree')


MultiStepLR = register()(lr_scheduler.MultiStepLR)
CosineAnnealingLR = register()(lr_scheduler.CosineAnnealingLR)
OneCycleLR = register()(lr_scheduler.OneCycleLR)
LambdaLR = register()(lr_scheduler.LambdaLR)
ConstantLR = register()(lr_scheduler.ConstantLR)
