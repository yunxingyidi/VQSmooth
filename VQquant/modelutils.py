import torch
import torch.nn as nn
import Smoothquant.group_quant as group_quant


DEV = torch.device("cuda:0")

def find_layers(module, layers=(nn.Conv2d, nn.Linear, group_quant.GroupQLinear), name=""):
    if isinstance(module, layers):
        return {name: module}
    if isinstance(module, nn.Module) and module.__class__.__name__ == "GroupQLinear":
        return {name: module}
    res = {}
    for name1, child in module.named_children():
        res.update(
            find_layers(child, layers, name + "." + name1 if name else name1)
        )
    return res