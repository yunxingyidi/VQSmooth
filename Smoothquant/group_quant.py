from distutils.core import setup_keywords

import torch
from openpyxl.styles.builtins import output
from torch import nn
from functools import partial
import GroupQuant
from dataclasses import dataclass

# @torch.no_grad()
# def quantize_activation_per_token_absmax(t, n_bits=8):
#     t_shape = t.shape
#     t.view(-1, t_shape[-1])
#     scales = t.abs().max(dim=-1, keepdim=True)[0]
#     q_max = 2 ** (n_bits - 1) - 1
#     scales.clamp_(min=1e-5).div_(q_max)
#     t.div_(scales).round_().mul_(scales)
#     return t

@torch.no_grad()
def quantize_activation_per_tensor_absmax(t, n_bits=8, residual_group=32):
    t_shape = t.shape
    t.view(-1, t_shape[-1])
    scales = t.abs().max()
    q_max = 2 ** (n_bits - 1) - 1
    scales.clamp_(min=1e-5).div_(q_max)
    t.div_(scales).round_().mul_(scales)
    return t

# @torch.no_grad()
# def quantize_activation_per_token_absmax(
#     t: torch.Tensor,
#     n_bits: int = 8,
#     residual_group: int = 32,
#     residual_bits: int = 4,
# ):
#     """
#     Per-token quantization with group-wise scale residual approximation

#     Args:
#         t: [B, T, H] activation
#         n_bits: activation quant bits
#         residual_group: group size along H
#         residual_bits: E, bits for residual code e
#         R_min_ratio, R_max_ratio: search range relative to delta_base
#         R_steps: number of R candidates

#     Returns:
#         t_dq: dequantized activation after scale residual approximation
#     """
#     B, T, H = t.shape
#     assert H % residual_group == 0
#     G = H // residual_group
#     e_min = -(2 ** (residual_bits - 1))
#     e_max = 0

#     q_max = 2 ** (n_bits - 1) - 1

#     # ===== Step 0: preserve fp activation =====
#     t_fp = t.clone()

#     # ===== Step 1: per-(token, group) true scale Δ_i =====
#     t_group = t_fp.view(B, T, G, residual_group)
#     delta = t_group.abs().max(dim=-1)[0].clamp_(min=1e-5) / q_max          # [B, T, G]

#     # ===== Step 2: base scale Δ_base (per token) =====
#     delta_base = delta.max(dim=-1, keepdim=True)[0]       # [B, T, 1]

#     # ===== Step 3: initialize residual step R_init =====
#     R = delta_base / (2 ** residual_bits)  # [B,T,1]

#     # ===== Step 4: residual code e =====
#     e = torch.round((delta - delta_base) / R)
#     e = torch.clamp(e, e_min, e_max)  # [B,T,G]

#     # ===== Step 5: reconstructed scale =====
#     delta_rec = delta_base + e * R
#     delta_rec = torch.clamp(delta_rec, min=1e-5)  # [B,T,G]

#     # ===== Step 6: quantize & dequantize =====
#     scale = delta_rec.unsqueeze(-1)  # [B,T,G,1]
#     t_q = torch.round(t_group / scale).clamp(-q_max, q_max)
#     t_dq = (t_q * scale).view(B, T, H).to(t.dtype)

#     return t_dq

@torch.no_grad()
def quantize_activation_per_token_absmax(
    t: torch.Tensor,
    n_bits: int = 8,
    residual_group: int = 32,
    residual_bits: int = 4,
):
    B, T, H = t.shape
    assert H % residual_group == 0
    G = H // residual_group

    t_q_cuda = torch.empty_like(t, dtype=torch.int8, device="cuda")
    t_fp_rec = torch.empty_like(t, dtype=torch.bfloat16, device="cuda")
    delta_base_cuda = torch.empty(B, T, device="cuda", dtype=torch.bfloat16)
    e_cuda = torch.empty(B, T, G, device="cuda", dtype=torch.int8)

    GroupQuant.quant_forward(
        t,
        t_q_cuda,
        delta_base_cuda,
        e_cuda,
        n_bits,
        residual_bits
    )

    GroupQuant.dequant_forward(
        t_q_cuda,
        delta_base_cuda,
        e_cuda,
        t_fp_rec,
        residual_bits
    )
    return t_fp_rec

@dataclass
class QuantTensor:
    def __init__(
        self,
        x,
        residual_group: int = 32
    ):
        super().__init__()
        B, T, H = x.shape
        assert H % residual_group == 0
        G = H // residual_group

        self.activation = torch.empty_like(
            x, device="cuda", dtype=torch.int8
        )
        self.delta_base = torch.empty(
            B, T, device="cuda", dtype=torch.bfloat16
        )
        self.e = torch.empty(
            B, T, G, device="cuda", dtype=torch.int8
        )
        self.quantize(x)

    def quantize(self, x, n_bits=8, residual_bits=4):
        GroupQuant.quant_forward(
            x,
            self.activation,
            self.delta_base,
            self.e,
            n_bits,
            residual_bits
        )
    def dequantize(self, residual_bits=4):
        t_fp_rec = torch.empty_like(self.activation, dtype=torch.bfloat16, device="cuda")

        GroupQuant.dequant_forward(
            self.activation,
            self.delta_base,
            self.e,
            t_fp_rec,
            residual_bits
        )
        return t_fp_rec

class GroupQLinear(nn.Module):
    def __init__(
        self,
        in_features,
        out_features,
        bias=True,
        fake_quant=False,
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        self.fake_quant = fake_quant

        self.register_buffer(
            "weight",
            torch.randn(
                self.out_features,
                self.in_features,
                dtype=torch.float16,
                requires_grad=False,
            ),
        )
        if bias:
            self.register_buffer(
                "bias",
                torch.zeros(
                    (1, self.out_features), dtype=torch.float16, requires_grad=False
                ),
            )
        else:
            self.register_buffer("bias", None)

    def to(self, *args, **kwargs):
        super(GroupQLinear, self).to(*args, **kwargs)
        self.weight = self.weight.to(*args, **kwargs)
        if self.bias is not None:
            self.bias = self.bias.to(*args, **kwargs)
        return self

    @torch.no_grad()
    def forward(self, x):
        if not self.fake_quant:
            dq_x = x.dequantize()
            y = torch.functional.F.linear(dq_x, self.weight, self.bias)
            q_y = QuantTensor(y)
            return q_y
        else:
            d_x = quantize_activation_per_token_absmax(x, n_bits=8)
            y = torch.functional.F.linear(d_x, self.weight, self.bias)
            y = quantize_activation_per_token_absmax(y, n_bits=8)
            return y

    @staticmethod
    def from_float(
        module, fake_quant
    ):
        assert isinstance(module, torch.nn.Linear)
        new_module = GroupQLinear(
            module.in_features,
            module.out_features,
            module.bias is not None,
            fake_quant,
        )
        new_module.weight = module.weight

        if module.bias is not None:
            new_module.bias = module.bias
        return new_module

    def __repr__(self):
        return f"GroupQLinear({self.in_features}, {self.out_features}, bias={self.bias is not None}, act_quant={self.act_quant_name}, output_quant={self.output_quant_name})"


def quantize_llama_like(
    model
):
    from transformers.models.llama.modeling_llama import (
        LlamaAttention,
        LlamaMLP,
    )
    from transformers.models.mistral.modeling_mistral import (
        MistralAttention,
        MistralMLP,
    )
    from modeling_llama import (
        QuantLlamaMLP,
        QuantLlamaAttention,
    )

    layer_counter = 0

    for name, m in model.model.named_modules():
        parent_name = name.rsplit(".", 1)[0]
        child_name = name.split(".")[-1]
        parent = model.model.get_submodule(parent_name)
        if isinstance(m, (LlamaMLP, MistralMLP)):
            setattr(parent, child_name, QuantLlamaMLP(m, m.config, False))
        elif isinstance(m, (LlamaAttention, MistralAttention)):
            setattr(parent, child_name, QuantLlamaAttention(m, m.config, layer_idx=layer_counter))
            layer_counter += 1
    return model

def quantize_model(
    model, act_quant="per_token", quantize_bmm_input=False
):
    from transformers.models.llama.modeling_llama import LlamaPreTrainedModel
    from transformers.models.mistral.modeling_mistral import MistralPreTrainedModel

    if isinstance(model, (LlamaPreTrainedModel, MistralPreTrainedModel)):
        return quantize_llama_like(
            model,
        )
    else:
        raise ValueError(f"Unsupported model type: {type(model)}")