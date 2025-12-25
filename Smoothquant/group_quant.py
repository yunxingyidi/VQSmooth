import torch
from openpyxl.styles.builtins import output
from torch import nn
from functools import partial

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

@torch.no_grad()
def quantize_activation_per_token_absmax(
    t: torch.Tensor,
    n_bits: int = 8,
    residual_group: int = 32,
    residual_bits: int = 4,
    R_min_ratio: float = 0.0,
    R_max_ratio: float = 1.0,
    R_steps: int = 64,
):
    """
    Per-token quantization with group-wise scale residual approximation

    Args:
        t: [B, T, H] activation
        n_bits: activation quant bits
        residual_group: group size along H
        residual_bits: E, bits for residual code e
        R_min_ratio, R_max_ratio: search range relative to delta_base
        R_steps: number of R candidates

    Returns:
        t_dq: dequantized activation after scale residual approximation
    """
    B, T, H = t.shape
    assert H % residual_group == 0
    G = H // residual_group

    q_max = 2 ** (n_bits - 1) - 1

    # ===== Step 0: preserve fp activation =====
    t_fp = t.clone()

    # ===== Step 1: per-(token, group) true scale Δ_i =====
    t_group = t_fp.view(B, T, G, residual_group)
    delta = t_group.abs().max(dim=-1)[0].clamp_(min=1e-5) / q_max          # [B, T, G]

    # ===== Step 2: base scale Δ_base (per token) =====
    delta_base = delta.max(dim=-1, keepdim=True)[0]       # [B, T, 1]

    # ===== Step 3: initialize residual step R_init =====
    R_init = (delta_base - delta).abs().mean(dim=-1, keepdim=True)
    R_init = R_init / max(residual_bits, 1)
    R_init = torch.clamp(R_init, min=1e-8)                # [B, T, 1]

    # ===== Step 4: residual code e_i =====
    e = torch.floor((delta - delta_base) / R_init)        # [B, T, G]
    e_min = -(2 ** (residual_bits - 1) - 1)
    e = torch.clamp(e, e_min, 0)

    # ===== Step 5: search optimal R (per token) =====
    R_candidates = torch.linspace(
        R_min_ratio, R_max_ratio, R_steps, device=t.device
    ).view(1, 1, -1) * delta_base                          # [B,T,R]

    best_R = torch.zeros_like(delta_base)
    best_loss = None

    for k in range(R_candidates.shape[-1]):
        Rk = R_candidates[..., k:k+1]                     # [B,T,1]
        delta_hat = delta_base + e * Rk                    # [B,T,G]
        loss = ((delta - delta_hat) ** 2).mean(dim=-1)    # [B,T]

        if best_loss is None:
            best_loss = loss
            best_R = Rk
        else:
            mask = loss < best_loss
            best_loss = torch.where(mask, loss, best_loss)
            best_R = torch.where(mask.unsqueeze(-1), Rk, best_R)

    # ===== Step 6: reconstructed group scale =====
    delta_rec = delta_base + e * best_R                    # [B,T,G]
    delta_rec = torch.clamp(delta_rec, min=1e-5)

    # ===== Step 7: quantize & dequantize with reconstructed scale =====
    scale_rec = delta_rec.unsqueeze(-1)                    # [B,T,G,1]
    t_q = torch.round(t_group / scale_rec) * scale_rec

    t_dq = t_q.view(B, T, H).to(t.dtype)
    # diff = t_dq - t
    # print("mean abs diff:", diff.abs().mean().item())
    # print("max  abs diff:", diff.abs().max().item())
    return t_dq


class GroupQLinear(nn.Module):
    def __init__(
        self,
        in_features,
        out_features,
        bias=True,
        act_quant="per_token",
        quantize_output=False,
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

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

        if act_quant == "per_token":
            self.act_quant_name = "per_token"
            self.act_quant = partial(quantize_activation_per_token_absmax, n_bits=8)
        elif act_quant == "per_tensor":
            self.act_quant_name = "per_tensor"
            self.act_quant = partial(quantize_activation_per_tensor_absmax, n_bits=8)
        else:
            raise ValueError(f"Invalid act_quant: {act_quant}")

        if quantize_output:
            self.output_quant_name = self.act_quant_name
            self.output_quant = self.act_quant
        else:
            self.output_quant_name = "None"
            self.output_quant = nn.Identity()

    def to(self, *args, **kwargs):
        super(GroupQLinear, self).to(*args, **kwargs)
        self.weight = self.weight.to(*args, **kwargs)
        if self.bias is not None:
            self.bias = self.bias.to(*args, **kwargs)
        return self

    @torch.no_grad()
    def forward(self, x):
        q_x = self.act_quant(x)
        y = torch.functional.F.linear(q_x, self.weight, self.bias)
        q_y = self.output_quant(y)
        return q_y

    @staticmethod
    def from_float(
        module, act_quant="per_token", quantize_output=False
    ):
        assert isinstance(module, torch.nn.Linear)
        new_module = GroupQLinear(
            module.in_features,
            module.out_features,
            module.bias is not None,
            act_quant=act_quant,
            quantize_output=quantize_output,
        )
        new_module.weight = module.weight

        if module.bias is not None:
            new_module.bias = module.bias
        return new_module

    def __repr__(self):
        return f"GroupQLinear({self.in_features}, {self.out_features}, bias={self.bias is not None}, act_quant={self.act_quant_name}, output_quant={self.output_quant_name})"


def quantize_llama_like(
    model, act_quant="per_token", quantize_bmm_input=False
):
    from transformers.models.llama.modeling_llama import (
        LlamaAttention,
        LlamaMLP,
    )

    from transformers.models.mistral.modeling_mistral import (
        MistralAttention,
        MistralMLP,
    )

    for name, m in model.model.named_modules():
        if isinstance(m, (LlamaMLP, MistralMLP)):
            m.gate_proj = GroupQLinear.from_float(
                m.gate_proj, act_quant=act_quant
            )
            m.up_proj = GroupQLinear.from_float(
                m.up_proj, act_quant=act_quant
            )
            m.down_proj = GroupQLinear.from_float(
                m.down_proj, act_quant=act_quant
            )
        elif isinstance(m, (LlamaAttention, MistralAttention)):
            # Her we simulate quantizing BMM inputs by quantizing the output of q_proj, k_proj, v_proj
            m.q_proj = GroupQLinear.from_float(
                m.q_proj,
                act_quant=act_quant,
                quantize_output=quantize_bmm_input,
            )
            m.k_proj = GroupQLinear.from_float(
                m.k_proj,
                act_quant=act_quant,
                quantize_output=quantize_bmm_input,
            )
            m.v_proj = GroupQLinear.from_float(
                m.v_proj,
                act_quant=act_quant,
                quantize_output=quantize_bmm_input,
            )
            m.o_proj = GroupQLinear.from_float(
                m.o_proj, act_quant=act_quant
            )
    return model


def quantize_model(
    model, act_quant="per_token", quantize_bmm_input=False
):
    from transformers.models.llama.modeling_llama import LlamaPreTrainedModel
    from transformers.models.mistral.modeling_mistral import MistralPreTrainedModel

    if isinstance(model, (LlamaPreTrainedModel, MistralPreTrainedModel)):
        return quantize_llama_like(
            model,
            act_quant=act_quant,
            quantize_bmm_input=quantize_bmm_input,
        )
    else:
        raise ValueError(f"Unsupported model type: {type(model)}")