from dataclasses import dataclass

import torch

try:
    import torch_npu  # noqa: F401
except ModuleNotFoundError:
    torch_npu = None


@dataclass
class GroupQuantizedTensor:
    activation: torch.Tensor
    delta_base: torch.Tensor
    e: torch.Tensor
    residual_group: int


@torch.no_grad()
def group_quantize(
    t: torch.Tensor,
    n_bits: int = 8,
    residual_group: int = 32,
    residual_bits: int = 4,
) -> GroupQuantizedTensor:
    if t.dim() != 3:
        raise ValueError(f"Expected [B, T, H], got {tuple(t.shape)}")

    bsz, seqlen, hidden = t.shape
    if hidden % residual_group != 0:
        raise ValueError(
            f"Hidden size {hidden} must be divisible by residual_group {residual_group}"
        )

    n_groups = hidden // residual_group
    q_max = 2 ** (n_bits - 1) - 1
    e_min = -(2 ** (residual_bits - 1))

    t_group = t.reshape(bsz, seqlen, n_groups, residual_group)
    delta = t_group.abs().amax(dim=-1).clamp_(min=1e-5) / q_max
    delta_base = delta.amax(dim=-1)

    r = (delta_base / (1 << residual_bits)).unsqueeze(-1)
    safe_r = torch.where(r == 0, torch.full_like(r, 1e-5), r)

    e = torch.round((delta - delta_base.unsqueeze(-1)) / safe_r)
    e = e.clamp_(min=e_min, max=0).to(torch.int8)

    delta_rec = delta_base.unsqueeze(-1) + e.to(delta.dtype) * r
    delta_rec = delta_rec.clamp_(min=1e-5)

    t_q = torch.round(t_group / delta_rec.unsqueeze(-1))
    t_q = t_q.clamp_(-q_max, q_max).to(torch.int8)

    return GroupQuantizedTensor(
        activation=t_q.reshape(bsz, seqlen, hidden),
        delta_base=delta_base.to(torch.bfloat16),
        e=e,
        residual_group=residual_group,
    )


@torch.no_grad()
def group_dequantize(
    activation: torch.Tensor,
    delta_base: torch.Tensor,
    e: torch.Tensor,
    residual_group: int = 32,
    residual_bits: int = 4,
    output_dtype: torch.dtype = torch.bfloat16,
) -> torch.Tensor:
    if activation.dim() != 3:
        raise ValueError(f"Expected [B, T, H], got {tuple(activation.shape)}")

    bsz, seqlen, hidden = activation.shape
    n_groups = e.shape[-1]
    actual_group = hidden // n_groups
    if actual_group != residual_group:
        residual_group = actual_group

    act_group = activation.reshape(bsz, seqlen, n_groups, residual_group).to(torch.float32)
    delta_base_fp = delta_base.to(torch.float32).unsqueeze(-1)
    r = delta_base_fp / (1 << residual_bits)
    delta_rec = (delta_base_fp + e.to(torch.float32) * r).clamp_(min=1e-5)

    return (act_group * delta_rec.unsqueeze(-1)).reshape(bsz, seqlen, hidden).to(output_dtype)


@torch.no_grad()
def group_quantize_dequantize(
    t: torch.Tensor,
    n_bits: int = 8,
    residual_group: int = 32,
    residual_bits: int = 4,
    output_dtype: torch.dtype = torch.bfloat16,
) -> torch.Tensor:
    qt = group_quantize(
        t,
        n_bits=n_bits,
        residual_group=residual_group,
        residual_bits=residual_bits,
    )
    return group_dequantize(
        qt.activation,
        qt.delta_base,
        qt.e,
        residual_group=qt.residual_group,
        residual_bits=residual_bits,
        output_dtype=output_dtype,
    )


@torch.no_grad()
def fake_quantize_activation_per_token(
    t: torch.Tensor,
    n_bits: int = 8,
    residual_group: int = 32,
    residual_bits: int = 4,
) -> torch.Tensor:
    print("Group Quant")
    return group_quantize_dequantize(
        t,
        n_bits=n_bits,
        residual_group=residual_group,
        residual_bits=residual_bits,
        output_dtype=t.dtype,
    )
