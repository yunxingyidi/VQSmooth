import torch
from torch.autograd.profiler import record_function
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from Smoothquant.group_quant import QuantTensor


@torch.no_grad()
def reconstruct_activation_scales(
    delta_base: torch.Tensor,
    e: torch.Tensor,
    residual_bits: int = 4,
) -> torch.Tensor:
    """Reconstruct per-(token, group) activation scales from residual codes."""
    delta_base_fp = delta_base.to(torch.float32).unsqueeze(-1)
    step = delta_base_fp / (1 << residual_bits)
    return (delta_base_fp + e.to(torch.float32) * step).clamp_(min=1e-5)


@torch.no_grad()
def mul_int8_matrix(
    activation: torch.Tensor,
    weight: torch.Tensor,
    n_group: int,
) -> torch.Tensor:
    """Compute grouped matmul and return per-group accumulations."""
    if activation.dtype != torch.int8:
        raise TypeError(f"Expected int8 activation, got {activation.dtype}")
    if weight.dtype != torch.int8:
        raise TypeError(f"Expected int8 weight, got {weight.dtype}")
    if activation.dim() != 3:
        raise ValueError(f"Expected activation with shape [B, T, H], got {tuple(activation.shape)}")
    if weight.dim() != 2:
        raise ValueError(f"Expected weight with shape [O, H], got {tuple(weight.shape)}")
    if n_group <= 0:
        raise ValueError(f"Expected positive n_group, got {n_group}")

    bsz, seqlen, hidden = activation.shape
    out_features, in_features = weight.shape
    if hidden != in_features:
        raise ValueError(
            f"Activation hidden size {hidden} does not match weight input size {in_features}"
        )
    if hidden % n_group != 0:
        raise ValueError(f"Hidden size {hidden} must be divisible by n_group {n_group}")

    group_size = hidden // n_group

    # Ascend NPU int32 batched matmul can be unstable for this layout; use bf16 fast path.
    if activation.device.type == "npu":
        act_group = activation.reshape(bsz * seqlen, n_group, group_size)
        act_group = act_group.permute(1, 0, 2).to(torch.bfloat16)

        weight_group = weight.reshape(out_features, n_group, group_size)
        weight_group = weight_group.permute(1, 2, 0).to(torch.bfloat16)

        group_output = torch.matmul(act_group, weight_group)
        return group_output.permute(1, 0, 2).reshape(bsz, seqlen, n_group, out_features)

    # CPU/CUDA: keep int32 accumulation for numerical consistency.
    # act_group = activation.reshape(bsz * seqlen, n_group, group_size).to(torch.int32)
    # weight_group = weight.reshape(out_features, n_group, group_size).to(torch.int32)
    # weight_group = weight_group.permute(1, 2, 0).contiguous()

    # act_group = act_group.permute(1, 0, 2).contiguous()
    # group_output = torch.matmul(act_group, weight_group)
    # return group_output.permute(1, 0, 2).reshape(bsz, seqlen, n_group, out_features)


@torch.no_grad()
def fuse_dequant_with_linear(
    x: "QuantTensor",
    weight_q: torch.Tensor,
    weight_scales: torch.Tensor,
    residual_bits: int = 4,
) -> torch.Tensor:
    """
    Fuse residual-group activation dequantization with per-channel weight dequantization.

    Returns a dequantized linear output with shape [B, T, O].
    """
    if x.activation.dtype != torch.int8:
        raise TypeError(f"Expected int8 activation, got {x.activation.dtype}")
    if weight_q.dtype != torch.int8:
        raise TypeError(f"Expected int8 weight, got {weight_q.dtype}")
    if x.activation.dim() != 3:
        raise ValueError(f"Expected activation with shape [B, T, H], got {tuple(x.activation.shape)}")
    if weight_q.dim() != 2:
        raise ValueError(f"Expected weight with shape [O, H], got {tuple(weight_q.shape)}")

    bsz, seqlen, hidden = x.activation.shape
    out_features, in_features = weight_q.shape
    if hidden != in_features:
        raise ValueError(
            f"Activation hidden size {hidden} does not match weight input size {in_features}"
        )

    n_group = x.e.shape[-1]
    if n_group <= 0:
        raise ValueError(f"Expected positive group count, got {n_group}")
    if hidden % n_group != 0:
        raise ValueError(f"Hidden size {hidden} must be divisible by n_group {n_group}")

    if weight_scales.numel() != out_features:
        raise ValueError(
            f"Expected weight_scales with {out_features} elements, got {weight_scales.numel()}"
        )

    activation_scales = reconstruct_activation_scales(
        x.delta_base,
        x.e,
        residual_bits=residual_bits,
    )
    if activation_scales.shape != (bsz, seqlen, n_group):
        raise ValueError(
            "Expected activation_scales with shape "
            f"{(bsz, seqlen, n_group)}, got {tuple(activation_scales.shape)}"
        )

    with record_function("vqsmooth.groupqlinear.matrix_multiplication"):
        group_output = mul_int8_matrix(x.activation, weight_q, n_group)
        # if group_output.dtype != torch.bfloat16:
        #     group_output = group_output.to(torch.bfloat16)
    with record_function("vqsmooth.groupqlinear.dequant"):
        group_output = group_output * activation_scales.to(torch.bfloat16).view(bsz, seqlen, n_group, 1)
        group_output = group_output * weight_scales.to(torch.bfloat16).view(1, 1, 1, out_features)
    return group_output.sum(dim=2)
