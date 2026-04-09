import torch
from typing import TYPE_CHECKING

try:
    import torch_npu  # noqa: F401
except ModuleNotFoundError:
    torch_npu = None

if TYPE_CHECKING:
    from Smoothquant.group_quant import QuantTensor


_GROUPED_MATMUL_AUX_CACHE = {}
_QUANT_REDUCE_SUM_WEIGHT_CACHE = {}
_FUSE_WEIGHT_DQ_CACHE = {}


@torch.no_grad()
def build_group_reduce_matmul_weight_cache(
    weight_q: torch.Tensor,
    n_group: int,
) -> torch.Tensor:
    """Pack weight into the current custom-op-ready layout [group, groupSize, outFeature]."""
    out_features, hidden = weight_q.shape
    group_size = hidden // n_group
    return (
        weight_q.reshape(out_features, n_group, group_size)
        .permute(1, 2, 0)
        .contiguous()
    )


@torch.no_grad()
def _get_grouped_matmul_aux(activation: torch.Tensor, bsz: int, seqlen: int, n_group: int, out_features: int):
    key = (str(activation.device), bsz, seqlen, n_group, out_features)
    cached = _GROUPED_MATMUL_AUX_CACHE.get(key)
    if cached is None:
        step = bsz * seqlen
        group_list = torch.arange(
            step,
            (n_group + 1) * step,
            step,
            device=activation.device,
            dtype=torch.int64,
        )
        scale = torch.ones((n_group, out_features), device=activation.device, dtype=torch.bfloat16)
        cached = (group_list, scale)
        _GROUPED_MATMUL_AUX_CACHE[key] = cached
    return cached


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
def build_quant_reduce_sum_weight_cache(
    weight_q: torch.Tensor,
    weight_scales: torch.Tensor,
    n_group: int,
):
    out_features, hidden = weight_q.shape
    group_size = hidden // n_group
    x2_nd = (
        weight_q.reshape(out_features, n_group, group_size)
        .permute(1, 2, 0)
        .contiguous()
    )
    x2_nz = torch_npu.npu_format_cast(x2_nd, 29)
    x2_scale = weight_scales.to(torch.bfloat16).contiguous()
    return x2_nz, x2_scale


@torch.no_grad()
def ensure_nz_format(weight_qrs_nz: torch.Tensor) -> torch.Tensor:
    """Ensure a cached quant_reduce_sum weight tensor keeps NZ storage format."""
    if torch_npu is None or not hasattr(torch_npu, "get_npu_format"):
        return weight_qrs_nz
    if torch_npu.get_npu_format(weight_qrs_nz) == 29:
        return weight_qrs_nz
    return torch_npu.npu_format_cast(weight_qrs_nz.contiguous(), 29)


@torch.no_grad()
def _get_quant_reduce_sum_weight_cache(
    weight_q: torch.Tensor,
    weight_scales: torch.Tensor,
    n_group: int,
):
    """Cache weight layout conversion for quant_reduce_sum."""
    key = (
        str(weight_q.device),
        int(weight_q.data_ptr()),
        tuple(weight_q.shape),
        tuple(weight_q.stride()),
        int(weight_scales.data_ptr()),
        tuple(weight_scales.shape),
        n_group,
    )
    cached = _QUANT_REDUCE_SUM_WEIGHT_CACHE.get(key)
    if cached is None:
        cached = build_quant_reduce_sum_weight_cache(weight_q, weight_scales, n_group)
        _QUANT_REDUCE_SUM_WEIGHT_CACHE[key] = cached
    return cached


@torch.no_grad()
def prepare_quant_reduce_sum_activation(
    x: "QuantTensor",
    residual_bits: int = 4,
):
    """Prepare activation tensors for npu_quant_matmul_reduce_sum."""
    bsz, seqlen, hidden = x.activation.shape
    n_group = x.e.shape[-1]
    group_size = hidden // n_group

    x1 = (
        x.activation.reshape(bsz, seqlen, n_group, group_size)
        .permute(2, 0, 1, 3)
        .reshape(n_group, bsz * seqlen, group_size)
        .contiguous()
    )
    x1_scale = reconstruct_activation_scales(
        x.delta_base,
        x.e,
        residual_bits=residual_bits,
    ).permute(2, 0, 1).reshape(n_group, bsz * seqlen).contiguous()
    return x1, x1_scale


@torch.no_grad()
def prepare_group_reduce_matmul_activation(
    x: "QuantTensor",
    residual_bits: int = 4,
):
    """Pack activation into the current custom-op-ready layout [group, token, groupSize]."""
    bsz, seqlen, hidden = x.activation.shape
    n_group = x.e.shape[-1]
    group_size = hidden // n_group
    packed_activation = (
        x.activation.reshape(bsz, seqlen, n_group, group_size)
        .permute(2, 0, 1, 3)
        .reshape(n_group, bsz * seqlen, group_size)
        .contiguous()
    )
    activation_scale = reconstruct_activation_scales(
        x.delta_base,
        x.e,
        residual_bits=residual_bits,
    ).reshape(bsz * seqlen, n_group).contiguous()
    return packed_activation, activation_scale


@torch.no_grad()
def mul_int8_matrix(
    activation: torch.Tensor,
    weight: torch.Tensor,
    n_group: int,
) -> torch.Tensor:
    """Compute grouped matmul and return per-group accumulations."""
    bsz, seqlen, hidden = activation.shape
    out_features, in_features = weight.shape

    group_size = hidden // n_group

    act_group = activation.reshape(bsz, seqlen, n_group, group_size)
    act_group = act_group.permute(2, 0, 1, 3).reshape(n_group, bsz * seqlen, group_size).contiguous()

    weight_group = weight.reshape(out_features, n_group, group_size)
    weight_group = weight_group.permute(1, 2, 0).contiguous()

    grouped_input = [act_group.reshape(n_group * bsz * seqlen, group_size)]
    grouped_weight = [weight_group]
    group_list, scale = _get_grouped_matmul_aux(activation, bsz, seqlen, n_group, out_features)

    grouped_output = torch_npu.npu_grouped_matmul(
        grouped_input,
        grouped_weight,
        scale=[scale],
        group_list=group_list,
        group_type=0,
        split_item=2,
        output_dtype=torch.bfloat16,
    )[0]
    return grouped_output.reshape(n_group, bsz, seqlen, out_features).permute(1, 2, 0, 3)


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

    bsz, seqlen, hidden = x.activation.shape
    out_features, in_features = weight_q.shape

    n_group = x.e.shape[-1]

    activation_scales = reconstruct_activation_scales(
        x.delta_base,
        x.e,
        residual_bits=residual_bits,
    )

    group_output = mul_int8_matrix(x.activation, weight_q, n_group)
    if group_output.dtype != torch.bfloat16:
        group_output = group_output.to(torch.bfloat16)
    group_output = group_output * activation_scales.to(torch.bfloat16).view(bsz, seqlen, n_group, 1)
    group_output = group_output * weight_scales.to(torch.bfloat16).view(1, 1, 1, out_features)
    return group_output.sum(dim=2)


@torch.no_grad()
def fuse_dequant_with_group_reduce_matmul_cached(
    x: "QuantTensor",
    packed_weight_q: torch.Tensor,
    weight_scales: torch.Tensor,
    residual_bits: int = 4,
    accum_dtype: torch.dtype = torch.float32,
    prescale_weight: bool = True,
) -> torch.Tensor:
    """
    Baseline-equivalent implementation:
    dequantize activation to bf16 + dequantize weight to bf16, then run one matmul.
    """
    bsz, seqlen, hidden = x.activation.shape
    n_group, group_size, out_features = packed_weight_q.shape

    if n_group * group_size != hidden:
        raise ValueError(
            f"Packed weight hidden {n_group * group_size} does not match activation hidden {hidden}"
        )

    compute_dtype = torch.bfloat16
    bt = bsz * seqlen

    if hasattr(x, "dequantize") and callable(getattr(x, "dequantize")):
        act_dq = x.dequantize(residual_bits=residual_bits)
    else:
        packed_activation, activation_scale = prepare_group_reduce_matmul_activation(
            x,
            residual_bits=residual_bits,
        )  # [G, BT, K], [BT, G]
        if packed_activation.shape[0] != n_group:
            raise ValueError(
                f"Packed activation group count {packed_activation.shape[0]} "
                f"does not match packed weight group count {n_group}"
            )
        if packed_activation.shape[-1] != group_size:
            raise ValueError(
                f"Packed activation group size {packed_activation.shape[-1]} "
                f"does not match packed weight group size {group_size}"
            )
        act_group = packed_activation.permute(1, 0, 2).to(torch.float32)  # [BT, G, K]
        act_scale = activation_scale.to(torch.float32).unsqueeze(-1)       # [BT, G, 1]
        act_dq = (act_group * act_scale).reshape(bsz, seqlen, hidden)

    act_dq = act_dq.to(compute_dtype).reshape(bt, hidden)

    # [G, K, O] -> [O, H], cache dequantized weight to reduce per-call prep overhead.
    cache_key = (
        str(packed_weight_q.device),
        int(packed_weight_q.data_ptr()),
        tuple(packed_weight_q.shape),
        tuple(packed_weight_q.stride()),
        str(packed_weight_q.dtype),
        int(weight_scales.data_ptr()),
        tuple(weight_scales.shape),
        str(weight_scales.dtype),
        bool(prescale_weight),
        str(compute_dtype),
    )
    weight_dq_oh = _FUSE_WEIGHT_DQ_CACHE.get(cache_key)
    if weight_dq_oh is None:
        weight_q_oh = packed_weight_q.permute(2, 0, 1).reshape(out_features, hidden).to(compute_dtype)
        if prescale_weight:
            weight_dq_oh = weight_q_oh * weight_scales.to(compute_dtype).view(out_features, 1)
        else:
            weight_dq_oh = weight_q_oh
        _FUSE_WEIGHT_DQ_CACHE[cache_key] = weight_dq_oh

    out = torch.matmul(act_dq, weight_dq_oh.transpose(0, 1))
    if not prescale_weight:
        out = out * weight_scales.to(compute_dtype).view(1, out_features)
    if accum_dtype != compute_dtype:
        out = out.to(accum_dtype)
    return out.reshape(bsz, seqlen, out_features)

@torch.no_grad()
def fuse_dequant_with_quant_reduce_sum_cached(
    x: "QuantTensor",
    weight_qrs_nz: torch.Tensor,
    weight_scales: torch.Tensor,
    residual_bits: int = 4,
) -> torch.Tensor:
    if torch_npu is None or not hasattr(torch_npu, "npu_quant_matmul_reduce_sum"):
        raise ModuleNotFoundError("torch_npu.npu_quant_matmul_reduce_sum is required")

    bsz, seqlen, hidden = x.activation.shape
    out_features, in_features = weight_qrs_nz.shape

    n_group = x.e.shape[-1]
    x1, x1_scale = prepare_quant_reduce_sum_activation(
        x,
        residual_bits=residual_bits,
    )   # x1: [G, BT, K]

    x1_scale = x1_scale.permute(1, 0).contiguous()   # [BT, G]

    x2_nz, x2_scale = _get_quant_reduce_sum_weight_cache(
        weight_qrs_nz,
        weight_scales,
        n_group,
    )
    x1_scale = x1_scale / (x1_scale.max(dim=1, keepdim=True)[0] + 1e-6)
    x2_scale = x2_scale / (x2_scale.max() + 1e-6)

    out = torch_npu.npu_quant_matmul_reduce_sum(
        x1,
        x2_nz,
        x1_scale=x1_scale,
        x2_scale=x2_scale,
    )

    return out.reshape(bsz, seqlen, out_features)
