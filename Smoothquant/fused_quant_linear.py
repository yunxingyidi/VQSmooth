import torch
from Smoothquant.custom_quant_matmul import run_custom_quant_matmul
try:
    import torch_npu  # noqa: F401
except ModuleNotFoundError:
    torch_npu = None

try:
    from VQquant.VectorQuant import vector_quant_npu as VectorQuant
except ModuleNotFoundError:
    try:
        import VectorQuant
    except ModuleNotFoundError:
        VectorQuant = None


@torch.no_grad()
def prewarm_weight_quant_runtime_cache(module, device):
    return get_weight_quant_runtime_cache(module, device)


@torch.no_grad()
def weight_quant_int8_npu_forward(module, x_quant):
    return forward_weight_quant_int8_npu(module, x_quant)


@torch.no_grad()
def weight_quant_int8_fallback_forward(module, x_quant):
    return forward_weight_quant_int8_fallback_fast(module, x_quant)


@torch.no_grad()
def get_weight_quant_runtime_cache(module, device):
    device = torch.device(device)
    cache_key = str(device)
    cached = module._weight_quant_runtime_cache.get(cache_key)
    if cached is not None:
        return cached

    if module.weight is not None:
        raise RuntimeError("weight_quant path expects VQ-backed weights, but got dense weight.")
    if VectorQuant is None:
        raise ModuleNotFoundError(
            "VectorQuant extension is not available. "
            "Build/install the VectorQuant extension or disable VQ weight quantization."
        )

    int8_cache_key = f"{device}:int8"
    weight_q_int8 = module._decoded_weight_cache.get(int8_cache_key)
    if weight_q_int8 is None:
        weight_q_int8 = torch.zeros(
            (module.out_features, module.in_features),
            dtype=torch.int8,
            device=device,
        )
        VectorQuant.dequant_forward(
            module.weight_indices.to(device),
            module.weight_codebook.to(device),
            weight_q_int8,
        )
        module._decoded_weight_cache[int8_cache_key] = weight_q_int8

    runtime = {
        "weight_q_int8": weight_q_int8,
        "weight_q_t_int8": weight_q_int8.transpose(0, 1).contiguous(),
        "weight_rhs_bf16": weight_q_int8.transpose(0, 1).to(torch.bfloat16).contiguous(),
        "weight_scales_1d": module.weight_scales.to(device=device, dtype=torch.bfloat16).view(-1),
        "bias_1d": (
            module.bias.to(device=device, dtype=torch.bfloat16).view(-1)
            if module.bias is not None
            else None
        ),
    }
    module._weight_quant_runtime_cache[cache_key] = runtime
    return runtime


@torch.no_grad()
def forward_weight_quant_int8_npu(module, x_quant):
    bsz, seqlen, hidden = x_quant.activation.shape
    runtime = get_weight_quant_runtime_cache(module, x_quant.activation.device)

    x1 = x_quant.activation.reshape(bsz * seqlen, hidden).contiguous()
    pertoken_scale = x_quant.scales.reshape(bsz * seqlen).to(torch.float32)
    bias = runtime["bias_1d"]

    # Prefer project custom kernel when enabled and available.
    y2d = run_custom_quant_matmul(
        x=x1,
        weight_t_int8=runtime["weight_q_t_int8"],
        weight_scales_1d=runtime["weight_scales_1d"],
        pertoken_scale=pertoken_scale,
        bias=bias,
        output_dtype=torch.bfloat16,
    )
    if y2d is not None:
        return y2d.reshape(bsz, seqlen, module.out_features)

    if torch_npu is None or not hasattr(torch_npu, "npu_quant_matmul"):
        return None

    try:
        y2d = torch_npu.npu_quant_matmul(
            x1,
            runtime["weight_q_t_int8"],
            runtime["weight_scales_1d"],
            pertoken_scale=pertoken_scale,
            bias=bias,
            output_dtype=torch.bfloat16,
        )
    except RuntimeError:
        return None
    return y2d.reshape(bsz, seqlen, module.out_features)


@torch.no_grad()
def forward_weight_quant_int8_fallback_fast(module, x_quant):
    bsz, seqlen, hidden = x_quant.activation.shape
    bt = bsz * seqlen
    runtime = get_weight_quant_runtime_cache(module, x_quant.activation.device)

    x2d = x_quant.activation.reshape(bt, hidden).to(torch.bfloat16)
    y2d = torch.matmul(x2d, runtime["weight_rhs_bf16"])
    y2d.mul_(x_quant.scales.reshape(bt, 1))
    y2d.mul_(runtime["weight_scales_1d"].view(1, -1))
    if runtime["bias_1d"] is not None:
        y2d.add_(runtime["bias_1d"])
    return y2d.reshape(bsz, seqlen, module.out_features)
