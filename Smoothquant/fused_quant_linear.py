import torch

try:
    from VQquant.VectorQuant import vector_quant_npu as VectorQuant
except ModuleNotFoundError:
    try:
        import VectorQuant
    except ModuleNotFoundError:
        VectorQuant = None


def decode_vq_weight(
    weight_indices: torch.Tensor,
    weight_codebook: torch.Tensor,
    out_features: int,
    in_features: int,
    device: torch.device | str,
    dtype: torch.dtype = torch.bfloat16,
) -> torch.Tensor:
    device = torch.device(device)
    if VectorQuant is None:
        raise ModuleNotFoundError(
            "VectorQuant extension is not available. "
            "Build/install the VectorQuant extension or provide a pre-decoded weight."
        )

    weight_q = torch.zeros(
        (out_features, in_features),
        dtype=dtype,
        device=device,
    )
    VectorQuant.dequant_forward(
        weight_indices.to(device),
        weight_codebook.to(device),
        weight_q,
    )
    return weight_q


def maybe_decode_weight(
    *,
    weight_q: torch.Tensor | None,
    weight_indices: torch.Tensor | None,
    weight_codebook: torch.Tensor | None,
    out_features: int,
    in_features: int,
    device: torch.device | str,
    dtype: torch.dtype = torch.bfloat16,
) -> torch.Tensor:
    if weight_q is not None:
        return weight_q.to(device=device, dtype=dtype)

    if weight_indices is None or weight_codebook is None:
        raise ValueError(
            "Expected either `weight_q` or both `weight_indices` and `weight_codebook`."
        )

    return decode_vq_weight(
        weight_indices=weight_indices,
        weight_codebook=weight_codebook,
        out_features=out_features,
        in_features=in_features,
        device=device,
        dtype=dtype,
    )


def maybe_dequantize_activation(x):
    if hasattr(x, "dequantize"):
        return x.dequantize()
    return x


def fused_quant_linear(
    x,
    *,
    out_features: int,
    in_features: int,
    bias: torch.Tensor | None = None,
    weight_q: torch.Tensor | None = None,
    weight_indices: torch.Tensor | None = None,
    weight_codebook: torch.Tensor | None = None,
    return_quant_tensor: bool = False,
    quant_tensor_cls=None,
    decoded_weight_cache: dict | None = None,
    cache_key=None,
    dtype: torch.dtype = torch.bfloat16,
):
    """
    Unified entry point for quantized linear forward.

    This is currently a fallback implementation that:
    1. materializes/decode weights if needed,
    2. dequantizes activation if `x` is a QuantTensor-like object,
    3. runs `torch.nn.functional.linear`,
    4. optionally repacks the output with `quant_tensor_cls`.

    Later, the body of this function can be replaced by a fused custom kernel
    without changing call sites.
    """
    if hasattr(x, "activation"):
        device = x.activation.device
    else:
        device = x.device

    cache_hit = False
    if decoded_weight_cache is not None and cache_key is not None:
        cached = decoded_weight_cache.get(cache_key)
        if cached is not None and cached.device == device and cached.dtype == dtype:
            decoded_weight = cached
            cache_hit = True
        else:
            decoded_weight = maybe_decode_weight(
                weight_q=weight_q,
                weight_indices=weight_indices,
                weight_codebook=weight_codebook,
                out_features=out_features,
                in_features=in_features,
                device=device,
                dtype=dtype,
            )
            decoded_weight_cache[cache_key] = decoded_weight
    else:
        decoded_weight = maybe_decode_weight(
            weight_q=weight_q,
            weight_indices=weight_indices,
            weight_codebook=weight_codebook,
            out_features=out_features,
            in_features=in_features,
            device=device,
            dtype=dtype,
        )

    x_dq = maybe_dequantize_activation(x)
    bias_term = bias.squeeze(0) if bias is not None and bias.dim() > 1 else bias
    y = torch.nn.functional.linear(x_dq, decoded_weight, bias_term)

    if return_quant_tensor:
        if quant_tensor_cls is None:
            raise ValueError("`quant_tensor_cls` is required when `return_quant_tensor=True`.")
        return quant_tensor_cls(y)

    return y


def fused_quant_linear_vq_cached(
    x_quant,
    *,
    out_features: int,
    in_features: int,
    bias: torch.Tensor | None,
    weight_indices: torch.Tensor,
    weight_codebook: torch.Tensor,
    decoded_weight_cache: dict,
    cache_key,
    return_quant_tensor: bool = False,
    quant_tensor_cls=None,
    dtype: torch.dtype = torch.bfloat16,
):
    """
    Fast path specialized for the current hot inference path:
    - activation is a per-token QuantTensor-like object with `.activation` and `.dequantize()`
    - weight is stored as VQ indices + codebook
    - decoded weight is cached externally in a dict
    - output is usually returned as bf16

    Compared with `fused_quant_linear`, this version intentionally avoids:
    - dynamic input-type dispatch
    - optional weight source branches
    - generic cache validation logic
    """
    device = x_quant.activation.device
    decoded_weight = decoded_weight_cache.get(cache_key)
    if decoded_weight is None:
        decoded_weight = decode_vq_weight(
            weight_indices=weight_indices,
            weight_codebook=weight_codebook,
            out_features=out_features,
            in_features=in_features,
            device=device,
            dtype=dtype,
        )
        decoded_weight_cache[cache_key] = decoded_weight

    x_dq = x_quant.dequantize()
    bias_term = bias.squeeze(0) if bias is not None and bias.dim() > 1 else bias
    y = torch.nn.functional.linear(x_dq, decoded_weight, bias_term)

    if return_quant_tensor:
        if quant_tensor_cls is None:
            raise ValueError("`quant_tensor_cls` is required when `return_quant_tensor=True`.")
        return quant_tensor_cls(y)

    return y
