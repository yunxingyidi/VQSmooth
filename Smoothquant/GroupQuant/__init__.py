try:
    from .group_quant_npu import (
        GroupQuantizedTensor,
        fake_quantize_activation_per_token,
        group_dequantize,
        group_quantize,
        group_quantize_dequantize,
    )
except ModuleNotFoundError as exc:
    if exc.name != "torch":
        raise
