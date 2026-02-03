import mindspore as ms
from mindspore.ops import Custom


group_quant = Custom(
    "GroupQuant",   # ⚠️ 不是 .so 文件
    out_shape=lambda x: (
        x,
        (x[0], x[1]),
        (x[0], x[1], 32),
    ),
    out_dtype=lambda x: (
        ms.int8,
        ms.float16,
        ms.int8,
    ),
    func_type="tbe",
    reg_info="quant_op_info.py"
)