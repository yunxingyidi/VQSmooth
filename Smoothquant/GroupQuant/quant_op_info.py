from mindspore.ops.op_info_register import (
    TBERegOp, DataType, op_info_register
)

group_quant_op_info = (
    TBERegOp("GroupQuant")             # 算子名（非常重要）
    .fusion_type("OPAQUE")
    .async_flag(False)
    .binfile_name("group_quant.so")    # 生成的二进制名
    .compute_cost(10)
    .kernel_name("group_quant")         # 对应 BuildCCE 的 kernel_name
    .partial_flag(True)

    # ---------- inputs ----------
    .input(0, "x", False, "required", "all")

    # ---------- outputs ----------
    .output(0, "y", False, "required", "all")
    .output(1, "delta_base", False, "required", "all")
    .output(2, "e", False, "required", "all")

    # ---------- dtype ----------
    .dtype_format(
        DataType.F16_Default,   # x
        DataType.I8_Default,    # y
        DataType.F16_Default,   # delta_base
        DataType.I8_Default     # e
    )

    .get_op_info()
)


@op_info_register(group_quant_op_info)
def _group_quant_op_info():
    """GroupQuant op register"""
    return

