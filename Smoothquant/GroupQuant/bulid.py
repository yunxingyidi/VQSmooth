from quant_tik import group_quant_kernel

if __name__ == "__main__":
    # ⚠️ 这里的 shape 必须是“静态确定的”
    group_quant_kernel(
        B=1,
        T=128,
        H=1024,
        G=32,
        n_bits=8,
        residual_bits=4,
        kernel_name="group_quant"
    )
