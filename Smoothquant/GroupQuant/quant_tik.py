from tbe import tik
import math

def group_quant_kernel(
    B, T, H, G,
    n_bits, residual_bits,
    kernel_name="group_quant"
):
    tik_inst = tik.Tik()

    group_size = H // G
    q_max = (1 << (n_bits - 1)) - 1
    r_min = -(1 << (residual_bits - 1))

    # =====================================================
    # GM Tensors
    # =====================================================
    t_fp_gm = tik_inst.Tensor(
        "float16", (B * T * H,),
        scope=tik.scope_gm, name="t_fp"
    )

    t_q_gm = tik_inst.Tensor(
        "int8", (B * T * H,),
        scope=tik.scope_gm, name="t_q"
    )

    delta_base_gm = tik_inst.Tensor(
        "float16", (B * T,),
        scope=tik.scope_gm, name="delta_base"
    )

    e_gm = tik_inst.Tensor(
        "int8", (B * T * G,),
        scope=tik.scope_gm, name="e"
    )

    # =====================================================
    # UB Tensors
    # =====================================================
    ub_x = tik_inst.Tensor(
        "float16", (group_size,),
        scope=tik.scope_ubuf, name="ub_x"
    )

    ub_abs = tik_inst.Tensor(
        "float16", (group_size,),
        scope=tik.scope_ubuf, name="ub_abs"
    )

    ub_q_fp = tik_inst.Tensor(
        "float16", (group_size,),
        scope=tik.scope_ubuf, name="ub_q_fp"
    )

    ub_q_i8 = tik_inst.Tensor(
        "int8", (group_size,),
        scope=tik.scope_ubuf, name="ub_q_i8"
    )

    ub_delta = tik_inst.Tensor(
        "float16", (G,),
        scope=tik.scope_ubuf, name="ub_delta"
    )

    ub_delta_base = tik_inst.Tensor(
        "float16", (B * T,),
        scope=tik.scope_ubuf, name="ub_delta_base"
    )

    ub_e_fp = tik_inst.Tensor(
        "float16", (G,),
        scope=tik.scope_ubuf, name="ub_e_fp"
    )

    ub_e_i8 = tik_inst.Tensor(
        "int8", (G,),
        scope=tik.scope_ubuf, name="ub_e_i8"
    )

    ub_delta_vec = tik_inst.Tensor(
        "float16", (group_size,),
        scope=tik.scope_ubuf, name="ub_delta_vec"
    )

    ub_qmax = tik_inst.Tensor(
        "float16", (group_size,),
        scope=tik.scope_ubuf, name="ub_qmax"
    )

    ub_zero = tik_inst.Tensor(
        "float16", (G,),
        scope=tik.scope_ubuf, name="ub_zero"
    )

    ub_rmin = tik_inst.Tensor(
        "float16", (G,),
        scope=tik.scope_ubuf, name="ub_rmin"
    )

    # 常量初始化
    tik_inst.vector_dup(G, ub_zero, 0.0, (G + 127) // 128, 1, 1)
    tik_inst.vector_dup(G, ub_rmin, float(r_min), (G + 127) // 128, 1, 1)

    burst_g = (group_size + 15) // 16

    # =====================================================
    # Main
    # =====================================================
    with tik_inst.for_range(0, B * T, block_num=B * T) as bt:

        base_offset = bt * H

        # -------------------------------------------------
        # Step1: compute delta[g]
        # -------------------------------------------------
        with tik_inst.for_range(0, G) as g:

            offset = base_offset + g * group_size

            tik_inst.data_move(
                ub_x,
                t_fp_gm[offset],
                0, 1, burst_g, 0, 0
            )

            tik_inst.vec_abs(group_size, ub_abs, ub_x, 1, 1, 1)

            # scalar reduce max
            absmax = tik_inst.Tensor(
                "float16",
                (1,),
                scope=tik.scope_ubuf, name="absmax"
            )
            work_tensor = tik_inst.Tensor(
                "float16",
                (128,),
                scope=tik.scope_ubuf, name="work_tensor"
            )
            tik_inst.vec_reduce_max(group_size, absmax, ub_abs, work_tensor, 1, 1)

            tmp_absmax = tik_inst.Scalar("float16")
            tmp_absmax.set_as(absmax[0])
            ub_delta[g].set_as(tmp_absmax / q_max)

        # -------------------------------------------------
        # Step2: delta_base = max(delta)
        # -------------------------------------------------
        delta_base = tik_inst.Scalar("float16")

        delta_max = tik_inst.Tensor(
            "float16",
            (1,),
            scope=tik.scope_ubuf, name="delta_max"
        )
        delta_work_tensor = tik_inst.Tensor(
            "float16",
            (128,),
            scope=tik.scope_ubuf, name="work_tensor"
        )

        tik_inst.vec_reduce_max(G, delta_max, ub_delta, delta_work_tensor, 1, 1)

        delta_base.set_as(delta_max[0])

        ub_delta_base[bt].set_as(delta_base)

        # -------------------------------------------------
        # Step3: residual e
        # -------------------------------------------------
        R = tik_inst.Scalar("float16")
        R.set_as(delta_base / (1 << residual_bits))


        with tik_inst.for_range(0, G) as g:
            val = tik_inst.Scalar("float16")
            val.set_as(ub_delta[g])
            tmp = tik_inst.Scalar("float16")
            tmp.set_as((val - delta_base) / R)

            ub_e_fp[g].set_as(tmp)

        tik_inst.vec_max(128, ub_e_fp, ub_e_fp, ub_rmin,
                      (G + 127) // 128, 1, 1, 1)

        tik_inst.vec_min(128, ub_e_fp, ub_e_fp, ub_zero,
                      (G + 127) // 128, 1, 1, 1)

        tik_inst.vec_conv(128, "round", ub_e_i8, ub_e_fp,
                       (G + 127) // 128, 1, 1)

        tik_inst.data_move(
            e_gm[bt * G],
            ub_e_i8,
            0, 1, (G + 15) // 16, 0, 0
        )

        # -------------------------------------------------
        # Step4: final quant
        # -------------------------------------------------
        with tik_inst.for_range(0, G) as g:

            e_scalar = tik_inst.Scalar("float16")
            e_scalar.set_as(ub_e_i8[g])

            delta_rec = tik_inst.Scalar("float16")
            delta_rec.set_as(delta_base + e_scalar * R)

            # clamp min
            with tik_inst.if_scope(delta_rec < 1e-6):
                delta_rec.set_as(1e-6)

            # expand scalar to vector
            tik_inst.vector_dup(
                128, ub_delta_vec,
                delta_rec,
                burst_g, 1, 1
            )

            offset = base_offset + g * group_size

            tik_inst.data_move(
                ub_x,
                t_fp_gm[offset: offset + group_size],
                0, 1, burst_g, 0, 0
            )

            tik_inst.vdiv(
                128,
                ub_q_fp,
                ub_x,
                ub_delta_vec,
                burst_g, 1, 1, 1, 1
            )

            tik_inst.vround(
                128, ub_q_fp, ub_q_fp,
                burst_g, 1, 1
            )

            # clamp
            tik_inst.vector_dup(
                128, ub_qmax,
                float(q_max),
                burst_g, 1, 1
            )

            tik_inst.vmin(
                128, ub_q_fp, ub_q_fp, ub_qmax,
                burst_g, 1, 1, 1, 1
            )

            tik_inst.vneg(
                128, ub_qmax, ub_qmax,
                burst_g, 1, 1
            )

            tik_inst.vmax(
                128, ub_q_fp, ub_q_fp, ub_qmax,
                burst_g, 1, 1, 1, 1
            )

            tik_inst.vconv(
                128, ub_q_i8, ub_q_fp,
                burst_g, 1, 1
            )

            tik_inst.data_move(
                t_q_gm[offset: offset + group_size],
                ub_q_i8,
                0, 1, burst_g, 0, 0
            )

    tik_inst.BuildCCE(
        kernel_name=kernel_name,
        inputs=[t_fp_gm],
        outputs=[t_q_gm, delta_base_gm, e_gm]
    )

    return tik_inst


if __name__ == "__main__":
    from tbe.common.platform import get_soc_spec

    print("SOC:", get_soc_spec("SOC_VERSION"))
    group_quant_kernel(
        B=1,
        T=4096,
        H=4096,
        G=128,
        n_bits=8,
        residual_bits=4,
        kernel_name="group_quant"
    )