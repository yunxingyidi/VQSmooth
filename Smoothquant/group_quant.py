import sys, os
sys.path.append(os.path.dirname(__file__))
from concurrent.futures import ThreadPoolExecutor
import torch
import torch_npu
from torch import nn
from dataclasses import dataclass
from Smoothquant.GroupQuant.group_quant_npu import (
    fake_quantize_activation_per_token,
    group_dequantize,
    group_quantize,
    group_quantize_dequantize,
)
from VQquant.smooth_vq import SmoothVQ
from VQquant.vector_quant import VectorQuantizer
# from fuse_linear import (
#     build_group_reduce_matmul_weight_cache,
#     build_quant_reduce_sum_weight_cache,
#     ensure_nz_format,
#     fuse_dequant_with_group_reduce_matmul_cached,
#     fuse_dequant_with_quant_reduce_sum_cached,
#     reconstruct_activation_scales,
# )

# try:
#     from custom_op.group_reduce_matmul.python import group_reduce_matmul as group_reduce_matmul_custom_op
# except Exception:
#     group_reduce_matmul_custom_op = None

try:
    from VQquant.VectorQuant import vector_quant_npu as VectorQuant
except ModuleNotFoundError:
    try:
        import VectorQuant
    except ModuleNotFoundError:
        VectorQuant = None

# @torch.no_grad()
# def quantize_weight_per_channel_absmax(t, n_bits=8):
#     scales = t.abs().max(dim=-1, keepdim=True)[0]
#     q_max = 2 ** (n_bits - 1) - 1
#     scales.clamp_(min=1e-5).div_(q_max)
#     t_q = (t / scales).round().clamp(-q_max, q_max)
#     return t_q, scales

# @torch.no_grad()
# def dequantize_weight_per_channel_absmax(t_q, scales):
#     if scales.dim() == 1:
#         if t_q.shape[0] == scales.numel():
#             scales = scales.unsqueeze(-1)
#         elif t_q.shape[-1] == scales.numel():
#             scales = scales.unsqueeze(0)
#     return t_q.float() * scales

@torch.no_grad()
def quantize_activation_per_tensor_absmax(t, n_bits=8, residual_group=32):
    t_shape = t.shape
    t.view(-1, t_shape[-1])
    scales = t.abs().max()
    q_max = 2 ** (n_bits - 1) - 1
    scales.clamp_(min=1e-5).div_(q_max)
    t.div_(scales).round_().mul_(scales)
    return t

# @torch.no_grad()
# def quantize_activation_per_token_absmax(
#     t: torch.Tensor,
#     n_bits: int = 8,
#     residual_group: int = 32,
#     residual_bits: int = 4,
# ):
#     """
#     Per-token quantization with group-wise scale residual approximation

#     Args:
#         t: [B, T, H] activation
#         n_bits: activation quant bits
#         residual_group: group size along H
#         residual_bits: E, bits for residual code e
#         R_min_ratio, R_max_ratio: search range relative to delta_base
#         R_steps: number of R candidates

#     Returns:
#         t_dq: dequantized activation after scale residual approximation
#     """
#     B, T, H = t.shape
#     assert H % residual_group == 0
#     G = H // residual_group
#     e_min = -(2 ** (residual_bits - 1))
#     e_max = 0

#     q_max = 2 ** (n_bits - 1) - 1

#     # ===== Step 0: preserve fp activation =====
#     t_fp = t.clone()

#     # ===== Step 1: per-(token, group) true scale Δ_i =====
#     t_group = t_fp.view(B, T, G, residual_group)
#     delta = t_group.abs().max(dim=-1)[0].clamp_(min=1e-5) / q_max          # [B, T, G]

#     # ===== Step 2: base scale Δ_base (per token) =====
#     delta_base = delta.max(dim=-1, keepdim=True)[0]       # [B, T, 1]

#     # ===== Step 3: initialize residual step R_init =====
#     R = delta_base / (2 ** residual_bits)  # [B,T,1]

#     # ===== Step 4: residual code e =====
#     e = torch.round((delta - delta_base) / R)
#     e = torch.clamp(e, e_min, e_max)  # [B,T,G]

#     # ===== Step 5: reconstructed scale =====
#     delta_rec = delta_base + e * R
#     delta_rec = torch.clamp(delta_rec, min=1e-5)  # [B,T,G]

#     # ===== Step 6: quantize & dequantize =====
#     scale = delta_rec.unsqueeze(-1)  # [B,T,G,1]
#     t_q = torch.round(t_group / scale).clamp(-q_max, q_max)
#     t_dq = (t_q * scale).view(B, T, H).to(t.dtype)

#     return t_dq

@torch.no_grad()
def quantize_activation_per_token_absmax(
    t: torch.Tensor,
    n_bits: int = 8,
    residual_group: int = 32,
    residual_bits: int = 4,
):
    return group_quantize_dequantize(
        t,
        n_bits=n_bits,
        residual_group=residual_group,
        residual_bits=residual_bits,
        output_dtype=torch.bfloat16,
    )

@dataclass
class QuantTensor:
    def __init__(
        self,
        x,
        residual_group: int = 32
    ):
        super().__init__()
        B, T, H = x.shape
        assert H % residual_group == 0
        G = H // residual_group
        device = x.device

        self.activation = torch.empty_like(
            x, device=device, dtype=torch.int8
        )
        self.delta_base = torch.empty(
            B, T, device=device, dtype=torch.bfloat16
        )
        self.e = torch.empty(
            B, T, G, device=device, dtype=torch.int8
        )
        self.residual_group = residual_group
        self.quantize(x)

    def quantize(self, x, n_bits=8, residual_bits=4):
        qt = group_quantize(
            x,
            n_bits=n_bits,
            residual_group=self.residual_group,
            residual_bits=residual_bits,
        )
        self.activation.copy_(qt.activation)
        self.delta_base.copy_(qt.delta_base)
        self.e.copy_(qt.e)

    def dequantize(self, residual_bits=4):
        return group_dequantize(
            self.activation,
            self.delta_base,
            self.e,
            residual_group=self.residual_group,
            residual_bits=residual_bits,
            output_dtype=torch.bfloat16,
        )

class GroupQLinear(nn.Module):
    def __init__(
        self,
        in_features,
        out_features,
        n_groups,
        codebook_width,
        bias=True,
        fake_quant=False,
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.n_groups = n_groups
        self.weight = None
        self.codebook_width = codebook_width

        self.fake_quant = fake_quant
        self.skip_output_quant = False

        self.register_buffer(
            "weight_indices",
            torch.zeros(
                self.n_groups,
                self.out_features,
                dtype=torch.uint8,
                requires_grad=False,
            ),
        )
        self.register_buffer(
            "weight_codebook",
            torch.zeros(
                2 ** self.codebook_width,
                self.in_features,
                dtype=torch.bfloat16,
                requires_grad=False,
            ),
        )
        # self.register_buffer(
        #     "weight_scales",
        #     torch.randn(
        #         self.out_features,
        #         dtype=torch.bfloat16,
        #         requires_grad=False,
        #     ),
        # )
        if bias:
            self.register_buffer(
                "bias",
                torch.zeros(
                    (1, self.out_features), dtype=torch.bfloat16, requires_grad=False
                ),
            )
        else:
            self.register_buffer("bias", None)

    def to(self, *args, **kwargs):
        super(GroupQLinear, self).to(*args, **kwargs)
        return self

    @torch.no_grad()
    def _decode_weight_q(self, device):
        if self.weight is not None:
            return self.weight.to(device=device, dtype=torch.bfloat16)

        if VectorQuant is None:
            raise ModuleNotFoundError(
                "VectorQuant extension is not available. "
                "Build/install the VectorQuant extension or disable VQ weight quantization."
            )

        weight_q = torch.zeros(
            (self.out_features, self.in_features),
            dtype=torch.bfloat16,
            device=device,
        )
        VectorQuant.dequant_forward(
            self.weight_indices.to(device),
            self.weight_codebook.to(device),
            weight_q,
        )
        return weight_q


    @torch.no_grad()
    def forward(self, x):
        if not self.fake_quant:
            weight_q = self._decode_weight_q(x.activation.device)
            # weight_dq = dequantize_weight_per_channel_absmax(self.weight, self.weight_scales)
            bias = self.bias.squeeze(0) if self.bias is not None else None
            y = torch.nn.functional.linear(x.dequantize(), weight_q, bias)
            q_y = QuantTensor(y)
            return q_y
        else:
            d_x = fake_quantize_activation_per_token(x)
            # weight_dq = dequantize_weight_per_channel_absmax(self.weight, self.weight_scales)
            y = torch.functional.F.linear(d_x, self.weight, self.bias)
            y = fake_quantize_activation_per_token(y)
            return y

    @staticmethod
    def from_float(
        module, args, QClass, quant_device=None, target_device=None
    ):
        assert isinstance(module, torch.nn.Linear)
        if quant_device is None:
            quant_device = module.weight.device
        if target_device is None:
            target_device = module.weight.device

        quant_device = torch.device(quant_device)
        target_device = torch.device(target_device)
        if quant_device.type == "npu":
            torch.npu.set_device(quant_device)

        n_groups = int(module.in_features / args.sub_vector)
        new_module = GroupQLinear(
            in_features=module.in_features,
            out_features=module.out_features,
            n_groups= n_groups,
            bias=module.bias is not None,
            fake_quant=args.fake_quant,
            codebook_width=args.codebook_width
        )
        if not args.eval_only:
            # weight_q, weight_scales = quantize_weight_per_channel_absmax(
            #     module.weight.detach().float()
            # )
            # new_module.weight_scales.copy_(
            #     weight_scales.squeeze(-1).to(
            #         device=new_module.weight_scales.device,
            #         dtype=new_module.weight_scales.dtype,
            #     )
            # )

            smoothvq = SmoothVQ(module)
            smoothvq.quantizer = QClass()
            smoothvq.quantizer.configure(codebook_width=args.codebook_width)
            if args.fake_quant:
                vq_weight = smoothvq.fasterquant(fake_quant=args.fake_quant)
                new_module.weight = vq_weight.to(
                    device=new_module.weight.device,
                    dtype=new_module.weight.dtype,
                )
            else:
                indices, codebook = smoothvq.fasterquant(fake_quant=args.fake_quant)
                new_module.weight_indices.copy_(
                    indices.to(
                        device=new_module.weight_indices.device,
                        dtype=new_module.weight_indices.dtype,
                    )
                )
                new_module.weight_codebook.copy_(
                    codebook.to(
                        device=new_module.weight_codebook.device,
                        dtype=new_module.weight_codebook.dtype,
                    )
                )
            smoothvq.free()
            if quant_module is not module:
                del quant_module
                
            # else:
            #     # quant_module = module
            #     indices, codebook = smoothvq.fasterquant(fake_quant=args.fake_quant)
            #     new_module.weight_indices = indices.to(target_device)
            #     new_module.weight_codebook = codebook.to(target_device)
        # else:
        #     new_module.weight_indices = module.indices
        #     new_module.weight_codebook = module.codebook

        new_module = new_module.to(target_device)
        # new_module.build_group_reduce_matmul_cache(n_groups)

        if module.bias is not None:
            new_module.bias = module.bias.to(target_device)

        return new_module

    def __repr__(self):
        return f"GroupQLinear({self.in_features}, {self.out_features}, bias={self.bias is not None}, act_quant={self.act_quant_name}, output_quant={self.output_quant_name})"


def quantize_llama_like(
    model,
    args
):
    from transformers.models.llama.modeling_llama import (
        LlamaAttention,
        LlamaMLP,
    )
    from transformers.models.mistral.modeling_mistral import (
        MistralAttention,
        MistralMLP,
    )
    from modeling_llama import (
        QuantLlamaMLP,
        QuantLlamaAttention,
    )

    layer_counter = 0

    for name, m in model.model.named_modules():
        parent_name = name.rsplit(".", 1)[0]
        child_name = name.split(".")[-1]
        parent = model.model.get_submodule(parent_name)
        QClass = lambda: VectorQuantizer(
            sub_vector=args.sub_vector,
            assignment_chunk_size=args.assignment_chunk_size,
            kmeans_iters=args.kmeans_iters,
            codebook_width=args.codebook_width,
        )

        if isinstance(m, (LlamaMLP, MistralMLP)):
            print(parent_name, child_name)
            setattr(parent, child_name, QuantLlamaMLP(m, m.config, args, QClass=QClass))
            layer_counter += 1
        if isinstance(m, (LlamaAttention, MistralAttention)):
            print(parent_name, child_name)
            setattr(
                parent,
                child_name,
                QuantLlamaAttention(
                    m,
                    m.config,
                    args=args,
                    QClass=QClass,
                ),
            )
            layer_counter += 1
    return model


def quantize_linears_in_parallel(linear_specs, args, QClass, quant_devices, target_device):
    if not quant_devices:
        quant_devices = [str(target_device)]

    devices = [torch.device(device.strip()) for device in quant_devices if device.strip()]
    if not devices:
        devices = [torch.device(target_device)]

    if len(devices) == 1:
        device = devices[0]
        return {
            name: GroupQLinear.from_float(
                module,
                args,
                QClass=QClass,
                quant_device=device,
                target_device=target_device,
            )
            for name, module in linear_specs
        }

    print(f"Running {str(linear_specs)} VQ on devices: {', '.join(str(device) for device in devices)}")
    futures = {}
    results = {}
    with ThreadPoolExecutor(max_workers=min(len(linear_specs), len(devices))) as executor:
        for idx, (name, module) in enumerate(linear_specs):
            device = devices[idx % len(devices)]
            futures[name] = executor.submit(
                GroupQLinear.from_float,
                module,
                args,
                QClass,
                device,
                target_device,
            )
        for name, future in futures.items():
            results[name] = future.result()

    return results

def quantize_model(
    model,
    args
):
    from transformers.models.llama.modeling_llama import LlamaPreTrainedModel
    from transformers.models.mistral.modeling_mistral import MistralPreTrainedModel

    if isinstance(model, (LlamaPreTrainedModel, MistralPreTrainedModel)):
        return quantize_llama_like(
            model,
            args
        )
    else:
        raise ValueError(f"Unsupported model type: {type(model)}")
