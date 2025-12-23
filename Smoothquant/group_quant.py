import torch
from torch import nn
from functools import partial

@torch.no_grad()
def quantize_activation_per_token_absmax(t, n_bits=8):
    t_shape = t.shape
    t.view(-1, t_shape[-1])
    scales = t.abs().max(dim=-1, keepdim=True)[0]
    q_max = 2 ** (n_bits - 1) - 1
    scales.clamp_(min=1e-5).div_(q_max)
    t.div_(scales).round_().mul_(scales)
    return t

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
# def quantize_activation_per_token_absmax(t, n_bits=8, residual_group=32):
    # orig_shape = t.shape
    # H = orig_shape[-1]
    # assert H % residual_group == 0
    # G = H // residual_group
    # q_max = 2 ** (n_bits - 1) - 1
    #
    # t_hat = torch.empty_like(t)
    #
    # for i in range(G):
    #     start = i * residual_group
    #     end = (i + 1) * residual_group
    #     cluster = t[..., start:end]  # [..., residual_group]
    #
    #     # Step 2: cluster-wise scale
    #     delta_i = cluster.abs().amax(dim=-1, keepdim=True) / q_max
    #
    #     # Step 3: base scale
    #     delta_base = delta_i.max(dim=-2, keepdim=True).values
    #
    #     # Step 4: residual step
    #     R = (delta_i - delta_base).abs().mean(dim=-2, keepdim=True)
    #
    #     # Step 5: 量化 + 残差补偿
    #     t_q = (cluster / delta_base).round() * delta_base
    #     t_hat[..., start:end] = t_q + R  # 不再 expand_as 整个 t
    #
    # return t_hat


class GroupQLinear(nn.Module):
    def __init__(
        self,
        in_features,
        out_features,
        bias=True,
        act_quant="per_token",
        quantize_output=False,
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        self.register_buffer(
            "weight",
            torch.randn(
                self.out_features,
                self.in_features,
                dtype=torch.float16,
                requires_grad=False,
            ),
        )
        if bias:
            self.register_buffer(
                "bias",
                torch.zeros(
                    (1, self.out_features), dtype=torch.float16, requires_grad=False
                ),
            )
        else:
            self.register_buffer("bias", None)

        if act_quant == "per_token":
            self.act_quant_name = "per_token"
            self.act_quant = partial(quantize_activation_per_token_absmax, n_bits=8)
        elif act_quant == "per_tensor":
            self.act_quant_name = "per_tensor"
            self.act_quant = partial(quantize_activation_per_tensor_absmax, n_bits=8)
        else:
            raise ValueError(f"Invalid act_quant: {act_quant}")

        if quantize_output:
            self.output_quant_name = self.act_quant_name
            self.output_quant = self.act_quant
        else:
            self.output_quant_name = "None"
            self.output_quant = lambda x: x

    def to(self, *args, **kwargs):
        super(GroupQLinear, self).to(*args, **kwargs)
        self.weight = self.weight.to(*args, **kwargs)
        if self.bias is not None:
            self.bias = self.bias.to(*args, **kwargs)
        return self

    @torch.no_grad()
    def forward(self, x):
        q_x = self.act_quant(x)
        y = torch.functional.F.linear(q_x, self.weight, self.bias)
        q_y = self.output_quant(y)
        return q_y

    @staticmethod
    def from_float(
        module, act_quant="per_token", quantize_output=False
    ):
        assert isinstance(module, torch.nn.Linear)
        new_module = GroupQLinear(
            module.in_features,
            module.out_features,
            module.bias is not None,
            act_quant=act_quant,
            quantize_output=quantize_output,
        )
        new_module.weight = module.weight

        if module.bias is not None:
            new_module.bias = module.bias
        return new_module

    def __repr__(self):
        return f"GroupQLinear({self.in_features}, {self.out_features}, bias={self.bias is not None}, act_quant={self.act_quant_name}, output_quant={self.output_quant_name})"


def quantize_llama_like(
    model, act_quant="per_token", quantize_bmm_input=False
):
    from transformers.models.llama.modeling_llama import (
        LlamaAttention,
        LlamaMLP,
    )

    from transformers.models.mistral.modeling_mistral import (
        MistralAttention,
        MistralMLP,
    )

    for name, m in model.model.named_modules():
        if isinstance(m, (LlamaMLP, MistralMLP)):
            m.gate_proj = GroupQLinear.from_float(
                m.gate_proj, act_quant=act_quant
            )
            m.up_proj = GroupQLinear.from_float(
                m.up_proj, act_quant=act_quant
            )
            m.down_proj = GroupQLinear.from_float(
                m.down_proj, act_quant=act_quant
            )
        elif isinstance(m, (LlamaAttention, MistralAttention)):
            # Her we simulate quantizing BMM inputs by quantizing the output of q_proj, k_proj, v_proj
            m.q_proj = GroupQLinear.from_float(
                m.q_proj,
                act_quant=act_quant,
                quantize_output=quantize_bmm_input,
            )
            m.k_proj = GroupQLinear.from_float(
                m.k_proj,
                act_quant=act_quant,
                quantize_output=quantize_bmm_input,
            )
            m.v_proj = GroupQLinear.from_float(
                m.v_proj,
                act_quant=act_quant,
                quantize_output=quantize_bmm_input,
            )
            m.o_proj = GroupQLinear.from_float(
                m.o_proj, act_quant=act_quant
            )
    return model


def quantize_model(
    model, act_quant="per_token", quantize_bmm_input=False
):
    from transformers.models.llama.modeling_llama import LlamaPreTrainedModel
    from transformers.models.mistral.modeling_mistral import MistralPreTrainedModel

    if isinstance(model, (LlamaPreTrainedModel, MistralPreTrainedModel)):
        return quantize_llama_like(
            model,
            act_quant=act_quant,
            quantize_bmm_input=quantize_bmm_input,
        )
    else:
        raise ValueError(f"Unsupported model type: {type(model)}")