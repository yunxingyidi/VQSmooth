import sys, os
sys.path.append(os.path.dirname(__file__))
import math
import torch
import torch.nn.functional as F
import torch.utils.checkpoint
from torch import nn
from group_quant import GroupQLinear, QuantTensor
from transformers.models.llama.configuration_llama import LlamaConfig
from transformers.models.llama.modeling_llama import  LlamaRotaryEmbedding, repeat_kv, apply_rotary_pos_emb
from transformers.utils import logging
from typing import Optional, Tuple
from transformers.cache_utils import Cache
from transformers.models.llama.modeling_llama import ACT2FN

logger = logging.get_logger(__name__)
class QuantLlamaMLP(nn.Module):
    def __init__(self, float_mlp, config, args, QClass=None):
        super().__init__()
        self.config = config
        self.fake_quant = args.fake_quant
        self.QClass = QClass
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size
        self.gate_proj = GroupQLinear.from_float(float_mlp.gate_proj, args, QClass=QClass)
        self.up_proj = GroupQLinear.from_float(float_mlp.up_proj, args, QClass=QClass)
        self.down_proj = GroupQLinear.from_float(float_mlp.down_proj, args, QClass=QClass)

        self.act_fn = ACT2FN[config.hidden_act]

    def forward(self, x):
        # down_proj = self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
        if not self.fake_quant:
            x_quant = QuantTensor(x)
            gate_q = self.gate_proj(x_quant)  # QuantTensor

            up_q = self.up_proj(x_quant)

            activation = self.act_fn(gate_q.dequantize()) * up_q.dequantize()
            hidden_f = QuantTensor(activation)

            out_q = self.down_proj(hidden_f)
            return out_q.dequantize()
        else:
            gate_q = self.gate_proj(x)
            up_q = self.up_proj(x)
            out_q = self.down_proj(self.act_fn(gate_q) * up_q)
            return out_q

class QuantLlamaAttention(nn.Module):
    """Multi-headed attention from 'Attention Is All You Need' paper"""
    def __init__(self, float_attention, config: LlamaConfig, layer_idx: Optional[int] = None, args=None, QClass=None):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        if layer_idx is None:
            logger.warning_once(
                f"Instantiating {self.__class__.__name__} without passing a `layer_idx` is not recommended and will "
                "lead to errors during the forward call if caching is used. Please make sure to provide a `layer_idx` "
                "when creating this class."
            )

        self.attention_dropout = config.attention_dropout
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = getattr(config, "head_dim", self.hidden_size // self.num_heads)
        self.num_key_value_heads = config.num_key_value_heads
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads
        self.max_position_embeddings = config.max_position_embeddings
        self.rope_theta = config.rope_theta
        self.is_causal = True

        self.q_proj = GroupQLinear.from_float(float_attention.q_proj, args, QClass=QClass)
        self.k_proj = GroupQLinear.from_float(float_attention.k_proj, args, QClass=QClass)
        self.v_proj = GroupQLinear.from_float(float_attention.v_proj, args, QClass=QClass)
        self.o_proj = GroupQLinear.from_float(float_attention.o_proj, args, QClass=QClass)
        self.fake_quant = args.fake_quant

        self.rotary_emb = LlamaRotaryEmbedding(config=self.config)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Cache] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        cache_position: Optional[torch.LongTensor] = None,
        position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,  # will become mandatory in v4.46
        **kwargs,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        bsz, q_len, _ = hidden_states.size()

        if self.config.pretraining_tp > 1:
            key_value_slicing = (self.num_key_value_heads * self.head_dim) // self.config.pretraining_tp
            query_slices = self.q_proj.weight.split(
                (self.num_heads * self.head_dim) // self.config.pretraining_tp, dim=0
            )
            key_slices = self.k_proj.weight.split(key_value_slicing, dim=0)
            value_slices = self.v_proj.weight.split(key_value_slicing, dim=0)

            query_states = [F.linear(hidden_states, query_slices[i]) for i in range(self.config.pretraining_tp)]
            query_states = torch.cat(query_states, dim=-1)

            key_states = [F.linear(hidden_states, key_slices[i]) for i in range(self.config.pretraining_tp)]
            key_states = torch.cat(key_states, dim=-1)

            value_states = [F.linear(hidden_states, value_slices[i]) for i in range(self.config.pretraining_tp)]
            value_states = torch.cat(value_states, dim=-1)

        else:
            if not self.fake_quant:
                q_hidden_states = QuantTensor(hidden_states)
                query_states = self.q_proj(q_hidden_states)
                key_states = self.k_proj(q_hidden_states)
                value_states = self.v_proj(q_hidden_states)

                query_states = query_states.dequantize()
                key_states = key_states.dequantize()
                value_states = value_states.dequantize()
            else:
                query_states = self.q_proj(hidden_states)
                key_states = self.k_proj(hidden_states)
                value_states = self.v_proj(hidden_states)

        query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)

        if position_embeddings is None:
            logger.warning_once(
                "The attention layers in this model are transitioning from computing the RoPE embeddings internally "
                "through `position_ids` (2D tensor with the indexes of the tokens), to using externally computed "
                "`position_embeddings` (Tuple of tensors, containing cos and sin). In v4.46 `position_ids` will be "
                "removed and `position_embeddings` will be mandatory."
            )
            cos, sin = self.rotary_emb(value_states, position_ids)
        else:
            cos, sin = position_embeddings

        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

        if past_key_value is not None:
            # sin and cos are specific to RoPE models; cache_position needed for the static cache
            cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
            key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx, cache_kwargs)
            print(f"Past key shape: {past_key_value.key.shape if past_key_value.key is not None else None}")
            print(f"Past value shape: {past_key_value.value.shape if past_key_value.value is not None else None}")

        key_states = repeat_kv(key_states, self.num_key_value_groups)
        value_states = repeat_kv(value_states, self.num_key_value_groups)

        causal_mask = self.make_causal_mask(bsz, q_len, hidden_states.device, hidden_states.dtype)

        if attention_mask is not None:
            attn_mask = (1.0 - attention_mask) * float('-inf')
            attn_mask = attn_mask.expand(bsz, 1, q_len, q_len)
            final_mask = causal_mask + attn_mask
        else:
            final_mask = causal_mask

        attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) / math.sqrt(self.head_dim)
        attn_weights = attn_weights + final_mask

        # softmax
        attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
        attn_weights = nn.functional.dropout(attn_weights, p=self.attention_dropout, training=self.training)

        attn_output = torch.matmul(attn_weights, value_states)
        attn_output = attn_output.transpose(1, 2).contiguous().reshape(bsz, q_len, -1)

        if not output_attentions:
            attn_weights = None

        if not self.fake_quant:
            attn_output = QuantTensor(attn_output)
            attn_output = self.o_proj(attn_output)
            return attn_output.dequantize(), attn_weights, past_key_value
        else:
            attn_output = self.o_proj(attn_output)
            return attn_output, attn_weights, past_key_value

    def make_causal_mask(self, bsz, q_len, device, dtype):
        mask = torch.triu(torch.full((q_len, q_len), float('-inf'), device=device, dtype=dtype), diagonal=1)
        return mask.unsqueeze(0).expand(bsz, 1, q_len, q_len)