from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import torch
from transformers.cache_utils import DynamicCache

from Smoothquant.GroupQuant.group_quant_npu import group_dequantize
from group_quant import KVQuantTensor


# @dataclass
# class CachedKVQuantTensor:
#     activation: torch.Tensor
#     delta_base: torch.Tensor
#     e: torch.Tensor
#     residual_group: int = 32

#     def __post_init__(self) -> None:
#         self.batch_size = self.activation.shape[0]
#         self.num_heads = self.activation.shape[1]
#         self.seq_len = self.activation.shape[2]
#         self.head_dim = self.activation.shape[3]

#     def dequantize(self, residual_bits: int = 4) -> torch.Tensor:
#         flat = group_dequantize(
#             self.activation.reshape(self.batch_size * self.num_heads, self.seq_len, self.head_dim),
#             self.delta_base.reshape(self.batch_size * self.num_heads, self.seq_len),
#             self.e.reshape(self.batch_size * self.num_heads, self.seq_len, -1),
#             residual_group=self.residual_group,
#             residual_bits=residual_bits,
#             output_dtype=torch.bfloat16,
#         )
#         return flat.reshape(self.batch_size, self.num_heads, self.seq_len, self.head_dim)


class QuantizedDynamicCache(DynamicCache):
    def __init__(self, residual_group: int = 32) -> None:
        super().__init__()
        self.residual_group = residual_group
        self.key_cache: list[KVQuantTensor] = []
        self.value_cache: list[KVQuantTensor] = []

    def __getitem__(self, layer_idx: int):
        if layer_idx < len(self):
            key = self.key_cache[layer_idx]
            value = self.value_cache[layer_idx]
            return key.dequantize(), value.dequantize()
        raise KeyError(f"Cache only has {len(self)} layers, attempted to access layer with index {layer_idx}")

    def __iter__(self):
        for layer_idx in range(len(self)):
            key = self.key_cache[layer_idx]
            value = self.value_cache[layer_idx]
            yield (key.dequantize(), value.dequantize())

    def get_seq_length(self, layer_idx: Optional[int] = 0) -> int:
        if len(self.key_cache) <= layer_idx:
            return 0
        return self.key_cache[layer_idx].seq_len

    def get_max_length(self) -> Optional[int]:
        return None

    def reorder_cache(self, beam_idx: torch.LongTensor):
        for layer_idx in range(len(self.key_cache)):
            key = self.key_cache[layer_idx]
            value = self.value_cache[layer_idx]
            device = key.activation.device
            beam = beam_idx.to(device)
            key.activation = key.activation.index_select(0, beam)
            key.delta_base = key.delta_base.index_select(0, beam)
            key.e = key.e.index_select(0, beam)
            key.batch_size = key.activation.shape[0]
            value.activation = value.activation.index_select(0, beam)
            value.delta_base = value.delta_base.index_select(0, beam)
            value.e = value.e.index_select(0, beam)
            value.batch_size = value.activation.shape[0]

    def update(
        self,
        key_states: KVQuantTensor,
        value_states: KVQuantTensor,
        layer_idx: int,
        cache_kwargs=None,
    ) -> Tuple[KVQuantTensor, KVQuantTensor]:
        if cache_kwargs is None:
            return super().update(key_states.activation, value_states.activation, layer_idx, cache_kwargs)

        k_delta_base = cache_kwargs.get("k_delta_base")
        v_delta_base = cache_kwargs.get("v_delta_base")
        k_e = cache_kwargs.get("k_e")
        v_e = cache_kwargs.get("v_e")

        if any(item is None for item in (k_delta_base, v_delta_base, k_e, v_e)):
            return super().update(key_states, value_states, layer_idx, cache_kwargs)

        if layer_idx == 0:
            self._seen_tokens += key_states.shape[-2]

        # key_chunk = CachedKVQuantTensor(
        #     activation=key_states,
        #     delta_base=k_delta_base,
        #     e=k_e,
        #     residual_group=self.residual_group,
        # )
        # value_chunk = CachedKVQuantTensor(
        #     activation=value_states,
        #     delta_base=v_delta_base,
        #     e=v_e,
        #     residual_group=self.residual_group,
        # )

        if len(self.key_cache) <= layer_idx:
            self.key_cache.append(key_states)
            self.value_cache.append(value_states)
        else:
            cached_k = self.key_cache[layer_idx]
            cached_v = self.value_cache[layer_idx]
            cached_k.activation = torch.cat([cached_k.activation, key_states.activation], dim=-2)
            cached_k.delta_base = torch.cat([cached_k.delta_base, key_states.delta_base], dim=-1)
            cached_k.e = torch.cat([cached_k.e, key_states.e], dim=-2)
            cached_k.seq_len = cached_k.activation.shape[-2]
            cached_v.activation = torch.cat([cached_v.activation, value_states.activation], dim=-2)
            cached_v.delta_base = torch.cat([cached_v.delta_base, value_states.delta_base], dim=-1)
            cached_v.e = torch.cat([cached_v.e, value_states.e], dim=-2)
            cached_v.seq_len = cached_v.activation.shape[-2]

        return self.key_cache[layer_idx], self.value_cache[layer_idx]
