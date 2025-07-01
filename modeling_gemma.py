import torch
from torch import nn
from typing import Optional, Tuple, List
from torch.nn import CrossEntropyLoss
import math
from modeling_siglip import SiglipVisionConfig, SiglipVisionModel

class KVCache():

    def __init__(self) -> None:
        self.key_cache: List[torch.Tensor] = []
        self.value_cache: List[torch.Tensor] = []

    def num_items(self) ->  int:
        if len(self.key_cache) == 0:
            return 0
        else:
            # Why - shape[-2] ? The key_cache shape is [Batch_Size, Num_Heads_KV, Seq_Len, Head_Dim]
            return self.key_cache[0].shape[-2]

    def update(
            self, 
            key_states: torch.Tensor,
            value_states: torch.Tensor,
            layer_idx: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if len(self.key_cache) <= layer_idx:
            # layer KV not yet added to the KV-Cache, hence create it.
            self.key_cache.append(key_states)
            self.value_cache.append(value_states)
        else:
            # else concatenate the new keys with the ones that already exist.
            # tensor shape is... [Batch_Size, Num_Heads_KV, Seq_Len, Head_Dim]
            self.key_cache[layer_idx] = torch.cat([self.key_cache[layer_idx], key_states], dim=-2)
            self.value_cache[layer_idx] = torch.cat([self.value_cache[layer_idx], value_states], dim=-2)

        # then return the existing keys + the new ones
        return self.key_cache[layer_idx], self.value_cache[layer_idx]
    

