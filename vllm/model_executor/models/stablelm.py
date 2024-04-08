# coding=utf-8
# Adapted from
# https://github.com/huggingface/transformers/blob/v4.28.0/src/transformers/models/gpt_neox/modeling_gpt_neox.py
# Copyright 2023 The vLLM team.
# Copyright 2022 EleutherAI The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Inference-only GPT-NeoX model compatible with HuggingFace weights.

The input of the model is flattened to a 1D tensor of tokens. The model uses
InputMetadata to extract the original 2D shape of the input.
"""
from typing import Dict, List, Optional, Tuple

import torch
from torch import nn

from vllm.model_executor.input_metadata import InputMetadata
from vllm.model_executor.layers.activation import get_act_fn
from vllm.model_executor.layers.activation import SiluAndMul
from vllm.model_executor.layers.rotary_embedding import get_rope
from vllm.model_executor.layers.sampler import Sampler
from vllm.model_executor.weight_utils import (default_weight_loader,
                                              hf_model_weights_iterator)
from vllm.model_executor.parallel_utils.parallel_state import (
    get_tensor_model_parallel_rank, get_tensor_model_parallel_world_size)
from vllm.model_executor.layers.vocab_parallel_embedding import (
    VocabParallelEmbedding, ParallelLMHead)
from vllm.model_executor.layers.linear import (LinearMethodBase,
                                               MergedColumnParallelLinear,
                                               QKVParallelLinear,
                                               RowParallelLinear)
from vllm.model_executor.layers.attention import PagedAttention

from vllm.model_executor.sampling_metadata import SamplingMetadata
from vllm.sequence import SamplerOutput
from transformers import AutoTokenizer

KVCache = Tuple[torch.Tensor, torch.Tensor]


class LayerNormPerHead(torch.nn.Module):
    def __init__(
        self,
        head_dim: int,
        num_heads: int,
        eps: float = 1e-5,
        bias: bool = False,
    ):
        super().__init__()
        self.head_dim = head_dim
        self.num_heads = num_heads
        self.norms = torch.torch.nn.ModuleList(
            [nn.LayerNorm(head_dim, eps=eps, bias=bias) for _ in range(self.num_heads)]
        )

    def forward(self, x: torch.Tensor):
        # Split along the num_heads axis to get per-head inputs
        # [batch_size, num_heads, seq_len, head_dim] -> [batch_size, 1, seq_len, head_dim] * num_heads
        heads = torch.split(x, 1, dim=1)
        # Normalize and put the heads back together
        return torch.cat([norm(x) for norm, x in zip(self.norms, heads)], dim=1)


class StableLMAttention(nn.Module):

    def __init__(self,
                 config,
                 linear_method: Optional[LinearMethodBase] = None
                 ):
        super().__init__()
        self.total_num_heads = config.num_attention_heads
        self.hidden_size = config.hidden_size
        self.head_size = self.hidden_size // self.total_num_heads

        tensor_model_parallel_world_size = (
            get_tensor_model_parallel_world_size())
        assert self.total_num_heads % tensor_model_parallel_world_size == 0
        self.num_heads = (self.total_num_heads //
                          tensor_model_parallel_world_size)
        self.qkv_bias = getattr(config, "use_qkv_bias", False)
        self.total_num_kv_heads = getattr(config, "num_key_value_heads", self.total_num_heads)
        self.q_size = self.num_heads * self.head_size
        if (self.total_num_kv_heads != self.total_num_heads):
            self.num_kv_heads = self.total_num_kv_heads // tensor_model_parallel_world_size
            self.q_norm = LayerNormPerHead(self.head_size, self.num_heads)
            self.k_norm = LayerNormPerHead(self.head_size, self.num_kv_heads)
            self.kv_size = self.num_kv_heads * self.head_size
        else:
            self.num_kv_heads = self.num_heads
            self.kv_size = self.q_size
            self.q_norm = nn.Identity()
            self.k_norm = nn.Identity()
        self.qkv_proj = QKVParallelLinear(
            config.hidden_size,
            self.head_size,
            self.total_num_heads,
            self.total_num_kv_heads,
            bias=self.qkv_bias,
            linear_method=linear_method
        )

        self.o_proj = RowParallelLinear(
            config.hidden_size,
            config.hidden_size,
            bias=False,
            linear_method=linear_method,
        )

        scaling = self.head_size**-0.5
        rotary_dim = int(self.head_size * config.rope_pct)
        rope_theta = getattr(config, "rope_theta", 10000)
        max_position_embeddings = getattr(config, "max_position_embeddings",
                                          8192)
        assert rotary_dim % 2 == 0
        self.rotary_emb = get_rope(
            self.head_size,
            rotary_dim=rotary_dim,
            max_position=max_position_embeddings,
            base=rope_theta,
        )
        self.attn = PagedAttention(self.num_heads, self.head_size, scaling, num_kv_heads=self.num_kv_heads)

    def forward(
        self,
        position_ids: torch.Tensor,
        hidden_states: torch.Tensor,
        kv_cache: KVCache,
        input_metadata: InputMetadata,
    ) -> torch.Tensor:
        bsz, q_len, _ = hidden_states.size()
        qkv, _ = self.qkv_proj(hidden_states)
        q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)
        q = self.q_norm(q.view(
            bsz, q_len, self.num_heads, self.head_size
        ).transpose(1, 2)).transpose(1, 2).reshape(bsz, q_len, self.q_size).contiguous()
        k = self.k_norm(k.view(
            bsz, q_len, self.num_kv_heads, self.head_size
        ).transpose(1, 2)).transpose(1, 2).reshape(bsz, q_len, self.kv_size).contiguous()
        q, k = self.rotary_emb(position_ids, q, k)
        k_cache, v_cache = kv_cache
        attn_output = self.attn(q, k, v.contiguous(), k_cache, v_cache, input_metadata)
        output, _ = self.o_proj(attn_output)
        return output


class StableLMMLP(nn.Module):

    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        hidden_act: str,
        linear_method: Optional[LinearMethodBase] = None,
    ):
        super().__init__()
        self.gate_up_proj = MergedColumnParallelLinear(
            hidden_size, [intermediate_size] * 2,
            bias=False,
            linear_method=linear_method)
        self.down_proj = RowParallelLinear(intermediate_size,
                                           hidden_size,
                                           bias=False,
                                           linear_method=linear_method)
        if hidden_act != "silu":
            raise ValueError(f"Unsupported activation: {hidden_act}. "
                             "Only silu is supported for now.")
        self.act_fn = SiluAndMul()

    def forward(self, x):
        gate_up, _ = self.gate_up_proj(x)
        x = self.act_fn(gate_up)
        x, _ = self.down_proj(x)
        return x



class StableLMLayer(nn.Module):

    def __init__(self, config,
        linear_method: Optional[LinearMethodBase] = None,):
        super().__init__()
        use_ln_bias = getattr(config, "num_key_value_heads", None) == getattr(config, "num_attention_heads", None)
        self.input_layernorm = nn.LayerNorm(config.hidden_size,
                                            eps=config.norm_eps,
                                            bias=use_ln_bias)
        self.parallel_attn_mlp = getattr(config, "num_key_value_heads", None) != getattr(config, "num_attention_heads", None)
        if not self.parallel_attn_mlp:
            self.post_attention_layernorm = nn.LayerNorm(config.hidden_size,
                                                         eps=config.norm_eps)
        self.self_attn = StableLMAttention(config, linear_method)
        self.mlp = StableLMMLP(config.hidden_size, config.intermediate_size, 'silu', linear_method)

    def forward(
        self,
        position_ids: torch.Tensor,
        hidden_states: torch.Tensor,
        kv_cache: KVCache,
        input_metadata: InputMetadata,
    ) -> torch.Tensor:
        residual = hidden_states
        attn_input = self.input_layernorm(hidden_states)
        hidden_states = self.self_attn(
            position_ids=position_ids,
            hidden_states=attn_input,
            kv_cache=kv_cache,
            input_metadata=input_metadata,
        )
        if not self.parallel_attn_mlp:
            hidden_states = residual + hidden_states
            residual = hidden_states
            mlp_input = self.post_attention_layernorm(hidden_states)
        else:
            mlp_input = attn_input
        mlp_output = self.mlp(mlp_input)
        if self.parallel_attn_mlp:
            hidden_states = residual + mlp_output + hidden_states
        else:
            hidden_states = mlp_output + residual
        return hidden_states


class StableLMModel(nn.Module):

    def __init__(
        self,
        config,
        linear_method: Optional[LinearMethodBase] = None,
    ):
        super().__init__()
        self.config = config

        self.embed_tokens = VocabParallelEmbedding(config.vocab_size,
                                               config.hidden_size)
        self.layers = nn.ModuleList(
            [StableLMLayer(config) for _ in range(config.num_hidden_layers)])
        use_ln_bias = getattr(config, "num_key_value_heads", None) is None
        self.norm = nn.LayerNorm(config.hidden_size, eps=config.norm_eps, bias=use_ln_bias)

    def forward(
        self,
        input_ids: torch.Tensor,
        position_ids: torch.Tensor,
        kv_caches: List[KVCache],
        input_metadata: InputMetadata,
    ) -> torch.Tensor:
        hidden_states = self.embed_tokens(input_ids)
        for i in range(len(self.layers)):
            layer = self.layers[i]
            hidden_states = layer(
                position_ids,
                hidden_states,
                kv_caches[i],
                input_metadata,
            )
        hidden_states = self.norm(hidden_states)
        return hidden_states


class StableLMEpochForCausalLM(nn.Module):

    def __init__(self,
                 config,
                 linear_method: Optional[LinearMethodBase] = None,):
        super().__init__()
        self.config = config
        self.model = StableLMModel(config, linear_method)
        self.lm_head = ParallelLMHead(config.vocab_size, config.hidden_size)
        self.sampler = Sampler(100289)

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        kv_caches: List[KVCache],
        input_metadata: InputMetadata,
    ) -> torch.Tensor:
        hidden_states = self.model(input_ids, positions, kv_caches,
                                      input_metadata)
        return hidden_states

    def sample(
        self,
        hidden_states: torch.Tensor,
        sampling_metadata: SamplingMetadata,
    ) -> Optional[SamplerOutput]:
        next_tokens = self.sampler(self.lm_head.weight, hidden_states,
                                   sampling_metadata)
        return next_tokens


    def load_weights(self,
                     model_name_or_path: str,
                     cache_dir: Optional[str] = None,
                     load_format: str = "auto",
                     revision: Optional[str] = None):
        stacked_params_mapping = [
            # (param_name, shard_name, shard_id)
            ("qkv_proj", "q_proj", "q"),
            ("qkv_proj", "k_proj", "k"),
            ("qkv_proj", "v_proj", "v"),
            ("gate_up_proj", "gate_proj", 0),
            ("gate_up_proj", "up_proj", 1),
        ]
        # Since the vocab is a subset of the embedding and output weights,
        # we need to set the vocab size of the sampler to the size of the
        # tokenizer's vocab.
        tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, trust_remote_code=True)
        print(tokenizer.eos_token, tokenizer.eos_token_id)
        #self.sampler.vocab_size = tokenizer.vocab_size
        print(tokenizer.vocab_size)
        params_dict = dict(self.named_parameters())
        tp_world_size = get_tensor_model_parallel_world_size()
        tp_rank = get_tensor_model_parallel_rank()
        n_kv_heads = getattr(self.config, "num_key_value_heads", None)
        if n_kv_heads is not None:
            n_kv_heads = n_kv_heads // tp_world_size
        num_heads = self.config.num_attention_heads // tp_world_size
        for name, loaded_weight in hf_model_weights_iterator(
                model_name_or_path, cache_dir, load_format, revision):
            if "rotary_emb.inv_freq" in name:
                continue
            if ("rotary_emb.cos_cached" in name
                    or "rotary_emb.sin_cached" in name):
                # Models trained using ColossalAI may include these tensors in
                # the checkpoint. Skip them.
                continue
            for (param_name, weight_name, shard_id) in stacked_params_mapping:
                if weight_name not in name:
                    continue
                name = name.replace(weight_name, param_name)
                # Skip loading extra bias for GPTQ models.
                if name.endswith(".bias") and name not in params_dict:
                    continue
                param = params_dict[name]
                weight_loader = param.weight_loader
                weight_loader(param, loaded_weight, shard_id)
                break
            else:
                # Skip loading extra bias for GPTQ models.
                if name.endswith(".bias") and name not in params_dict:
                    continue
                # Handle splitting up the list of norms per head...
                if (n_kv_heads is None) or (get_tensor_model_parallel_world_size() > 1):
                    # don't need to worry about it if tp == 1
                    if "q_norm.norms" in name:
                        norm_indx = int(name.split("q_norm.norms.")[1].split(".")[0])
                        if (norm_indx >= (tp_rank * num_heads)) and (norm_indx < ((tp_rank + 1) * num_heads)):
                            name = name.replace(f"q_norm.norms.{norm_indx}", f"q_norm.norms.{norm_indx % num_heads}")
                        else:
                            continue
                    elif "k_norm.norms" in name:
                        norm_indx = int(name.split("k_norm.norms.")[1].split(".")[0])
                        if (norm_indx >= (tp_rank * n_kv_heads)) and (norm_indx < ((tp_rank + 1) * n_kv_heads)):
                            name = name.replace(f"k_norm.norms.{norm_indx}", f"k_norm.norms.{norm_indx % n_kv_heads}")
                        else:
                            continue
                param = params_dict[name]
                weight_loader = getattr(param, "weight_loader",
                                        default_weight_loader)
                weight_loader(param, loaded_weight)