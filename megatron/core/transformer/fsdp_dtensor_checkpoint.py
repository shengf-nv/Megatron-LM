# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
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

import re
import torch

try:
    from megatron.core.distributed.fsdp.src.megatron_fsdp.param_and_grad_buffer import (
        make_fsdp_dtensor,
    )

    HAVE_MEGATRON_FSDP = True
except ImportError:
    HAVE_MEGATRON_FSDP = False

from megatron.core import parallel_state
from megatron.core.tensor_parallel.layers import copy_tensor_model_parallel_attributes
from megatron.training.global_vars import get_args


def handle_experts_in_state_dict(model, model_state_dict, optimizer_state_dict):
    """
    Handle experts in model and optimizer state dicts.
    """
    assert HAVE_MEGATRON_FSDP, "This function requires Megatron-FSDP to be installed."
    args = get_args()

    ep_size = parallel_state.get_expert_model_parallel_world_size()
    ep_rank = parallel_state.get_expert_model_parallel_rank()
    num_local_experts = args.num_experts // ep_size
    local_expert_start = ep_rank * num_local_experts
    local_expert_end = local_expert_start + num_local_experts

    def get_expert_index_from_key(key):
        """Extract expert index from various expert key formats.

        Supported formats:
        - GroupedMLP: 'mlp.experts.linear_fc1.weight0', 'mlp.experts.linear_fc2.weight0'
        - SequentialMLP: 'mlp.experts.local_experts.0.linear_fc1.weight', 'mlp.experts.local_experts.0.linear_fc2.weight'

        Returns:
            int: Expert index if found, None otherwise.
        """
        # GroupedMLP: index is at the end after 'weight'
        if 'mlp.experts.linear_fc1.weight' in key or 'mlp.experts.linear_fc2.weight' in key:
            # Handle SwiGLU weight{idx}_w and weight{idx}_v format
            if key.endswith('_w') or key.endswith('_v'):
                index_str = key.split('weight')[-1][:-2]
            # Handle regular weight{idx} format
            else:
                index_str = key.split('weight')[-1]
            if index_str.isdigit():
                return int(index_str)
        # SequentialMLP: index is between 'local_experts.' and next '.'
        elif 'mlp.experts.local_experts' in key:
            index_str = key.split('local_experts.')[-1].split('.', 1)[0]
            if index_str.isdigit():
                return int(index_str)
        return None

    def should_keep_expert_key(expert_index):
        """Determine if this rank should keep this expert key based on expert index"""
        if expert_index is None:
            # If we can't determine expert index, keep the key (non-expert weights)
            return True

        # Check if this expert belongs to this rank
        return local_expert_start <= expert_index < local_expert_end

    def replace_expert_index_in_key(key, expert_index, state_dict):
        """Replace expert index in key with new index corresponding to the current rank"""
        new_expert_index = expert_index + local_expert_start
        # GroupedMLP: 'mlp.experts.linear_fc1.weight0', 'mlp.experts.linear_fc2.weight0'
        if 'mlp.experts.linear_fc1.weight' in key or 'mlp.experts.linear_fc2.weight' in key:
            # Handle SwiGLU weight{idx}_w and weight{idx}_v format
            if key.endswith('_w') or key.endswith('_v'):
                suffix = key[-2:]  # '_w' or '_v'
                new_key = key.replace(
                    f'weight{expert_index}{suffix}', f'weight{new_expert_index}{suffix}'
                )
            # Handle regular weight{idx} format
            else:
                new_key = key.replace(f'weight{expert_index}', f'weight{new_expert_index}')
        # SequentialMLP: index is between 'local_experts.' and next '.'
        elif 'mlp.experts.local_experts' in key:
            new_key = key.replace(
                f'local_experts.{expert_index}.', f'local_experts.{new_expert_index}.'
            )

        state_dict[new_key] = state_dict[key]
        del state_dict[key]

    # Process model state dict
    for key in list(model_state_dict.keys()):
        expert_index = get_expert_index_from_key(key)
        if not should_keep_expert_key(expert_index):
            replace_expert_index_in_key(key, expert_index, model_state_dict)

    # Process optimizer state param_group dict
    for key in list(optimizer_state_dict["param_to_group_meta"].keys()):
        expert_index = get_expert_index_from_key(key)
        if not should_keep_expert_key(expert_index):
            replace_expert_index_in_key(
                key, expert_index, optimizer_state_dict["param_to_group_meta"]
            )

    try:
        optimizer_state_dict = optimizer_state_dict["state"]
    except KeyError:
        optimizer_state_dict = {}

    # Process optimizer state dict
    if len(optimizer_state_dict) != 0:
        for key in list(optimizer_state_dict.keys()):
            expert_index = get_expert_index_from_key(key)
            if not should_keep_expert_key(expert_index):
                replace_expert_index_in_key(key, expert_index, optimizer_state_dict)


def handle_swiglu_in_state_dict(model, model_state_dict, optimizer_state_dict):
    """
    Handle SWiGLU in model and optimizer state dicts.
    """
    assert HAVE_MEGATRON_FSDP, "This function requires Megatron-FSDP to be installed."

    def intersection(s1, s2):
        # Only works for step=1
        start = max(s1.start, s2.start)
        stop = min(s1.stop, s2.stop)
        if start >= stop:
            return slice(0, 0)  # Empty slice if no intersection
        return slice(start, stop)

    def offset_slice(s, offset):
        return slice(s.start + offset, s.stop + offset)

    def is_swiglu_key(key):
        """
        Check if this key should be handled as SwiGLU linear_fc1 weight or bias.
        """
        # Non-expert MLP: 'mlp.linear_fc1.weight', 'mlp.linear_fc1.bias'
        # GroupedMLP: 'mlp.experts.linear_fc1.weight0', 'mlp.experts.linear_fc1.bias0'
        # SequentialMLP: 'mlp.experts.local_experts.0.linear_fc1.weight', 'mlp.experts.local_experts.0.linear_fc1.bias'
        return any(re.search(pat, key) for pat in [
            r"(.*)\.mlp\.linear_fc1\.weight$",
            r"(.*)\.mlp\.linear_fc1\.bias$",
            r"(.*)\.mlp\.experts\.linear_fc1\.weight(\d+)$",
            r"(.*)\.mlp\.experts\.linear_fc1\.bias(\d+)$",
            r"(.*)\.mlp\.experts\.local_experts\.(\d+)\.linear_fc1\.weight$",
            r"(.*)\.mlp\.experts\.local_experts\.(\d+)\.linear_fc1\.bias$",
        ])

    def split_swiglu_linear_fc1(data, dist_param, swiglu_shard_axis, is_expert_param):
        """
        Split the SWiGLU linear_fc1 parameter into two parts: weight_w and weight_v.
        """
        assert data.shape[swiglu_shard_axis] % 2 == 0, (
            f"SWiGLU weights must have an even size along the shard axis {swiglu_shard_axis}, "
            f"got {data.shape[swiglu_shard_axis]}"
        )

        fsdp_slice = dist_param.megatron_fsdp_slice
        megatron_fsdp_dist_index = dist_param.megatron_fsdp_dist_index

        tp_mesh = megatron_fsdp_dist_index.get_submesh(
            [megatron_fsdp_dist_index.tp_dim], is_expert_parallel=is_expert_param
        )
        data_size = data.numel() // tp_mesh.mesh.numel()
        w_slice = slice(0, data_size // 2)
        v_slice = slice(data_size // 2, data_size)

        view_shape = list(data.shape)
        view_shape[swiglu_shard_axis] = -1
        local_tensor = data.to_local()
        weight_w = local_tensor.view(-1)[
            offset_slice(intersection(fsdp_slice, w_slice), -fsdp_slice.start)
        ]
        weight_v = local_tensor.view(-1)[
            offset_slice(intersection(fsdp_slice, v_slice), -fsdp_slice.start)
        ]
        weight_w = weight_w.reshape(view_shape)
        weight_v = weight_v.reshape(view_shape)

        # Fake parameters w and v are used to provide the correct parameter
        # shape and Tensor-Parallelism information.
        per_tp_rank_shape = list(data.shape)
        if getattr(dist_param, "tensor_model_parallel", False):
            tp_dim = dist_param.partition_dim
            per_tp_rank_shape[tp_dim] //= tp_mesh.mesh.numel()
        linear_fc1_meta = torch.empty(*per_tp_rank_shape, device="meta")
        w_meta, v_meta = torch.chunk(linear_fc1_meta, 2, dim=swiglu_shard_axis)
        copy_tensor_model_parallel_attributes(w_meta, dist_param)
        copy_tensor_model_parallel_attributes(v_meta, dist_param)

        weight_w = make_fsdp_dtensor(
            weight_w.data,
            w_meta,
            dist_index=megatron_fsdp_dist_index,
            is_expert_param=is_expert_param,
            run_check=True,
            update_uneven_dtensor_chunk_meta=True,
        )
        weight_v = make_fsdp_dtensor(
            weight_v.data,
            v_meta,
            dist_index=megatron_fsdp_dist_index,
            is_expert_param=is_expert_param,
            run_check=True,
            update_uneven_dtensor_chunk_meta=True,
        )
        return weight_w, weight_v

    for key in list(model_state_dict.keys()):
        if is_swiglu_key(key):
            dist_param = model.get_parameter(f"module.{key}")
            weight_w, weight_v = split_swiglu_linear_fc1(
                model_state_dict[key],
                dist_param,
                swiglu_shard_axis=0,
                is_expert_param='mlp.experts' in key,
            )

            # Update the model state dict with the new keys
            model_state_dict[f"{key}_w"] = weight_w
            model_state_dict[f"{key}_v"] = weight_v
            del model_state_dict[key]

    try:
        optimizer_state_dict = optimizer_state_dict["state"]
    except KeyError:
        optimizer_state_dict = {}

    if len(optimizer_state_dict) != 0:
        for key in list(optimizer_state_dict.keys()):
            if not is_swiglu_key(key):
                continue
            optimizer_state_dict[f"{key}_w"] = optimizer_state_dict[key].copy()
            optimizer_state_dict[f"{key}_v"] = optimizer_state_dict[key].copy()
            for subkey in ["exp_avg", "exp_avg_sq"]:
                dist_param = model.get_parameter(key[len("module.") :])
                weight_w, weight_v = split_swiglu_linear_fc1(
                    optimizer_state_dict[key][subkey], dist_param, swiglu_shard_axis=0
                )
                # Update the optimizer state dict with the new keys
                optimizer_state_dict[f"{key}_w"][subkey] = weight_w
                optimizer_state_dict[f"{key}_v"][subkey] = weight_v
            del optimizer_state_dict[key]


def handle_fp8_extra_state_case(model_state_dict):
    """
    Handle the case where FP8 extra state is present in the model state dict.
    """
    assert HAVE_MEGATRON_FSDP, "This function requires Megatron-FSDP to be installed."

    for key in list(model_state_dict.keys()):
        if key.endswith('._extra_state'):
            del model_state_dict[key]


def flatten_state_dict(obj, parent_key="", sep="."):
    """
    Recursively flattens a nested state dict into a single-level dict with keys
    """
    items = {}
    if isinstance(obj, dict):
        for k, v in obj.items():
            new_key = f"{parent_key}{sep}{k}" if parent_key else str(k)
            items.update(flatten_state_dict(v, new_key, sep=sep))
    elif isinstance(obj, list):
        for i, v in enumerate(obj):
            new_key = f"{parent_key}{sep}{i}" if parent_key else str(i)
            items.update(flatten_state_dict(v, new_key, sep=sep))
    else:
        items[parent_key] = obj
    return items


def print_diff_in_state_dicts(state_dict_metadata, load_state_dict):
    """
    Print the differences between two state dicts: metadata state dict and load state dict.
    This function compares the keys and shapes of the tensors in both dicts.
    """
    state_dict_metadata = flatten_state_dict(state_dict_metadata)
    load_state_dict = flatten_state_dict(load_state_dict)
    meta_keys = set(state_dict_metadata.keys())
    load_keys = set(load_state_dict.keys())

    only_in_meta = meta_keys - load_keys
    only_in_load = load_keys - meta_keys
    in_both = meta_keys & load_keys

    print("Keys only in checkpoint metadata_state_dict:")
    for k in sorted(only_in_meta):
        print(f"  {k}")

    print("\nKeys only in load_state_dict:")
    for k in sorted(only_in_load):
        print(f"  {k}")

    print("\nKeys in both but with different shapes:")
    for k in sorted(in_both):
        v_meta = state_dict_metadata[k]
        v_load = load_state_dict[k]
        # If tensors, compare shape; else, compare type/values
        meta_shape = v_meta.size if hasattr(v_meta, "size") else type(v_meta)
        load_shape = v_load.shape if hasattr(v_load, "shape") else type(v_load)
        if meta_shape != load_shape:
            print(f"  {k}: meta shape={meta_shape}, load shape={load_shape}")
