import functools
from functools import partial

import torch
from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
    CheckpointImpl,
    apply_activation_checkpointing,
    checkpoint_wrapper,
)
from torch.distributed.fsdp import (
    CPUOffload,
    FullyShardedDataParallel as FSDP,
    MixedPrecision,
    ShardingStrategy,
)
from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy


non_reentrant_wrapper = partial(
    checkpoint_wrapper,
    checkpoint_impl=CheckpointImpl.NO_REENTRANT,
)


def cast_trainable_params_to_dtype(model, target_dtype=None):
    if target_dtype is None:
        for param in model.parameters():
            if param.is_floating_point():
                target_dtype = param.dtype
                break
    if target_dtype is None:
        return None

    for param in model.parameters():
        if param.requires_grad and param.is_floating_point() and param.dtype != target_dtype:
            param.data = param.data.to(dtype=target_dtype)
    return target_dtype


def validate_uniform_param_dtype(model):
    dtypes = {param.dtype for param in model.parameters() if param.is_floating_point()}
    if len(dtypes) > 1:
        raise ValueError(
            f"FSDP requires uniform floating param dtype per wrapped module, got: {sorted(str(dtype) for dtype in dtypes)}"
        )
    return next(iter(dtypes)) if dtypes else None


def prepare_model_for_fsdp_training(
    model,
    *,
    device,
    no_split_modules=(),
    enable_gradient_checkpointing=False,
    selective_checkpointing=1.0,
    sharding_strategy=None,
    use_lora=False,
    cpu_offload=False,
    master_weight_type="fp32",
    world_size=1,
):
    model = model.to(device)
    cast_trainable_params_to_dtype(model)
    validate_uniform_param_dtype(model)

    if enable_gradient_checkpointing:
        apply_fsdp_checkpointing(
            model,
            no_split_modules,
            selective_checkpointing or 1.0,
        )

    if world_size > 1:
        fsdp_kwargs = get_fsdp_kwargs(
            model,
            sharding_strategy=sharding_strategy or "full",
            use_lora=use_lora,
            cpu_offload=cpu_offload,
            master_weight_type=master_weight_type,
            no_split_modules=no_split_modules,
        )
        model = wrap_model_with_fsdp(model, **fsdp_kwargs)

    return model


def apply_fsdp_checkpointing(model, no_split_modules, p=1):
    block_idx = 0
    cut_off = 1 / 2
    p = eval(p) if isinstance(p, str) else p

    def selective_checkpointing(submodule):
        nonlocal block_idx
        nonlocal cut_off

        if isinstance(submodule, no_split_modules):
            block_idx += 1
            if block_idx * p >= cut_off:
                cut_off += 1
                return True
        return False

    apply_activation_checkpointing(
        model,
        checkpoint_wrapper_fn=non_reentrant_wrapper,
        check_fn=selective_checkpointing,
    )


def get_mixed_precision(master_weight_type="fp32"):
    weight_type = torch.float32 if master_weight_type == "fp32" else torch.bfloat16
    return MixedPrecision(
        param_dtype=weight_type,
        reduce_dtype=weight_type,
        buffer_dtype=weight_type,
        cast_forward_inputs=False,
    )


def get_fsdp_kwargs(
    model,
    sharding_strategy,
    use_lora=False,
    cpu_offload=False,
    master_weight_type="fp32",
    no_split_modules=(),
):
    auto_wrap_policy = functools.partial(
        transformer_auto_wrap_policy,
        transformer_layer_cls=no_split_modules,
    ) if no_split_modules else None

    mixed_precision = get_mixed_precision(master_weight_type)

    if sharding_strategy == "full":
        sharding_strategy = ShardingStrategy.FULL_SHARD
    elif sharding_strategy == "hybrid_full":
        sharding_strategy = ShardingStrategy.HYBRID_SHARD
    elif sharding_strategy == "none":
        sharding_strategy = ShardingStrategy.NO_SHARD
        auto_wrap_policy = None
    elif sharding_strategy == "hybrid_zero2":
        sharding_strategy = ShardingStrategy._HYBRID_SHARD_ZERO2
    else:
        sharding_strategy = ShardingStrategy.FULL_SHARD

    device_id = torch.cuda.current_device()
    cpu_offload = CPUOffload(offload_params=True) if cpu_offload else None
    fsdp_kwargs = {
        "auto_wrap_policy": auto_wrap_policy,
        "mixed_precision": mixed_precision,
        "sharding_strategy": sharding_strategy,
        "device_id": device_id,
        "limit_all_gathers": True,
        "cpu_offload": cpu_offload,
    }
    if use_lora:
        fsdp_kwargs["use_orig_params"] = True
    return fsdp_kwargs


def wrap_model_with_fsdp(model, **fsdp_kwargs):
    return FSDP(model, **fsdp_kwargs)
