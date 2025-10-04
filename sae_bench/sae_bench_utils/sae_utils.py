import re
from dataclasses import dataclass
from typing import Any

from sae_lens import SAE


@dataclass
class NormalizedSAEConfig:
    architecture: str
    d_sae: int
    d_in: int
    dtype: str
    apply_b_dec_to_input: bool
    hook_layer: int
    hook_name: str
    context_size: int | None
    dataset_trust_remote_code: bool
    model_name: str
    hook_head_index: int | None
    model_from_pretrained_kwargs: dict
    prepend_bos: bool


def get_cfg_meta_field(cfg: Any, field: str) -> Any | None:
    # SAELens v6 moves some cfg properties to `cfg.metadata` that were previously on the cfg object
    if hasattr(cfg, field):
        return getattr(cfg, field)
    if hasattr(cfg, "metadata") and hasattr(cfg.metadata, field):
        return getattr(cfg.metadata, field)
    return None


def norm_cfg(sae: SAE) -> NormalizedSAEConfig:
    """
    Handle differences in SAE cfg between SAELens v5 and v6 and SAEbench / dictionary learning.
    """
    hook_name = get_cfg_meta_field(sae.cfg, "hook_name")
    hook_layer = get_cfg_meta_field(sae.cfg, "hook_layer")
    if hook_name is not None and hook_layer is None:
        match = re.search(r"\d+", str(hook_name))
        if match:
            hook_layer = int(match.group(0))
    context_size = get_cfg_meta_field(sae.cfg, "context_size")
    dataset_trust_remote_code = get_cfg_meta_field(sae.cfg, "dataset_trust_remote_code")
    model_name = get_cfg_meta_field(sae.cfg, "model_name")
    hook_head_index = get_cfg_meta_field(sae.cfg, "hook_head_index")
    model_from_pretrained_kwargs = get_cfg_meta_field(
        sae.cfg, "model_from_pretrained_kwargs"
    )
    prepend_bos = get_cfg_meta_field(sae.cfg, "prepend_bos")
    if prepend_bos is None:
        prepend_bos = True
    if hook_layer is None:
        raise ValueError("Cound not determine Hook layer from SAE cfg")
    if hook_name is None:
        raise ValueError("Cound not determine Hook name from SAE cfg")
    if model_name is None:
        raise ValueError("Cound not determine Model name from SAE cfg")

    architecture = sae.cfg.architecture
    if callable(sae.cfg.architecture):
        architecture = sae.cfg.architecture()
    return NormalizedSAEConfig(
        architecture=architecture or "standard",  # type: ignore
        d_sae=sae.cfg.d_sae,
        d_in=sae.cfg.d_in,
        dtype=str(sae.cfg.dtype),
        apply_b_dec_to_input=sae.cfg.apply_b_dec_to_input or False,
        hook_layer=hook_layer,
        hook_name=hook_name,
        context_size=context_size,
        dataset_trust_remote_code=dataset_trust_remote_code or True,
        model_name=model_name,
        hook_head_index=hook_head_index,
        model_from_pretrained_kwargs=model_from_pretrained_kwargs or {},
        prepend_bos=prepend_bos,
    )
