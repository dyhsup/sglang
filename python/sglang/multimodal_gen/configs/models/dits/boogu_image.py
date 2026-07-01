# SPDX-License-Identifier: Apache-2.0
"""Boogu-Image DiT (transformer) architecture config.

Mirrors the ``register_to_config`` defaults of
``boogu.models.transformers.transformer_boogu.BooguImageTransformer2DModel``.
"""

from dataclasses import dataclass, field
from typing import Any

from sglang.multimodal_gen.configs.models.dits.base import DiTArchConfig, DiTConfig


def _default_instruction_feature_configs() -> dict[str, Any]:
    return dict(
        instruction_feat_dim=1024,
        reduce_type="mean",
        num_instruction_feat_layers=1,
    )


def _default_prompt_tuning_configs() -> dict[str, Any]:
    return dict(use_prompt_tuning=False)


@dataclass
class BooguImageArchConfig(DiTArchConfig):
    patch_size: int = 2
    in_channels: int = 16
    out_channels: int | None = None
    hidden_size: int = 2304
    num_layers: int = 26
    num_double_stream_layers: int = 2
    num_refiner_layers: int = 2
    num_attention_heads: int = 24
    num_kv_heads: int = 8
    multiple_of: int = 256
    ffn_dim_multiplier: float | None = None
    norm_eps: float = 1e-5
    axes_dim_rope: tuple[int, int, int] = (40, 40, 40)
    axes_lens: tuple[int, int, int] = (2048, 1664, 1664)
    instruction_feature_configs: dict[str, Any] = field(
        default_factory=_default_instruction_feature_configs
    )
    prompt_tuning_configs: dict[str, Any] = field(
        default_factory=_default_prompt_tuning_configs
    )
    timestep_scale: float = 1.0

    def __post_init__(self):
        super().__post_init__()
        self.out_channels = self.out_channels or self.in_channels
        self.num_channels_latents = self.out_channels


@dataclass
class BooguImageDitConfig(DiTConfig):
    arch_config: DiTArchConfig = field(default_factory=BooguImageArchConfig)

    prefix: str = "boogu_image"
