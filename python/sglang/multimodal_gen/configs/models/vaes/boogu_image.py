# SPDX-License-Identifier: Apache-2.0
"""Boogu-Image VAE config.

Boogu-Image uses a standard diffusers ``AutoencoderKL``. The real
``block_out_channels`` / ``scaling_factor`` / ``shift_factor`` values are read
from the checkpoint's ``vae/config.json`` via ``update_model_arch`` at load
time; the defaults below mirror a typical SD3/Flux-style 8x VAE so that
``vae_scale_factor`` is well defined before weights are loaded.
"""

from dataclasses import dataclass, field

from sglang.multimodal_gen.configs.models.vaes.base import VAEArchConfig, VAEConfig


@dataclass
class BooguImageVAEArchConfig(VAEArchConfig):
    spatial_compression_ratio: int = 8

    in_channels: int = 3
    out_channels: int = 3
    latent_channels: int = 16
    block_out_channels: tuple[int, ...] = (128, 256, 512, 512)

    scaling_factor: float = 1.0
    shift_factor: float | None = None


@dataclass
class BooguImageVAEConfig(VAEConfig):
    arch_config: BooguImageVAEArchConfig = field(
        default_factory=BooguImageVAEArchConfig
    )

    use_tiling: bool = False
    use_temporal_tiling: bool = False
    use_parallel_tiling: bool = False

    def __post_init__(self):
        self.blend_num_frames = (
            self.tile_sample_min_num_frames - self.tile_sample_stride_num_frames
        ) * 2

    def post_init(self):
        self.arch_config.vae_scale_factor = 2 ** (
            len(self.arch_config.block_out_channels) - 1
        )
        self.arch_config.spatial_compression_ratio = self.arch_config.vae_scale_factor
