# SPDX-License-Identifier: Apache-2.0
"""PipelineConfig for the Boogu-Image ti2i pipeline.

Boogu-Image keeps 4D latents ``(B, C, H', W')`` like GLM-Image, so we inherit
from :class:`SpatialImagePipelineConfig`. Pre-processing (instruction encoding,
reference-image VAE encoding, latent/timestep preparation) and the
dual/triple-CFG denoising loop are owned by the model-specific stages, so the
standard ``DenoisingStage`` callbacks here are intentionally minimal.
"""

from dataclasses import dataclass, field

import torch
from diffusers.image_processor import VaeImageProcessor

from sglang.multimodal_gen.configs.models import DiTConfig, VAEConfig
from sglang.multimodal_gen.configs.models.dits.boogu_image import BooguImageDitConfig
from sglang.multimodal_gen.configs.models.vaes.boogu_image import BooguImageVAEConfig
from sglang.multimodal_gen.configs.pipeline_configs.base import (
    ModelTaskType,
    SpatialImagePipelineConfig,
)


@dataclass
class BooguImagePipelineConfig(SpatialImagePipelineConfig):
    """Configuration for the Boogu-Image pipeline."""

    vae_precision: str = "bf16"

    # Boogu performs its own dual/triple CFG inside the denoising stage, so the
    # framework-level embedded-guidance path is disabled.
    should_use_guidance: bool = False
    task_type: ModelTaskType = ModelTaskType.TI2I

    vae_tiling: bool = False
    vae_sp: bool = False
    enable_autocast: bool = False

    dit_config: DiTConfig = field(default_factory=BooguImageDitConfig)
    vae_config: VAEConfig = field(default_factory=BooguImageVAEConfig)

    def __post_init__(self):
        self.vae_scale_factor = self.vae_config.get_vae_scale_factor()
        self.image_processor = VaeImageProcessor(vae_scale_factor=self.vae_scale_factor)

    def get_decode_scale_and_shift(self, device, dtype, vae):
        # Boogu encodes with z = (z0 - shift) * scaling and therefore decodes
        # with z0 = z / scaling + shift. Read factors from the VAE config.
        scaling_factor = getattr(vae.config, "scaling_factor", None)
        shift_factor = getattr(vae.config, "shift_factor", None)
        return scaling_factor, shift_factor

    def post_decoding(self, frames, server_args):
        # DecodingStage already de-normalizes VAE output to [0, 1]. Return a
        # CHW/THWC-style tensor (not PIL) so the framework's output materializer
        # (_sample_to_uint8_frames) can convert it to uint8 frames.
        return frames
