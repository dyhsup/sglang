# SPDX-License-Identifier: Apache-2.0
"""Boogu-Image DiT for SGLang Diffusion.

The Boogu transformer (mixed double-/single-stream architecture with a
Qwen3-VL instruction encoder) already ships as a fully-featured diffusers
``ModelMixin`` in the ``boogu`` package. Re-porting its ~1600 lines of custom
blocks / attention / rope into SGLang would be error-prone and offers no
benefit for a first single-GPU adapter, so this module *wraps* the original
implementation and exposes it through the SGLang :class:`BaseDiT` interface
(``EntryClass`` discovery, ``config``/``hf_config`` construction, the standard
``forward`` contract).

The original Boogu ``forward`` signature is preserved unchanged; the
model-specific denoising stage calls it directly with Boogu's argument names.
"""

from typing import Any

import torch

from sglang.multimodal_gen.configs.models.dits.boogu_image import BooguImageDitConfig
from sglang.multimodal_gen.runtime.models.dits.base import BaseDiT
from sglang.multimodal_gen.runtime.platforms import AttentionBackendEnum
from sglang.multimodal_gen.runtime.utils.logging_utils import init_logger

logger = init_logger(__name__)

# Names of the diffusers ``register_to_config`` kwargs accepted by the original
# Boogu transformer __init__. Anything else in the HF config.json is ignored.
_BOOGU_INIT_KEYS = (
    "patch_size",
    "in_channels",
    "out_channels",
    "hidden_size",
    "num_layers",
    "num_double_stream_layers",
    "num_refiner_layers",
    "num_attention_heads",
    "num_kv_heads",
    "multiple_of",
    "ffn_dim_multiplier",
    "norm_eps",
    "axes_dim_rope",
    "axes_lens",
    "instruction_feature_configs",
    "prompt_tuning_configs",
    "timestep_scale",
)


class BooguImageTransformer2DModel(BaseDiT):
    """SGLang wrapper around ``boogu`` ``BooguImageTransformer2DModel``."""

    _fsdp_shard_conditions: list = []
    _compile_conditions: list = []
    # The real Boogu transformer is nested under ``self.model`` (see __init__),
    # so its parameters are named ``model.<...>`` while the checkpoint stores
    # them as bare ``<...>``. Prepend ``model.`` to every checkpoint key so the
    # FSDP weight loader matches them. ``(?!model\.)`` avoids double-prefixing.
    param_names_mapping: dict = {r"^(?!model\.)(.*)$": r"model.\1"}
    reverse_param_names_mapping: dict = {}
    lora_param_names_mapping: dict = {}

    # Boogu attention runs through the original diffusers attention processors,
    # so advertise only the always-available torch SDPA backend.
    _supported_attention_backends: set[AttentionBackendEnum] = {
        AttentionBackendEnum.TORCH_SDPA,
    }

    def __init__(
        self,
        config: BooguImageDitConfig,
        hf_config: dict[str, Any],
        quant_config: Any | None = None,
    ) -> None:
        super().__init__(config=config, hf_config=hf_config)

        from boogu.models.transformers.transformer_boogu import (
            BooguImageTransformer2DModel as _BooguTransformer,
        )

        arch_config = config.arch_config

        # Prefer values parsed from the checkpoint's config.json (placed into
        # arch_config via update_model_arch); fall back to arch_config defaults.
        def _get(name, default):
            value = hf_config.get(name, None)
            if value is None:
                value = getattr(arch_config, name, default)
            return value

        init_kwargs = {}
        for key in _BOOGU_INIT_KEYS:
            default = getattr(arch_config, key, None)
            init_kwargs[key] = _get(key, default)

        self.model = _BooguTransformer(**init_kwargs)

        # Required BaseDiT instance attributes.
        self.in_channels = init_kwargs["in_channels"]
        self.out_channels = self.model.out_channels
        self.patch_size = init_kwargs["patch_size"]
        self.hidden_size = init_kwargs["hidden_size"]
        self.num_attention_heads = init_kwargs["num_attention_heads"]
        self.num_channels_latents = self.out_channels

        # Expose the inner config so stages can read e.g. axes_dim_rope.
        self.__post_init__()

    @property
    def config(self):
        # Many call-sites (and the denoising stage) read transformer.config.*;
        # delegate to the wrapped diffusers config which carries every field.
        return self.model.config

    @config.setter
    def config(self, value):
        # BaseDiT.__init__ assigns self.config = config; store it but the
        # property getter above always returns the inner diffusers config once
        # the model is built. Keep the SGLang config around for completeness.
        self._sgl_config = value

    def forward(
        self,
        hidden_states,
        timestep,
        instruction_hidden_states,
        freqs_cis,
        instruction_attention_mask,
        ref_image_hidden_states=None,
        attention_kwargs=None,
        return_dict: bool = False,
        **kwargs,
    ):
        return self.model(
            hidden_states=hidden_states,
            timestep=timestep,
            instruction_hidden_states=instruction_hidden_states,
            freqs_cis=freqs_cis,
            instruction_attention_mask=instruction_attention_mask,
            ref_image_hidden_states=ref_image_hidden_states,
            attention_kwargs=attention_kwargs,
            return_dict=return_dict,
        )


EntryClass = BooguImageTransformer2DModel
