# SPDX-License-Identifier: Apache-2.0
"""Boogu-Image ti2i (text+image-to-image) composed pipeline."""

import os

from sglang.multimodal_gen.runtime.pipelines_core import LoRAPipeline
from sglang.multimodal_gen.runtime.pipelines_core.composed_pipeline_base import (
    ComposedPipelineBase,
)
from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.boogu_image import (
    BooguImageBeforeDenoisingStage,
    BooguImageDenoisingStage,
)
from sglang.multimodal_gen.runtime.server_args import ServerArgs
from sglang.multimodal_gen.runtime.utils.logging_utils import init_logger

logger = init_logger(__name__)


class BooguImagePipeline(LoRAPipeline, ComposedPipelineBase):
    # Must match the diffusers model_index.json `_class_name`.
    pipeline_name = "BooguImagePipeline"

    # NOTE: "scheduler" is intentionally NOT listed here. Boogu's
    # FlowMatchEulerDiscreteScheduler (with `do_shift`/`seq_len`/`time_shift_version`)
    # shares its class name with SGLang's built-in scheduler, so the framework's
    # name-based ModelRegistry would load the wrong (incompatible) class. We load
    # Boogu's own scheduler directly from the checkpoint in create_pipeline_stages.
    _required_config_modules = [
        "mllm",
        "processor",
        "vae",
        "transformer",
    ]

    def _load_boogu_scheduler(self):
        from boogu.schedulers.scheduling_flow_match_euler_discrete_time_shifting import (
            FlowMatchEulerDiscreteScheduler,
        )

        scheduler_path = os.path.join(self.model_path, "scheduler")
        scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(scheduler_path)
        self.add_module("scheduler", scheduler)
        return scheduler

    def create_pipeline_stages(self, server_args: ServerArgs):
        scheduler = self._load_boogu_scheduler()

        # 1. Model-specific pre-processing (instruction + ref-image encoding,
        #    latent / timestep / rope preparation).
        self.add_stage(
            BooguImageBeforeDenoisingStage(
                mllm=self.get_module("mllm"),
                processor=self.get_module("processor"),
                vae=self.get_module("vae"),
                transformer=self.get_module("transformer"),
                scheduler=scheduler,
            ),
            "boogu_image_before_denoising_stage",
        )

        # 2. Dual/triple-CFG denoising loop (Boogu-specific transformer call).
        self.add_stage(
            BooguImageDenoisingStage(
                transformer=self.get_module("transformer"),
                scheduler=scheduler,
            ),
            "boogu_image_denoising_stage",
        )

        # 3. Standard VAE decoding.
        self.add_standard_decoding_stage()


EntryClass = [BooguImagePipeline]
