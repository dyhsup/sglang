# SPDX-License-Identifier: Apache-2.0
"""Model-specific stages for the Boogu-Image ti2i (text+image-to-image) pipeline.

Boogu-Image does not fit the standard t2i/ti2i stage templates:

* Its "text encoder" is a Qwen3-VL MLLM that jointly encodes the instruction
  text *and* the reference image(s) into ``instruction_hidden_states`` plus an
  ``instruction_attention_mask``.
* Its DiT takes the reference-image VAE latents as a separate
  ``ref_image_hidden_states`` argument (concatenated on the *sequence* dim
  inside the transformer), not channel-concatenated as the standard
  ``DenoisingStage`` expects.
* It uses a dual/triple classifier-free guidance scheme combining a text guidance
  scale and an image guidance scale.

Following the GLM-Image precedent we therefore use the Hybrid style: a single
``BooguImageBeforeDenoisingStage`` performs all pre-processing, and a custom
``BooguImageDenoisingStage`` runs the dual/triple-CFG denoising loop. VAE
decoding is handled by the framework-standard ``DecodingStage``.
"""

import inspect
from typing import List, Optional, Union

import PIL
import torch
from diffusers.utils.torch_utils import randn_tensor

from sglang.multimodal_gen.runtime.distributed import get_local_torch_device
from sglang.multimodal_gen.runtime.managers.forward_context import set_forward_context
from sglang.multimodal_gen.runtime.managers.memory_managers.component_manager import (
    ComponentUse,
)
from sglang.multimodal_gen.runtime.models.vision_utils import load_image
from sglang.multimodal_gen.runtime.pipelines_core.schedule_batch import Req
from sglang.multimodal_gen.runtime.pipelines_core.stages.base import PipelineStage
from sglang.multimodal_gen.runtime.server_args import ServerArgs
from sglang.multimodal_gen.runtime.utils.logging_utils import init_logger
from sglang.multimodal_gen.runtime.utils.precision import get_module_dtype

logger = init_logger(__name__)


def _image_path_to_list(image_path: Union[str, List[str]]) -> List[str]:
    return image_path if isinstance(image_path, list) else [image_path]


def retrieve_timesteps(
    scheduler,
    num_inference_steps: Optional[int] = None,
    device=None,
    timesteps: Optional[List[int]] = None,
    **kwargs,
):
    """Set scheduler timesteps, passing through extra kwargs the scheduler accepts."""
    accepted = set(inspect.signature(scheduler.set_timesteps).parameters.keys())
    extra = {k: v for k, v in kwargs.items() if k in accepted}
    if timesteps is not None:
        scheduler.set_timesteps(timesteps=timesteps, device=device, **extra)
    else:
        scheduler.set_timesteps(num_inference_steps, device=device, **extra)
    timesteps = scheduler.timesteps
    return timesteps, len(timesteps)


class BooguImageBeforeDenoisingStage(PipelineStage):
    """All Boogu pre-processing: instruction encoding, ref-image VAE encoding,
    latent / timestep / rope preparation."""

    def __init__(self, mllm, processor, vae, transformer, scheduler) -> None:
        super().__init__()
        self.mllm = mllm
        self.processor = processor
        self.vae = vae
        self.transformer = transformer
        self.scheduler = scheduler

        self.vae_scale_factor = (
            2 ** (len(self.vae.config.block_out_channels) - 1)
            if getattr(self, "vae", None) is not None
            else 8
        )

        # System prompt matching the Boogu reference pipeline (ti2i editing).
        self.SYSTEM_PROMPT_4_TI2I = (
            "Describe the key features of the input image (color, shape, size, "
            "texture, objects, background), then explain how the user's text "
            "instruction should alter or modify the image. Generate a new image "
            "that meets the user's requirements while maintaining consistency "
            "with the original input where appropriate."
        )

    def component_uses(
        self, server_args: ServerArgs, stage_name: str | None = None
    ) -> list[ComponentUse]:
        stage_name = self._component_stage_name(stage_name)
        uses: list[ComponentUse] = []
        if self.mllm is not None:
            uses.append(ComponentUse(stage_name, "mllm"))
        if self.vae is not None:
            uses.append(ComponentUse(stage_name, "vae"))
        return uses

    # --- instruction (text + image) encoding via Qwen3-VL --------------------

    def _apply_chat_template(self, instruction, input_pil_images):
        system_role = {
            "role": "system",
            "content": [{"type": "text", "text": self.SYSTEM_PROMPT_4_TI2I}],
        }
        user_text_content = [{"type": "text", "text": instruction}]
        if input_pil_images:
            images_content = [
                {"type": "image", "image": img} for img in input_pil_images
            ]
            user = {"role": "user", "content": images_content + user_text_content}
        else:
            user = {"role": "user", "content": user_text_content}
        return [system_role, user]

    @torch.no_grad()
    def _encode_instruction(
        self,
        instruction: str,
        input_pil_images: Optional[List[PIL.Image.Image]],
        device,
        max_sequence_length: int,
    ):
        prompts = [self._apply_chat_template(instruction, input_pil_images)]
        vlm_inputs = self.processor.apply_chat_template(
            prompts,
            padding="longest",
            max_length=max_sequence_length,
            truncation=False,
            padding_side="right",
            return_tensors="pt",
            tokenize=True,
            return_dict=True,
        )
        for k in vlm_inputs:
            if isinstance(vlm_inputs[k], torch.Tensor):
                vlm_inputs[k] = vlm_inputs[k].to(device)

        instruction_mask = vlm_inputs["attention_mask"]
        feats = self.mllm(**vlm_inputs, output_hidden_states=False).last_hidden_state

        dtype = get_module_dtype(self.mllm, torch.bfloat16)
        feats = feats.to(dtype=dtype, device=device)
        instruction_mask = instruction_mask.to(device=device)
        return feats, instruction_mask

    # --- reference image VAE encoding ----------------------------------------

    def _encode_vae(self, img: torch.Tensor) -> torch.Tensor:
        vae_dtype = get_module_dtype(self.vae, torch.float32)
        z0 = self.vae.encode(img.to(dtype=vae_dtype)).latent_dist.sample()
        if self.vae.config.shift_factor is not None:
            z0 = z0 - self.vae.config.shift_factor
        if self.vae.config.scaling_factor is not None:
            z0 = z0 * self.vae.config.scaling_factor
        return z0.to(dtype=vae_dtype)

    @torch.no_grad()
    def _prepare_ref_latents(self, images, device):
        """Encode each reference image to a [C, H, W] latent (one per sample)."""
        from diffusers.image_processor import VaeImageProcessor

        image_processor = VaeImageProcessor(
            vae_scale_factor=self.vae_scale_factor * 2, do_resize=True
        )
        ref_latents = []
        for img in images:
            img = img.convert("RGB")
            pixel = image_processor.preprocess(img).to(device=device)
            ref_latents.append(self._encode_vae(pixel).squeeze(0))
        return ref_latents

    def _prepare_latents(
        self, batch_size, num_channels_latents, height, width, dtype, device, generator
    ):
        height = int(height) // self.vae_scale_factor
        width = int(width) // self.vae_scale_factor
        shape = (batch_size, num_channels_latents, height, width)
        return randn_tensor(shape, generator=generator, device=device, dtype=dtype)

    @torch.no_grad()
    def forward(self, batch: Req, server_args: ServerArgs) -> Req:
        device = get_local_torch_device()
        pipeline_config = server_args.pipeline_config
        sampling = batch.sampling_params

        instruction = batch.prompt
        negative_instruction = batch.negative_prompt or ""
        max_sequence_length = getattr(sampling, "max_sequence_length", 1280)
        align_res = getattr(sampling, "align_res", True)

        assert batch.image_path is not None, "Boogu-Image ti2i requires image_path"
        input_pil_images = [
            load_image(p).convert("RGB")
            for p in _image_path_to_list(batch.image_path)
        ]

        # 1. Encode reference image(s) into VAE latents (one per sample image).
        with self.use_declared_component(component_name="vae", module=self.vae) as vae:
            assert vae is not None
            self.vae = vae
            ref_latents_list = self._prepare_ref_latents(input_pil_images, device)
        # Boogu's transformer expects ref_image_hidden_states as List[List[tensor]]
        # (outer=batch, inner=images per sample).
        ref_latents = [ref_latents_list]

        # 2. Encode instruction (+ images) and negative instruction via Qwen3-VL.
        with self.use_declared_component(component_name="mllm", module=self.mllm) as mllm:
            assert mllm is not None
            self.mllm = mllm
            instruction_embeds, instruction_attention_mask = self._encode_instruction(
                instruction, input_pil_images, device, max_sequence_length
            )
            negative_instruction_embeds, negative_instruction_attention_mask = (
                self._encode_instruction(
                    negative_instruction, input_pil_images, device, max_sequence_length
                )
            )

        # 3. Resolve output resolution. For single-sample ti2i with align_res the
        #    output follows the reference image latent size.
        if align_res and len(ref_latents_list) >= 1:
            ref0 = ref_latents_list[0]
            height = ref0.shape[-2] * self.vae_scale_factor
            width = ref0.shape[-1] * self.vae_scale_factor
        else:
            height = batch.height
            width = batch.width
        batch.height = height
        batch.width = width

        # 4. Initial noise latents.
        dtype = get_module_dtype(self.transformer, torch.bfloat16)
        generator = torch.Generator(device=device).manual_seed(int(batch.seed))
        latent_channels = self.transformer.config.in_channels
        latents = self._prepare_latents(
            1, latent_channels, height, width, instruction_embeds.dtype, device, generator
        )

        # 5. Rotary embeddings (precomputed freqs_cis tables).
        from boogu.models.transformers.rope import BooguImageRotaryPosEmbed

        freqs_cis = BooguImageRotaryPosEmbed.get_freqs_cis(
            self.transformer.config.axes_dim_rope,
            self.transformer.config.axes_lens,
            theta=10000,
        )

        # 6. Timesteps (flow-match scheduler, token-count aware shift).
        timesteps, num_inference_steps = retrieve_timesteps(
            self.scheduler,
            batch.num_inference_steps,
            device,
            num_tokens=latents.shape[-2] * latents.shape[-1],
        )

        # 7. Populate batch for the denoising stage.
        batch.prompt_embeds = [instruction_embeds]
        batch.negative_prompt_embeds = [negative_instruction_embeds]
        batch.latents = latents
        batch.image_latent = ref_latents
        batch.timesteps = timesteps
        batch.scheduler = self.scheduler
        batch.num_inference_steps = num_inference_steps
        batch.sigmas = None
        batch.generator = generator
        batch.raw_latent_shape = latents.shape

        # Boogu-specific tensors carried via the generic extra dict.
        batch.extra["boogu_freqs_cis"] = freqs_cis
        batch.extra["boogu_instruction_attention_mask"] = instruction_attention_mask
        batch.extra["boogu_negative_instruction_attention_mask"] = (
            negative_instruction_attention_mask
        )
        batch.extra["boogu_text_guidance_scale"] = float(batch.guidance_scale)
        batch.extra["boogu_image_guidance_scale"] = float(
            getattr(sampling, "image_guidance_scale", 1.0)
        )
        batch.extra["boogu_enable_teacache"] = bool(
            getattr(batch, "enable_teacache", False)
            or getattr(sampling, "enable_teacache", False)
        )
        batch.extra["boogu_teacache_rel_l1_thresh"] = float(
            getattr(sampling, "teacache_rel_l1_thresh", 0.05) or 0.05
        )
        batch.extra["boogu_cfg_gate_step"] = float(
            getattr(sampling, "cfg_gate_step", 1.0) or 1.0
        )
        return batch


class BooguImageDenoisingStage(PipelineStage):
    """Dual/triple classifier-free-guidance denoising loop for Boogu-Image.

    Optionally enables Boogu's native TeaCache (opt-in via ``enable_teacache``
    on the request; threshold ``teacache_rel_l1_thresh`` default 0.05). Each CFG
    condition branch (full / drop-text / drop-all / drop-image) keeps an
    isolated TeaCache state to avoid cross-branch cache pollution, mirroring the
    original Boogu ``processing`` loop.
    """

    def __init__(self, transformer, scheduler) -> None:
        super().__init__()
        self.transformer = transformer
        self.scheduler = scheduler

    def component_uses(
        self, server_args: ServerArgs, stage_name: str | None = None
    ) -> list[ComponentUse]:
        stage_name = self._component_stage_name(stage_name)
        return [ComponentUse(stage_name, "transformer", memory_intensive=True)]

    def _predict(
        self,
        latents,
        t,
        instruction_embeds,
        freqs_cis,
        instruction_attention_mask,
        ref_image_hidden_states,
    ):
        timestep = t.expand(latents.shape[0]).to(latents.dtype)
        return self.transformer(
            hidden_states=latents,
            timestep=timestep,
            instruction_hidden_states=instruction_embeds,
            freqs_cis=freqs_cis,
            instruction_attention_mask=instruction_attention_mask,
            ref_image_hidden_states=ref_image_hidden_states,
        )

    @torch.no_grad()
    def forward(self, batch: Req, server_args: ServerArgs) -> Req:
        latents = batch.latents
        ref_latents = batch.image_latent
        instruction_embeds = batch.prompt_embeds[0]
        negative_instruction_embeds = batch.negative_prompt_embeds[0]
        freqs_cis = batch.extra["boogu_freqs_cis"]
        instruction_attention_mask = batch.extra["boogu_instruction_attention_mask"]
        negative_instruction_attention_mask = batch.extra[
            "boogu_negative_instruction_attention_mask"
        ]
        text_guidance_scale = batch.extra["boogu_text_guidance_scale"]
        image_guidance_scale = batch.extra["boogu_image_guidance_scale"]

        timesteps = batch.timesteps
        dtype = latents.dtype
        n_steps = len(timesteps)

        # --- CFG gating (opt-in via request `cfg_gate_step`) ------------------
        # After a fraction of the denoising trajectory the guidance directions
        # (delta_text / delta_image) change slowly, so we stop recomputing the
        # extra CFG branches and reuse the last-computed deltas. This is the
        # ERNIE/DenoisingStage-style residual-reuse trick applied to Boogu's
        # dual/triple CFG. cfg_gate_step is a fraction in [0, 1]; e.g. 0.5 means
        # the last 50% of steps reuse cached deltas. Default 1.0 = disabled.
        cfg_gate_frac = float(batch.extra.get("boogu_cfg_gate_step", 1.0) or 1.0)
        cfg_gate_frac = min(max(cfg_gate_frac, 0.0), 1.0)
        gate_start = int(round(cfg_gate_frac * n_steps))
        cached_delta_text = None
        cached_delta_image = None

        # --- TeaCache setup (opt-in via request `enable_teacache`) ------------
        from boogu.utils.teacache_util import TeaCacheParams

        enable_teacache = bool(batch.extra.get("boogu_enable_teacache", False))
        teacache_thresh = float(batch.extra.get("boogu_teacache_rel_l1_thresh", 0.05))
        # Per-CFG-branch isolated TeaCache states so residual caches never mix.
        tc_cond = TeaCacheParams()
        tc_drop_text = TeaCacheParams()
        tc_drop_all = TeaCacheParams()
        tc_drop_image = TeaCacheParams()

        def _predict_cached(
            step_i, tc_params, latents_, t, embeds, mask, ref,
        ):
            """Run one transformer prediction, optionally driving TeaCache for
            this specific CFG branch."""
            if enable_teacache:
                self.transformer.enable_teacache = True
                self.transformer.teacache_rel_l1_thresh = teacache_thresh
                tc_params.is_first_or_last_step = (
                    step_i == 0 or step_i == n_steps - 1
                )
                self.transformer.teacache_params = tc_params
            else:
                self.transformer.enable_teacache = False
            return self._predict(latents_, t, embeds, freqs_cis, mask, ref)

        with self.use_declared_component(
            component_name="transformer", module=self.transformer
        ) as transformer:
            assert transformer is not None
            self.transformer = transformer

            with self.progress_bar(total=n_steps, batch=batch) as progress_bar:
                for i, t in enumerate(timesteps):
                    # CFG gating: reuse cached guidance deltas for the tail of
                    # the trajectory. The first/last step always recompute.
                    reuse = (
                        gate_start < n_steps
                        and i >= gate_start
                        and i != n_steps - 1
                        and cached_delta_text is not None
                    )
                    with set_forward_context(current_timestep=t, attn_metadata=None):
                        model_pred = _predict_cached(
                            i, tc_cond, latents, t,
                            instruction_embeds, instruction_attention_mask,
                            ref_latents,
                        )

                        if text_guidance_scale > 1.0 and image_guidance_scale > 1.0:
                            # Triple CFG: full, drop-text (keep ref), drop-all.
                            if reuse:
                                delta_text = cached_delta_text
                                delta_image = cached_delta_image
                            else:
                                model_pred_drop_text = _predict_cached(
                                    i, tc_drop_text, latents, t,
                                    negative_instruction_embeds,
                                    negative_instruction_attention_mask,
                                    ref_latents,
                                )
                                model_pred_drop_all = _predict_cached(
                                    i, tc_drop_all, latents, t,
                                    negative_instruction_embeds,
                                    negative_instruction_attention_mask,
                                    None,
                                )
                                delta_text = model_pred - model_pred_drop_text
                                delta_image = model_pred_drop_text - model_pred_drop_all
                                cached_delta_text = delta_text
                                cached_delta_image = delta_image
                            model_pred = (
                                model_pred
                                + (text_guidance_scale - 1) * delta_text
                                + (image_guidance_scale - 1) * delta_image
                            )
                        elif text_guidance_scale > 1.0:
                            # Text-only guidance, keep reference condition.
                            if reuse:
                                delta_text = cached_delta_text
                            else:
                                model_pred_drop_text = _predict_cached(
                                    i, tc_drop_text, latents, t,
                                    negative_instruction_embeds,
                                    negative_instruction_attention_mask,
                                    ref_latents,
                                )
                                delta_text = model_pred - model_pred_drop_text
                                cached_delta_text = delta_text
                            model_pred = (
                                model_pred + (text_guidance_scale - 1) * delta_text
                            )
                        elif image_guidance_scale > 1.0:
                            # Image-only guidance, drop reference condition.
                            if reuse:
                                delta_image = cached_delta_image
                            else:
                                model_pred_drop_image = _predict_cached(
                                    i, tc_drop_image, latents, t,
                                    instruction_embeds, instruction_attention_mask,
                                    None,
                                )
                                delta_image = model_pred - model_pred_drop_image
                                cached_delta_image = delta_image
                            model_pred = (
                                model_pred + (image_guidance_scale - 1) * delta_image
                            )

                    latents = self.scheduler.step(
                        model_pred, t, latents, return_dict=False
                    )[0]
                    latents = latents.to(dtype=dtype)
                    progress_bar.update()

            # Reset transformer cache flag so it doesn't leak to next request.
            self.transformer.enable_teacache = False

        batch.latents = latents
        return batch
