# SPDX-License-Identifier: Apache-2.0
from dataclasses import dataclass

from sglang.multimodal_gen.configs.sample.sampling_params import SamplingParams


@dataclass
class BooguImageSamplingParams(SamplingParams):
    """Sampling params for Boogu-Image ti2i (text+image-to-image) editing.

    Boogu uses a dual/triple classifier-free guidance scheme:
    - ``guidance_scale`` maps to Boogu's ``text_guidance_scale``.
    - ``image_guidance_scale`` is an extra Boogu-specific knob for guiding by
      the reference image.

    For single-sample ti2i the output resolution follows the reference image
    (``align_res=True``) so ``height``/``width`` may be left as ``None``.
    """

    negative_prompt: str = ""
    num_frames: int = 1

    # Denoising stage. text_guidance_scale is exposed as the generic
    # guidance_scale so existing CLI/serving plumbing keeps working.
    guidance_scale: float = 4.0
    num_inference_steps: int = 50

    # Boogu-specific guidance / resolution controls.
    image_guidance_scale: float = 1.0
    # TeaCache 阈值（仅当请求 enable_teacache=True 时生效）。0.05 画质近无损。
    teacache_rel_l1_thresh: float = 0.05
    # CFG gating：去噪轨迹后段复用已缓存的 guidance delta，减少额外 CFG 分支前向。
    # 取值为 [0,1] 的比例，例如 0.5 表示后 50% 步复用缓存 delta。1.0 = 关闭。
    cfg_gate_step: float = 1.0
    max_input_image_pixels: int = 2048 * 2048
    max_input_image_side_length: int = 2048 * 2
    max_vlm_input_pil_pixels: int = 384 * 384
    max_vlm_input_pil_side_length: int = 384 * 2
    max_sequence_length: int = 1280
    align_res: bool = True
