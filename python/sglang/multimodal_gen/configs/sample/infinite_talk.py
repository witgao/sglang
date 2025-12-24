from dataclasses import dataclass

from sglang.multimodal_gen.configs.sample.sampling_params import SamplingParams


@dataclass
class InfiniteTalkSamplingParam(SamplingParams):
    guidance_scale: float = 5.0

