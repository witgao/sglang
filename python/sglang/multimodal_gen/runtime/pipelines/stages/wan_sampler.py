import torch
import torch.nn.functional as F
from tqdm import tqdm
import logging
from PIL import Image
import numpy as np
import datetime
import os
from sglang.multimodal_gen.runtime.pipelines_core.stages.base import PipelineStage
from sglang.multimodal_gen.runtime.server_args import ServerArgs
from sglang.multimodal_gen.runtime.pipelines_core.schedule_batch import Req

from sglang.multimodal_gen.runtime.models.original import OriginalWanTransformerModel
from sglang.multimodal_gen.runtime.models.original.wanvideo.nodes_sampler import WanVideoSampler

def wan_sampler(model, text_embeds, image_embeds, multitalk_embeds):
    sampler = WanVideoSampler()
    return sampler.process(model=model, image_embeds=image_embeds, shift=11.0, steps=6, cfg=1.0, seed=2, scheduler="dpm++_sde",
                           riflex_freq_index=0, text_embeds=text_embeds, rope_function="comfy", add_noise_to_samples=True, multitalk_embeds=multitalk_embeds)
    
class WanSamplerStage(PipelineStage):

    def __init__(self, transformer: OriginalWanTransformerModel):
        super().__init__()
        self.transformer = transformer

    def forward(self, batch: Req, server_args: ServerArgs,) -> Req:        
        with torch.no_grad():
            text_embeds = {
                "prompt_embeds": batch.prompt_embeds[0],
                "negative_prompt_embeds": batch.negative_prompt_embeds[0],
            }
            image_embeds = batch.image_latent2
            audio_embeds = batch.audio_embeds
            sampler_result = wan_sampler(self.transformer, text_embeds=text_embeds,
                                     image_embeds=image_embeds, multitalk_embeds=audio_embeds)

        batch.samples = sampler_result[0]
        return batch