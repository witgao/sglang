import torch
import torch.nn.functional as F
from tqdm import tqdm
import logging

from sglang.multimodal_gen.runtime.pipelines_core.stages.base import PipelineStage
from sglang.multimodal_gen.runtime.server_args import ServerArgs
from sglang.multimodal_gen.runtime.pipelines_core.schedule_batch import Req

from sglang.multimodal_gen.runtime.models.original import OriginalWanVAEModel

class WanVideoPassImagesFromSamples:

    def decode(self, samples):
        video = samples.get("video", None)
        video.clamp_(-1.0, 1.0)
        video.add_(1.0).div_(2.0)
        return video.cpu().float(), samples.get("output_path", "")
    
class VAEDecoderStage(PipelineStage):

    def __init__(self, vae: OriginalWanVAEModel):
        super().__init__()
        self.vae = vae

    def forward(self, batch: Req, server_args: ServerArgs,) -> Req:
        samples = batch.samples
        decoder = WanVideoPassImagesFromSamples()
        images = decoder.decode(samples=samples)
        batch.images = images[0]
        return batch
