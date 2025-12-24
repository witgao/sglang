from typing import cast
import torch
import torch.nn.functional as F
import librosa
from tqdm import tqdm
import logging

from sglang.multimodal_gen.runtime.pipelines_core.stages.base import PipelineStage
from sglang.multimodal_gen.runtime.server_args import ServerArgs
from sglang.multimodal_gen.runtime.pipelines_core.schedule_batch import Req

from sglang.multimodal_gen.runtime.models.original import OriginCLIPVisionModelWithProjection, OriginCLIPImageProcessor


class ImageEncodeStage(PipelineStage):

    def __init__(self, image_processor: OriginCLIPImageProcessor, image_encoder: OriginCLIPVisionModelWithProjection):
        super().__init__()
        self.image_processor = image_processor
        self.image_encoder = image_encoder

    def forward(self, batch: Req, server_args: ServerArgs,) -> Req:
        image = batch.condition_image

        device = torch.device(torch.cuda.current_device())
        offload_device = torch.device("cpu")

        self.image_encoder.to(device)

        image = self.image_processor(images=image, return_tensors="pt").to(device)
        image_embeds = self.image_encoder(**image, output_hidden_states=True)
        image_embeds = image_embeds.hidden_states[-2]

        self.image_encoder.to(offload_device)

        batch.image_embeds.append(image_embeds)
        
        return batch
