from typing import cast
from sglang.multimodal_gen.runtime.pipelines_core.stages.base import PipelineStage
from sglang.multimodal_gen.runtime.server_args import ServerArgs
from sglang.multimodal_gen.runtime.pipelines_core.schedule_batch import Req


from sglang.multimodal_gen.runtime.utils.image_utils import ImageUtils
from sglang.multimodal_gen.runtime.utils.audio_utils import AudioUtils


class InputStage(PipelineStage):

    def __init__(self):
        super().__init__()

    def forward(self, batch: Req, server_args: ServerArgs,) -> Req:
        image_path = batch.image_path
        if image_path:
            image, mask = ImageUtils.load(image_path)
            batch.condition_image = image

            if batch.width != None and batch.height != None:
                image, width, height, mask = ImageUtils.resize(image=image, width=batch.width, height=batch.height, keep_proportion="pad",
                                                               upscale_method="lanczos", divisible_by=0, pad_color="0, 0, 0",
                                                               crop_position="center", device="cpu")
                batch.condition_image = image
                batch.width = width
                batch.height = height

        audio_path = batch.audio_path
        if audio_path:
            audio, sample_rate = AudioUtils.load(audio_path)
            batch.condition_audio = audio
            batch.original_condition_audio_sample_rate = sample_rate

        return batch
