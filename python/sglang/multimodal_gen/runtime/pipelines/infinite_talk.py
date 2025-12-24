from sglang.multimodal_gen.runtime.pipelines_core.composed_pipeline_base import (
    ComposedPipelineBase,
)
from sglang.multimodal_gen.runtime.pipelines_core.lora_pipeline import LoRAPipeline
from sglang.multimodal_gen.runtime.server_args import ServerArgs

from sglang.multimodal_gen.runtime.pipelines_core.stages import (
    ImageEncodingStage,
    ConditioningStage,
    DecodingStage,
    DenoisingStage,
    ImageVAEEncodingStage,
    InputValidationStage,
    LatentPreparationStage,
    TextEncodingStage,
    TimestepPreparationStage,
)

from sglang.multimodal_gen.runtime.pipelines.stages import (
    InputStage,
    OutputStage,
    AudioEncodeStage,
    ImageEncodeStage,
    TextEncodeStage,
    VAEEncoderStage,
    VAEDecoderStage,
    WanSamplerStage
)


class InfiniteTalkPipeline(LoRAPipeline, ComposedPipelineBase):
    pipeline_name = "InfiniteTalkPipeline"

    _required_config_modules = [
        "audio_processor",
        "audio_encoder",
        "image_processor",
        "image_encoder",
        "text_encoder",
        "tokenizer",
        "vae",
        "transformer"
    ]

    def initialize_pipeline(self, server_args: ServerArgs):
        pass

    def create_pipeline_stages(self, server_args: ServerArgs):
        self.add_stage(
            stage_name="InputStage",
            stage=InputStage()
        )
        self.add_stage(
            stage_name="AudioEncodeStage",
            stage=AudioEncodeStage(audio_processor=self.get_module("audio_processor"),
                                   audio_encoder=self.get_module("audio_encoder")),
        )
        self.add_stage(
            stage_name="image_encoding_stage",
            stage=ImageEncodingStage(
                image_encoder=self.get_module("image_encoder"),
                image_processor=self.get_module("image_processor"),
            ),
        )
        # self.add_stage(
        #     stage_name="ImageEncodeStage",
        #     stage=ImageEncodeStage(image_processor=self.get_module("image_processor"),
        #                            image_encoder=self.get_module("image_encoder")),
        # )
        self.add_stage(
            stage_name="prompt_encoding_stage",
            stage=TextEncodingStage(
                text_encoders=[self.get_module("text_encoder")],
                tokenizers=[self.get_module("tokenizer")],
            ),
        )
        # self.add_stage(
        #     stage_name="TextEncodeStage",
        #     stage=TextEncodeStage(
        #         text_encoder=self.get_module("text_encoder")),
        # )
        self.add_stage(
            stage_name="VAEEncoderStage",
            stage=VAEEncoderStage(vae=self.get_module("vae")),
        )
        self.add_stage(
            stage_name="WanSamplerStage",
            stage=WanSamplerStage(transformer=self.get_module("transformer")),
        )
        self.add_stage(
            stage_name="VAEDecoderStage",
            stage=VAEDecoderStage(vae=self.get_module("vae")),
        )
        self.add_stage(
            stage_name="OutputStage",
            stage=OutputStage(),
        )


EntryClass = InfiniteTalkPipeline
