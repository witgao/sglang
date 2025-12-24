from dataclasses import dataclass, field
import torch
from collections.abc import Callable
from sglang.multimodal_gen.configs.models import EncoderConfig
from sglang.multimodal_gen.configs.models.encoders import (
    CLIPVisionConfig,
    T5Config,
    BaseEncoderOutput,  
)

from sglang.multimodal_gen.configs.pipeline_configs.base import (
    ModelTaskType,
    PipelineConfig,
)

def t5_postprocess_text(outputs: BaseEncoderOutput, _text_inputs) -> torch.Tensor:
    mask: torch.Tensor = outputs.attention_mask
    hidden_state: torch.Tensor = outputs.last_hidden_state
    seq_lens = mask.gt(0).sum(dim=1).long()
    assert torch.isnan(hidden_state).sum() == 0
    prompt_embeds = [u[:v] for u, v in zip(hidden_state, seq_lens, strict=True)]
    # prompt_embeds_tensor: torch.Tensor = torch.stack(
    #     [
    #         torch.cat([u, u.new_zeros(512 - u.size(0), u.size(1))])
    #         for u in prompt_embeds
    #     ],
    #     dim=0,
    # )
    return prompt_embeds

@dataclass
class InfiniteTalkConfig(PipelineConfig):
    task_type: ModelTaskType = ModelTaskType.I2V
    
    # Text encoding stage
    text_encoder_configs: tuple[EncoderConfig, ...] = field(
        default_factory=lambda: (T5Config(),)
    )
    text_encoder_precisions: tuple[str, ...] = field(default_factory=lambda: ("bf16",))
    postprocess_text_funcs: tuple[Callable[[BaseEncoderOutput], torch.Tensor], ...] = (
        field(default_factory=lambda: (t5_postprocess_text,))
    )

    image_encoder_config: EncoderConfig = field(
        default_factory=CLIPVisionConfig)
    image_encoder_precision: str = "fp32"
    image_encoder_extra_args: dict = field(
        default_factory=lambda: dict(
            output_hidden_states=True,
        )
    )

    def postprocess_image(self, image):
        return image.hidden_states[-2]

