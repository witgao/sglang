from dataclasses import dataclass, field
import torch
from collections.abc import Callable
from sglang.multimodal_gen.configs.models import EncoderConfig
from sglang.multimodal_gen.configs.models.encoders import (
    CLIPVisionConfig,
    T5Config,
    BaseEncoderOutput,  
)
from sglang.multimodal_gen.configs.models import DiTConfig, EncoderConfig, VAEConfig
from sglang.multimodal_gen.configs.models.vaes import WanVAEConfig
from sglang.multimodal_gen.configs.models.dits import WanVideoConfig

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
    return prompt_embeds

@dataclass
class InfiniteTalkConfig(PipelineConfig):
    task_type: ModelTaskType = ModelTaskType.I2V
    
    # DiT
    dit_config: DiTConfig = field(default_factory=WanVideoConfig)
    
    # VAE
    vae_config: VAEConfig = field(default_factory=WanVAEConfig)
    vae_tiling: bool = False
    vae_sp: bool = False
    
    # Denoising stage
    flow_shift: float | None = 7.0

    # Text encoding stage
    text_encoder_configs: tuple[EncoderConfig, ...] = field(
        default_factory=lambda: (T5Config(),)
    )
    postprocess_text_funcs: tuple[Callable[[BaseEncoderOutput], torch.Tensor], ...] = (
        field(default_factory=lambda: (t5_postprocess_text,))
    )

    # Image encoding stage
    image_encoder_config: EncoderConfig = field(
        default_factory=CLIPVisionConfig)
    image_encoder_precision: str = "fp32"
    image_encoder_extra_args: dict = field(
        default_factory=lambda: dict(
            output_hidden_states=True,
        )
    )
    
    # Precision for each component
    precision: str = "bf16"
    vae_precision: str = "bf16"
    text_encoder_precisions: tuple[str, ...] = field(default_factory=lambda: ("bf16",))

    def postprocess_image(self, image):
        return image.hidden_states[-2]

