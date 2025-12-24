
import torch
import torch.nn.functional as F
from diffusers.loaders import FromOriginalModelMixin
from transformers import Wav2Vec2Config, Wav2Vec2Model, Wav2Vec2FeatureExtractor
from transformers.modeling_outputs import BaseModelOutput

import os
import json
from accelerate import init_empty_weights


from sglang.multimodal_gen.runtime.utils.model_utils import ModelUtils


def check_device_same(first_device, second_device):
    if first_device.type != second_device.type:
        return False

    if first_device.type == "cuda" and first_device.index is None:
        first_device = torch.device("cuda", index=0)

    if second_device.type == "cuda" and second_device.index is None:
        second_device = torch.device("cuda", index=0)

    return first_device == second_device

# simplified version of the accelerate function https://github.com/huggingface/accelerate/blob/main/src/accelerate/utils/modeling.py


def set_module_tensor_to_device(module, tensor_name, device, value=None, dtype=None):
    """
    A helper function to set a given tensor (parameter of buffer) of a module on a specific device (note that doing
    `param.to(device)` creates a new tensor not linked to the parameter, which is why we need this function).

    Args:
        module (`torch.nn.Module`):
            The module in which the tensor we want to move lives.
        tensor_name (`str`):
            The full name of the parameter/buffer.
        device (`int`, `str` or `torch.device`):
            The device on which to set the tensor.
        value (`torch.Tensor`, *optional*):
            The value of the tensor (useful when going from the meta device to any other device).
        dtype (`torch.dtype`, *optional*):
            If passed along the value of the parameter will be cast to this `dtype`. Otherwise, `value` will be cast to
            the dtype of the existing parameter in the model.
    """
    # Recurse if needed
    if "." in tensor_name:
        splits = tensor_name.split(".")
        for split in splits[:-1]:
            new_module = getattr(module, split)
            if new_module is None:
                raise ValueError(f"{module} has no attribute {split}.")
            module = new_module
        tensor_name = splits[-1]

    if tensor_name not in module._parameters and tensor_name not in module._buffers:
        raise ValueError(
            f"{module} does not have a parameter or a buffer named {tensor_name}.")
    is_buffer = tensor_name in module._buffers
    old_value = getattr(module, tensor_name)

    if old_value.device == torch.device("meta") and device not in ["meta", torch.device("meta")] and value is None:
        raise ValueError(
            f"{tensor_name} is on the meta device, we need a `value` to put in on {device}.")

    param = module._parameters[tensor_name] if tensor_name in module._parameters else None
    param_cls = type(param)

    if value is not None:
        if dtype is None:
            value = value.to(old_value.dtype)
        elif not str(value.dtype).startswith(("torch.uint", "torch.int", "torch.bool")):
            value = value.to(dtype)

    device_quantization = None
    with torch.no_grad():
        if value is None:
            new_value = old_value.to(device)
            if dtype is not None and device in ["meta", torch.device("meta")]:
                if not str(old_value.dtype).startswith(("torch.uint", "torch.int", "torch.bool")):
                    new_value = new_value.to(dtype)

                if not is_buffer:
                    module._parameters[tensor_name] = param_cls(
                        new_value, requires_grad=old_value.requires_grad)
        elif isinstance(value, torch.Tensor):
            new_value = value.to(device)
        else:
            new_value = torch.tensor(value, device=device)
        if device_quantization is not None:
            device = device_quantization
        if is_buffer:
            module._buffers[tensor_name] = new_value
        elif value is not None or not check_device_same(torch.device(device), module._parameters[tensor_name].device):
            param_cls = type(module._parameters[tensor_name])
            new_value = param_cls(new_value, requires_grad=False).to(device)
            module._parameters[tensor_name] = new_value

    # if device != "cpu":
    #    mm.soft_empty_cache()


def get_mask_from_lengths(lengths, max_len=None):
    lengths = lengths.to(torch.long)
    if max_len is None:
        max_len = torch.max(lengths).item()

    ids = torch.arange(0, max_len).unsqueeze(0).expand(
        lengths.shape[0], -1).to(lengths.device)
    mask = ids < lengths.unsqueeze(1).expand(-1, max_len)

    return mask


def linear_interpolation(features, seq_len):
    features = features.transpose(1, 2)
    output_features = F.interpolate(
        features, size=seq_len, align_corners=True, mode='linear')
    return output_features.transpose(1, 2)


class OriginalWav2Vec2Model(Wav2Vec2Model, FromOriginalModelMixin):

    def __init__(
            self, config: Wav2Vec2Config, base_dtype
    ):
        super().__init__(config)

        feature_extractor_config = {
            "do_normalize": False,
            "feature_size": 1,
            "padding_side": "right",
            "padding_value": 0.0,
            "return_attention_mask": False,
            "sampling_rate": 16000
        }
        self.wav2vec_feature_extractor = Wav2Vec2FeatureExtractor(
            **feature_extractor_config)
        self.base_dtype = base_dtype

    def forward(
        self,
        input_values,
        seq_len,
        attention_mask=None,
        mask_time_indices=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        # self.config.output_attentions = True

        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        extract_features = self.feature_extractor(input_values)
        extract_features = extract_features.transpose(1, 2)
        extract_features = linear_interpolation(
            extract_features, seq_len=seq_len)

        if attention_mask is not None:
            # compute reduced attention_mask corresponding to feature vectors
            attention_mask = self._get_feature_vector_attention_mask(
                extract_features.shape[1], attention_mask, add_adapter=False
            )

        hidden_states, extract_features = self.feature_projection(
            extract_features)
        hidden_states = self._mask_hidden_states(
            hidden_states, mask_time_indices=mask_time_indices, attention_mask=attention_mask
        )

        encoder_outputs = self.encoder(
            hidden_states,
            attention_mask=attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        hidden_states = encoder_outputs[0]

        if self.adapter is not None:
            hidden_states = self.adapter(hidden_states)

        if not return_dict:
            return (hidden_states, ) + encoder_outputs[1:]
        return BaseModelOutput(
            last_hidden_state=hidden_states,
            hidden_states=encoder_outputs.hidden_states,
            attentions=encoder_outputs.attentions,
        )

    def feature_extract(
        self,
        input_values,
        seq_len,
    ):
        extract_features = self.feature_extractor(input_values)
        extract_features = extract_features.transpose(1, 2)
        extract_features = linear_interpolation(
            extract_features, seq_len=seq_len)

        return extract_features

    def encode(
        self,
        extract_features,
        attention_mask=None,
        mask_time_indices=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        # self.config.output_attentions = True

        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if attention_mask is not None:
            # compute reduced attention_mask corresponding to feature vectors
            attention_mask = self._get_feature_vector_attention_mask(
                extract_features.shape[1], attention_mask, add_adapter=False
            )

        hidden_states, extract_features = self.feature_projection(
            extract_features)
        hidden_states = self._mask_hidden_states(
            hidden_states, mask_time_indices=mask_time_indices, attention_mask=attention_mask
        )

        encoder_outputs = self.encoder(
            hidden_states,
            attention_mask=attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        hidden_states = encoder_outputs[0]

        if self.adapter is not None:
            hidden_states = self.adapter(hidden_states)

        if not return_dict:
            return (hidden_states, ) + encoder_outputs[1:]
        return BaseModelOutput(
            last_hidden_state=hidden_states,
            hidden_states=encoder_outputs.hidden_states,
            attentions=encoder_outputs.attentions,
        )

    @classmethod
    def from_single_file(cls, pretrained_model_link_or_path_or_dict=None, **kwargs):
        base_dtype = torch.float16
        offload_device = torch.device("cpu")

        wav2vec2_config = Wav2Vec2Config(**kwargs)
        with init_empty_weights():
            wav2vec2 = OriginalWav2Vec2Model(
                wav2vec2_config, base_dtype=base_dtype).eval()

        sd = ModelUtils.load_torch_file(pretrained_model_link_or_path_or_dict)

        for name, param in wav2vec2.named_parameters():
            key = "wav2vec2." + name
            if "original0" in name:
                key = "wav2vec2.encoder.pos_conv_embed.conv.weight_g"
            elif "original1" in name:
                key = "wav2vec2.encoder.pos_conv_embed.conv.weight_v"
            value = sd[key]
            set_module_tensor_to_device(
                wav2vec2, name, device=offload_device, dtype=base_dtype, value=value)

        return wav2vec2


EntryClass = [OriginalWav2Vec2Model]
