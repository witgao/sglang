from typing import cast
import torch
import torch.nn.functional as F
import librosa
from tqdm import tqdm
import logging

from sglang.multimodal_gen.runtime.pipelines_core.stages.base import PipelineStage
from sglang.multimodal_gen.runtime.server_args import ServerArgs
from sglang.multimodal_gen.runtime.pipelines_core.schedule_batch import Req

from sglang.multimodal_gen.runtime.models.original import OriginalUMT5Model


def check_device_same(first_device, second_device):
    if first_device.type != second_device.type:
        return False

    if first_device.type == "cuda" and first_device.index is None:
        first_device = torch.device("cuda", index=0)

    if second_device.type == "cuda" and second_device.index is None:
        second_device = torch.device("cuda", index=0)

    return first_device == second_device


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


class TextEncodeStage(PipelineStage):

    def __init__(self, text_encoder: OriginalUMT5Model):
        super().__init__()
        self.text_encoder = text_encoder

    def parse_prompt_weights(self, prompt):
        """Extract text and weights from prompts with (text:weight) format"""
        import re

        # Parse all instances of (text:weight) in the prompt
        pattern = r'\((.*?):([\d\.]+)\)'
        matches = re.findall(pattern, prompt)

        # Replace each match with just the text part
        cleaned_prompt = prompt
        weights = {}

        for match in matches:
            text, weight = match
            orig_text = f"({text}:{weight})"
            cleaned_prompt = cleaned_prompt.replace(orig_text, text)
            weights[text] = float(weight)

        return cleaned_prompt, weights

    def text_encode(self, positive_prompt, negative_prompt, t5):
        device = torch.device(torch.cuda.current_device())
        offload_device = torch.device("cpu")

        encoder = t5
        dtype = t5.dtype

        positive_prompts = []
        all_weights = []

        # encoder.model.to(device)

        # Split positive prompts and process each with weights
        if "|" in positive_prompt:
            logging.info(
                "Multiple positive prompts detected, splitting by '|'")
            positive_prompts_raw = [p.strip()
                                    for p in positive_prompt.split('|')]
        else:
            positive_prompts_raw = [positive_prompt.strip()]

        for p in positive_prompts_raw:
            cleaned_prompt, weights = self.parse_prompt_weights(p)
            positive_prompts.append(cleaned_prompt)
            all_weights.append(weights)

        if encoder.quantization == "fp8_e4m3fn":
            cast_dtype = torch.float8_e4m3fn
        else:
            cast_dtype = encoder.dtype

        params_to_keep = {'norm', 'pos_embedding', 'token_embedding'}
        for name, param in encoder.model.named_parameters():
            dtype_to_use = dtype if any(
                keyword in name for keyword in params_to_keep) else cast_dtype
            value = encoder.state_dict[name] if hasattr(
                encoder, 'state_dict') else encoder.model.state_dict()[name]
            set_module_tensor_to_device(
                encoder.model, name, device=device, dtype=dtype_to_use, value=value)
        if hasattr(encoder, 'state_dict'):
            del encoder.state_dict

        with torch.autocast(device_type=device.type, dtype=encoder.dtype, enabled=encoder.quantization != 'disabled'):
            context = encoder(positive_prompts, device)
            # Apply weights to embeddings if any were extracted
            for i, weights in enumerate(all_weights):
                for text, weight in weights.items():
                    logging.info(
                        f"Applying weight {weight} to prompt: {text}")
                    if len(weights) > 0:
                        context[i] = context[i] * weight

            context_null = encoder([negative_prompt], device)

        encoder.model.to(offload_device)

        return (context, context_null)

    def forward(self, batch: Req, server_args: ServerArgs,) -> Req:
        positive_prompt = batch.prompt
        negative_prompt = batch.negative_prompt

        prompt_embeds, negative_prompt_embeds = self.text_encode(
            positive_prompt=positive_prompt,
            negative_prompt=negative_prompt,
            t5=self.text_encoder
        )

        batch.prompt_embeds = prompt_embeds
        batch.negative_prompt_embeds = negative_prompt_embeds
        return batch
