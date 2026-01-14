

import torch
from torch import nn
from diffusers.loaders import FromOriginalModelMixin

from sglang.multimodal_gen.runtime.utils.model_utils import ModelUtils
from sglang.multimodal_gen.runtime.models.original.infinite_talk import OriginalInfiniteTalkModel

from accelerate import init_empty_weights
from tqdm import tqdm
import logging
from typing import Optional

from .wanvideo.model import WanModel

def _replace_linear(model, compute_dtype, state_dict, prefix="", patches=None, scale_weights=None, compile_args=None):
   
    has_children = list(model.children())
    if not has_children:
        return
    
    allow_compile = False

    for name, module in model.named_children():
        if compile_args is not None:
            allow_compile = compile_args.get("allow_unmerged_lora_compile", False)
        module_prefix = prefix + name + "."
        module_prefix = module_prefix.replace("_orig_mod.", "")
        _replace_linear(module, compute_dtype, state_dict, module_prefix, patches, scale_weights, compile_args)

        if isinstance(module, nn.Linear) and "loras" not in module_prefix:
            in_features = state_dict[module_prefix + "weight"].shape[1]
            out_features = state_dict[module_prefix + "weight"].shape[0]
            if scale_weights is not None:
                scale_key = f"{module_prefix}scale_weight"

            with init_empty_weights():
                model._modules[name] = CustomLinear(
                    in_features,
                    out_features,
                    module.bias is not None,
                    compute_dtype=compute_dtype,
                    scale_weight=scale_weights.get(scale_key) if scale_weights else None,
                    allow_compile=allow_compile
                )
            model._modules[name].source_cls = type(module)
            model._modules[name].requires_grad_(False)

    return model

def set_lora_params(module, patches, module_prefix="", device=torch.device("cpu")):
    remove_lora_from_module(module)
    # Recursively set lora_diffs and lora_strengths for all CustomLinear layers
    for name, child in module.named_children():
        params = list(child.parameters())
        if params:
            device = params[0].device
        else:
            device = torch.device("cpu")
        child_prefix = (f"{module_prefix}{name}.")
        set_lora_params(child, patches, child_prefix, device)
    if isinstance(module, CustomLinear):
        key = f"{module_prefix}weight"
        patch = patches.get(key, [])
        #print(f"Processing LoRA patches for {key}: {len(patch)} patches found")
        if len(patch) == 0:
            key = key.replace("_orig_mod.", "")
            patch = patches.get(key, [])
            #print(f"Processing LoRA patches for {key}: {len(patch)} patches found")
        if len(patch) != 0:
            lora_diffs = []
            for p in patch:
                lora_obj = p[1]
                if "head" in key:
                    continue  # For now skip LoRA for head layers
                elif hasattr(lora_obj, "weights"):
                    lora_diffs.append(lora_obj.weights)
                elif isinstance(lora_obj, tuple) and lora_obj[0] == "diff":
                    lora_diffs.append(lora_obj[1])
                else:
                    continue
            lora_strengths = [p[0] for p in patch]
            module.set_lora_diffs(lora_diffs, device=device)
            module.lora_strengths = lora_strengths
            module.step = 0  # Initialize step for LoRA scheduling


class CustomLinear(nn.Linear):
    def __init__(
        self,
        in_features,
        out_features,
        bias=False,
        compute_dtype=None,
        device=None,
        scale_weight=None,
        allow_compile=False
    ) -> None:
        super().__init__(in_features, out_features, bias, device)
        self.compute_dtype = compute_dtype
        self.lora_diffs = []
        self.step = 0
        self.scale_weight = scale_weight
        self.lora_strengths = []
        self.allow_compile = allow_compile

        if not allow_compile:
            self._get_weight_with_lora = torch.compiler.disable()(self._get_weight_with_lora)
    
    def set_lora_diffs(self, lora_diffs, device=torch.device("cpu")):
        self.lora_diffs = []
        for i, diff in enumerate(lora_diffs):
            if len(diff) > 1:
                self.register_buffer(f"lora_diff_{i}_0", diff[0].to(device, self.compute_dtype))
                self.register_buffer(f"lora_diff_{i}_1", diff[1].to(device, self.compute_dtype))
                setattr(self, f"lora_diff_{i}_2", diff[2])
                self.lora_diffs.append((f"lora_diff_{i}_0", f"lora_diff_{i}_1", f"lora_diff_{i}_2"))
            else:
                self.register_buffer(f"lora_diff_{i}_0", diff[0].to(device, self.compute_dtype))
                self.lora_diffs.append(f"lora_diff_{i}_0")

    def _get_weight_with_lora(self, weight):
        """Apply LoRA outside compiled region"""
        if not hasattr(self, "lora_diff_0_0"):
            return weight
        
        for lora_diff_names, lora_strength in zip(self.lora_diffs, self.lora_strengths):
            if isinstance(lora_strength, list):
                lora_strength = lora_strength[self.step]
                if lora_strength == 0.0:
                    continue
            elif lora_strength == 0.0:
                continue
            if isinstance(lora_diff_names, tuple):
                lora_diff_0 = getattr(self, lora_diff_names[0])
                lora_diff_1 = getattr(self, lora_diff_names[1])
                lora_diff_2 = getattr(self, lora_diff_names[2])
                patch_diff = torch.mm(
                    lora_diff_0.flatten(start_dim=1),
                    lora_diff_1.flatten(start_dim=1)
                ).reshape(weight.shape) + 0
                alpha = lora_diff_2 / lora_diff_1.shape[0] if lora_diff_2 is not None else 1.0
                scale = lora_strength * alpha
                weight = weight.add(patch_diff, alpha=scale)
            else:
                lora_diff = getattr(self, lora_diff_names)
                weight = weight.add(lora_diff, alpha=lora_strength)
        return weight

    def forward(self, input):
        if self.bias is not None:
            bias = self.bias.to(input)
        else:
            bias = None
        weight = self.weight.to(input)

        if self.scale_weight is not None:
            if weight.numel() < input.numel():
                weight = weight * self.scale_weight
            else:
                input = input * self.scale_weight

        weight = self._get_weight_with_lora(weight)

        return torch.nn.functional.linear(input, weight, bias)
    
def remove_lora_from_module(module):
    for name, submodule in module.named_modules():
        if hasattr(submodule, "lora_diffs"):
            for i in range(len(submodule.lora_diffs)):
                if hasattr(submodule, f"lora_diff_{i}_0"):
                    delattr(submodule, f"lora_diff_{i}_0")
                if hasattr(submodule, f"lora_diff_{i}_1"):
                    delattr(submodule, f"lora_diff_{i}_1")
                if hasattr(submodule, f"lora_diff_{i}_2"):
                    delattr(submodule, f"lora_diff_{i}_2")


def check_device_same(first_device, second_device):
    if first_device.type != second_device.type:
        return False

    if first_device.type == "cuda" and first_device.index is None:
        first_device = torch.device("cuda", index=0)

    if second_device.type == "cuda" and second_device.index is None:
        second_device = torch.device("cuda", index=0)

    return first_device == second_device

def set_module_tensor_to_device(module, tensor_name, device, value=None, dtype=None):
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
        raise ValueError(f"{module} does not have a parameter or a buffer named {tensor_name}.")
    is_buffer = tensor_name in module._buffers
    old_value = getattr(module, tensor_name)

    if old_value.device == torch.device("meta") and device not in ["meta", torch.device("meta")] and value is None:
        raise ValueError(f"{tensor_name} is on the meta device, we need a `value` to put in on {device}.")

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
                    module._parameters[tensor_name] = param_cls(new_value, requires_grad=old_value.requires_grad)
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
            
def compile_model(transformer, compile_args=None):
    if compile_args is None:
        return transformer
    if hasattr(torch, '_dynamo') and hasattr(torch._dynamo, 'config'):
        torch._dynamo.config.cache_size_limit = compile_args["dynamo_cache_size_limit"]
        torch._dynamo.config.force_parameter_static_shapes = compile_args["force_parameter_static_shapes"]
        try:
            if hasattr(torch._dynamo.config, 'allow_unspec_int_on_nn_module'):
                torch._dynamo.config.allow_unspec_int_on_nn_module = True
        except Exception as e:
            log.warning(f"Could not set allow_unspec_int_on_nn_module: {e}")
        try:
            torch._dynamo.config.recompile_limit = compile_args["dynamo_recompile_limit"]
        except Exception as e:
            log.warning(f"Could not set recompile_limit: {e}")

    if compile_args["compile_transformer_blocks_only"]:
        for i, block in enumerate(transformer.blocks):
            if hasattr(block, "_orig_mod"):
                block = block._orig_mod
            transformer.blocks[i] = torch.compile(block, fullgraph=compile_args["fullgraph"], dynamic=compile_args["dynamic"], backend=compile_args["backend"], mode=compile_args["mode"])
        if transformer.vace_layers is not None:
            for i, block in enumerate(transformer.vace_blocks):
                if hasattr(block, "_orig_mod"):
                    block = block._orig_mod
                transformer.vace_blocks[i] = torch.compile(block, fullgraph=compile_args["fullgraph"], dynamic=compile_args["dynamic"], backend=compile_args["backend"], mode=compile_args["mode"])
    else:
        transformer = torch.compile(transformer, fullgraph=compile_args["fullgraph"], dynamic=compile_args["dynamic"], backend=compile_args["backend"], mode=compile_args["mode"])
    return transformer

device = torch.device(torch.cuda.current_device())
offload_device = torch.device("cpu")


def add_patches(model, lora_sd, strength_patch=1.0, strength_model=1.0):
    patches = {}
    
    model_sd = model.state_dict()
    for k in lora_sd:
        offset = None
        function = None
        if isinstance(k, str):
            key = k
        else:
            offset = k[1]
            key = k[0]
            if len(k) > 2:
                function = k[2]

        if key in model_sd:
            current_patches = patches.get(key, [])
            current_patches.append((strength_patch, lora_sd[k], strength_model, offset, function))
            patches[key] = current_patches

    return patches


def load_weights(transformer, sd=None, weight_dtype=None, base_dtype=None, 
                 transformer_load_device=None, block_swap_args=None, gguf=False, reader=None, patcher=None, compile_args=None):
    params_to_keep = {"time_in", "patch_embedding", "time_", "modulation", "text_embedding", 
                      "adapter", "add", "ref_conv", "casual_audio_encoder", "cond_encoder", "frame_packer", "audio_proj_glob", "face_encoder", "fuser_block"}
    param_count = sum(1 for _ in transformer.named_parameters())
    # pbar = ProgressBar(param_count)
    cnt = 0
    block_idx = vace_block_idx = None

    named_params = transformer.named_parameters()

    for name, param in tqdm(named_params,
            desc=f"Loading transformer parameters to {transformer_load_device}",
            total=param_count,
            leave=True):
        block_idx = vace_block_idx = None
        if "vace_blocks." in name:
            try:
                vace_block_idx = int(name.split("vace_blocks.")[1].split(".")[0])
            except Exception:
                vace_block_idx = None
        elif "blocks." in name and "face" not in name:
            try:
                block_idx = int(name.split("blocks.")[1].split(".")[0])
            except Exception:
                block_idx = None

        if "loras" in name or "controlnet" in name:
            continue

        # GGUF: skip GGUFParameter params
        # if gguf and isinstance(param, GGUFParameter):
        #     continue
        
        key = name.replace("_orig_mod.", "")
        value=sd[key]

        if gguf:
            dtype_to_use = torch.float32 if "patch_embedding" in name or "motion_encoder" in name else base_dtype
        else:
            dtype_to_use = base_dtype if any(keyword in name for keyword in params_to_keep) else weight_dtype
            dtype_to_use = weight_dtype if value.dtype == weight_dtype else dtype_to_use
            scale_key = key.replace(".weight", ".scale_weight")
            if scale_key in sd:
                dtype_to_use = value.dtype
            if "bias" in name or "img_emb" in name:
                dtype_to_use = base_dtype
            if "patch_embedding" in name or "motion_encoder" in name:
                dtype_to_use = torch.float32
            if "modulation" in name or "norm" in name:
                dtype_to_use = value.dtype if value.dtype == torch.float32 else base_dtype

        load_device = transformer_load_device
        if block_swap_args is not None:
            load_device = device
            if block_idx is not None:
                if block_idx >= len(transformer.blocks) - block_swap_args.get("blocks_to_swap", 0):
                    load_device = offload_device
            elif vace_block_idx is not None:
                if vace_block_idx >= len(transformer.vace_blocks) - block_swap_args.get("vace_blocks_to_swap", 0):
                    load_device = offload_device
        # Set tensor to device
        set_module_tensor_to_device(transformer, name, device=load_device, dtype=dtype_to_use, value=value)
        cnt += 1

def model_lora_keys_unet(model, key_map={}):
    sd = model.state_dict()
    sdk = sd.keys()

    for k in sdk:
        if k.endswith(".weight"):
            key_lora = k[len("diffusion_model."):-len(".weight")].replace(".", "_")
            key_map["lora_unet_{}".format(key_lora)] = k
            key_map["{}".format(k[:-len(".weight")])] = k #generic lora format without any weird key names
        else:
            key_map["{}".format(k)] = k #generic lora format for not .weight without any weird key names

    return key_map

class LoRAAdapter:
    name = "lora"

    def __init__(self, loaded_keys, weights):
        self.loaded_keys = loaded_keys
        self.weights = weights

    @classmethod
    def load(
        cls,
        x: str,
        lora: dict[str, torch.Tensor],
        alpha: float,
        dora_scale: torch.Tensor,
        loaded_keys: set[str] = None,
    ) -> Optional["LoRAAdapter"]:
        if loaded_keys is None:
            loaded_keys = set()

        reshape_name = "{}.reshape_weight".format(x)
        regular_lora = "{}.lora_up.weight".format(x)
        diffusers_lora = "{}_lora.up.weight".format(x)
        diffusers2_lora = "{}.lora_B.weight".format(x)
        diffusers3_lora = "{}.lora.up.weight".format(x)
        mochi_lora = "{}.lora_B".format(x)
        transformers_lora = "{}.lora_linear_layer.up.weight".format(x)
        qwen_default_lora = "{}.lora_B.default.weight".format(x)
        A_name = None

        if regular_lora in lora.keys():
            A_name = regular_lora
            B_name = "{}.lora_down.weight".format(x)
            mid_name = "{}.lora_mid.weight".format(x)
        elif diffusers_lora in lora.keys():
            A_name = diffusers_lora
            B_name = "{}_lora.down.weight".format(x)
            mid_name = None
        elif diffusers2_lora in lora.keys():
            A_name = diffusers2_lora
            B_name = "{}.lora_A.weight".format(x)
            mid_name = None
        elif diffusers3_lora in lora.keys():
            A_name = diffusers3_lora
            B_name = "{}.lora.down.weight".format(x)
            mid_name = None
        elif mochi_lora in lora.keys():
            A_name = mochi_lora
            B_name = "{}.lora_A".format(x)
            mid_name = None
        elif transformers_lora in lora.keys():
            A_name = transformers_lora
            B_name = "{}.lora_linear_layer.down.weight".format(x)
            mid_name = None
        elif qwen_default_lora in lora.keys():
            A_name = qwen_default_lora
            B_name = "{}.lora_A.default.weight".format(x)
            mid_name = None

        if A_name is not None:
            mid = None
            if mid_name is not None and mid_name in lora.keys():
                mid = lora[mid_name]
                loaded_keys.add(mid_name)
            reshape = None
            if reshape_name in lora.keys():
                try:
                    reshape = lora[reshape_name].tolist()
                    loaded_keys.add(reshape_name)
                except:
                    pass
            weights = (lora[A_name], lora[B_name], alpha, mid, dora_scale, reshape)
            loaded_keys.add(A_name)
            loaded_keys.add(B_name)
            return cls(loaded_keys, weights)
        else:
            return None

def load_lora(lora, to_load, log_missing=True):
    patch_dict = {}
    loaded_keys = set()
    for x in to_load:
        alpha_name = "{}.alpha".format(x)
        alpha = None
        if alpha_name in lora.keys():
            alpha = lora[alpha_name].item()
            loaded_keys.add(alpha_name)

        dora_scale_name = "{}.dora_scale".format(x)
        dora_scale = None
        if dora_scale_name in lora.keys():
            dora_scale = lora[dora_scale_name]
            loaded_keys.add(dora_scale_name)

        adapter = LoRAAdapter.load(x, lora, alpha, dora_scale, loaded_keys)
        if adapter is not None:
            patch_dict[to_load[x]] = ("diff", adapter.weights)
            loaded_keys.update(adapter.loaded_keys)
        

        w_norm_name = "{}.w_norm".format(x)
        b_norm_name = "{}.b_norm".format(x)
        w_norm = lora.get(w_norm_name, None)
        b_norm = lora.get(b_norm_name, None)

        if w_norm is not None:
            loaded_keys.add(w_norm_name)
            patch_dict[to_load[x]] = ("diff", (w_norm,))
            if b_norm is not None:
                loaded_keys.add(b_norm_name)
                patch_dict["{}.bias".format(to_load[x][:-len(".weight")])] = ("diff", (b_norm,))

        diff_name = "{}.diff".format(x)
        diff_weight = lora.get(diff_name, None)
        if diff_weight is not None:
            patch_dict[to_load[x]] = ("diff", (diff_weight,))
            loaded_keys.add(diff_name)

        diff_bias_name = "{}.diff_b".format(x)
        diff_bias = lora.get(diff_bias_name, None)
        if diff_bias is not None:
            patch_dict["{}.bias".format(to_load[x][:-len(".weight")])] = ("diff", (diff_bias,))
            loaded_keys.add(diff_bias_name)

        set_weight_name = "{}.set_weight".format(x)
        set_weight = lora.get(set_weight_name, None)
        if set_weight is not None:
            patch_dict[to_load[x]] = ("set", (set_weight,))
            loaded_keys.add(set_weight_name)

    if log_missing:
        for x in lora.keys():
            if x not in loaded_keys:
                logging.warning("lora key not loaded: {}".format(x))

    return patch_dict

def add_lora_weights(model, lora):
    lora_path = lora["path"]
    lora_strength = lora["strength"]
    
    lora_sd = ModelUtils.load_torch_file(lora_path, safe_load=True)
    lora_sd = {key.replace("diffusion_model.", "", 1): value for key, value in lora_sd.items()}
    
    key_map = {}
    key_map = model_lora_keys_unet(model, key_map)
    loaded = load_lora(lora_sd, key_map)
    patches = add_patches(model, loaded, lora_strength)
    del lora_sd
    return patches

class OriginalWanTransformerModel(FromOriginalModelMixin):

    def __init__(self):
        super().__init__()

    @classmethod
    def from_single_file(cls, model=None, **kwargs):
        lora = {
            "path": "/data1/gaozhenyu/studio/sglang/models/InfiniteTalk/transformer/lightx2v_I2V_14B_480p_cfg_step_distill_rank64_bf16.safetensors",
            "strength": 1.0,
            "low_mem_load": False,
            "merge_loras": False,
        }
    
        infinite_talk_model_path = "/data1/gaozhenyu/studio/sglang/models/InfiniteTalk/transformer/Wan2_1-InfiniteTalk-Single_fp8_e4m3fn_scaled_KJ.safetensors"

        base_precision = "fp16_fast"
        load_device = "offload_device"
        compile_args = None
        attention_mode = "sageattn"
        quantization = "fp8_e4m3fn_scaled"
        rms_norm_function = None
                  
        device = torch.device(torch.cuda.current_device())
        offload_device = torch.device("cpu")

        transformer = None
 
        transformer_load_device = device if load_device == "main_device" else offload_device

        base_dtype = {"fp8_e4m3fn": torch.float8_e4m3fn, "fp8_e4m3fn_fast": torch.float8_e4m3fn, "bf16": torch.bfloat16, "fp16": torch.float16, "fp16_fast": torch.float16, "fp32": torch.float32}[base_precision]
        
        if base_precision == "fp16_fast":
            torch.backends.cuda.matmul.allow_fp16_accumulation = True

        model_path = model

        sd = ModelUtils.load_torch_file(model_path, device=transformer_load_device, safe_load=True)

        patch_size=(1, 2, 2)
        model_type = "i2v"
        in_features = out_features = dim = 5120
        num_heads = 40
        num_layers = 40
        in_channels = 36 
        ffn_dim = ffn2_dim = 13824 
        out_dim = 16
        
        TRANSFORMER_CONFIG= {
            "dim": dim,
            "in_features": in_features,
            "out_features": out_features,
            "patch_size": patch_size,
            "ffn_dim": ffn_dim,
            "ffn2_dim": ffn2_dim,
            "eps": 1e-06,
            "freq_dim": 256,
            "in_dim": in_channels,
            "model_type": model_type,
            "out_dim": out_dim,
            "text_len": 512,
            "num_heads": num_heads,
            "num_layers": num_layers,
            "attention_mode": attention_mode,
            "rope_func": "comfy",
            "main_device": device,
            "offload_device": offload_device,
            "dtype": base_dtype,
            "rms_norm_function": rms_norm_function,

        }

        with init_empty_weights():
            transformer = WanModel(**TRANSFORMER_CONFIG).eval()

        if infinite_talk_model_path is not None:
            from .wanvideo.multitalk.multitalk import SingleStreamMultiAttention
            
            infinite_talk_model = OriginalInfiniteTalkModel.from_single_file(infinite_talk_model_path)

            for block in transformer.blocks:
                with init_empty_weights():
                    block.norm_x = nn.LayerNorm(dim, transformer.eps, elementwise_affine=True)
                    block.audio_cross_attn = SingleStreamMultiAttention(
                            dim=dim,
                            encoder_hidden_states_dim=768,
                            num_heads=num_heads,
                        qkv_bias=True,
                        class_range=24,
                        class_interval=4,
                        attention_mode=attention_mode,
                    )
            transformer.multitalk_audio_proj = infinite_talk_model

            extra_sd = {}
            extra_sd_temp = ModelUtils.load_torch_file(infinite_talk_model_path, device=transformer_load_device, safe_load=True)
                
            for k, v in extra_sd_temp.items():
                extra_sd[k.replace("audio_proj.", "multitalk_audio_proj.")] = v
                
            sd.update(extra_sd)
            del extra_sd
     
        scale_weights = {}
        if "fp8" in quantization:
            for k, v in sd.items():
                if k.endswith(".scale_weight"):
                    scale_weights[k] = v.to(device, base_dtype)

        if quantization in ["fp8_e4m3fn", "fp8_e4m3fn_fast"]:
            weight_dtype = torch.float8_e4m3fn
        elif quantization in ["fp8_e5m2", "fp8_e5m2_fast"]:
            weight_dtype = torch.float8_e5m2
        else:
            weight_dtype = base_dtype
        
        if lora is not None:
            patches = add_lora_weights(transformer, lora)
            
        if lora is not None:
            transformer = _replace_linear(transformer, base_dtype, sd, scale_weights=scale_weights, compile_args=compile_args)
            transformer.patched_linear = True


        block_swap_args = {
            "blocks_to_swap": 40,
            "offload_img_emb": False,
            "offload_txt_emb": False,
            "use_non_blocking": True,
            "vace_blocks_to_swap": 0,
            "prefetch_blocks": 1,
            "block_swap_debug": False
        }
        if block_swap_args is not None:
            transformer.use_non_blocking = block_swap_args.get("use_non_blocking", False)
            transformer.blocks_to_swap = block_swap_args.get("blocks_to_swap", 0)
            transformer.vace_blocks_to_swap = block_swap_args.get("vace_blocks_to_swap", 0)
            transformer.prefetch_blocks = block_swap_args.get("prefetch_blocks", 0)
            transformer.block_swap_debug = block_swap_args.get("block_swap_debug", False)
            transformer.offload_img_emb = block_swap_args.get("offload_img_emb", False)
            transformer.offload_txt_emb = block_swap_args.get("offload_txt_emb", False)

        
        load_weights(transformer, sd, weight_dtype, base_dtype=base_dtype, transformer_load_device=device, 
                         block_swap_args=block_swap_args, compile_args=compile_args)

        if len(patches) != 0: #handle patched linear layers (unmerged loras, fp8 scaled)
            set_lora_params(transformer, patches)

        transformer.lora_scheduling_enabled = False
        
        transformer = compile_model(transformer, compile_args)

        return transformer


EntryClass = [OriginalWanTransformerModel]
