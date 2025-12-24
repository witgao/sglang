

import torch
from torch import nn
from torch.nn import Module, ModuleList
import torch.nn.functional as F
from diffusers.loaders import FromOriginalModelMixin
from diffusers.models.modeling_utils import ModelMixin

from sglang.multimodal_gen.runtime.utils.model_utils import ModelUtils
from sglang.multimodal_gen.runtime.models.original.infinite_talk import OriginalInfiniteTalkModel

from accelerate import init_empty_weights
import re
import numpy as np
import gc
import math
from tqdm import tqdm
import logging
import os
import sglang.comfy.model_management as mm
import sglang.comfy
from sglang.comfy.sd import load_lora_for_models
from .wanvideo.custom_linear import _replace_linear
from .wanvideo.utils import set_module_tensor_to_device
from .wanvideo.wanvideo.modules.model import WanModel, LoRALinearLayer
from .wanvideo.utils import apply_lora



device = mm.get_torch_device()
offload_device = mm.unet_offload_device()

class WanVideoLoraSelect:

    def getlorapath(self, lora, strength, unique_id, blocks={}, prev_lora=None, low_mem_load=False, merge_loras=True):
        if not merge_loras:
            low_mem_load = False  # Unmerged LoRAs don't need low_mem_load
        loras_list = []

        if not isinstance(strength, list):
            strength = round(strength, 4)
            if strength == 0.0:
                if prev_lora is not None:
                    loras_list.extend(prev_lora)
                return (loras_list,)

        lora_path = lora

        # Load metadata from the safetensors file
        metadata = {}
        try:
            from safetensors.torch import safe_open
            with safe_open(lora_path, framework="pt", device="cpu") as f:
                metadata = f.metadata()
        except Exception as e:
            logging.info(f"Could not load metadata from {lora}: {e}")

        lora = {
            "path": lora_path,
            "strength": strength,
            "name": os.path.splitext(lora)[0],
            "blocks": blocks.get("selected_blocks", {}),
            "layer_filter": blocks.get("layer_filter", ""),
            "low_mem_load": low_mem_load,
            "merge_loras": merge_loras,
        }
        if prev_lora is not None:
            loras_list.extend(prev_lora)

        loras_list.append(lora)
        return (loras_list,)
    
class WanVideoModel(sglang.comfy.model_base.BaseModel):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.pipeline = {}

    def __getitem__(self, k):
        return self.pipeline[k]

    def __setitem__(self, k, v):
        self.pipeline[k] = v

try:
    from sglang.comfy.latent_formats import Wan21, Wan22
    latent_format = Wan21
except: #for backwards compatibility
    logging.warning("WARNING: Wan21 latent format not found, update ComfyUI for better live video preview")
    from sglang.comfy.latent_formats import HunyuanVideo
    latent_format = HunyuanVideo

class WanVideoModelConfig:
    def __init__(self, dtype, latent_format=latent_format):
        self.unet_config = {}
        self.unet_extra_config = {}
        self.latent_format = latent_format
        #self.latent_format.latent_channels = 16
        self.manual_cast_dtype = dtype
        self.sampling_settings = {"multiplier": 1.0}
        self.memory_usage_factor = 2.0
        self.unet_config["disable_unet_model_creation"] = True

def standardize_lora_key_format(lora_sd):
    new_sd = {}
    for k, v in lora_sd.items():
        # aitoolkit/lycoris format
        if k.startswith("lycoris_blocks_"):
            k = k.replace("lycoris_blocks_", "blocks.")
            k = k.replace("_cross_attn_", ".cross_attn.")
            k = k.replace("_self_attn_", ".self_attn.")
            k = k.replace("_ffn_net_0_proj", ".ffn.0")
            k = k.replace("_ffn_net_2", ".ffn.2")
            k = k.replace("to_out_0", "o")
        # Diffusers format
        if k.startswith('transformer.'):
            k = k.replace('transformer.', 'diffusion_model.')
        if k.startswith('pipe.dit.'): #unianimate-dit/diffsynth
            k = k.replace('pipe.dit.', 'diffusion_model.')
        if k.startswith('blocks.'):
            k = k.replace('blocks.', 'diffusion_model.blocks.')
        if k.startswith('vace_blocks.'):
            k = k.replace('vace_blocks.', 'diffusion_model.vace_blocks.')
        k = k.replace('.default.', '.')
        k = k.replace('.diff_m', '.modulation.diff')

        # Fun LoRA format
        if k.startswith('lora_unet__'):
            # Split into main path and weight type parts
            parts = k.split('.')
            main_part = parts[0]  # e.g. lora_unet__blocks_0_cross_attn_k
            weight_type = '.'.join(parts[1:]) if len(parts) > 1 else None  # e.g. lora_down.weight
            
            # Process the main part - convert from underscore to dot format
            if 'blocks_' in main_part:
                # Extract components
                components = main_part[len('lora_unet__'):].split('_')
                
                # Start with diffusion_model
                new_key = "diffusion_model"
                
                # Add blocks.N
                if components[0] == 'blocks':
                    new_key += f".blocks.{components[1]}"
                    
                    # Handle different module types
                    idx = 2
                    if idx < len(components):
                        if components[idx] == 'self' and idx+1 < len(components) and components[idx+1] == 'attn':
                            new_key += ".self_attn"
                            idx += 2
                        elif components[idx] == 'cross' and idx+1 < len(components) and components[idx+1] == 'attn':
                            new_key += ".cross_attn"
                            idx += 2
                        elif components[idx] == 'ffn':
                            new_key += ".ffn"
                            idx += 1
                    
                    # Add the component (k, q, v, o) and handle img suffix
                    if idx < len(components):
                        component = components[idx]
                        idx += 1
                        
                        # Check for img suffix
                        if idx < len(components) and components[idx] == 'img':
                            component += '_img'
                            idx += 1
                            
                        new_key += f".{component}"
                
                # Handle weight type - this is the critical fix
                if weight_type:
                    if weight_type == 'alpha':
                        new_key += '.alpha'
                    elif weight_type == 'lora_down.weight' or weight_type == 'lora_down':
                        new_key += '.lora_A.weight'
                    elif weight_type == 'lora_up.weight' or weight_type == 'lora_up':
                        new_key += '.lora_B.weight'
                    else:
                        # Keep original weight type if not matching our patterns
                        new_key += f'.{weight_type}'
                        # Add .weight suffix if missing
                        if not new_key.endswith('.weight'):
                            new_key += '.weight'
                
                k = new_key
            else:
                # For other lora_unet__ formats (head, embeddings, etc.)
                new_key = main_part.replace('lora_unet__', 'diffusion_model.')
                
                # Fix specific component naming patterns
                new_key = new_key.replace('_self_attn', '.self_attn')
                new_key = new_key.replace('_cross_attn', '.cross_attn')
                new_key = new_key.replace('_ffn', '.ffn')
                new_key = new_key.replace('blocks_', 'blocks.')
                new_key = new_key.replace('head_head', 'head.head')
                new_key = new_key.replace('img_emb', 'img_emb')
                new_key = new_key.replace('text_embedding', 'text.embedding')
                new_key = new_key.replace('time_embedding', 'time.embedding')
                new_key = new_key.replace('time_projection', 'time.projection')
                
                # Replace remaining underscores with dots, carefully
                parts = new_key.split('.')
                final_parts = []
                for part in parts:
                    if part in ['img_emb', 'self_attn', 'cross_attn']:
                        final_parts.append(part)  # Keep these intact
                    else:
                        final_parts.append(part.replace('_', '.'))
                new_key = '.'.join(final_parts)
                
                # Handle weight type
                if weight_type:
                    if weight_type == 'alpha':
                        new_key += '.alpha'
                    elif weight_type == 'lora_down.weight' or weight_type == 'lora_down':
                        new_key += '.lora_A.weight'
                    elif weight_type == 'lora_up.weight' or weight_type == 'lora_up':
                        new_key += '.lora_B.weight'
                    else:
                        new_key += f'.{weight_type}'
                        if not new_key.endswith('.weight'):
                            new_key += '.weight'
                
                k = new_key
                
            # Handle special embedded components
            special_components = {
                'time.projection': 'time_projection',
                'img.emb': 'img_emb',
                'text.emb': 'text_emb',
                'time.emb': 'time_emb',
            }
            for old, new in special_components.items():
                if old in k:
                    k = k.replace(old, new)

        # Fix diffusion.model -> diffusion_model
        if k.startswith('diffusion.model.'):
            k = k.replace('diffusion.model.', 'diffusion_model.')
            
        # Finetrainer format
        if '.attn1.' in k:
            k = k.replace('.attn1.', '.cross_attn.')
            k = k.replace('.to_k.', '.k.')
            k = k.replace('.to_q.', '.q.')
            k = k.replace('.to_v.', '.v.')
            k = k.replace('.to_out.0.', '.o.')
        elif '.attn2.' in k:
            k = k.replace('.attn2.', '.cross_attn.')
            k = k.replace('.to_k.', '.k.')
            k = k.replace('.to_q.', '.q.')
            k = k.replace('.to_v.', '.v.')
            k = k.replace('.to_out.0.', '.o.')
            
        if "img_attn.proj" in k:
            k = k.replace("img_attn.proj", "img_attn_proj")
        if "img_attn.qkv" in k:
            k = k.replace("img_attn.qkv", "img_attn_qkv")
        if "txt_attn.proj" in k:
            k = k.replace("txt_attn.proj", "txt_attn_proj")
        if "txt_attn.qkv" in k:
            k = k.replace("txt_attn.qkv", "txt_attn_qkv")
        new_sd[k] = v
    return new_sd

def filter_state_dict_by_blocks(state_dict, blocks_mapping, layer_filter=[]):
    filtered_dict = {}

    if isinstance(layer_filter, str):
        layer_filters = [layer_filter] if layer_filter else []
    else:
        # Filter out empty strings
        layer_filters = [f for f in layer_filter if f] if layer_filter else []

    #print("layer_filter: ", layer_filters)

    for key in state_dict:
        if not any(filter_str in key for filter_str in layer_filters):
            if 'blocks.' in key:
                
                block_pattern = key.split('diffusion_model.')[1].split('.', 2)[0:2]
                block_key = f'{block_pattern[0]}.{block_pattern[1]}.'

                if block_key in blocks_mapping:
                    filtered_dict[key] = state_dict[key]
            else:
                filtered_dict[key] = state_dict[key]
    
    for key in filtered_dict:
        print(key)

    #from safetensors.torch import save_file
    #save_file(filtered_dict, "filtered_state_dict_2.safetensors")

    return filtered_dict
def add_lora_weights(patcher, lora, base_dtype, merge_loras=False):
    unianimate_sd = None
    control_lora=False
    #spacepxl's control LoRA patch
    for l in lora:
        logging.info(f"Loading LoRA: {l['name']} with strength: {l['strength']}")
        lora_path = l["path"]
        lora_strength = l["strength"]
        if isinstance(lora_strength, list):
            if merge_loras:
                raise ValueError("LoRA strength should be a single value when merge_loras=True")
            patcher.model.diffusion_model.lora_scheduling_enabled = True
        if lora_strength == 0:
            logging.warning(f"LoRA {lora_path} has strength 0, skipping...")
            continue
        lora_sd = ModelUtils.load_torch_file(lora_path, safe_load=True)
        if "dwpose_embedding.0.weight" in lora_sd: #unianimate
            from .wanvideo.unianimate.nodes import update_transformer
            logging.info("Unianimate LoRA detected, patching model...")
            patcher.model.diffusion_model, unianimate_sd = update_transformer(patcher.model.diffusion_model, lora_sd)

        lora_sd = standardize_lora_key_format(lora_sd)

        if l["blocks"]:
            lora_sd = filter_state_dict_by_blocks(lora_sd, l["blocks"], l.get("layer_filter", []))

        # Filter out any LoRA keys containing 'img' if the base model state_dict has no 'img' keys
        #if not any('img' in k for k in sd.keys()):
        #    lora_sd = {k: v for k, v in lora_sd.items() if 'img' not in k}
        
        if "diffusion_model.patch_embedding.lora_A.weight" in lora_sd:
            control_lora = True
        #stand-in LoRA patch
        if "diffusion_model.blocks.0.self_attn.q_loras.down.weight" in lora_sd:
            patch_stand_in_lora(patcher.model.diffusion_model, lora_sd, device, base_dtype, lora_strength)
        # normal LoRA patch
        else:
            patcher, _ = load_lora_for_models(patcher, None, lora_sd, lora_strength, 0)
        
        del lora_sd
    return patcher, control_lora, unianimate_sd
   
def rename_fuser_block(name):
    # map fuser blocks to main blocks
    new_name = name
    if "face_adapter.fuser_blocks." in name:
        match = re.search(r'face_adapter\.fuser_blocks\.(\d+)\.', name)
        if match:
            fuser_block_num = int(match.group(1))
            main_block_num = fuser_block_num * 5
            new_name = name.replace(f"face_adapter.fuser_blocks.{fuser_block_num}.", f"blocks.{main_block_num}.fuser_block.")
    return new_name

def load_weights(transformer, sd=None, weight_dtype=None, base_dtype=None, 
                 transformer_load_device=None, block_swap_args=None, gguf=False, reader=None, patcher=None, compile_args=None):
    params_to_keep = {"time_in", "patch_embedding", "time_", "modulation", "text_embedding", 
                      "adapter", "add", "ref_conv", "casual_audio_encoder", "cond_encoder", "frame_packer", "audio_proj_glob", "face_encoder", "fuser_block"}
    param_count = sum(1 for _ in transformer.named_parameters())
    # pbar = ProgressBar(param_count)
    cnt = 0
    block_idx = vace_block_idx = None

    if gguf:
        pass
        # logging.info("Using GGUF to load and assign model weights to device...")

        # # Prepare sd from GGUF readers

        # # handle possible non-GGUF weights
        # extra_sd = {}
        # for key, value in sd.items():
        #     if value.device != torch.device("meta"):
        #         extra_sd[key] = value

        # sd = {}
        # all_tensors = []
        # for r in reader:
        #     all_tensors.extend(r.tensors)
        # for tensor in all_tensors:
        #     name = rename_fuser_block(tensor.name)
        #     if "glob" not in name and "audio_proj" in name:
        #         name = name.replace("audio_proj", "multitalk_audio_proj")
        #     load_device = device
        #     if "vace_blocks." in name:
        #         try:
        #             vace_block_idx = int(name.split("vace_blocks.")[1].split(".")[0])
        #         except Exception:
        #             vace_block_idx = None
        #     elif "blocks." in name and "face" not in name:
        #         try:
        #             block_idx = int(name.split("blocks.")[1].split(".")[0])
        #         except Exception:
        #             block_idx = None

        #     if block_swap_args is not None:
        #         if block_idx is not None:
        #             if block_idx >= len(transformer.blocks) - block_swap_args.get("blocks_to_swap", 0):
        #                 load_device = offload_device
        #         elif vace_block_idx is not None:
        #             if vace_block_idx >= len(transformer.vace_blocks) - block_swap_args.get("vace_blocks_to_swap", 0):
        #                 load_device = offload_device
                        
        #     is_gguf_quant = tensor.tensor_type not in [GGMLQuantizationType.F32, GGMLQuantizationType.F16]
        #     weights = torch.from_numpy(tensor.data.copy()).to(load_device)
        #     sd[name] = GGUFParameter(weights, quant_type=tensor.tensor_type) if is_gguf_quant else weights
        # sd.update(extra_sd)
        # del all_tensors, extra_sd

        # if not getattr(transformer, "gguf_patched", False):
        #     transformer = _replace_with_gguf_linear(
        #         transformer, base_dtype, sd, patches=patcher.patches, compile_args=compile_args
        #     )
        #     transformer.gguf_patched = True
    else:
        logging.info("Using accelerate to load and assign model weights to device...")
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
        # if cnt % 100 == 0:
        #     pbar.update(100)

    #[print(name, param.device, param.dtype) for name, param in transformer.named_parameters()]

    # pbar.update_absolute(0)

def patch_control_lora(transformer, device):
    logging.info("Control-LoRA detected, patching model...")

    in_cls = transformer.patch_embedding.__class__ # nn.Conv3d
    old_in_dim = transformer.in_dim # 16
    new_in_dim = 32
    
    new_in = in_cls(
        new_in_dim,
        transformer.patch_embedding.out_channels,
        transformer.patch_embedding.kernel_size,
        transformer.patch_embedding.stride,
        transformer.patch_embedding.padding,
    ).to(device=device, dtype=torch.float32)
    
    new_in.weight.zero_()
    new_in.bias.zero_()
    
    new_in.weight[:, :old_in_dim].copy_(transformer.patch_embedding.weight)
    new_in.bias.copy_(transformer.patch_embedding.bias)
    
    transformer.patch_embedding = new_in
    transformer.expanded_patch_embedding = new_in

def patch_stand_in_lora(transformer, lora_sd, transformer_load_device, base_dtype, lora_strength):
    if "diffusion_model.blocks.0.self_attn.q_loras.down.weight" in lora_sd:                        
        logging.info("Stand-In LoRA detected")
        for block in transformer.blocks:
            block.self_attn.q_loras = LoRALinearLayer(transformer.dim, transformer.dim, rank=128, device=transformer_load_device, dtype=base_dtype, strength=lora_strength)
            block.self_attn.k_loras = LoRALinearLayer(transformer.dim, transformer.dim, rank=128, device=transformer_load_device, dtype=base_dtype, strength=lora_strength)
            block.self_attn.v_loras = LoRALinearLayer(transformer.dim, transformer.dim, rank=128, device=transformer_load_device, dtype=base_dtype, strength=lora_strength)
            for lora in [block.self_attn.q_loras, block.self_attn.k_loras, block.self_attn.v_loras]:
                for param in lora.parameters():
                    param.requires_grad = False
        for name, param in transformer.named_parameters():
            if "lora" in name:
                param.data.copy_(lora_sd["diffusion_model." + name].to(param.device, dtype=param.dtype))

def add_lora_weights(patcher, lora, base_dtype, merge_loras=False):
    unianimate_sd = None
    control_lora=False
    #spacepxl's control LoRA patch
    for l in lora:
        logging.info(f"Loading LoRA: {l['name']} with strength: {l['strength']}")
        lora_path = l["path"]
        lora_strength = l["strength"]
        if isinstance(lora_strength, list):
            if merge_loras:
                raise ValueError("LoRA strength should be a single value when merge_loras=True")
            patcher.model.diffusion_model.lora_scheduling_enabled = True
        if lora_strength == 0:
            logging.warning(f"LoRA {lora_path} has strength 0, skipping...")
            continue
        lora_sd = ModelUtils.load_torch_file(lora_path, safe_load=True)
        # if "dwpose_embedding.0.weight" in lora_sd: #unianimate
        #     from .wanvideo.unianimate.nodes import update_transformer
        #     logging.info("Unianimate LoRA detected, patching model...")
        #     patcher.model.diffusion_model, unianimate_sd = update_transformer(patcher.model.diffusion_model, lora_sd)

        lora_sd = standardize_lora_key_format(lora_sd)

        if l["blocks"]:
            lora_sd = filter_state_dict_by_blocks(lora_sd, l["blocks"], l.get("layer_filter", []))

        # Filter out any LoRA keys containing 'img' if the base model state_dict has no 'img' keys
        #if not any('img' in k for k in sd.keys()):
        #    lora_sd = {k: v for k, v in lora_sd.items() if 'img' not in k}
        
        if "diffusion_model.patch_embedding.lora_A.weight" in lora_sd:
            control_lora = True
        #stand-in LoRA patch
        if "diffusion_model.blocks.0.self_attn.q_loras.down.weight" in lora_sd:
            pass
            # patch_stand_in_lora(patcher.model.diffusion_model, lora_sd, device, base_dtype, lora_strength)
        # normal LoRA patch
        else:
            patcher, _ = load_lora_for_models(patcher, None, lora_sd, lora_strength, 0)
        
        del lora_sd
    return patcher, control_lora, unianimate_sd

class OriginalWanTransformerModel(FromOriginalModelMixin):

    def __init__(
        self
    ):
        super().__init__()

    @classmethod
    def from_single_file(cls, model=None, **kwargs):
        lora_select = WanVideoLoraSelect()
        lora_result = lora_select.getlorapath(lora="/data/gaozhenyu/studio/ComfyUI/models/loras/lightx2v_I2V_14B_480p_cfg_step_distill_rank64_bf16.safetensors", strength=1.0, low_mem_load=False, merge_loras=False, unique_id=0)
        infinite_talk_model = OriginalInfiniteTalkModel.from_single_file("/data/gaozhenyu/studio/ComfyUI/models/diffusion_models/Wan2_1-InfiniteTalk-Single_fp8_e4m3fn_scaled_KJ.safetensors")
        base_precision = "fp16_fast"
        load_device = "offload_device"
        quantization = "disabled"
        compile_args = None
        attention_mode = "sageattn"
        block_swap_args = {
            "blocks_to_swap": 40,
            "offload_img_emb": False,
            "offload_txt_emb": False,
            "use_non_blocking": True,
            "vace_blocks_to_swap": 0,
            "prefetch_blocks": 1,
            "block_swap_debug": False
        }
        lora = lora_result[0]
        vram_management_args = None
        extra_model = None
        vace_model = None
        fantasytalking_model = None
        multitalk_model = infinite_talk_model
        fantasyportrait_model = None
        rms_norm_function = None
                  
        device = torch.device(torch.cuda.current_device())
        offload_device = torch.device("cpu")

        assert not (vram_management_args is not None and block_swap_args is not None), "Can't use both block_swap_args and vram_management_args at the same time"
        if vace_model is not None:
            extra_model = vace_model
        lora_low_mem_load = merge_loras = False
        if lora is not None:
            merge_loras = any(l.get("merge_loras", True) for l in lora)
            lora_low_mem_load = any(l.get("low_mem_load", False) for l in lora)

        transformer = None
        mm.unload_all_models()
        mm.cleanup_models()
        mm.soft_empty_cache()

        if "sage" in attention_mode:
            try:
                from sageattention import sageattn
            except Exception as e:
                raise ValueError(f"Can't import SageAttention: {str(e)}")

        gguf = False
 
        transformer_load_device = device if load_device == "main_device" else offload_device
        if lora is not None and not merge_loras:
            transformer_load_device = offload_device

        base_dtype = {"fp8_e4m3fn": torch.float8_e4m3fn, "fp8_e4m3fn_fast": torch.float8_e4m3fn, "bf16": torch.bfloat16, "fp16": torch.float16, "fp16_fast": torch.float16, "fp32": torch.float32}[base_precision]
        
        if base_precision == "fp16_fast":
            if hasattr(torch.backends.cuda.matmul, "allow_fp16_accumulation"):
                torch.backends.cuda.matmul.allow_fp16_accumulation = True
            else:
                raise ValueError("torch.backends.cuda.matmul.allow_fp16_accumulation is not available in this version of torch, requires torch 2.7.0.dev2025 02 26 nightly minimum currently")
        else:
            try:
                if hasattr(torch.backends.cuda.matmul, "allow_fp16_accumulation"):
                    torch.backends.cuda.matmul.allow_fp16_accumulation = False
            except:
                pass
 

        model_path = model

        gguf_reader = None
        if not gguf:
            sd = ModelUtils.load_torch_file(model_path, device=transformer_load_device, safe_load=True)

        # Ovi
        extra_audio_model = False
        if any(key.startswith("video_model.") for key in sd.keys()):
            sd = {key.replace("video_model.", "", 1).replace("modulation.modulation", "modulation"): value for key, value in sd.items()}
        if any(key.startswith("audio_model.") for key in sd.keys()) and any(key.startswith("blocks.") for key in sd.keys()):
            extra_audio_model = True
                

        is_wananimate = "pose_patch_embedding.weight" in sd
        # rename WanAnimate face fuser block keys to insert into main blocks instead
        if is_wananimate:
            for key in list(sd.keys()):
                new_key = rename_fuser_block(key)
                if new_key != key:
                    sd[new_key] = sd.pop(key)

        is_scaled_fp8 = False

        if quantization == "disabled":
            for k, v in sd.items():
                if isinstance(v, torch.Tensor):
                    if v.dtype == torch.float8_e4m3fn:
                        quantization = "fp8_e4m3fn"
                        if "scaled_fp8" in sd:
                            is_scaled_fp8 = True
                            quantization = "fp8_e4m3fn_scaled"
                        break
                    elif v.dtype == torch.float8_e5m2:
                        quantization = "fp8_e5m2"
                        if "scaled_fp8" in sd:
                            is_scaled_fp8 = True
                            quantization = "fp8_e5m2_scaled"
                        break

        scale_weights = {}
        if "fp8" in quantization:
            for k, v in sd.items():
                if k.endswith(".scale_weight"):
                    is_scaled_fp8 = True
                    break

        if is_scaled_fp8 and "scaled" not in quantization:
            quantization = quantization + "_scaled"

        if torch.cuda.is_available():
            #only warning for now
            major, minor = torch.cuda.get_device_capability(device)
            logging.info(f"CUDA Compute Capability: {major}.{minor}")
            if compile_args is not None and "e4" in quantization and (major, minor) < (8, 9):
                logging.warning("WARNING: Torch.compile with fp8_e4m3fn weights on CUDA compute capability < 8.9 may not be supported. Please use fp8_e5m2, GGUF or higher precision instead, or check the latest triton version that adds support for older architectures https://github.com/woct0rdho/triton-windows/releases/tag/v3.5.0-windows.post21")

        if is_scaled_fp8 and "scaled" not in quantization:
            raise ValueError("The model is a scaled fp8 model, please set quantization to '_scaled'")
        if not is_scaled_fp8 and "scaled" in quantization:
            raise ValueError("The model is not a scaled fp8 model, please disable '_scaled' in quantization")

        if "vace_blocks.0.after_proj.weight" in sd and not "patch_embedding.weight" in sd:
            raise ValueError("You are attempting to load a VACE module as a WanVideo model, instead you should use the vace_model input and matching T2V base model")

        first_key = next(iter(sd))
        if first_key.startswith("audio_model.") and not extra_audio_model:
            sd = {key.replace("audio_model.", "", 1): value for key, value in sd.items()}
        if first_key.startswith("model.diffusion_model."):
            sd = {key.replace("model.diffusion_model.", "", 1): value for key, value in sd.items()}
        elif first_key.startswith("model."):
            sd = {key.replace("model.", "", 1): value for key, value in sd.items()}

        if "patch_embedding.weight" in sd:
            dim = sd["patch_embedding.weight"].shape[0]
            in_channels = sd["patch_embedding.weight"].shape[1]
        elif "patch_embedding.0.weight" in sd:
            dim = sd["patch_embedding.0.weight"].shape[0]
            in_channels = sd["patch_embedding.0.weight"].shape[1]
        else:
            raise ValueError("No patch_embedding weight found, is the selected model a full WanVideo model?")

        in_features = sd["blocks.0.self_attn.k.weight"].shape[1]
        out_features = sd["blocks.0.self_attn.k.weight"].shape[0]
        logging.info(f"Detected model in_channels: {in_channels}")

        if "blocks.0.ffn.0.bias" in sd:
            ffn_dim = sd["blocks.0.ffn.0.bias"].shape[0]
            ffn2_dim = sd["blocks.0.ffn.2.weight"].shape[1]
        else:
            ffn_dim = sd["blocks.0.ffn.w1.weight"].shape[0]
            ffn2_dim = sd["blocks.0.ffn.w1.weight"].shape[1]

        patch_size=(1, 2, 2)
        if "patch_embedding.0.weight" in sd:
            patch_size = [1]

        is_humo = "audio_proj.audio_proj_glob_1.layer.weight" in sd
        is_wananimate = "pose_patch_embedding.weight" in sd

        #lynx
        lynx_ip_layers = lynx_ref_layers = None
        if "blocks.0.self_attn.ref_adapter.to_k_ref.weight" in sd:
            logging.info("Lynx full reference adapter detected")
            lynx_ref_layers = "full"
        if "blocks.0.cross_attn.ip_adapter.registers" in sd:
            logging.info("Lynx full IP adapter detected")
            lynx_ip_layers = "full"
        elif "blocks.0.cross_attn.ip_adapter.to_v_ip.weight" in sd:
            logging.info("Lynx lite IP adapter detected")
            lynx_ip_layers = "lite"

        model_type = "t2v"
        if "audio_injector.injector.0.k.weight" in sd:
            model_type = "s2v"
        elif not "text_embedding.0.weight" in sd:
            model_type = "no_cross_attn" #minimaxremover
        elif "model_type.Wan2_1-FLF2V-14B-720P" in sd or "img_emb.emb_pos" in sd or "flf2v" in model.lower():
            model_type = "fl2v"
        elif in_channels in [36, 48]:
            if "blocks.0.cross_attn.k_img.weight" not in sd:
                model_type = "t2v"
            else:
                model_type = "i2v"
        elif in_channels == 16:
            model_type = "t2v"
        elif "control_adapter.conv.weight" in sd:
            model_type = "t2v"

        out_dim = 16
        if dim == 5120: #14B
            num_heads = 40
            num_layers = 40
        elif dim == 3072: #5B
            num_heads = 24
            num_layers = 30
            out_dim = 48
            model_type = "t2v" #5B no img crossattn
        elif dim == 4096: #longcat
            num_heads = 32
            num_layers = 48
        else: #1.3B
            num_heads = 12
            num_layers = 30

        vace_layers, vace_in_dim = None, None
        if "vace_blocks.0.after_proj.weight" in sd:
            if in_channels != 16:
                raise ValueError("VACE only works properly with T2V models.")
            model_type = "t2v"
            if dim == 5120:
                vace_layers = [0, 5, 10, 15, 20, 25, 30, 35]
            else:
                vace_layers = [0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28]
            vace_in_dim = 96

        logging.info(f"Model cross attention type: {model_type}, num_heads: {num_heads}, num_layers: {num_layers}")

        teacache_coefficients_map = {
            "1_3B": {
                "e": [2.39676752e+03, -1.31110545e+03, 2.01331979e+02, -8.29855975e+00, 1.37887774e-01],
                "e0": [-5.21862437e+04, 9.23041404e+03, -5.28275948e+02, 1.36987616e+01, -4.99875664e-02],
            },
            "14B": {
                "e": [-5784.54975374, 5449.50911966, -1811.16591783, 256.27178429, -13.02252404],
                "e0": [-3.03318725e+05, 4.90537029e+04, -2.65530556e+03, 5.87365115e+01, -3.15583525e-01],
            },
            "i2v_480": {
                "e": [-3.02331670e+02, 2.23948934e+02, -5.25463970e+01, 5.87348440e+00, -2.01973289e-01],
                "e0": [2.57151496e+05, -3.54229917e+04, 1.40286849e+03, -1.35890334e+01, 1.32517977e-01],
            },
            "i2v_720":{
                "e": [-114.36346466, 65.26524496, -18.82220707, 4.91518089, -0.23412683],
                "e0": [8.10705460e+03, 2.13393892e+03, -3.72934672e+02, 1.66203073e+01, -4.17769401e-02],
            },
            # Placeholders until TeaCache for Wan2.2 is obtained
            "14B_2.2": {
                "e": [-5784.54975374, 5449.50911966, -1811.16591783, 256.27178429, -13.02252404],
                "e0": [-3.03318725e+05, 4.90537029e+04, -2.65530556e+03, 5.87365115e+01, -3.15583525e-01],
            },
            "i2v_14B_2.2":{
                "e": [-114.36346466, 65.26524496, -18.82220707, 4.91518089, -0.23412683],
                "e0": [8.10705460e+03, 2.13393892e+03, -3.72934672e+02, 1.66203073e+01, -4.17769401e-02],
            },
        }

        magcache_ratios_map = {
            "1_3B": np.array([1.0]*2+[1.0124, 1.02213, 1.00166, 1.0041, 0.99791, 1.00061, 0.99682, 0.99762, 0.99634, 0.99685, 0.99567, 0.99586, 0.99416, 0.99422, 0.99578, 0.99575, 0.9957, 0.99563, 0.99511, 0.99506, 0.99535, 0.99531, 0.99552, 0.99549, 0.99541, 0.99539, 0.9954, 0.99536, 0.99489, 0.99485, 0.99518, 0.99514, 0.99484, 0.99478, 0.99481, 0.99479, 0.99415, 0.99413, 0.99419, 0.99416, 0.99396, 0.99393, 0.99388, 0.99386, 0.99349, 0.99349, 0.99309, 0.99304, 0.9927, 0.9927, 0.99228, 0.99226, 0.99171, 0.9917, 0.99137, 0.99135, 0.99068, 0.99063, 0.99005, 0.99003, 0.98944, 0.98942, 0.98849, 0.98849, 0.98758, 0.98757, 0.98644, 0.98643, 0.98504, 0.98503, 0.9836, 0.98359, 0.98202, 0.98201, 0.97977, 0.97978, 0.97717, 0.97718, 0.9741, 0.97411, 0.97003, 0.97002, 0.96538, 0.96541, 0.9593, 0.95933, 0.95086, 0.95089, 0.94013, 0.94019, 0.92402, 0.92414, 0.90241, 0.9026, 0.86821, 0.86868, 0.81838, 0.81939]),
            "14B": np.array([1.0]*2+[1.02504, 1.03017, 1.00025, 1.00251, 0.9985, 0.99962, 0.99779, 0.99771, 0.9966, 0.99658, 0.99482, 0.99476, 0.99467, 0.99451, 0.99664, 0.99656, 0.99434, 0.99431, 0.99533, 0.99545, 0.99468, 0.99465, 0.99438, 0.99434, 0.99516, 0.99517, 0.99384, 0.9938, 0.99404, 0.99401, 0.99517, 0.99516, 0.99409, 0.99408, 0.99428, 0.99426, 0.99347, 0.99343, 0.99418, 0.99416, 0.99271, 0.99269, 0.99313, 0.99311, 0.99215, 0.99215, 0.99218, 0.99215, 0.99216, 0.99217, 0.99163, 0.99161, 0.99138, 0.99135, 0.98982, 0.9898, 0.98996, 0.98995, 0.9887, 0.98866, 0.98772, 0.9877, 0.98767, 0.98765, 0.98573, 0.9857, 0.98501, 0.98498, 0.9838, 0.98376, 0.98177, 0.98173, 0.98037, 0.98035, 0.97678, 0.97677, 0.97546, 0.97543, 0.97184, 0.97183, 0.96711, 0.96708, 0.96349, 0.96345, 0.95629, 0.95625, 0.94926, 0.94929, 0.93964, 0.93961, 0.92511, 0.92504, 0.90693, 0.90678, 0.8796, 0.87945, 0.86111, 0.86189]),
            "i2v_480": np.array([1.0]*2+[0.98783, 0.98993, 0.97559, 0.97593, 0.98311, 0.98319, 0.98202, 0.98225, 0.9888, 0.98878, 0.98762, 0.98759, 0.98957, 0.98971, 0.99052, 0.99043, 0.99383, 0.99384, 0.98857, 0.9886, 0.99065, 0.99068, 0.98845, 0.98847, 0.99057, 0.99057, 0.98957, 0.98961, 0.98601, 0.9861, 0.98823, 0.98823, 0.98756, 0.98759, 0.98808, 0.98814, 0.98721, 0.98724, 0.98571, 0.98572, 0.98543, 0.98544, 0.98157, 0.98165, 0.98411, 0.98413, 0.97952, 0.97953, 0.98149, 0.9815, 0.9774, 0.97742, 0.97825, 0.97826, 0.97355, 0.97361, 0.97085, 0.97087, 0.97056, 0.97055, 0.96588, 0.96587, 0.96113, 0.96124, 0.9567, 0.95681, 0.94961, 0.94969, 0.93973, 0.93988, 0.93217, 0.93224, 0.91878, 0.91896, 0.90955, 0.90954, 0.92617, 0.92616]),
            "i2v_720": np.array([1.0]*2+[0.99428, 0.99498, 0.98588, 0.98621, 0.98273, 0.98281, 0.99018, 0.99023, 0.98911, 0.98917, 0.98646, 0.98652, 0.99454, 0.99456, 0.9891, 0.98909, 0.99124, 0.99127, 0.99102, 0.99103, 0.99215, 0.99212, 0.99515, 0.99515, 0.99576, 0.99572, 0.99068, 0.99072, 0.99097, 0.99097, 0.99166, 0.99169, 0.99041, 0.99042, 0.99201, 0.99198, 0.99101, 0.99101, 0.98599, 0.98603, 0.98845, 0.98844, 0.98848, 0.98851, 0.98862, 0.98857, 0.98718, 0.98719, 0.98497, 0.98497, 0.98264, 0.98263, 0.98389, 0.98393, 0.97938, 0.9794, 0.97535, 0.97536, 0.97498, 0.97499, 0.973, 0.97301, 0.96827, 0.96828, 0.96261, 0.96263, 0.95335, 0.9534, 0.94649, 0.94655, 0.93397, 0.93414, 0.91636, 0.9165, 0.89088, 0.89109, 0.8679, 0.86768]),
            "14B_2.2": np.array([1.0]*2+[0.99505, 0.99389, 0.99441, 0.9957, 0.99558, 0.99551, 0.99499, 0.9945, 0.99534, 0.99548, 0.99468, 0.9946, 0.99463, 0.99458, 0.9946, 0.99453, 0.99408, 0.99404, 0.9945, 0.99441, 0.99409, 0.99398, 0.99403, 0.99397, 0.99382, 0.99377, 0.99349, 0.99343, 0.99377, 0.99378, 0.9933, 0.99328, 0.99303, 0.99301, 0.99217, 0.99216, 0.992, 0.99201, 0.99201, 0.99202, 0.99133, 0.99132, 0.99112, 0.9911, 0.99155, 0.99155, 0.98958, 0.98957, 0.98959, 0.98958, 0.98838, 0.98835, 0.98826, 0.98825, 0.9883, 0.98828, 0.98711, 0.98709, 0.98562, 0.98561, 0.98511, 0.9851, 0.98414, 0.98412, 0.98284, 0.98282, 0.98104, 0.98101, 0.97981, 0.97979, 0.97849, 0.97849, 0.97557, 0.97554, 0.97398, 0.97395, 0.97171, 0.97166, 0.96917, 0.96913, 0.96511, 0.96507, 0.96263, 0.96257, 0.95839, 0.95835, 0.95483, 0.95475, 0.94942, 0.94936, 0.9468, 0.94678, 0.94583, 0.94594, 0.94843, 0.94872, 0.96949, 0.97015]),
            "i2v_14B_2.2": np.array([1.0]*2+[0.99512, 0.99559, 0.99559, 0.99561, 0.99595, 0.99577, 0.99512, 0.99512, 0.99546, 0.99534, 0.99543, 0.99531, 0.99496, 0.99491, 0.99504, 0.99499, 0.99444, 0.99449, 0.99481, 0.99481, 0.99435, 0.99435, 0.9943, 0.99431, 0.99411, 0.99406, 0.99373, 0.99376, 0.99413, 0.99405, 0.99363, 0.99359, 0.99335, 0.99331, 0.99244, 0.99243, 0.99229, 0.99229, 0.99239, 0.99236, 0.99163, 0.9916, 0.99149, 0.99151, 0.99191, 0.99192, 0.9898, 0.98981, 0.9899, 0.98987, 0.98849, 0.98849, 0.98846, 0.98846, 0.98861, 0.98861, 0.9874, 0.98738, 0.98588, 0.98589, 0.98539, 0.98534, 0.98444, 0.98439, 0.9831, 0.98309, 0.98119, 0.98118, 0.98001, 0.98, 0.97862, 0.97859, 0.97555, 0.97558, 0.97392, 0.97388, 0.97152, 0.97145, 0.96871, 0.9687, 0.96435, 0.96434, 0.96129, 0.96127, 0.95639, 0.95638, 0.95176, 0.95175, 0.94446, 0.94452, 0.93972, 0.93974, 0.93575, 0.9359, 0.93537, 0.93552, 0.96655, 0.96616]),
        }

        model_variant = "14B" #default to this
        if model_type == "i2v" or model_type == "fl2v":
            if "480" in model or "fun" in model.lower() or "a2" in model.lower() or "540" in model: #just a guess for the Fun model for now...
                model_variant = "i2v_480"
            elif "720" in model:
                model_variant = "i2v_720"
        elif model_type == "t2v":
            model_variant = "14B"
            
        if dim == 1536:
            model_variant = "1_3B"
        if dim == 3072:
            logging.info(f"5B model detected, no Teacache or MagCache coefficients available, consider using EasyCache for this model")
        
        if "high" in model.lower() or "low" in model.lower():
            if "i2v" in model.lower():
                model_variant = "i2v_14B_2.2"
            else:
                model_variant = "14B_2.2"
        
        logging.info(f"Model variant detected: {model_variant}")
        
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
            "teacache_coefficients": teacache_coefficients_map[model_variant],
            "magcache_ratios": magcache_ratios_map[model_variant],
            "vace_layers": vace_layers,
            "vace_in_dim": vace_in_dim,
            "inject_sample_info": True if "fps_embedding.weight" in sd else False,
            "add_ref_conv": True if "ref_conv.weight" in sd else False,
            "in_dim_ref_conv": sd["ref_conv.weight"].shape[1] if "ref_conv.weight" in sd else None,
            "add_control_adapter": True if "control_adapter.conv.weight" in sd else False,
            "use_motion_attn": True if "blocks.0.motion_attn.k.weight" in sd else False,
            "enable_adain": True if "audio_injector.injector_adain_layers.0.linear.weight" in sd else False,
            "cond_dim": sd["cond_encoder.weight"].shape[1] if "cond_encoder.weight" in sd else 0,
            "zero_timestep": model_type == "s2v",
            "humo_audio": is_humo,
            "is_wananimate": is_wananimate,
            "rms_norm_function": rms_norm_function,
            "lynx_ip_layers": lynx_ip_layers,
            "lynx_ref_layers": lynx_ref_layers,
            "is_longcat": dim == 4096,

        }

        with init_empty_weights():
            transformer = WanModel(**TRANSFORMER_CONFIG).eval()

        if multitalk_model is not None:
            multitalk_model_type = multitalk_model.get("model_type", "MultiTalk")
            logging.info(f"{multitalk_model_type} detected, patching model...")

            multitalk_model_path = multitalk_model["model_path"]
            if multitalk_model_path.endswith(".gguf") and not gguf:
                raise ValueError("Multitalk/InfiniteTalk model is a GGUF model, main model also has to be a GGUF model.")
            if "scaled" in multitalk_model and gguf:
                raise ValueError("fp8 scaled Multitalk/InfiniteTalk model can't be used with GGUF main model")

            # init audio module
            from .wanvideo.multitalk.multitalk import SingleStreamMultiAttention
            from .wanvideo.wanvideo.modules.model import WanLayerNorm
               
            for block in transformer.blocks:
                with init_empty_weights():
                    block.norm_x = WanLayerNorm(dim, transformer.eps, elementwise_affine=True)
                    block.audio_cross_attn = SingleStreamMultiAttention(
                            dim=dim,
                            encoder_hidden_states_dim=768,
                            num_heads=num_heads,
                        qkv_bias=True,
                        class_range=24,
                        class_interval=4,
                        attention_mode=attention_mode,
                    )
            transformer.multitalk_audio_proj = multitalk_model["proj_model"]
            transformer.multitalk_model_type = multitalk_model_type

            extra_model_path = multitalk_model["model_path"]
            extra_sd = {}
            if multitalk_model_path.endswith(".gguf"):
                pass
            else:
                extra_sd_temp = ModelUtils.load_torch_file(extra_model_path, device=transformer_load_device, safe_load=True)
                
            for k, v in extra_sd_temp.items():
                extra_sd[k.replace("audio_proj.", "multitalk_audio_proj.")] = v
                
            sd.update(extra_sd)
            del extra_sd
        
        # Bindweave text_projection
        if "text_projection.0.weight" in sd:
            logging.info("Bindweave model detected, adding text_projection to the model")
            text_dim = sd["text_projection.0.weight"].shape[0]
            transformer.text_projection = nn.Sequential(nn.Linear(sd["text_projection.0.weight"].shape[1], text_dim), nn.GELU(approximate='tanh'), nn.Linear(text_dim, text_dim))
       
        latent_format=Wan22 if dim == 3072 else Wan21
        comfy_model = WanVideoModel(
            WanVideoModelConfig(base_dtype, latent_format=latent_format),
            model_type=sglang.comfy.model_base.ModelType.FLOW,
            device=device,
        )

        comfy_model.diffusion_model = transformer
        comfy_model.load_device = transformer_load_device
        patcher = sglang.comfy.model_patcher.ModelPatcher(comfy_model, device, offload_device)
        patcher.model.is_patched = False
        
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
        
        params_to_keep = {"norm", "bias", "time_in", "patch_embedding", "time_", "img_emb", "modulation", "text_embedding", "adapter", "add", "ref_conv", "audio_proj"}

        control_lora = False

        if not merge_loras and control_lora:
            logging.warning("Control-LoRA patching is only supported with merge_loras=True")

        if lora is not None:
            patcher, control_lora, unianimate_sd = add_lora_weights(patcher, lora, base_dtype, merge_loras=merge_loras)
            if unianimate_sd is not None:
                logging.info("Merging UniAnimate weights to the model...")
                sd.update(unianimate_sd)
                del unianimate_sd
      
        if not gguf:
            if lora is not None and merge_loras:
                pass
            elif "scaled" in quantization or lora is not None:
                transformer = _replace_linear(transformer, base_dtype, sd, scale_weights=scale_weights, compile_args=compile_args)
                transformer.patched_linear = True

        patcher.model["base_dtype"] = base_dtype
        patcher.model["weight_dtype"] = weight_dtype
        patcher.model["base_path"] = model_path
        patcher.model["model_name"] = model
        patcher.model["quantization"] = quantization
        patcher.model["auto_cpu_offload"] = True if vram_management_args is not None else False
        patcher.model["control_lora"] = control_lora
        patcher.model["compile_args"] = compile_args
        patcher.model["gguf_reader"] = gguf_reader
        patcher.model["fp8_matmul"] = "fast" in quantization
        patcher.model["scale_weights"] = scale_weights
        patcher.model["sd"] = sd
        patcher.model["lora"] = lora

        if 'transformer_options' not in patcher.model_options:
            patcher.model_options['transformer_options'] = {}
        patcher.model_options["transformer_options"]["block_swap_args"] = block_swap_args
        patcher.model_options["transformer_options"]["merge_loras"] = merge_loras

        for model in mm.current_loaded_models:
            if model._model() == patcher:
                mm.current_loaded_models.remove(model)
        return patcher


EntryClass = [OriginalWanTransformerModel]
