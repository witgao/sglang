import os, gc, math, copy
import torch
import numpy as np
from tqdm import tqdm
import inspect
import logging
from .wanvideo.modules.model import rope_params
from .custom_linear import set_lora_params
from .wanvideo.schedulers import get_scheduler
from .multitalk.multitalk import timestep_transform, add_noise
from .utils import(log, print_memory, fourier_filter, optimized_scale, setup_radial_attention,
                   compile_model, dict_to_device, tangential_projection, get_raag_guidance)
# from .nodes_model_loading import load_weights
from contextlib import nullcontext

# from sglang.comfy import model_management as mm
from sglang.comfy.utils import load_torch_file

script_directory = os.path.dirname(os.path.abspath(__file__))

device = torch.device(torch.cuda.current_device())
offload_device = torch.device("cpu")

rope_functions = ["default", "comfy", "comfy_chunked"]

VAE_STRIDE = (4, 8, 8)
PATCH_SIZE = (1, 2, 2)

from .utils import set_module_tensor_to_device

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
    
class MetaParameter(torch.nn.Parameter):
    def __new__(cls, dtype, quant_type=None):
        data = torch.empty(0, dtype=dtype)
        self = torch.nn.Parameter(data, requires_grad=False)
        self.quant_type = quant_type
        return self

def offload_transformer(transformer):    
    transformer.teacache_state.clear_all()
    transformer.magcache_state.clear_all()
    transformer.easycache_state.clear_all()

    if transformer.patched_linear:
        for name, param in transformer.named_parameters():
            if "loras" in name or "controlnet" in name:
                continue
            module = transformer
            subnames = name.split('.')
            for subname in subnames[:-1]:
                module = getattr(module, subname)
            attr_name = subnames[-1]
            if param.data.is_floating_point():
                meta_param = torch.nn.Parameter(torch.empty_like(param.data, device='meta'), requires_grad=False)
                setattr(module, attr_name, meta_param)
            elif isinstance(param.data, GGUFParameter):
                quant_type = getattr(param, 'quant_type', None)
                setattr(module, attr_name, MetaParameter(param.data.dtype, quant_type))
            else:
                pass
    else:
        transformer.to(offload_device)

    for block in transformer.blocks:
        block.kv_cache = None
        if transformer.audio_model is not None and hasattr(block, 'audio_block'):
            block.audio_block = None

    # mm.soft_empty_cache()
    gc.collect()


def init_blockswap(transformer, block_swap_args, model):
    if not transformer.patched_linear:
        if block_swap_args is not None:
            for name, param in transformer.named_parameters():
                if "block" not in name or "control_adapter" in name or "face" in name:
                    param.data = param.data.to(device)
                elif block_swap_args["offload_txt_emb"] and "txt_emb" in name:
                    param.data = param.data.to(offload_device)
                elif block_swap_args["offload_img_emb"] and "img_emb" in name:
                    param.data = param.data.to(offload_device)

            transformer.block_swap(
                block_swap_args["blocks_to_swap"] - 1 ,
                block_swap_args["offload_txt_emb"],
                block_swap_args["offload_img_emb"],
                vace_blocks_to_swap = block_swap_args.get("vace_blocks_to_swap", None),
            )
        elif model["auto_cpu_offload"]:
            for module in transformer.modules():
                if hasattr(module, "offload"):
                    module.offload()
                if hasattr(module, "onload"):
                    module.onload()
            for block in transformer.blocks:
                block.modulation = torch.nn.Parameter(block.modulation.to(device))
            transformer.head.modulation = torch.nn.Parameter(transformer.head.modulation.to(device))
        else:
            transformer.to(device)

class WanVideoSampler:

    def process(self, model, image_embeds, shift, steps, cfg, seed, scheduler, riflex_freq_index, text_embeds=None,
        force_offload=True, samples=None, feta_args=None, denoise_strength=1.0, context_options=None,
        cache_args=None, teacache_args=None, flowedit_args=None, batched_cfg=False, slg_args=None, rope_function="default", loop_args=None,
        experimental_args=None, sigmas=None, unianimate_poses=None, fantasytalking_embeds=None, uni3c_embeds=None, multitalk_embeds=None, freeinit_args=None, start_step=0, end_step=-1, add_noise_to_samples=False):

        patcher = model
        transformer = model.diffusion_model

        dtype = model["base_dtype"]
        weight_dtype = model["weight_dtype"]
        gguf_reader = None

        vae = image_embeds.get("vae", None)
        tiled_vae = False

        transformer_options = model["transformer_options"]

        block_swap_args = transformer_options.get("block_swap_args", None)

        is_5b = False
        vae_upscale_factor = 8


        multitalk_sampling = image_embeds.get("multitalk_sampling", False)

        text_embeds = dict_to_device(text_embeds, device)

        seed_g = torch.Generator(device=torch.device("cpu"))
        seed_g.manual_seed(seed)

        #region Scheduler
        if denoise_strength < 1.0:
            if start_step != 0:
                raise ValueError("start_step must be 0 when denoise_strength is used")
            start_step = steps - int(steps * denoise_strength) - 1
            add_noise_to_samples = True #for now to not break old workflows

        sample_scheduler = None
      
        sample_scheduler, timesteps,_,_ = get_scheduler(scheduler, steps, start_step, end_step, shift, device, transformer.dim, flowedit_args, denoise_strength, sigmas=None, log_timesteps=True)

        total_steps = steps
        steps = len(timesteps)

        scheduler_step_args = {"generator": seed_g}
        step_sig = inspect.signature(sample_scheduler.step)
        for arg in list(scheduler_step_args.keys()):
            if arg not in step_sig.parameters:
                scheduler_step_args.pop(arg)

        cfg = [cfg] * (steps + 1)

        control_latents = control_camera_latents = clip_fea = clip_fea_neg = end_image = recammaster = camera_embed = unianim_data = mocha_embeds = None
        fun_or_fl2v_model = has_ref = drop_last = False
        phantom_latents = fun_ref_image = ATI_tracks = None
        add_cond = attn_cond = attn_cond_neg  = None

        #I2V
        # image_cond = image_embeds.get("image_embeds", None)
        # if image_cond is not None:
            # control_embeds = image_embeds.get("control_embeds", None)
            # pass
        # else: #t2v
        target_shape = image_embeds.get("target_shape", None)
            # if target_shape is None:
            #     raise ValueError("Empty image embeds must be provided for T2V models")

        has_ref = False

        noise = torch.randn(
                    16,
                    target_shape[1],
                    target_shape[2], #todo make this smarter
                    target_shape[3], #todo make this smarter
                    dtype=torch.float32,
                    device=torch.device("cpu"),
                    generator=seed_g)
            
        seq_len = math.ceil((noise.shape[2] * noise.shape[3]) / 4 * noise.shape[1])
           
        # CLIP image features
        # clip_fea = image_embeds.get("clip_context", None)
        # if clip_fea is not None:
        #     clip_fea = clip_fea.to(dtype)
        # clip_fea_neg = image_embeds.get("negative_clip_context", None)
        # if clip_fea_neg is not None:
        #     clip_fea_neg = clip_fea_neg.to(dtype)

        num_frames = 0

        humo_reference_count = 0

        humo_image_cond = None

        pos_latent = neg_latent = None

        # region WanAnim inputs
        # wananimate_loop = image_embeds.get("looping", False)
        # if wananimate_loop and context_options is not None:
        #     raise Exception("context_options are not compatible or necessary with WanAnim looping, since it creates the video in a loop.")
        # wananim_face_pixels = image_embeds.get("face_pixels", None)
        # wananim_is_masked = image_embeds.get("is_masked", False)
        # if not wananimate_loop: # create zero face pixels if mask is provided without face pixels, as masking seems to require face input to work properly
        #     if wananim_face_pixels is None and wananim_is_masked:
        #         if context_options is None:
        #             wananim_face_pixels = torch.zeros(1, 3, num_frames-1, 512, 512, dtype=torch.float32, device=offload_device)
        #         else:
        #             wananim_face_pixels = torch.zeros(1, 3, context_options["context_frames"]-1, 512, 512, dtype=torch.float32, device=device)

        # if image_cond is None:
        image_cond = None
        has_ref = False

        latent_video_length = noise.shape[1]

        # FantasyTalking
        audio_proj = multitalk_audio_embeds = None
        audio_scale = 1.0
        # if fantasytalking_embeds is not None:
        #     log.info(f"Audio proj shape: {audio_proj.shape}")
        # elif multitalk_embeds is not None:
            # Handle single or multiple speaker embeddings
        audio_features_in = multitalk_embeds
            # if audio_features_in is None:
            #     multitalk_audio_embeds = None
            # else:
                # if isinstance(audio_features_in, list):
        multitalk_audio_embeds = [emb.to(device, dtype) for emb in audio_features_in]
                # else:
                #     # keep backward-compatibility with single tensor input
                #     multitalk_audio_embeds = [audio_features_in.to(device, dtype)]

        audio_scale = 1.0
        audio_cfg_scale = 1.0
        ref_target_masks = None
        if not isinstance(audio_cfg_scale, list):
            audio_cfg_scale = [audio_cfg_scale] * (steps + 1)

        shapes = [tuple(e.shape) for e in multitalk_audio_embeds]
        log.info(f"Multitalk audio features shapes (per speaker): {shapes}")

       
        latent = noise

        uni3c_data = uni3c_data_input = None

        # Bindweave
        qwenvl_embeds_pos = None
        qwenvl_embeds_neg = None

        # mm.unload_all_models()
        # mm.soft_empty_cache()
        gc.collect()

        #blockswap init
        init_blockswap(transformer, block_swap_args, model)

        # Initialize Cache if enabled
        previous_cache_states = None
        transformer.enable_teacache = transformer.enable_magcache = transformer.enable_easycache = False
        cache_args = teacache_args if teacache_args is not None else cache_args #for backward compatibility on old workflows
        
        if previous_cache_states is None:
            self.cache_state = [None, None]
            # if phantom_latents is not None:
            #     log.info(f"Phantom latents shape: {phantom_latents.shape}")
            #     self.cache_state = [None, None, None]
            self.cache_state_source = [None, None]
            self.cache_states_context = []

        # Skip layer guidance (SLG)
        # if slg_args is not None:
        #     pass
        # else:
        transformer.slg_blocks = None

        # Setup radial attention
        # if transformer.attention_mode == "radial_sage_attention":
        #     setup_radial_attention(transformer, transformer_options, latent, seq_len, latent_video_length, context_options=context_options)


        # Experimental args
        use_cfg_zero_star = use_tangential = use_fresca = bidirectional_sampling = use_tsr = False
        raag_alpha = 0.0
        transformer.video_attention_split_steps = []

        # RoPE base freq scaling as used with CineScale
        ntk_alphas = [1.0, 1.0, 1.0]

        # Stand-In
        standin_input = None

        freqs = None
        transformer.rope_embedder.k = None
        transformer.rope_embedder.num_frames = None
        d = transformer.dim // transformer.num_heads

        # if mocha_embeds is not None:
        #     pass
        # elif "default" in rope_function or bidirectional_sampling: # original RoPE
        #     freqs = torch.cat([
        #         rope_params(1024, d - 4 * (d // 6), L_test=latent_video_length, k=riflex_freq_index),
        #         rope_params(1024, 2 * (d // 6)),
        #         rope_params(1024, 2 * (d // 6))
        #     ],
        #     dim=1)
        # elif "comfy" in rope_function: # comfy's rope
        transformer.rope_embedder.k = riflex_freq_index
        transformer.rope_embedder.num_frames = latent_video_length

        transformer.rope_func = rope_function
        for block in transformer.blocks:
            block.rope_func = rope_function
        # if transformer.vace_layers is not None:
        #     for block in transformer.vace_blocks:
        #         block.rope_func = rope_function

        lynx_embeds = None

        ttm_start_step = 0

        #region model pred
        def predict_with_cfg(z, cfg_scale, positive_embeds, negative_embeds, timestep, idx, image_cond=None, clip_fea=None,
                             control_latents=None, vace_data=None, unianim_data=None, audio_proj=None, control_camera_latents=None,
                             add_cond=None, cache_state=None, context_window=None, multitalk_audio_embeds=None, fantasy_portrait_input=None, reverse_time=False,
                             mtv_motion_tokens=None, s2v_audio_input=None, s2v_ref_motion=None, s2v_motion_frames=[1, 0], s2v_pose=None,
                             humo_image_cond=None, humo_image_cond_neg=None, humo_audio=None, humo_audio_neg=None, wananim_pose_latents=None,
                             wananim_face_pixels=None, uni3c_data=None, latent_model_input_ovi=None, flashvsr_LQ_latent=None,):
            nonlocal transformer
            nonlocal audio_cfg_scale

            autocast_enabled = ("fp8" in model["quantization"] and not transformer.patched_linear)
            with torch.autocast(device_type=device.type, dtype=dtype) if autocast_enabled else nullcontext():

                if use_cfg_zero_star and (idx <= zero_star_steps) and use_zero_init:
                    return z*0, None

                nonlocal patcher
                current_step_percentage = idx / len(timesteps)
                control_lora_enabled = False
                image_cond_input = None
                if image_cond is not None:
                    if reverse_time: # Flip the image condition
                        image_cond_input = torch.cat([
                            torch.flip(image_cond[:4], dims=[1]),
                            torch.flip(image_cond[4:], dims=[1])
                        ]).to(z)
                    else:
                        image_cond_input = image_cond.to(z)

                phantom_ref = None

                add_cond_input = None

                if not multitalk_sampling and multitalk_audio_embeds is not None:
                    pass
                elif multitalk_sampling and multitalk_audio_embeds is not None:
                    multitalk_audio_input = multitalk_audio_embeds

                uni3c_data_input = uni3c_data

                humo_audio_input = humo_audio_input_neg = None

                if image_cond is not None:
                    self.noise_front_pad_num = image_cond_input.shape[1] - z.shape[1]
                    if self.noise_front_pad_num > 0:
                        pad = torch.zeros((z.shape[0], self.noise_front_pad_num, z.shape[2], z.shape[3]), dtype=z.dtype, device=z.device)
                        z = torch.concat([pad, z], dim=1)
                        nonlocal seq_len
                        seq_len = math.ceil((z.shape[2] * z.shape[3]) / 4 * z.shape[1])
                else:
                    self.noise_front_pad_num = 0

                base_params = {
                    'x': [z], # latent
                    'y': [image_cond_input] if image_cond_input is not None else None, # image cond
                    'clip_fea': clip_fea, # clip features
                    'seq_len': seq_len, # sequence length
                    'device': device, # main device
                    'freqs': freqs, # rope freqs
                    't': timestep, # current timestep
                    'is_uncond': False, # is unconditional
                    'current_step': idx, # current step
                    'current_step_percentage': current_step_percentage, # current step percentage
                    'last_step': len(timesteps) - 1 == idx, # is last step
                    'control_lora_enabled': control_lora_enabled, # control lora toggle for patch embed selection
                    'enhance_enabled': False, # enhance-a-video toggle
                    'camera_embed': camera_embed, # recammaster embedding
                    'unianim_data': unianim_data, # unianimate input
                    'fun_ref': None, # Fun model reference latent
                    'fun_camera': None, # Fun model camera embed
                    'audio_proj': audio_proj if fantasytalking_embeds is not None else None, # FantasyTalking audio projection
                    'audio_scale': audio_scale, # FantasyTalking audio scale
                    "uni3c_data": uni3c_data_input, # Uni3C input
                    "controlnet": None, # TheDenk's controlnet input
                    "add_cond": add_cond_input, # additional conditioning input
                    "nag_params": text_embeds.get("nag_params", {}), # normalized attention guidance
                    "nag_context": text_embeds.get("nag_prompt_embeds", None), # normalized attention guidance context
                    "multitalk_audio": multitalk_audio_input if multitalk_audio_embeds is not None else None, # Multi/InfiniteTalk audio input
                    "ref_target_masks": ref_target_masks if multitalk_audio_embeds is not None else None, # Multi/InfiniteTalk reference target masks
                    "inner_t": None, # inner timestep for EchoShot
                    "standin_input": standin_input, # Stand-in reference input
                    "fantasy_portrait_input": fantasy_portrait_input, # Fantasy portrait input
                    "phantom_ref": phantom_ref, # Phantom reference input
                    "reverse_time": reverse_time, # Reverse RoPE toggle
                    "ntk_alphas": ntk_alphas, # RoPE freq scaling values
                    "mtv_motion_tokens": None, # MTV-Crafter motion tokens
                    "mtv_motion_rotary_emb": None, # MTV-Crafter RoPE
                    "mtv_strength": 1.0, # MTV-Crafter scaling
                    "mtv_freqs": None, # MTV-Crafter extra RoPE freqs
                    "s2v_audio_input": s2v_audio_input, # official speech-to-video audio input
                    "s2v_ref_latent": None, # speech-to-video reference latent
                    "s2v_ref_motion": s2v_ref_motion, # speech-to-video reference motion latent
                    "s2v_audio_scale": 1.0, # speech-to-video audio scale
                    "s2v_pose": s2v_pose if s2v_pose is not None else None, # speech-to-video pose control
                    "s2v_motion_frames": s2v_motion_frames, # speech-to-video motion frames,
                    "humo_audio": humo_audio, # humo audio input
                    "humo_audio_scale": 1,
                    "wananim_pose_latents": wananim_pose_latents.to(device) if wananim_pose_latents is not None else None, # WanAnimate pose latents
                    "wananim_face_pixel_values": None, # WanAnimate face images
                    "wananim_pose_strength": 1.0,
                    "wananim_face_strength": 1.0,
                    "lynx_embeds": lynx_embeds, # Lynx face and reference embeddings
                    "x_ovi": [latent_model_input_ovi.to(z)] if latent_model_input_ovi is not None else None, # Audio latent model input for Ovi
                    "seq_len_ovi": None, # Audio latent model sequence length for Ovi
                    "ovi_negative_text_embeds": None, # Audio latent model negative text embeds for Ovi
                    "flashvsr_LQ_latent": flashvsr_LQ_latent, # FlashVSR LQ latent for upsampling
                    "flashvsr_strength": 1.0, # FlashVSR strength
                    "num_cond_latents": None,
                }

                batch_size = 1

                if not math.isclose(cfg_scale, 1.0):
                    if negative_embeds is None:
                        raise ValueError("Negative embeddings must be provided for CFG scale > 1.0")
                    if len(positive_embeds) > 1:
                        negative_embeds = negative_embeds * len(positive_embeds)

                try:
                    if not batched_cfg:
                        #conditional (positive) pass
                        if pos_latent is not None: # for humo
                            base_params['x'] = [torch.cat([z[:, :-humo_reference_count], pos_latent], dim=1)]
                        base_params["add_text_emb"] = qwenvl_embeds_pos.to(device) if qwenvl_embeds_pos is not None else None # QwenVL embeddings for Bindweave
                        noise_pred_cond, noise_pred_ovi, cache_state_cond = transformer(
                            context=positive_embeds,
                            pred_id=cache_state[0] if cache_state else None,
                            vace_data=vace_data, attn_cond=attn_cond,
                            **base_params
                        )
                        noise_pred_cond = noise_pred_cond[0]
                        noise_pred_ovi = noise_pred_ovi[0] if noise_pred_ovi is not None else None
                        if math.isclose(cfg_scale, 1.0):
                            if use_fresca:
                                noise_pred_cond = fourier_filter(noise_pred_cond, fresca_scale_low, fresca_scale_high, fresca_freq_cutoff)
                            return noise_pred_cond, noise_pred_ovi, [cache_state_cond]

                        #unconditional (negative) pass
                        base_params['is_uncond'] = True
                        base_params['clip_fea'] = clip_fea_neg if clip_fea_neg is not None else clip_fea
                        base_params["add_text_emb"] = qwenvl_embeds_neg.to(device) if qwenvl_embeds_neg is not None else None # QwenVL embeddings for Bindweave
                        # if wananim_face_pixels is not None:
                        #     base_params['wananim_face_pixel_values'] = torch.zeros_like(wananim_face_pixels).to(device, torch.float32) - 1
                        # if humo_audio_input_neg is not None:
                        #     base_params['humo_audio'] = humo_audio_input_neg
                        if neg_latent is not None:
                            base_params['x'] = [torch.cat([z[:, :-humo_reference_count], neg_latent], dim=1)]

                        noise_pred_uncond, noise_pred_ovi_uncond, cache_state_uncond = transformer(
                            context=negative_embeds if humo_audio_input_neg is None else positive_embeds, #ti #t
                            pred_id=cache_state[1] if cache_state else None,
                            vace_data=vace_data, attn_cond=attn_cond_neg,
                            **base_params)
                        noise_pred_uncond = noise_pred_uncond[0]
                        noise_pred_ovi_uncond = noise_pred_ovi_uncond[0] if noise_pred_ovi_uncond is not None else None

                        #audio cfg (fantasytalking and multitalk)
                        if (fantasytalking_embeds is not None or multitalk_audio_embeds is not None):
                            if not math.isclose(audio_cfg_scale[idx], 1.0):
                                if cache_state is not None and len(cache_state) != 3:
                                    cache_state.append(None)

                                # Set audio parameters to None/zeros based on type
                                if fantasytalking_embeds is not None:
                                    base_params['audio_proj'] = None
                                    audio_context = positive_embeds
                                else:  # multitalk
                                    base_params['multitalk_audio'] = torch.zeros_like(multitalk_audio_input)[-1:]
                                    audio_context = negative_embeds
                                base_params['is_uncond'] = False
                                noise_pred_no_audio, _, cache_state_audio = transformer(
                                    context=audio_context,
                                    pred_id=cache_state[2] if cache_state else None,
                                    vace_data=vace_data,
                                    **base_params)

                                noise_pred = (noise_pred_uncond
                                    + cfg_scale * (noise_pred_no_audio[0] - noise_pred_uncond)
                                    + audio_cfg_scale[idx] * (noise_pred_cond - noise_pred_no_audio[0]))
                                return noise_pred, None,[cache_state_cond, cache_state_uncond, cache_state_audio]
                    #batched
                    else:
                        base_params['z'] = [z] * 2
                        base_params['y'] = [image_cond_input] * 2 if image_cond_input is not None else None
                        base_params['clip_fea'] = torch.cat([clip_fea, clip_fea], dim=0)
                        cache_state_uncond = None
                        [noise_pred_cond, noise_pred_uncond], _, cache_state_cond = transformer(
                            context=positive_embeds + negative_embeds, is_uncond=False,
                            pred_id=cache_state[0] if cache_state else None,
                            **base_params
                        )
                except Exception as e:
                    log.error(f"Error during model prediction: {e}")
                    if force_offload:
                        if not model["auto_cpu_offload"]:
                            offload_transformer(transformer)
                    raise e

                #https://github.com/WeichenFan/CFG-Zero-star/
                alpha = 1.0
                if use_cfg_zero_star:
                    alpha = optimized_scale(
                        noise_pred_cond.view(batch_size, -1),
                        noise_pred_uncond.view(batch_size, -1)
                    ).view(batch_size, 1, 1, 1)

                noise_pred_uncond_scaled = noise_pred_uncond * alpha

                if use_tangential:
                    noise_pred_uncond_scaled = tangential_projection(noise_pred_cond, noise_pred_uncond_scaled)

                # RAAG (RATIO-aware Adaptive Guidance)
                if raag_alpha > 0.0:
                    cfg_scale = get_raag_guidance(noise_pred_cond, noise_pred_uncond_scaled, cfg_scale, raag_alpha)
                    log.info(f"RAAG modified cfg: {cfg_scale}")

                #https://github.com/WikiChao/FreSca
                if use_fresca:
                    filtered_cond = fourier_filter(noise_pred_cond - noise_pred_uncond, fresca_scale_low, fresca_scale_high, fresca_freq_cutoff)
                    noise_pred = noise_pred_uncond_scaled + cfg_scale * filtered_cond * alpha
                else:
                    noise_pred = noise_pred_uncond_scaled + cfg_scale * (noise_pred_cond - noise_pred_uncond_scaled)
                del noise_pred_uncond_scaled, noise_pred_cond, noise_pred_uncond

                if latent_model_input_ovi is not None:
                    audio_cfg_scale = cfg_scale - 1.0 if cfg_scale > 4.0 else cfg_scale
             
                    noise_pred_ovi = noise_pred_ovi_uncond + audio_cfg_scale * (noise_pred_ovi - noise_pred_ovi_uncond)

                return noise_pred, noise_pred_ovi, [cache_state_cond, cache_state_uncond]

        callback= None

        # Differential diffusion prep
        masks = None

        #clear memory before sampling
        # mm.soft_empty_cache()
        gc.collect()
        try:
            torch.cuda.reset_peak_memory_stats(device)
            #torch.cuda.memory._record_memory_history(max_entries=100000)
        except:
            pass

        # Main sampling loop with FreeInit iterations
        iterations = 1
        current_latent = latent

        for iter_idx in range(iterations):

            # Reset per-iteration states
            self.cache_state = [None, None]
            self.cache_state_source = [None, None]
            self.cache_states_context = []

            # Set latent for denoising
            latent = current_latent

            try:
                #region main loop start
                for idx, t in enumerate(tqdm(timesteps, disable=multitalk_sampling)):
                    # if flowedit_args is not None:
                    #     if idx < skip_steps:
                    #         continue

                    # if bidirectional_sampling:
                    #     latent_flipped = torch.flip(latent, dims=[1])
                    #     latent_model_input_flipped = latent_flipped.to(device)

                    latent_model_input = latent.to(device)

                    current_step_percentage = idx / len(timesteps)

                    timestep = torch.tensor([t]).to(device)

                    if multitalk_sampling:
                        mode = image_embeds.get("multitalk_mode", "multitalk")
                        if mode == "auto":
                            mode = transformer.multitalk_model_type.lower()
                        log.info(f"Multitalk mode: {mode}")
                        cond_frame = None
                        offload = image_embeds.get("force_offload", False)
                        offloaded = False
                        tiled_vae = image_embeds.get("tiled_vae", False)
                        frame_num = clip_length = image_embeds.get("frame_window_size", 81)

                        clip_embeds = image_embeds.get("clip_context", None)
                        if clip_embeds is not None:
                            clip_embeds = clip_embeds.to(dtype)
                        colormatch = image_embeds.get("colormatch", "disabled")
                        motion_frame = image_embeds.get("motion_frame", 25)
                        target_w = image_embeds.get("target_w", None)
                        target_h = image_embeds.get("target_h", None)
                        original_images = cond_image = image_embeds.get("multitalk_start_image", None)
                        if original_images is None:
                            original_images = torch.zeros([noise.shape[0], 1, target_h, target_w], device=device)

                        output_path = image_embeds.get("output_path", "")
                        img_counter = 0

                        gen_video_list = []
                        is_first_clip = True
                        arrive_last_frame = False
                        cur_motion_frames_num = 1
                        audio_start_idx = iteration_count = step_iteration_count= 0
                        audio_end_idx = audio_start_idx + clip_length
                        indices = (torch.arange(4 + 1) - 2) * 1
                        current_condframe_index = 0

                        audio_embedding = multitalk_audio_embeds
                        human_num = len(audio_embedding)
                        audio_embs = None
                        cond_frame = None

                        uni3c_data = uni3c_data_input = None

                        encoded_silence = None

                        try:
                            silence_path = os.path.join(script_directory, "multitalk", "encoded_silence.safetensors")
                            encoded_silence = load_torch_file(silence_path)["audio_emb"].to(dtype)
                        except:
                             log.warning("No encoded silence file found, padding with end of audio embedding instead.")

                        total_frames = len(audio_embedding[0])
                        estimated_iterations = total_frames // (frame_num - motion_frame) + 1

                        if frame_num >= total_frames:
                            arrive_last_frame = True
                            estimated_iterations = 1

                        log.info(f"Sampling {total_frames} frames in {estimated_iterations} windows, at {latent.shape[3]*vae_upscale_factor}x{latent.shape[2]*vae_upscale_factor} with {steps} steps")

                        while True: # start video generation iteratively
                            self.cache_state = [None, None]

                            cur_motion_frames_latent_num = int(1 + (cur_motion_frames_num-1) // 4)
                            if mode == "infinitetalk":
                                cond_image = original_images[:, :, current_condframe_index:current_condframe_index+1] if cond_image is not None else None
                            if multitalk_embeds is not None:
                                audio_embs = []
                                # split audio with window size
                                for human_idx in range(human_num):
                                    center_indices = torch.arange(audio_start_idx, audio_end_idx, 1).unsqueeze(1) + indices.unsqueeze(0)
                                    center_indices = torch.clamp(center_indices, min=0, max=audio_embedding[human_idx].shape[0]-1)
                                    audio_emb = audio_embedding[human_idx][center_indices].unsqueeze(0).to(device)
                                    audio_embs.append(audio_emb)
                                audio_embs = torch.concat(audio_embs, dim=0).to(dtype)

                            h, w = (cond_image.shape[-2], cond_image.shape[-1]) if cond_image is not None else (target_h, target_w)
                            lat_h, lat_w = h // VAE_STRIDE[1], w // VAE_STRIDE[2]
                            seq_len = ((frame_num - 1) // VAE_STRIDE[0] + 1) * lat_h * lat_w // (PATCH_SIZE[1] * PATCH_SIZE[2])
                            latent_frame_num = (frame_num - 1) // 4 + 1

                            noise = torch.randn(
                                16, latent_frame_num,
                                lat_h, lat_w, dtype=torch.float32, device=torch.device("cpu"), generator=seed_g).to(device)

                            # Calculate the correct latent slice based on current iteration
                            if is_first_clip:
                                latent_start_idx = 0
                                latent_end_idx = noise.shape[1]
                            else:
                                new_frames_per_iteration = frame_num - motion_frame
                                new_latent_frames_per_iteration = ((new_frames_per_iteration - 1) // 4 + 1)
                                latent_start_idx = iteration_count * new_latent_frames_per_iteration
                                latent_end_idx = latent_start_idx + noise.shape[1]

                            if samples is not None:
                                noise_mask = samples.get("noise_mask", None)
                                input_samples = samples["samples"]
                                if input_samples is not None:
                                    input_samples = input_samples.squeeze(0).to(noise)
                                    # Check if we have enough frames in input_samples
                                    if latent_end_idx > input_samples.shape[1]:
                                        # We need more frames than available - pad the input_samples at the end
                                        pad_length = latent_end_idx - input_samples.shape[1]
                                        last_frame = input_samples[:, -1:].repeat(1, pad_length, 1, 1)
                                        input_samples = torch.cat([input_samples, last_frame], dim=1)
                                    input_samples = input_samples[:, latent_start_idx:latent_end_idx]
                                    if noise_mask is not None:
                                        original_image = input_samples.to(device)

                                    assert input_samples.shape[1] == noise.shape[1], f"Slice mismatch: {input_samples.shape[1]} vs {noise.shape[1]}"

                                    if add_noise_to_samples:
                                        latent_timestep = timesteps[0]
                                        noise = noise * latent_timestep / 1000 + (1 - latent_timestep / 1000) * input_samples
                                    else:
                                        noise = input_samples

                                # diff diff prep
                                if noise_mask is not None:
                                    if len(noise_mask.shape) == 4:
                                        noise_mask = noise_mask.squeeze(1)
                                    if audio_end_idx > noise_mask.shape[0]:
                                        noise_mask = noise_mask.repeat(audio_end_idx // noise_mask.shape[0], 1, 1)
                                    noise_mask = noise_mask[audio_start_idx:audio_end_idx]
                                    noise_mask = torch.nn.functional.interpolate(
                                        noise_mask.unsqueeze(0).unsqueeze(0),  # Add batch and channel dims [1,1,T,H,W]
                                        size=(noise.shape[1], noise.shape[2], noise.shape[3]),
                                        mode='trilinear',
                                        align_corners=False
                                    ).repeat(1, noise.shape[0], 1, 1, 1)

                                    thresholds = torch.arange(len(timesteps), dtype=original_image.dtype) / len(timesteps)
                                    thresholds = thresholds.reshape(-1, 1, 1, 1, 1).to(device)
                                    masks = (1-noise_mask.repeat(len(timesteps), 1, 1, 1, 1).to(device)) > thresholds

                            # zero padding and vae encode for img cond
                            if cond_image is not None or cond_frame is not None:
                                cond_ = cond_image if (is_first_clip or humo_image_cond is None) else cond_frame
                                cond_frame_num = cond_.shape[2]
                                video_frames = torch.zeros(1, 3, frame_num-cond_frame_num, target_h, target_w, device=device, dtype=vae.dtype)
                                padding_frames_pixels_values = torch.concat([cond_.to(device, vae.dtype), video_frames], dim=2)

                                # encode
                                vae.to(device)
                                y = vae.encode(padding_frames_pixels_values, device=device, tiled=tiled_vae, pbar=False).to(dtype)[0]

                                if mode == "multitalk":
                                    latent_motion_frames = y[:, :cur_motion_frames_latent_num] # C T H W
                                else:
                                    cond_ = cond_image if is_first_clip else cond_frame
                                    latent_motion_frames = vae.encode(cond_.to(device, vae.dtype), device=device, tiled=tiled_vae, pbar=False).to(dtype)[0]

                                vae.to(offload_device)

                                #motion_frame_index = cur_motion_frames_latent_num if mode == "infinitetalk" else 1
                                msk = torch.zeros(4, latent_frame_num, lat_h, lat_w, device=device, dtype=dtype)
                                msk[:, :1] = 1
                                y = torch.cat([msk, y]) # 4+C T H W
                                # mm.soft_empty_cache()
                            else:
                                y = None
                                latent_motion_frames = noise[:, :1]

                            partial_humo_cond_input = partial_humo_cond_neg_input = partial_humo_audio = partial_humo_audio_neg = None

                            if scheduler == "multitalk":
                                timesteps = list(np.linspace(1000, 1, steps, dtype=np.float32))
                                timesteps.append(0.)
                                timesteps = [torch.tensor([t], device=device) for t in timesteps]
                                timesteps = [timestep_transform(t, shift=shift, num_timesteps=1000) for t in timesteps]
                            else:
                                if isinstance(scheduler, dict):
                                    sample_scheduler = copy.deepcopy(scheduler["sample_scheduler"])
                                    timesteps = scheduler["timesteps"]
                                else:
                                    sample_scheduler, timesteps,_,_ = get_scheduler(scheduler, total_steps, start_step, end_step, shift, device, transformer.dim, flowedit_args, denoise_strength, sigmas=None)
                                timesteps = [torch.tensor([float(t)], device=device) for t in timesteps] + [torch.tensor([0.], device=device)]

                            # sample videos
                            latent = noise

                            # injecting motion frames
                            if not is_first_clip and mode == "multitalk":
                                latent_motion_frames = latent_motion_frames.to(latent.dtype).to(device)
                                motion_add_noise = torch.randn(latent_motion_frames.shape, device=torch.device("cpu"), generator=seed_g).to(device).contiguous()
                                add_latent = add_noise(latent_motion_frames, motion_add_noise, timesteps[0])
                                latent[:, :add_latent.shape[1]] = add_latent

                            if offloaded:
                                # Load weights
                                if transformer.patched_linear and gguf_reader is None:
                                    load_weights(model.diffusion_model, model["sd"], weight_dtype, base_dtype=dtype, transformer_load_device=device, block_swap_args=block_swap_args)
                                elif gguf_reader is not None: #handle GGUF
                                    load_weights(transformer, model["sd"], base_dtype=dtype, transformer_load_device=device, patcher=patcher, gguf=True, reader=gguf_reader, block_swap_args=block_swap_args)
                                #blockswap init
                                init_blockswap(transformer, block_swap_args, model)

                            # Use the appropriate prompt for this section
                            if len(text_embeds["prompt_embeds"]) > 1:
                                prompt_index = min(iteration_count, len(text_embeds["prompt_embeds"]) - 1)
                                positive = [text_embeds["prompt_embeds"][prompt_index]]
                                log.info(f"Using prompt index: {prompt_index}")
                            else:
                                positive = text_embeds["prompt_embeds"]

                            # unianimate slices
                            partial_unianim_data = None
                           
                            partial_fantasy_portrait_input = None

                            # mm.soft_empty_cache()
                            gc.collect()
                            # sampling loop
                            sampling_pbar = tqdm(total=len(timesteps)-1, desc=f"Sampling audio indices {audio_start_idx}-{audio_end_idx}", position=0, leave=True)
                            for i in range(len(timesteps)-1):
                                timestep = timesteps[i]
                                latent_model_input = latent.to(device)
                                if mode == "infinitetalk":
                                    if humo_image_cond is None or not is_first_clip:
                                        latent_model_input[:, :cur_motion_frames_latent_num] = latent_motion_frames

                                noise_pred, _, self.cache_state = predict_with_cfg(
                                    latent_model_input, cfg[min(i, len(timesteps)-1)], positive, text_embeds["negative_prompt_embeds"],
                                    timestep, i, y, clip_embeds, control_latents, None, partial_unianim_data, audio_proj, control_camera_latents, add_cond,
                                    cache_state=self.cache_state, multitalk_audio_embeds=audio_embs, fantasy_portrait_input=partial_fantasy_portrait_input,
                                    humo_image_cond=partial_humo_cond_input, humo_image_cond_neg=partial_humo_cond_neg_input, humo_audio=partial_humo_audio, humo_audio_neg=partial_humo_audio_neg,
                                    uni3c_data = uni3c_data)

                                if callback is not None:
                                    callback_latent = (latent_model_input.to(device) - noise_pred.to(device) * t.to(device) / 1000).detach().permute(1,0,2,3)
                                    callback(step_iteration_count, callback_latent, None, estimated_iterations*(len(timesteps)-1))
                                    del callback_latent

                                sampling_pbar.update(1)
                                step_iteration_count += 1

                                # update latent
                                # if use_tsr:
                                #     noise_pred = temporal_score_rescaling(noise_pred, latent, timestep, tsr_k, tsr_sigma)
                                if scheduler == "multitalk":
                                    noise_pred = -noise_pred
                                    dt = (timesteps[i] - timesteps[i + 1]) / 1000
                                    latent = latent + noise_pred * dt[:, None, None, None]
                                else:
                                    latent = sample_scheduler.step(noise_pred.unsqueeze(0), timestep, latent.unsqueeze(0).to(noise_pred.device), **scheduler_step_args)[0].squeeze(0)
                                del noise_pred, latent_model_input, timestep

                                # differential diffusion inpaint
                                if masks is not None:
                                    if i < len(timesteps) - 1:
                                        image_latent = add_noise(original_image.to(device), noise.to(device), timesteps[i+1])
                                        mask = masks[i].to(latent)
                                        latent = image_latent * mask + latent * (1-mask)

                                # injecting motion frames
                                if not is_first_clip and mode == "multitalk":
                                    latent_motion_frames = latent_motion_frames.to(latent.dtype).to(device)
                                    motion_add_noise = torch.randn(latent_motion_frames.shape, device=torch.device("cpu"), generator=seed_g).to(device).contiguous()
                                    add_latent = add_noise(latent_motion_frames, motion_add_noise, timesteps[i+1])
                                    latent[:, :add_latent.shape[1]] = add_latent
                                else:
                                    if humo_image_cond is None or not is_first_clip:
                                        latent[:, :cur_motion_frames_latent_num] = latent_motion_frames

                            del noise, latent_motion_frames
                            if offload:
                                offload_transformer(transformer)
                                offloaded = True
                            if humo_image_cond is not None and humo_reference_count > 0:
                                latent = latent[:,:-humo_reference_count]
                            vae.to(device)
                            videos = vae.decode(latent.unsqueeze(0).to(device, vae.dtype), device=device, tiled=tiled_vae, pbar=False)[0].cpu()

                            vae.to(offload_device)

                            sampling_pbar.close()


                            # optionally save generated samples to disk
                            if output_path:
                                pass
                            else:
                                gen_video_list.append(videos if is_first_clip else videos[:, cur_motion_frames_num:])

                            current_condframe_index += 1
                            iteration_count += 1

                            # decide whether is done
                            if arrive_last_frame:
                                break

                            # update next condition frames
                            is_first_clip = False
                            cur_motion_frames_num = motion_frame

                            cond_ = videos[:, -cur_motion_frames_num:].unsqueeze(0)
                            if mode == "infinitetalk":
                                cond_frame = cond_
                            else:
                                cond_image = cond_

                            del videos, latent

                            # Repeat audio emb
                            if multitalk_embeds is not None:
                                audio_start_idx += (frame_num - cur_motion_frames_num - humo_reference_count)
                                audio_end_idx = audio_start_idx + clip_length
                                if audio_end_idx >= len(audio_embedding[0]):
                                    arrive_last_frame = True
                                    miss_lengths = []
                                    source_frames = []
                                    for human_inx in range(human_num):
                                        source_frame = len(audio_embedding[human_inx])
                                        source_frames.append(source_frame)
                                        if audio_end_idx >= len(audio_embedding[human_inx]):
                                            print(f"Audio embedding for subject {human_inx} not long enough: {len(audio_embedding[human_inx])}, need {audio_end_idx}, padding...")
                                            miss_length = audio_end_idx - len(audio_embedding[human_inx]) + 3
                                            print(f"Padding length: {miss_length}")
                                            if encoded_silence is not None:
                                                add_audio_emb = encoded_silence[-1*miss_length:]
                                            else:
                                                add_audio_emb = torch.flip(audio_embedding[human_inx][-1*miss_length:], dims=[0])
                                            audio_embedding[human_inx] = torch.cat([audio_embedding[human_inx], add_audio_emb.to(device, dtype)], dim=0)
                                            miss_lengths.append(miss_length)
                                        else:
                                            miss_lengths.append(0)
                                if mode == "infinitetalk" and current_condframe_index >= original_images.shape[2]:
                                    last_frame = original_images[:, :, -1:, :, :]
                                    miss_length   = 1
                                    original_images = torch.cat([original_images, last_frame.repeat(1, 1, miss_length, 1, 1)], dim=2)

                        if not output_path:
                            gen_video_samples = torch.cat(gen_video_list, dim=1)
                        else:
                            gen_video_samples = torch.zeros(3, 1, 64, 64) # dummy output

                        if force_offload:
                            if not model["auto_cpu_offload"]:
                                offload_transformer(transformer)
                        try:
                            print_memory(device)
                            torch.cuda.reset_peak_memory_stats(device)
                        except:
                            pass
                        return {"video": gen_video_samples.permute(1, 2, 3, 0), "output_path": output_path},
            except Exception as e:
                log.error(f"Error during sampling: {e}")
                if force_offload:
                    if not model["auto_cpu_offload"]:
                        offload_transformer(transformer)
                raise e

        if force_offload:
            if not model["auto_cpu_offload"]:
                offload_transformer(transformer)

        try:
            print_memory(device)
            torch.cuda.reset_peak_memory_stats(device)
        except:
            pass
        return {}