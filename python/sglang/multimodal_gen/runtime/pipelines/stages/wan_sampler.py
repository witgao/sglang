import torch
from sglang.multimodal_gen.runtime.pipelines_core.stages.base import PipelineStage
from sglang.multimodal_gen.runtime.server_args import ServerArgs
from sglang.multimodal_gen.runtime.pipelines_core.schedule_batch import Req

from sglang.multimodal_gen.runtime.models.dits.wanvideo import WanTransformer3DModel
from sglang.multimodal_gen.runtime.managers.forward_context import set_forward_context

from sglang.multimodal_gen.runtime.utils.model_utils import ModelUtils

import os
import gc
from tqdm import tqdm
from sglang.multimodal_gen.runtime.models.original.wanvideo.schedulers import get_scheduler
from sglang.multimodal_gen.utils import dict_to_3d_list, masks_like

from contextlib import nullcontext

device = torch.device(torch.cuda.current_device())
offload_device = torch.device("cpu")

VAE_STRIDE = (4, 8, 8)
PATCH_SIZE = (1, 2, 2)

def offload_transformer(transformer):
    if getattr(transformer, "patched_linear", False):
        for name, param in transformer.named_parameters():
            if "loras" in name or "controlnet" in name:
                continue
            module = transformer
            subnames = name.split(".")
            for subname in subnames[:-1]:
                module = getattr(module, subname)
            attr_name = subnames[-1]
            if param.data.is_floating_point():
                meta_param = torch.nn.Parameter(
                    torch.empty_like(param.data, device="meta"), requires_grad=False
                )
                setattr(module, attr_name, meta_param)
    else:
        transformer.to(offload_device)

    for block in transformer.blocks:
        block.kv_cache = None
        # if transformer.audio_model is not None and hasattr(block, "audio_block"):
        #     block.audio_block = None

    gc.collect()

def list_to_device(tensor_list, device, dtype=None):
    """
    Move all tensors in a list to the specified device and optionally cast to dtype.
    """
    return [t.to(device, dtype=dtype) if dtype is not None else t.to(device) for t in tensor_list]

def dict_to_device(tensor_dict, device, dtype=None):
    """
    Move all tensors (and tensor lists) in a dict to the specified device and optionally cast to dtype.
    Supports values that are tensors or lists of tensors.
    """
    result = {}
    for k, v in tensor_dict.items():
        if isinstance(v, torch.Tensor):
            result[k] = v.to(device, dtype=dtype) if dtype is not None else v.to(device)
        elif isinstance(v, list) and all(isinstance(t, torch.Tensor) for t in v):
            result[k] = list_to_device(v, device, dtype)
        else:
            result[k] = v
    return result

def get_vae_shift_scale(vae, device, dtype):
    shift = getattr(vae, "shift_factor", None)
    scale = getattr(vae, "scaling_factor", None)
    if isinstance(shift, torch.Tensor):
        shift = shift.to(device=device, dtype=dtype)
    elif shift is not None:
        shift = torch.tensor(shift, device=device, dtype=dtype)
    if isinstance(scale, torch.Tensor):
        scale = scale.to(device=device, dtype=dtype)
    elif scale is not None:
        scale = torch.tensor(scale, device=device, dtype=dtype)
    return shift, scale

class WanSamplerStage(PipelineStage):
    def __init__(
        self, transformer: WanTransformer3DModel
    ):
        super().__init__()
        self.transformer = transformer

    def forward(
        self,
        batch: Req,
        server_args: ServerArgs,
    ) -> Req:
        with torch.no_grad():
            text_embeds = {
                "prompt_embeds": batch.prompt_embeds[0],
                "negative_prompt_embeds": batch.negative_prompt_embeds[0],
            }
            image_embeds = batch.image_latent2
            audio_embeds = batch.audio_embeds
            sampler_result = self.process(
                self.transformer,
                text_embeds=text_embeds,
                image_embeds=image_embeds,
                multitalk_embeds=audio_embeds,
            )

        batch.samples = sampler_result[0]
        return batch

    def process(self, transformer, text_embeds, image_embeds, multitalk_embeds):
        # transformer推理的dtype
        dtype = torch.bfloat16
        rope_function = "comfy"
        device = torch.device(torch.cuda.current_device())
        offload_device = torch.device("cpu")
        vae = image_embeds.get("vae", None)
        seed=2
        riflex_freq_index=0
        shift=11.0
        steps=4
        scheduler="unipc"
        cfg=1.0
        add_noise_to_samples=True
        
        # 加载文本embeds到GPU
        text_embeds = dict_to_device(text_embeds, device)

        # 设置schedule的随机种子
        seed_g = torch.Generator(device=torch.device("cpu"))
        seed_g.manual_seed(seed)
        scheduler_step_args = {"generator": seed_g}

        # 计算单个clip的帧数
        frame_num = clip_length = image_embeds.get("frame_window_size", 81)
        latent_frame_num = (frame_num - 1) // 4 + 1

        # 加载音频embeds到GPU
        audio_features_in = multitalk_embeds
        audio_embedding = [emb.to(device, dtype) for emb in audio_features_in]

        # TODO
        # transformer.slg_blocks = None
        # transformer.video_attention_split_steps = []
        # transformer.rope_embedder.k = None
        # transformer.rope_embedder.num_frames = None
        # transformer.rope_embedder.k = riflex_freq_index
        # transformer.rope_embedder.num_frames = latent_frame_num
        # transformer.rope_func = rope_function
        # for block in transformer.blocks:
        #     block.rope_func = rope_function
        
        # 滑动窗口大小，每次生成一个clip
        motion_frame = image_embeds.get("motion_frame", 25)
        target_w = image_embeds.get("target_w", None)
        target_h = image_embeds.get("target_h", None)
        
        # 加载图片embeds到GPU
        clip_embeds = image_embeds.get("clip_context", None)
        clip_embeds = clip_embeds.to(dtype)
       
        # 图片原图
        cond_image = image_embeds.get("multitalk_start_image", None)

        # 保存生成的结果
        gen_video_list = []
        # 是否首帧
        is_first_clip = True
        # 是否结束
        arrive_last_frame = False
        
        # 计算滑动窗口
        cur_motion_frames_num = 1
        # 音频窗口
        audio_start_idx = iteration_count = step_iteration_count = 0
        audio_end_idx = audio_start_idx + clip_length

        # 每一个窗口的音频embeds
        audio_embs = None
        # 每一个窗口的图片embeds
        cond_frame = None

        # 静音的音频embeds，用于最后一段补齐
        encoded_silence = None
        silence_path = os.path.join(
            "/data1/gaozhenyu/studio/sglang/python/sglang/multimodal_gen/runtime/models/original/wanvideo/",
            "multitalk",
            "encoded_silence.safetensors",
        )
        encoded_silence = ModelUtils.load_torch_file(silence_path)["audio_emb"].to(dtype)

        # 音频总帧数
        total_frames = len(audio_embedding[0])

        # 如果总帧数小于clip大小，则标记结束
        if frame_num >= total_frames:
            arrive_last_frame = True

        # transformer.to(device=device, dtype=dtype)
        
        # 开始根据音频循环生成
        while True:  # start video generation iteratively
            # cur_motion_frames_num第一次是1，后面就是motion_frame，压缩4倍，计算latent的帧数
            cur_motion_frames_latent_num = int(1 + (cur_motion_frames_num - 1) // 4)
            
            audio_embs, arrive_last_frame = self.prepare_audio_embeds(audio_embedding, audio_start_idx, audio_end_idx, dtype, encoded_silence)
            vae_dtype = next(vae.parameters()).dtype
            noise, latent_motion_frames, y, seq_len = self.prepare_latent(cond_image, frame_num,
                                                                          seed_g, target_w, target_h, dtype,
                                                                          is_first_clip, cond_frame, vae)
        
            # 创建schedule
            transformer_dim = getattr(
                transformer, "dim", getattr(transformer, "hidden_size", None)
            )
            if transformer_dim is None and hasattr(transformer, "config"):
                transformer_dim = getattr(transformer.config, "hidden_size", None)
            if transformer_dim is None:
                transformer_dim = 5120
            sample_scheduler, timesteps, _, _ = get_scheduler(
                scheduler,
                steps,
                0,
                -1,
                shift,
                device,
                transformer_dim,
                None,
                1.0,
                sigmas=None,
            )
            timesteps = [torch.tensor([float(t)], device=device) for t in timesteps] + [
                torch.tensor([0.0], device=device)
            ]

            latent = noise
            
            sampling_pbar = tqdm(
                total=len(timesteps) - 1,
                desc=f"Sampling audio indices {audio_start_idx}-{audio_end_idx}",
                position=0,
                leave=True,
            )
            
            # 开始单clip生成
            for i in range(len(timesteps) - 1):
                timestep = timesteps[i]
                latent_model_input = latent.to(device)
                # 将上一个clip的尾帧拷贝到第一帧，保证连续性
                latent_model_input[:, :cur_motion_frames_latent_num] = (
                    latent_motion_frames
                )
                # 推理
                noise_pred, _, _ = self.predict_with_cfg(
                    transformer,
                    seq_len,
                    text_embeds,
                    latent_model_input,
                    timestep,
                    y,
                    clip_embeds,
                    multitalk_audio_embeds=audio_embs,
                )

                sampling_pbar.update(1)
                step_iteration_count += 1

                # 更新step和latent
                latent = sample_scheduler.step(
                    noise_pred.unsqueeze(0),
                    timestep,
                    latent.unsqueeze(0).to(noise_pred.device),
                    **scheduler_step_args,
                )[0].squeeze(0)
                del noise_pred, latent_model_input, timestep

                # 将参考帧再次更新到latent中，防止被重新生成
                latent[:, :cur_motion_frames_latent_num] = latent_motion_frames

            del noise, latent_motion_frames

            vae.to(device)
            # 解码
            z = latent.unsqueeze(0).to(device, vae_dtype)
            shift2, scale = get_vae_shift_scale(vae, z.device, z.dtype)
            if scale is not None:
                z = z / scale
            if shift2 is not None:
                z = z + shift2
            videos = vae.decode(z)[0].cpu()

            vae.to(offload_device)

            sampling_pbar.close()

            # 第一次把生成的81帧都加入，后面要去除前面的参考窗口帧，只保留新生成的帧
            gen_video_list.append(
                    videos if is_first_clip else videos[:, cur_motion_frames_num:]
                )

            iteration_count += 1

            if arrive_last_frame:
                break

            is_first_clip = False
            # 第一次参考窗口是1，后面都是motion_frame
            cur_motion_frames_num = motion_frame
            
            # 将最后的参考帧保存，用于下一个clip的参考
            cond_frame = videos[:, -cur_motion_frames_num:].unsqueeze(0)

            del videos, latent

            audio_start_idx += (frame_num - cur_motion_frames_num)
            audio_end_idx = audio_start_idx + clip_length
        
        gen_video_samples = torch.cat(gen_video_list, dim=1)
        
        # offload_transformer(transformer)
        torch.cuda.reset_peak_memory_stats(device)
        return (
            {
                "video": gen_video_samples.permute(1, 2, 3, 0),
            },
        )
                            
    def prepare_latent(self, cond_image, frame_num, seed_g, target_w, target_h, dtype, is_first_clip, cond_frame, vae):
        vae_dtype = next(vae.parameters()).dtype
        
        # 每次输入81帧进行生成，其中第一帧是参考帧，剩下80帧是黑帧。image和latent是1:4关系(vae时间维度压缩)，所以需要21帧latent,其中第一个latent是参考，后面20个是生成的数据。
        # 参考帧是不断更新，第一次是参考图，后面是上次生成的结果。参考帧的数量也是变化的，第一次是1帧，后面是motion_frame的大小，会把上一个clip的最后motion_frame帧作为参考帧。
        # vae在时间维度是4倍压缩，在宽高维度上是8倍压缩。所以一个80*W*H的视频序列，对应的latent是20*W/8*H/8
        # 创建latent
        # 原图宽高
        h, w = (
            (cond_image.shape[-2], cond_image.shape[-1])
        )
        # latent宽高，下采样8倍
        lat_h, lat_w = h // 8, w // 8
        # latent帧数，压缩4倍
        latent_frame_num = (frame_num - 1) // 4 + 1
        # token按照2*2个patch进行分割
        seq_len = (
            latent_frame_num
            * lat_h
            * lat_w
            // (2 * 2)
        )
        # 生成latent，作为推理的输入和输出
        noise = torch.randn(
            16,
            latent_frame_num,
            lat_h,
            lat_w,
            dtype=dtype,
            device=torch.device("cpu"),
            generator=seed_g,
        ).to(device)


        # cond_frame_num是参考图帧数 1
        cond_ = cond_image
        cond_frame_num = cond_.shape[2]
        # 用黑帧补齐frame_num帧，
        video_frames = torch.zeros(
            1,
            3,
            frame_num - cond_frame_num,
            target_h,
            target_w,
            device=device,
            dtype=vae_dtype,
        )
        # 合并参考帧和黑帧
        padding_frames_pixels_values = torch.concat(
            [cond_.to(device, vae_dtype), video_frames], dim=2
        )

        # encode
        vae.to(device)

        shift, scale = get_vae_shift_scale(vae, device, dtype)
        enc = vae.encode(
            padding_frames_pixels_values,
        )
        y = enc.mode() if hasattr(enc, "mode") else enc
        if shift is not None:
            y = y - shift
        if scale is not None:
            y = y * scale
        y = y.to(dtype)[0]  # [16, 21, 60, 60]
        
        # noise是输入，用来生成的噪声，latent_motion_frames是参考帧，放到输入的噪声里。这个是告诉模型从参考帧开始继续生成，为了多个clip之前生成连贯。所以参考帧是上一个clip的尾帧。
        # y是生成的参考对象，与noise的长度一样，但是只有第一帧是原始图片，后面都是黑帧，这是用来做条件引导的。y的通道是4+16，比noise多4个通道，是标记mask，1告诉模型只是参考，0告诉模型可以自由发挥。
        
        # 如果是首帧，用参考图作为输入，如果不是，则用上一帧生成的结果作为输入
        cond_ = cond_image if is_first_clip else cond_frame
        # latent_motion_frames只有一帧，只用来在latent中固定前面的帧。
        enc = vae.encode(
            cond_.to(device, vae_dtype),
        )
        latent_motion_frames = enc.mode() if hasattr(enc, "mode") else enc
        if shift is not None:
            latent_motion_frames = latent_motion_frames - shift
        if scale is not None:
            latent_motion_frames = latent_motion_frames * scale
        latent_motion_frames = latent_motion_frames.to(dtype)[0]  # [16, 1, 60, 60]

        vae.to(offload_device)

        msk = torch.zeros(
            4,
            latent_frame_num,
            lat_h,
            lat_w,
            device=device,
            dtype=dtype,
        )
        # 第一帧mask标记为1
        msk[:, :1] = 1 
        y = torch.cat([msk, y])  # 4+C T H W
        
        return noise, latent_motion_frames, y, seq_len
    
    def timestep_prepara():
        pass
    
    def denoising():
        pass

    def predict_with_cfg(
        self,
        transformer,
        seq_len,
        text_embeds,
        z,
        timestep,
        image_cond,
        clip_fea=None,
        multitalk_audio_embeds=None,
    ):
        # scaled模型，已经在自定义的layer中实现了cast
        # autocast_enabled = (
        #     "fp8" in model["quantization"] and not transformer.patched_linear
        # )
        with (
            # torch.autocast(device_type=device.type, dtype=dtype)
            # if autocast_enabled
            nullcontext()
        ):
            try:
                timestep_value = int(timestep.item()) if torch.is_tensor(timestep) else int(timestep)
                prompt_embeds = text_embeds["prompt_embeds"]
                if isinstance(prompt_embeds, list):
                    prompt_embeds = prompt_embeds[0]
                if prompt_embeds.dim() == 2:
                    prompt_embeds = prompt_embeds.unsqueeze(0)
                
                prompt_embeds = prompt_embeds.to(z.device, z.dtype)

                latent_model_input = z
                if image_cond is not None:
                    image_cond_input = image_cond.to(z)
                    latent_model_input = torch.cat(
                        [latent_model_input, image_cond_input], dim=0
                    )
                latent_model_input = latent_model_input.unsqueeze(0)

                # latent_model_input = self.scheduler.scale_model_input(
                #             latent_model_input, timestep_value
                # )
                
                encoder_hidden_states_image = clip_fea
                encoder_hidden_states_image = encoder_hidden_states_image.to(
                    z.device, z.dtype
                )
                encoder_hidden_states_image = [encoder_hidden_states_image]

                with set_forward_context(
                    current_timestep=timestep_value, attn_metadata=None, forward_batch=None
                ):
                    noise_pred = transformer(
                        hidden_states=latent_model_input,
                        encoder_hidden_states=[prompt_embeds],
                        timestep=timestep,
                        guidance=None,
                        encoder_hidden_states_image=encoder_hidden_states_image,
                        mask_strategy=dict_to_3d_list(None, t_max=50, l_max=60, h_max=24),
                    )
                return noise_pred[0], None, [None]
            except Exception as e:
                # offload_transformer(transformer)
                raise e
    
    def prepare_audio_embeds(self, audio_embedding, audio_start_idx, audio_end_idx, dtype, encoded_silence):
        audio_embs = []
        indices = (torch.arange(4 + 1) - 2) * 1
        human_num = len(audio_embedding)
        arrive_last_frame = False

        # 最后一段补齐静音
        if audio_end_idx >= len(audio_embedding[0]):
            arrive_last_frame = True
            miss_lengths = []
            source_frames = []
            for human_inx in range(human_num):
                source_frame = len(audio_embedding[human_inx])
                source_frames.append(source_frame)
                if audio_end_idx >= len(audio_embedding[human_inx]):
                    print(
                        f"Audio embedding for subject {human_inx} not long enough: {len(audio_embedding[human_inx])}, need {audio_end_idx}, padding..."
                    )
                    miss_length = (
                        audio_end_idx - len(audio_embedding[human_inx]) + 3
                    )
                    print(f"Padding length: {miss_length}")
                    if encoded_silence is not None:
                        add_audio_emb = encoded_silence[-1 * miss_length :]
                    else:
                        add_audio_emb = torch.flip(
                            audio_embedding[human_inx][-1 * miss_length :],
                            dims=[0],
                        )
                    audio_embedding[human_inx] = torch.cat(
                        [
                            audio_embedding[human_inx],
                            add_audio_emb.to(device, dtype),
                        ],
                        dim=0,
                    )
                    miss_lengths.append(miss_length)
                else:
                    miss_lengths.append(0)       
                    
        # 根据窗口大小，加载对应窗口的每个人的音频embeds
        for human_idx in range(human_num):
            # 构建81个小窗口，每个小窗口是当前索引加上前后+2的索引 [81, 5]
            center_indices = torch.arange(
                audio_start_idx, audio_end_idx, 1
            ).unsqueeze(1) + indices.unsqueeze(0)
            # trim 防止越界 [81, 5]
            center_indices = torch.clamp(
                center_indices,
                min=0,
                max=audio_embedding[human_idx].shape[0] - 1,
            )
            # 从[100, 12, 768] -> [81, 5, 12, 768] -> [1, 81, 5, 12, 768]
            audio_emb = (
                audio_embedding[human_idx][center_indices]
                .unsqueeze(0)
                .to(device)
            )
            audio_embs.append(audio_emb)
            # 合并 [1, 81, 5, 12, 768] -> [n, 81, 5, 12, 768]
        audio_embs = torch.concat(audio_embs, dim=0).to(dtype)
    
        return audio_embs, arrive_last_frame
        