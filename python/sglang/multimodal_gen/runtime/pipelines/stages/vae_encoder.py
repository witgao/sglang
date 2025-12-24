import torch
import torch.nn.functional as F
from tqdm import tqdm
import logging
from PIL import Image
import numpy as np
import datetime
import os
from sglang.multimodal_gen.runtime.pipelines_core.stages.base import PipelineStage
from sglang.multimodal_gen.runtime.server_args import ServerArgs
from sglang.multimodal_gen.runtime.pipelines_core.schedule_batch import Req

from sglang.multimodal_gen.runtime.models.original import OriginalWanVAEModel


def bislerp(samples, width, height):
    def slerp(b1, b2, r):
        '''slerps batches b1, b2 according to ratio r, batches should be flat e.g. NxC'''

        c = b1.shape[-1]

        # norms
        b1_norms = torch.norm(b1, dim=-1, keepdim=True)
        b2_norms = torch.norm(b2, dim=-1, keepdim=True)

        # normalize
        b1_normalized = b1 / b1_norms
        b2_normalized = b2 / b2_norms

        # zero when norms are zero
        b1_normalized[b1_norms.expand(-1, c) == 0.0] = 0.0
        b2_normalized[b2_norms.expand(-1, c) == 0.0] = 0.0

        # slerp
        dot = (b1_normalized*b2_normalized).sum(1)
        omega = torch.acos(dot)
        so = torch.sin(omega)

        # technically not mathematically correct, but more pleasing?
        res = (torch.sin((1.0-r.squeeze(1))*omega)/so).unsqueeze(1)*b1_normalized + \
            (torch.sin(r.squeeze(1)*omega)/so).unsqueeze(1) * b2_normalized
        res *= (b1_norms * (1.0-r) + b2_norms * r).expand(-1, c)

        # edge cases for same or polar opposites
        res[dot > 1 - 1e-5] = b1[dot > 1 - 1e-5]
        res[dot < 1e-5 - 1] = (b1 * (1.0-r) + b2 * r)[dot < 1e-5 - 1]
        return res

    def generate_bilinear_data(length_old, length_new, device):
        coords_1 = torch.arange(
            length_old, dtype=torch.float32, device=device).reshape((1, 1, 1, -1))
        coords_1 = torch.nn.functional.interpolate(
            coords_1, size=(1, length_new), mode="bilinear")
        ratios = coords_1 - coords_1.floor()
        coords_1 = coords_1.to(torch.int64)

        coords_2 = torch.arange(
            length_old, dtype=torch.float32, device=device).reshape((1, 1, 1, -1)) + 1
        coords_2[:, :, :, -1] -= 1
        coords_2 = torch.nn.functional.interpolate(
            coords_2, size=(1, length_new), mode="bilinear")
        coords_2 = coords_2.to(torch.int64)
        return ratios, coords_1, coords_2

    orig_dtype = samples.dtype
    samples = samples.float()
    n, c, h, w = samples.shape
    h_new, w_new = (height, width)

    # linear w
    ratios, coords_1, coords_2 = generate_bilinear_data(
        w, w_new, samples.device)
    coords_1 = coords_1.expand((n, c, h, -1))
    coords_2 = coords_2.expand((n, c, h, -1))
    ratios = ratios.expand((n, 1, h, -1))

    pass_1 = samples.gather(-1, coords_1).movedim(1, -1).reshape((-1, c))
    pass_2 = samples.gather(-1, coords_2).movedim(1, -1).reshape((-1, c))
    ratios = ratios.movedim(1, -1).reshape((-1, 1))

    result = slerp(pass_1, pass_2, ratios)
    result = result.reshape(n, h, w_new, c).movedim(-1, 1)

    # linear h
    ratios, coords_1, coords_2 = generate_bilinear_data(
        h, h_new, samples.device)
    coords_1 = coords_1.reshape((1, 1, -1, 1)).expand((n, c, -1, w_new))
    coords_2 = coords_2.reshape((1, 1, -1, 1)).expand((n, c, -1, w_new))
    ratios = ratios.reshape((1, 1, -1, 1)).expand((n, 1, -1, w_new))

    pass_1 = result.gather(-2, coords_1).movedim(1, -1).reshape((-1, c))
    pass_2 = result.gather(-2, coords_2).movedim(1, -1).reshape((-1, c))
    ratios = ratios.movedim(1, -1).reshape((-1, 1))

    result = slerp(pass_1, pass_2, ratios)
    result = result.reshape(n, h_new, w_new, c).movedim(-1, 1)
    return result.to(orig_dtype)


def lanczos(samples, width, height):
    images = [Image.fromarray(np.clip(
        255. * image.movedim(0, -1).cpu().numpy(), 0, 255).astype(np.uint8)) for image in samples]
    images = [image.resize(
        (width, height), resample=Image.Resampling.LANCZOS) for image in images]
    images = [torch.from_numpy(np.array(image).astype(
        np.float32) / 255.0).movedim(-1, 0) for image in images]
    result = torch.stack(images)
    return result.to(samples.device, samples.dtype)

def common_upscale(samples, width, height, upscale_method, crop):
    orig_shape = tuple(samples.shape)
    if len(orig_shape) > 4:
        samples = samples.reshape(
            samples.shape[0], samples.shape[1], -1, samples.shape[-2], samples.shape[-1])
        samples = samples.movedim(2, 1)
        samples = samples.reshape(-1,
                                  orig_shape[1], orig_shape[-2], orig_shape[-1])
    if crop == "center":
        old_width = samples.shape[-1]
        old_height = samples.shape[-2]
        old_aspect = old_width / old_height
        new_aspect = width / height
        x = 0
        y = 0
        if old_aspect > new_aspect:
            x = round((old_width - old_width * (new_aspect / old_aspect)) / 2)
        elif old_aspect < new_aspect:
            y = round((old_height - old_height * (old_aspect / new_aspect)) / 2)
        s = samples.narrow(-2, y, old_height - y *
                           2).narrow(-1, x, old_width - x * 2)
    else:
        s = samples

    if upscale_method == "bislerp":
        out = bislerp(s, width, height)
    elif upscale_method == "lanczos":
        out = lanczos(s, width, height)
    else:
        out = torch.nn.functional.interpolate(
            s, size=(height, width), mode=upscale_method)

    if len(orig_shape) == 4:
        return out

    out = out.reshape((orig_shape[0], -1, orig_shape[1]) + (height, width))
    return out.movedim(2, 1).reshape(orig_shape[:-2] + (height, width))

class VAEEncoderStage(PipelineStage):

    def __init__(self, vae: OriginalWanVAEModel):
        super().__init__()
        self.vae = vae

    def process(self, vae, width, height, frame_window_size, motion_frame, force_offload, colormatch, start_image=None, tiled_vae=False, clip_embeds=None, mode="multitalk", output_path=""):

        H = height
        W = width
        VAE_STRIDE = (4, 8, 8)

        num_frames = ((frame_window_size - 1) // 4) * 4 + 1

        # Resize and rearrange the input image dimensions
        if start_image is not None:
            resized_start_image = common_upscale(
                start_image.movedim(-1, 1), W, H, "lanczos", "disabled").movedim(0, 1)
            resized_start_image = resized_start_image * 2 - 1
            resized_start_image = resized_start_image.unsqueeze(0)

        target_shape = (16, (num_frames - 1) // VAE_STRIDE[0] + 1,
                        height // VAE_STRIDE[1],
                        width // VAE_STRIDE[2])

        if output_path:
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = os.path.join(
                output_path, f"{timestamp}_{mode}_output")
            os.makedirs(output_path, exist_ok=True)

        image_embeds = {
            "multitalk_sampling": True,
            "multitalk_start_image": resized_start_image if start_image is not None else None,
            "frame_window_size": num_frames,
            "motion_frame": motion_frame,
            "target_h": H,
            "target_w": W,
            "tiled_vae": tiled_vae,
            "force_offload": force_offload,
            "vae": vae,
            "target_shape": target_shape,
            "clip_context": clip_embeds,
            "colormatch": colormatch,
            "multitalk_mode": mode,
            "output_path": output_path
        }

        return (image_embeds, output_path)
    
    def image_to_video_encode(self, vae, image, width, height, clip_embeds):
        return self.process(width=width, height=height, frame_window_size=81, motion_frame=9, force_offload=False, colormatch="disabled",
                           start_image=image, clip_embeds=clip_embeds, vae=vae, tiled_vae=False, mode="infinitetalk")
    
    def forward(self, batch: Req, server_args: ServerArgs,) -> Req:
        image = batch.condition_image
        width = batch.width
        height = batch.height
        vae_encode_result = self.image_to_video_encode(self.vae, image, width, height, batch.image_embeds[0])[0]
        batch.image_latent2 = vae_encode_result
        return batch
