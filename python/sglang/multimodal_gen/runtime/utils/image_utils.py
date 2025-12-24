from PIL import Image, ImageOps, ImageSequence
import torch
import torch.nn.functional as F
import numpy as np
import math


def _bislerp(samples, width, height):
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


def _lanczos(samples, width, height):
    images = [Image.fromarray(np.clip(
        255. * image.movedim(0, -1).cpu().numpy(), 0, 255).astype(np.uint8)) for image in samples]
    images = [image.resize(
        (width, height), resample=Image.Resampling.LANCZOS) for image in images]
    images = [torch.from_numpy(np.array(image).astype(
        np.float32) / 255.0).movedim(-1, 0) for image in images]
    result = torch.stack(images)
    return result.to(samples.device, samples.dtype)


def _common_upscale(samples, width, height, upscale_method, crop):
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
            x = round((old_width - old_width *
                      (new_aspect / old_aspect)) / 2)
        elif old_aspect < new_aspect:
            y = round((old_height - old_height *
                      (old_aspect / new_aspect)) / 2)
        s = samples.narrow(-2, y, old_height - y *
                           2).narrow(-1, x, old_width - x * 2)
    else:
        s = samples

    if upscale_method == "bislerp":
        out = _bislerp(s, width, height)
    elif upscale_method == "lanczos":
        out = _lanczos(s, width, height)
    else:
        out = torch.nn.functional.interpolate(
            s, size=(height, width), mode=upscale_method)

    if len(orig_shape) == 4:
        return out

    out = out.reshape((orig_shape[0], -1, orig_shape[1]) + (height, width))
    return out.movedim(2, 1).reshape(orig_shape[:-2] + (height, width))


class ImageUtils:

    @staticmethod
    def load(image_path):
        img = Image.open(image_path)

        output_images = []
        output_masks = []
        w, h = None, None

        excluded_formats = ['MPO']

        for i in ImageSequence.Iterator(img):
            i = ImageOps.exif_transpose(i)

            if i.mode == 'I':
                i = i.point(lambda i: i * (1 / 255))
            image = i.convert("RGB")

            if len(output_images) == 0:
                w = image.size[0]
                h = image.size[1]

            if image.size[0] != w or image.size[1] != h:
                continue

            image = np.array(image).astype(np.float32) / 255.0
            image = torch.from_numpy(image)[None,]
            if 'A' in i.getbands():
                mask = np.array(i.getchannel('A')).astype(np.float32) / 255.0
                mask = 1. - torch.from_numpy(mask)
            elif i.mode == 'P' and 'transparency' in i.info:
                mask = np.array(i.convert('RGBA').getchannel(
                    'A')).astype(np.float32) / 255.0
                mask = 1. - torch.from_numpy(mask)
            else:
                mask = torch.zeros((64, 64), dtype=torch.float32, device="cpu")
            output_images.append(image)
            output_masks.append(mask.unsqueeze(0))

        if len(output_images) > 1 and img.format not in excluded_formats:
            output_image = torch.cat(output_images, dim=0)
            output_mask = torch.cat(output_masks, dim=0)
        else:
            output_image = output_images[0]
            output_mask = output_masks[0]

        return (output_image, output_mask)

    @staticmethod
    def pad(image, left, right, top, bottom, extra_padding, color, pad_mode, mask=None, target_width=None, target_height=None):
        B, H, W, C = image.shape
        # Resize masks to image dimensions if necessary
        if mask is not None:
            BM, HM, WM = mask.shape
            if HM != H or WM != W:
                mask = F.interpolate(mask.unsqueeze(1), size=(
                    H, W), mode='nearest-exact').squeeze(1)

        # Parse background color
        bg_color = [int(x.strip())/255.0 for x in color.split(",")]
        if len(bg_color) == 1:
            bg_color = bg_color * 3  # Grayscale to RGB
        bg_color = torch.tensor(
            bg_color, dtype=image.dtype, device=image.device)

        # Calculate padding sizes with extra padding
        if target_width is not None and target_height is not None:
            if extra_padding > 0:
                image = _common_upscale(image.movedim(-1, 1), W - extra_padding,
                                         H - extra_padding, "lanczos", "disabled").movedim(1, -1)
                B, H, W, C = image.shape

            padded_width = target_width
            padded_height = target_height
            pad_left = (padded_width - W) // 2
            pad_right = padded_width - W - pad_left
            pad_top = (padded_height - H) // 2
            pad_bottom = padded_height - H - pad_top
        else:
            pad_left = left + extra_padding
            pad_right = right + extra_padding
            pad_top = top + extra_padding
            pad_bottom = bottom + extra_padding

            padded_width = W + pad_left + pad_right
            padded_height = H + pad_top + pad_bottom

        # Pillarbox blur mode
        if pad_mode == "pillarbox_blur":
            def _gaussian_blur_nchw(img_nchw, sigma_px):
                if sigma_px <= 0:
                    return img_nchw
                radius = max(1, int(3.0 * float(sigma_px)))
                k = 2 * radius + 1
                x = torch.arange(-radius, radius + 1,
                                 device=img_nchw.device, dtype=img_nchw.dtype)
                k1 = torch.exp(-(x * x) /
                               (2.0 * float(sigma_px) * float(sigma_px)))
                k1 = k1 / k1.sum()
                kx = k1.view(1, 1, 1, k)
                ky = k1.view(1, 1, k, 1)
                c = img_nchw.shape[1]
                kx = kx.repeat(c, 1, 1, 1)
                ky = ky.repeat(c, 1, 1, 1)
                img_nchw = F.conv2d(
                    img_nchw, kx, padding=(0, radius), groups=c)
                img_nchw = F.conv2d(
                    img_nchw, ky, padding=(radius, 0), groups=c)
                return img_nchw

            out_image = torch.zeros(
                (B, padded_height, padded_width, C), dtype=image.dtype, device=image.device)
            for b in range(B):
                scale_fill = max(
                    padded_width / float(W), padded_height / float(H)) if (W > 0 and H > 0) else 1.0
                bg_w = max(1, int(round(W * scale_fill)))
                bg_h = max(1, int(round(H * scale_fill)))
                src_b = image[b].movedim(-1, 0).unsqueeze(0)
                bg = _common_upscale(src_b, bg_w, bg_h,
                                      "bilinear", crop="disabled")
                y0 = max(0, (bg_h - padded_height) // 2)
                x0 = max(0, (bg_w - padded_width) // 2)
                y1 = min(bg_h, y0 + padded_height)
                x1 = min(bg_w, x0 + padded_width)
                bg = bg[:, :, y0:y1, x0:x1]
                if bg.shape[2] != padded_height or bg.shape[3] != padded_width:
                    pad_h = padded_height - bg.shape[2]
                    pad_w = padded_width - bg.shape[3]
                    pad_top_fix = max(0, pad_h // 2)
                    pad_bottom_fix = max(0, pad_h - pad_top_fix)
                    pad_left_fix = max(0, pad_w // 2)
                    pad_right_fix = max(0, pad_w - pad_left_fix)
                    bg = F.pad(bg, (pad_left_fix, pad_right_fix,
                               pad_top_fix, pad_bottom_fix), mode="replicate")
                sigma = max(
                    1.0, 0.006 * float(min(padded_height, padded_width)))
                bg = _gaussian_blur_nchw(bg, sigma_px=sigma)
                if C >= 3:
                    r, g, bch = bg[:, 0:1], bg[:, 1:2], bg[:, 2:3]
                    luma = 0.2126 * r + 0.7152 * g + 0.0722 * bch
                    gray = torch.cat([luma, luma, luma], dim=1)
                    desat = 0.20
                    rgb = torch.cat([r, g, bch], dim=1)
                    rgb = rgb * (1.0 - desat) + gray * desat
                    bg[:, 0:3, :, :] = rgb
                dim = 0.35
                bg = torch.clamp(bg * dim, 0.0, 1.0)
                out_image[b] = bg.squeeze(0).movedim(0, -1)
            out_image[:, pad_top:pad_top+H, pad_left:pad_left+W, :] = image
            # Mask handling for pillarbox_blur
            if mask is not None:
                fg_mask = mask
                out_masks = torch.ones(
                    (B, padded_height, padded_width), dtype=image.dtype, device=image.device)
                out_masks[:, pad_top:pad_top+H, pad_left:pad_left+W] = fg_mask
            else:
                out_masks = torch.ones(
                    (B, padded_height, padded_width), dtype=image.dtype, device=image.device)
                out_masks[:, pad_top:pad_top+H, pad_left:pad_left+W] = 0.0
            return (out_image, out_masks)

        # Standard pad logic (edge/color)
        out_image = torch.zeros(
            (B, padded_height, padded_width, C), dtype=image.dtype, device=image.device)
        for b in range(B):
            if pad_mode == "edge":
                # Pad with edge color (mean)
                top_edge = image[b, 0, :, :]
                bottom_edge = image[b, H-1, :, :]
                left_edge = image[b, :, 0, :]
                right_edge = image[b, :, W-1, :]
                out_image[b, :pad_top, :, :] = top_edge.mean(dim=0)
                out_image[b, pad_top+H:, :, :] = bottom_edge.mean(dim=0)
                out_image[b, :, :pad_left, :] = left_edge.mean(dim=0)
                out_image[b, :, pad_left+W:, :] = right_edge.mean(dim=0)
                out_image[b, pad_top:pad_top+H,
                          pad_left:pad_left+W, :] = image[b]
            elif pad_mode == "edge_pixel":
                # Pad with exact edge pixel values
                for y in range(pad_top):
                    out_image[b, y, pad_left:pad_left+W, :] = image[b, 0, :, :]
                for y in range(pad_top+H, padded_height):
                    out_image[b, y, pad_left:pad_left +
                              W, :] = image[b, H-1, :, :]
                for x in range(pad_left):
                    out_image[b, pad_top:pad_top+H, x, :] = image[b, :, 0, :]
                for x in range(pad_left+W, padded_width):
                    out_image[b, pad_top:pad_top+H, x, :] = image[b, :, W-1, :]
                out_image[b, :pad_top, :pad_left, :] = image[b, 0, 0, :]
                out_image[b, :pad_top, pad_left+W:, :] = image[b, 0, W-1, :]
                out_image[b, pad_top+H:, :pad_left, :] = image[b, H-1, 0, :]
                out_image[b, pad_top+H:, pad_left +
                          W:, :] = image[b, H-1, W-1, :]
                out_image[b, pad_top:pad_top+H,
                          pad_left:pad_left+W, :] = image[b]
            else:
                # Pad with specified background color
                out_image[b, :, :, :] = bg_color.unsqueeze(0).unsqueeze(0)
                out_image[b, pad_top:pad_top+H,
                          pad_left:pad_left+W, :] = image[b]

        if mask is not None:
            out_masks = torch.nn.functional.pad(
                mask,
                (pad_left, pad_right, pad_top, pad_bottom),
                mode='replicate'
            )
        else:
            out_masks = torch.ones(
                (B, padded_height, padded_width), dtype=image.dtype, device=image.device)
            for m in range(B):
                out_masks[m, pad_top:pad_top+H, pad_left:pad_left+W] = 0.0

        return (out_image, out_masks)

    @staticmethod
    def resize(image, width, height, keep_proportion, upscale_method, divisible_by, pad_color, crop_position, device="cpu", mask=None, per_batch=64):
        B, H, W, C = image.shape

        if device == "gpu":
            if upscale_method == "lanczos":
                raise Exception("Lanczos is not supported on the GPU")
            device = torch.device(torch.cuda.current_device())
        else:
            device = torch.device("cpu")

        pillarbox_blur = keep_proportion == "pillarbox_blur"

        # Initialize padding variables
        pad_left = pad_right = pad_top = pad_bottom = 0

        if keep_proportion in ["resize", "total_pixels"] or keep_proportion.startswith("pad") or pillarbox_blur:
            if keep_proportion == "total_pixels":
                total_pixels = width * height
                aspect_ratio = W / H
                new_height = int(math.sqrt(total_pixels / aspect_ratio))
                new_width = int(math.sqrt(total_pixels * aspect_ratio))

            # If one of the dimensions is zero, calculate it to maintain the aspect ratio
            elif width == 0 and height == 0:
                new_width = W
                new_height = H
            elif width == 0 and height != 0:
                ratio = height / H
                new_width = round(W * ratio)
                new_height = height
            elif height == 0 and width != 0:
                ratio = width / W
                new_width = width
                new_height = round(H * ratio)
            elif width != 0 and height != 0:
                ratio = min(width / W, height / H)
                new_width = round(W * ratio)
                new_height = round(H * ratio)
            else:
                new_width = width
                new_height = height

            if keep_proportion.startswith("pad") or pillarbox_blur:
                # Calculate padding based on position
                if crop_position == "center":
                    pad_left = (width - new_width) // 2
                    pad_right = width - new_width - pad_left
                    pad_top = (height - new_height) // 2
                    pad_bottom = height - new_height - pad_top
                elif crop_position == "top":
                    pad_left = (width - new_width) // 2
                    pad_right = width - new_width - pad_left
                    pad_top = 0
                    pad_bottom = height - new_height
                elif crop_position == "bottom":
                    pad_left = (width - new_width) // 2
                    pad_right = width - new_width - pad_left
                    pad_top = height - new_height
                    pad_bottom = 0
                elif crop_position == "left":
                    pad_left = 0
                    pad_right = width - new_width
                    pad_top = (height - new_height) // 2
                    pad_bottom = height - new_height - pad_top
                elif crop_position == "right":
                    pad_left = width - new_width
                    pad_right = 0
                    pad_top = (height - new_height) // 2
                    pad_bottom = height - new_height - pad_top

            width = new_width
            height = new_height
        else:
            if width == 0:
                width = W
            if height == 0:
                height = H

        if divisible_by > 1:
            width = width - (width % divisible_by)
            height = height - (height % divisible_by)

        # Preflight estimate (log-only when batching is active)
        if per_batch != 0 and B > per_batch:
            try:
                bytes_per_elem = image.element_size()  # typically 4 for float32
                est_total_bytes = B * height * width * C * bytes_per_elem
                est_mb = est_total_bytes / (1024 * 1024)
                print(
                    f"resize estimated output ~{est_mb:.2f} MB; batching {per_batch}/{B}")
            except:
                pass

        def _process_subbatch(in_image, in_mask, pad_left, pad_right, pad_top, pad_bottom):
            # Avoid unnecessary clones; only move if needed
            out_image = in_image if in_image.device == device else in_image.to(
                device)
            out_mask = None if in_mask is None else (
                in_mask if in_mask.device == device else in_mask.to(device))

            # Crop logic
            if keep_proportion == "crop":
                old_height = out_image.shape[-3]
                old_width = out_image.shape[-2]
                old_aspect = old_width / old_height
                new_aspect = width / height
                if old_aspect > new_aspect:
                    crop_w = round(old_height * new_aspect)
                    crop_h = old_height
                else:
                    crop_w = old_width
                    crop_h = round(old_width / new_aspect)
                if crop_position == "center":
                    x = (old_width - crop_w) // 2
                    y = (old_height - crop_h) // 2
                elif crop_position == "top":
                    x = (old_width - crop_w) // 2
                    y = 0
                elif crop_position == "bottom":
                    x = (old_width - crop_w) // 2
                    y = old_height - crop_h
                elif crop_position == "left":
                    x = 0
                    y = (old_height - crop_h) // 2
                elif crop_position == "right":
                    x = old_width - crop_w
                    y = (old_height - crop_h) // 2
                out_image = out_image.narrow(-2, x,
                                             crop_w).narrow(-3, y, crop_h)
                if out_mask is not None:
                    out_mask = out_mask.narrow(-1, x,
                                               crop_w).narrow(-2, y, crop_h)

            out_image = _common_upscale(
                out_image.movedim(-1, 1), width, height, upscale_method, crop="disabled").movedim(1, -1)
            if out_mask is not None:
                if upscale_method == "lanczos":
                    out_mask = _common_upscale(out_mask.unsqueeze(1).repeat(
                        1, 3, 1, 1), width, height, upscale_method, crop="disabled").movedim(1, -1)[:, :, :, 0]
                else:
                    out_mask = _common_upscale(out_mask.unsqueeze(
                        1), width, height, upscale_method, crop="disabled").squeeze(1)

            # Pad logic
            if (keep_proportion.startswith("pad") or pillarbox_blur) and (pad_left > 0 or pad_right > 0 or pad_top > 0 or pad_bottom > 0):
                padded_width = width + pad_left + pad_right
                padded_height = height + pad_top + pad_bottom
                if divisible_by > 1:
                    width_remainder = padded_width % divisible_by
                    height_remainder = padded_height % divisible_by
                    if width_remainder > 0:
                        extra_width = divisible_by - width_remainder
                        pad_right += extra_width
                    if height_remainder > 0:
                        extra_height = divisible_by - height_remainder
                        pad_bottom += extra_height

                pad_mode = (
                    "pillarbox_blur" if pillarbox_blur else
                    "edge" if keep_proportion == "pad_edge" else
                    "edge_pixel" if keep_proportion == "pad_edge_pixel" else
                    "color"
                )
                out_image, out_mask = ImageUtils.pad(
                    out_image, pad_left, pad_right, pad_top, pad_bottom, 0, pad_color, pad_mode, mask=out_mask)

            return out_image, out_mask

        # If batching disabled (per_batch==0) or batch fits, process whole batch
        if per_batch == 0 or B <= per_batch:
            out_image, out_mask = _process_subbatch(
                image, mask, pad_left, pad_right, pad_top, pad_bottom)
        else:
            chunks = []
            mask_chunks = [] if mask is not None else None
            total_batches = (B + per_batch - 1) // per_batch
            current_batch = 0
            for start_idx in range(0, B, per_batch):
                current_batch += 1
                end_idx = min(start_idx + per_batch, B)
                sub_img = image[start_idx:end_idx]
                sub_mask = mask[start_idx:end_idx] if mask is not None else None
                sub_out_img, sub_out_mask = _process_subbatch(
                    sub_img, sub_mask, pad_left, pad_right, pad_top, pad_bottom)
                chunks.append(sub_out_img.cpu())
                if mask is not None:
                    mask_chunks.append(sub_out_mask.cpu()
                                       if sub_out_mask is not None else None)
                print(
                    f"resize batch {current_batch}/{total_batches} · images {end_idx}/{B}")
            out_image = torch.cat(chunks, dim=0)
            if mask is not None and any(m is not None for m in mask_chunks):
                out_mask = torch.cat(
                    [m for m in mask_chunks if m is not None], dim=0)
            else:
                out_mask = None

        return (out_image.cpu(), out_image.shape[2], out_image.shape[1], out_mask.cpu() if out_mask is not None else torch.zeros(64, 64, device=torch.device("cpu"), dtype=torch.float32))
