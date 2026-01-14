import math
import torch
import torch.nn as nn
from einops import rearrange
from contextlib import nullcontext
import logging

# Flash Attention imports
try:
    import flash_attn_interface
    FLASH_ATTN_3_AVAILABLE = True
except Exception as e:
    FLASH_ATTN_3_AVAILABLE = False

try:
    import flash_attn
    FLASH_ATTN_2_AVAILABLE = True
except Exception as e:
    FLASH_ATTN_2_AVAILABLE = False
        
# Sage Attention imports
try:
    from sageattention import sageattn
    @torch.compiler.disable()
    def sageattn_func(q, k, v, attn_mask=None, dropout_p=0, is_causal=False, tensor_layout="HND"):
        if not (q.dtype == k.dtype == v.dtype):
            return sageattn(q, k.to(q.dtype), v.to(q.dtype), attn_mask=attn_mask, dropout_p=dropout_p, is_causal=is_causal, tensor_layout=tensor_layout)
        elif q.dtype == torch.float32:
            return sageattn(q.to(torch.float16), k.to(torch.float16), v.to(torch.float16), attn_mask=attn_mask, dropout_p=dropout_p, is_causal=is_causal, tensor_layout=tensor_layout).to(torch.float32)
        else:
            return sageattn(q, k, v, attn_mask=attn_mask, dropout_p=dropout_p, is_causal=is_causal, tensor_layout=tensor_layout)

    def sageattn_func_compiled(q, k, v, attn_mask=None, dropout_p=0, is_causal=False, tensor_layout="HND"):
        if not (q.dtype == k.dtype == v.dtype):
            return sageattn(q, k.to(q.dtype), v.to(q.dtype), attn_mask=attn_mask, dropout_p=dropout_p, is_causal=is_causal, tensor_layout=tensor_layout)
        elif q.dtype == torch.float32:
            return sageattn(q.to(torch.float16), k.to(torch.float16), v.to(torch.float16), attn_mask=attn_mask, dropout_p=dropout_p, is_causal=is_causal, tensor_layout=tensor_layout).to(torch.float32)
        else:
            return sageattn(q, k, v, attn_mask=attn_mask, dropout_p=dropout_p, is_causal=is_causal, tensor_layout=tensor_layout)
except Exception as e:
    logging.warning(f"Warning: Could not load sageattention: {str(e)}")
    if isinstance(e, ModuleNotFoundError):
        logging.warning("sageattention package is not installed, sageattention will not be available")
    elif isinstance(e, ImportError) and "DLL" in str(e):
        logging.warning("sageattention DLL loading error, sageattention will not be available")
    sageattn_func = None

try:
    from sageattn3 import sageattn3_blackwell as sageattn_blackwell
except:
    try:
        from sageattn import sageattn_blackwell
    except:
        SAGE3_AVAILABLE = False

try: 
    from sageattention import sageattn_varlen
    @torch.compiler.disable()
    def sageattn_varlen_func(q, k, v, q_lens, k_lens, max_seqlen_q, max_seqlen_k, dropout_p=0, is_causal=False):
        cu_seqlens_q = torch.tensor([0] + list(torch.cumsum(torch.tensor(q_lens), dim=0)), device=q.device, dtype=torch.int32)
        cu_seqlens_k = torch.tensor([0] + list(torch.cumsum(torch.tensor(k_lens), dim=0)), device=q.device, dtype=torch.int32)
        if not (q.dtype == k.dtype == v.dtype):
            return sageattn_varlen(q, k.to(q.dtype), v.to(q.dtype), cu_seqlens_q, cu_seqlens_k, max_seqlen_q, max_seqlen_k, dropout_p=dropout_p, is_causal=is_causal)
        elif q.dtype == torch.float32:
            return sageattn_varlen(q.to(torch.float16), k.to(torch.float16), v.to(torch.float16), cu_seqlens_q, cu_seqlens_k, max_seqlen_q, max_seqlen_k, dropout_p=dropout_p, is_causal=is_causal).to(torch.float32)
        else:
            return sageattn_varlen(q, k, v, cu_seqlens_q, cu_seqlens_k, max_seqlen_q, max_seqlen_k, dropout_p=dropout_p, is_causal=is_causal)
except: 
    sageattn_varlen_func = None


def flash_attention(
    q,
    k,
    v,
    q_lens=None,
    k_lens=None,
    dropout_p=0.,
    softmax_scale=None,
    q_scale=None,
    causal=False,
    window_size=(-1, -1),
    deterministic=False,
    dtype=torch.bfloat16,
    version=None,
):
    """
    q:              [B, Lq, Nq, C1].
    k:              [B, Lk, Nk, C1].
    v:              [B, Lk, Nk, C2]. Nq must be divisible by Nk.
    q_lens:         [B].
    k_lens:         [B].
    dropout_p:      float. Dropout probability.
    softmax_scale:  float. The scaling of QK^T before applying softmax.
    causal:         bool. Whether to apply causal attention mask.
    window_size:    (left right). If not (-1, -1), apply sliding window local attention.
    deterministic:  bool. If True, slightly slower and uses more memory.
    dtype:          torch.dtype. Apply when dtype of q/k/v is not float16/bfloat16.
    """
    half_dtypes = (torch.float16, torch.bfloat16)
    #assert dtype in half_dtypes
    #assert q.device.type == 'cuda' and q.size(-1) <= 256

    # params
    b, lq, lk, out_dtype = q.size(0), q.size(1), k.size(1), q.dtype

    def half(x):
        return x if x.dtype in half_dtypes else x.to(dtype)

    # preprocess query
    if q_lens is None:
        q = half(q.flatten(0, 1))
        q_lens = torch.tensor(
            [lq] * b, dtype=torch.int32).to(
                device=q.device, non_blocking=True)
    else:
        q = half(torch.cat([u[:v] for u, v in zip(q, q_lens)]))

    # preprocess key, value
    if k_lens is None:
        k = half(k.flatten(0, 1))
        v = half(v.flatten(0, 1))
        k_lens = torch.tensor(
            [lk] * b, dtype=torch.int32).to(
                device=k.device, non_blocking=True)
    else:
        k = half(torch.cat([u[:v] for u, v in zip(k, k_lens)]))
        v = half(torch.cat([u[:v] for u, v in zip(v, k_lens)]))

    q = q.to(v.dtype)
    k = k.to(v.dtype)

    if q_scale is not None:
        q = q * q_scale

    if version is not None and version == 3 and not FLASH_ATTN_3_AVAILABLE:
        logging.warning('Flash attention 3 is not available, use flash attention 2 instead.')

    # apply attention
    if (version is None or version == 3) and FLASH_ATTN_3_AVAILABLE:
        # Note: dropout_p, window_size are not supported in FA3 now.
        x = flash_attn_interface.flash_attn_varlen_func(
            q=q,
            k=k,
            v=v,
            cu_seqlens_q=torch.cat([q_lens.new_zeros([1]), q_lens]).cumsum(
                0, dtype=torch.int32).to(q.device, non_blocking=True),
            cu_seqlens_k=torch.cat([k_lens.new_zeros([1]), k_lens]).cumsum(
                0, dtype=torch.int32).to(q.device, non_blocking=True),
            seqused_q=None,
            seqused_k=None,
            max_seqlen_q=lq,
            max_seqlen_k=lk,
            softmax_scale=softmax_scale,
            causal=causal,
            deterministic=deterministic)[0].unflatten(0, (b, lq))
    else:
        assert FLASH_ATTN_2_AVAILABLE
        x = flash_attn.flash_attn_varlen_func(
            q=q,
            k=k,
            v=v,
            cu_seqlens_q=torch.cat([q_lens.new_zeros([1]), q_lens]).cumsum(
                0, dtype=torch.int32).to(q.device, non_blocking=True),
            cu_seqlens_k=torch.cat([k_lens.new_zeros([1]), k_lens]).cumsum(
                0, dtype=torch.int32).to(q.device, non_blocking=True),
            max_seqlen_q=lq,
            max_seqlen_k=lk,
            dropout_p=dropout_p,
            softmax_scale=softmax_scale,
            causal=causal,
            window_size=window_size,
            deterministic=deterministic).unflatten(0, (b, lq))

    # output
    return x.type(out_dtype)


def attention(
    q,
    k,
    v,
    q_lens=None,
    k_lens=None,
    max_seqlen_q=None,
    max_seqlen_k=None,
    dropout_p=0.,
    softmax_scale=None,
    q_scale=None,
    causal=False,
    window_size=(-1, -1),
    deterministic=False,
    dtype=torch.bfloat16,
    attention_mode='sdpa',
    attn_mask=None,
):  
    if "flash" in attention_mode:
        if attention_mode == 'flash_attn_2':
            fa_version = 2
        elif attention_mode == 'flash_attn_3':
            fa_version = 3
        return flash_attention(
            q=q,
            k=k,
            v=v,
            q_lens=q_lens,
            k_lens=k_lens,
            dropout_p=dropout_p,
            softmax_scale=softmax_scale,
            q_scale=q_scale,
            causal=causal,
            window_size=window_size,
            deterministic=deterministic,
            dtype=dtype,
            version=fa_version,
        )
    elif attention_mode == 'sdpa':
        if not (q.dtype == k.dtype == v.dtype):
            return torch.nn.functional.scaled_dot_product_attention(q.transpose(1, 2), k.transpose(1, 2).to(q.dtype), v.transpose(1, 2).to(q.dtype), attn_mask=attn_mask).transpose(1, 2).contiguous()
        return torch.nn.functional.scaled_dot_product_attention(q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2), attn_mask=attn_mask).transpose(1, 2).contiguous()
    elif attention_mode == 'sageattn_3':
        return sageattn_blackwell(
            q.transpose(1,2), 
            k.transpose(1,2), 
            v.transpose(1,2), 
            per_block_mean=False #seems necessary for reasonable VRAM usage, not sure of other implications
            ).transpose(1,2).contiguous()
    elif attention_mode == 'sageattn_varlen':
        return sageattn_varlen_func(
                q,k,v,
                q_lens=q_lens,
                k_lens=k_lens,
                max_seqlen_k=max_seqlen_k,
                max_seqlen_q=max_seqlen_q
            )
    elif attention_mode == 'sageattn_compiled':
        return sageattn_func_compiled(q, k, v, tensor_layout="NHD").contiguous()
    else:
        return sageattn_func(q, k, v, tensor_layout="NHD").contiguous()


def apply_rope_comfy(xq, xk, freqs_cis):
    xq_ = xq.to(dtype=freqs_cis.dtype).reshape(*xq.shape[:-1], -1, 1, 2)
    xk_ = xk.to(dtype=freqs_cis.dtype).reshape(*xk.shape[:-1], -1, 1, 2)
    xq_out = freqs_cis[..., 0] * xq_[..., 0] + freqs_cis[..., 1] * xq_[..., 1]
    xk_out = freqs_cis[..., 0] * xk_[..., 0] + freqs_cis[..., 1] * xk_[..., 1]
    return xq_out.reshape(*xq.shape).type_as(xq), xk_out.reshape(*xk.shape).type_as(xk)

def rope_riflex(pos, dim, i, theta, L_test, k, ntk_factor=1.0):
    assert dim % 2 == 0
    # if mm.is_device_mps(pos.device) or mm.is_intel_xpu() or mm.is_directml_enabled():
    #     device = torch.device("cpu")
    # else:
    device = pos.device

    if ntk_factor != 1.0:
        theta *= ntk_factor

    scale = torch.linspace(0, (dim - 2) / dim, steps=dim//2, dtype=torch.float64, device=device)
    omega = 1.0 / (theta**scale)

    # RIFLEX modification - adjust last frequency component if L_test and k are provided
    if i==0 and k > 0 and L_test:
        omega[k-1] = 0.9 * 2 * torch.pi / L_test

    out = torch.einsum("...n,d->...nd", pos.to(dtype=torch.float32, device=device), omega)
    out = torch.stack([torch.cos(out), -torch.sin(out), torch.sin(out), torch.cos(out)], dim=-1)
    out = rearrange(out, "b n d (i j) -> b n d i j", i=2, j=2)
    return out.to(dtype=torch.float32, device=pos.device)

class EmbedND_RifleX(nn.Module):
    def __init__(self, dim, theta, axes_dim, num_frames, k):
        super().__init__()
        self.dim = dim
        self.theta = theta
        self.axes_dim = axes_dim
        self.num_frames = num_frames
        self.k = k

    def forward(self, ids, ntk_factor=[1.0,1.0,1.0]):
        n_axes = ids.shape[-1]
        emb = torch.cat(
            [rope_riflex(
                ids[..., i], 
                self.axes_dim[i], 
                i, #f h w
                self.theta, 
                self.num_frames, 
                self.k,
                ntk_factor[i])
            for i in range(n_axes)],
            dim=-3,
        )
        return emb.unsqueeze(1)



def sinusoidal_embedding_1d(dim, position):
    # preprocess
    assert dim % 2 == 0
    half = dim // 2
    position = position.type(torch.float32)

    # calculation
    sinusoid = torch.outer(
        position, torch.pow(10000, -torch.arange(half).to(position).div(half)))
    x = torch.cat([torch.cos(sinusoid), torch.sin(sinusoid)], dim=1)
    return x

@torch.autocast(device_type=torch.device(torch.cuda.current_device()).type, enabled=False)
@torch.compiler.disable()
def rope_apply(x, grid_sizes, freqs, reverse_time=False):
    x_ndim = grid_sizes.shape[-1]
    if x_ndim == 3:
        return rope_apply_3d(x, grid_sizes, freqs, reverse_time=reverse_time)
    else:
        return rope_apply_1d(x, grid_sizes, freqs)

def rope_apply_3d(x, grid_sizes, freqs, reverse_time=False):
    n, c = x.size(2), x.size(3) // 2

    # split freqs
    freqs = freqs.split([c - 2 * (c // 3), c // 3, c // 3], dim=1)

    # loop over samples
    output = []
    for i, (f, h, w) in enumerate(grid_sizes.tolist()):
        seq_len = f * h * w

        # precompute multipliers
        x_i = torch.view_as_complex(x[i, :seq_len].to(torch.float64).reshape(
            seq_len, n, -1, 2))
        if reverse_time:
            time_freqs = freqs[0][:f].view(f, 1, 1, -1)
            time_freqs = torch.flip(time_freqs, dims=[0])
            time_freqs = time_freqs.expand(f, h, w, -1)
            
            spatial_freqs = torch.cat([
                freqs[1][:h].view(1, h, 1, -1).expand(f, h, w, -1),
                freqs[2][:w].view(1, 1, w, -1).expand(f, h, w, -1)
            ], dim=-1)
            
            freqs_i = torch.cat([time_freqs, spatial_freqs], dim=-1).reshape(seq_len, 1, -1)
        else:
            freqs_i = torch.cat([
                freqs[0][:f].view(f, 1, 1, -1).expand(f, h, w, -1),
                freqs[1][:h].view(1, h, 1, -1).expand(f, h, w, -1),
                freqs[2][:w].view(1, 1, w, -1).expand(f, h, w, -1)
            ],
                                dim=-1).reshape(seq_len, 1, -1)

        # apply rotary embedding
        x_i = torch.view_as_real(x_i * freqs_i).flatten(2)
        x_i = torch.cat([x_i, x[i, seq_len:]])

        # append to collection
        output.append(x_i)
    return torch.stack(output).to(x.dtype)


def rope_apply_1d(x, grid_sizes, freqs):
    n, c = x.size(2), x.size(3) // 2 ## b l h d
    c_rope = freqs.shape[1]  # number of complex dims to rotate
    assert c_rope <= c, "RoPE dimensions cannot exceed half of hidden size"
    
    # loop over samples
    output = []
    for i, (l, ) in enumerate(grid_sizes.tolist()):
        seq_len = l
        # precompute multipliers
        x_i = torch.view_as_complex(x[i, :seq_len].to(torch.float64).reshape(
            seq_len, n, -1, 2)) # [l n d//2]
        x_i_rope = x_i[:, :, :c_rope] * freqs[:seq_len, None, :]  # [L, N, c_rope]
        x_i_passthrough = x_i[:, :, c_rope:]  # untouched dims
        x_i = torch.cat([x_i_rope, x_i_passthrough], dim=2)

        # apply rotary embedding
        x_i = torch.view_as_real(x_i).flatten(2)
        x_i = torch.cat([x_i, x[i, seq_len:]])

        # append to collection
        output.append(x_i)
    return torch.stack(output).to(x.dtype)

# class WanLayerNorm(nn.LayerNorm):

#     def __init__(self, dim, eps=1e-6, elementwise_affine=False):
#         super().__init__(dim, elementwise_affine=elementwise_affine, eps=eps)

#     def forward(self, x):
#         r"""
#         Args:
#             x(Tensor): Shape [B, L, C]
#         """
#         return super().forward(x)

class WanRMSNorm(nn.Module):

    def __init__(self, dim, eps=1e-5):
        super().__init__()
        self.dim = dim
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x, num_chunks=1):
        r"""
        Args:
            x(Tensor): Shape [B, L, C]
        """
        use_chunked = num_chunks > 1
        if use_chunked:
            return self.forward_chunked(x, num_chunks)
        else:
            return self._norm(x.to(self.weight.dtype)) * self.weight

    def _norm(self, x):
        return x * (torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)).to(x.dtype)

    def forward_chunked(self, x, num_chunks=4):
        output = torch.empty_like(x)
        
        chunk_sizes = [x.shape[1] // num_chunks + (1 if i < x.shape[1] % num_chunks else 0) 
                    for i in range(num_chunks)]
        
        start_idx = 0
        for size in chunk_sizes:
            end_idx = start_idx + size
            
            chunk = x[:, start_idx:end_idx, :]
            
            norm_factor = torch.rsqrt(chunk.pow(2).mean(dim=-1, keepdim=True) + self.eps)
            output[:, start_idx:end_idx, :] = chunk * norm_factor.to(chunk.dtype) * self.weight

            start_idx = end_idx
            
        return output
    
class WanFusedRMSNorm(nn.RMSNorm):
    def forward(self, x, num_chunks=1):
        use_chunked = num_chunks > 1
        if use_chunked:
            return self.forward_chunked(x, num_chunks)
        else:
            return super().forward(x)

    def forward_chunked(self, x, num_chunks=4):
        output = torch.empty_like(x)
        
        chunk_sizes = [x.shape[1] // num_chunks + (1 if i < x.shape[1] % num_chunks else 0) 
                    for i in range(num_chunks)]
        
        start_idx = 0
        for size in chunk_sizes:
            end_idx = start_idx + size
            chunk = x[:, start_idx:end_idx, :]
            output[:, start_idx:end_idx, :] = super().forward(chunk)
            start_idx = end_idx
            
        return output

class WanSelfAttention(nn.Module):

    def __init__(self,
                 in_features,
                 out_features,
                 num_heads,
                 qk_norm=True,
                 eps=1e-6,
                 attention_mode="sdpa",
                 rms_norm_function="default"):
        assert out_features % num_heads == 0
        super().__init__()
        self.dim = min(in_features, out_features)
        self.num_heads = num_heads
        self.head_dim = out_features // num_heads
        self.qk_norm = qk_norm
        self.eps = eps
        self.attention_mode = attention_mode

        # layers
        # 把Token映射成Query、Key、Value
        self.q = nn.Linear(in_features, out_features)
        self.k = nn.Linear(in_features, out_features)
        self.v = nn.Linear(in_features, out_features)
        self.o = nn.Linear(in_features, out_features)

        norm_dim = self.dim

        # 通过RMSNorm对Q、K进行归一化，稳定注意力分布
        if rms_norm_function=="pytorch":
            self.norm_q = WanFusedRMSNorm(norm_dim, eps=eps) if qk_norm else nn.Identity()
            self.norm_k = WanFusedRMSNorm(norm_dim, eps=eps) if qk_norm else nn.Identity()
        else:
            self.norm_q = WanRMSNorm(norm_dim, eps=eps) if qk_norm else nn.Identity()
            self.norm_k = WanRMSNorm(norm_dim, eps=eps) if qk_norm else nn.Identity()

    def qkv_fn(self, x):
        b, s, n, d = *x.shape[:2], self.num_heads, self.head_dim
        q = self.norm_q(self.q(x).to(self.norm_q.weight.dtype)).to(x.dtype).view(b, s, n, d)
        k = self.norm_k(self.k(x).to(self.norm_k.weight.dtype)).to(x.dtype).view(b, s, n, d)
        v = self.v(x).view(b, s, n, d)
        return q, k, v
    
    def forward(self, q, k, v, seq_lens, lynx_ref_feature=None, lynx_ref_scale=1.0, attention_mode_override=None):
        r"""
        Args:
            x(Tensor): Shape [B, L, num_heads, C / num_heads]
            seq_lens(Tensor): Shape [B]
            grid_sizes(Tensor): Shape [B, 3], the second dimension contains (F, H, W)
            freqs(Tensor): Rope freqs, shape [1024, C / num_heads / 2]
        """
        attention_mode = self.attention_mode
        if attention_mode_override is not None:
            attention_mode = attention_mode_override

        x = attention(q, k, v, k_lens=seq_lens, attention_mode=attention_mode)

        # output
        return self.o(x.flatten(2))
    
    def normalized_attention_guidance(self, b, n, d, q, context, nag_context=None, nag_params={}):
        # NAG text attention
        context_positive = context
        context_negative = nag_context
        nag_scale = nag_params['nag_scale']
        nag_alpha = nag_params['nag_alpha']
        nag_tau = nag_params['nag_tau']

        k_positive = self.norm_k(self.k(context_positive).to(self.norm_k.weight.dtype)).view(b, -1, n, d).to(q.dtype)
        v_positive = self.v(context_positive).view(b, -1, n, d)
        k_negative = self.norm_k(self.k(context_negative).to(self.norm_k.weight.dtype)).view(b, -1, n, d).to(q.dtype)
        v_negative = self.v(context_negative).view(b, -1, n, d)

        x_positive = attention(q, k_positive, v_positive, attention_mode=self.attention_mode)
        x_positive = x_positive.flatten(2)

        x_negative = attention(q, k_negative, v_negative, attention_mode=self.attention_mode)
        x_negative = x_negative.flatten(2)

        nag_guidance = x_positive * nag_scale - x_negative * (nag_scale - 1)
        
        norm_positive = torch.norm(x_positive, p=1, dim=-1, keepdim=True)
        norm_guidance = torch.norm(nag_guidance, p=1, dim=-1, keepdim=True)
        
        scale = norm_guidance / norm_positive
        scale = torch.nan_to_num(scale, nan=10.0)
        
        mask = scale > nag_tau
        adjustment = (norm_positive * nag_tau) / (norm_guidance + 1e-7)
        nag_guidance = torch.where(mask, nag_guidance * adjustment, nag_guidance)
        del mask, adjustment
        
        return nag_guidance * nag_alpha + x_positive * (1 - nag_alpha)

class WanI2VCrossAttention(WanSelfAttention):

    def __init__(self, in_features, out_features, num_heads, qk_norm=True, eps=1e-6, attention_mode='sdpa', rms_norm_function="default", **kwargs):
        super().__init__(in_features, out_features, num_heads, qk_norm, eps, rms_norm_function=rms_norm_function)
        # 图像条件的k、v层
        self.k_img = nn.Linear(in_features, out_features)
        self.v_img = nn.Linear(in_features, out_features)
        # 通过RMSNorm对k进行归一化
        self.norm_k_img = WanRMSNorm(out_features, eps=eps) if qk_norm else nn.Identity()
        self.attention_mode = attention_mode

    def forward(self, x, context, clip_embed=None, nag_params={}, nag_context=None, is_uncond=False, rope_func="comfy",  **kwargs):
        r"""
        Args:
            x(Tensor): Shape [B, L1, C]
            context(Tensor): Shape [B, L2, C]
        """
        b, n, d = x.size(0), self.num_heads, self.head_dim
        # compute query
        q = self.norm_q(self.q(x).to(self.norm_q.weight.dtype),num_chunks=2 if rope_func == "comfy_chunked" else 1).view(b, -1, n, d).to(x.dtype)

        if nag_context is not None and not is_uncond:
            x_text = self.normalized_attention_guidance(b, n, d, q, context, nag_context, nag_params)
        else:
            # text attention
            k = self.norm_k(self.k(context).to(self.norm_k.weight.dtype)).view(b, -1, n, d).to(x.dtype)
            v = self.v(context).view(b, -1, n, d)
            x_text = attention(q, k, v, attention_mode=self.attention_mode).flatten(2)

        #img attention
        if clip_embed is not None:
            k_img = self.norm_k_img(self.k_img(clip_embed).to(self.norm_k_img.weight.dtype)).view(b, -1, n, d).to(x.dtype)
            v_img = self.v_img(clip_embed).view(b, -1, n, d)
            img_x = attention(q, k_img, v_img, attention_mode=self.attention_mode).flatten(2)
            x = x_text + img_x
        else:
            x = x_text

        return self.o(x)
 
class WanAttentionBlock(nn.Module):

    def __init__(self, in_features, out_features, ffn_dim, ffn2_dim, num_heads,
                qk_norm=True, eps=1e-6, attention_mode="sdpa", rope_func="comfy", rms_norm_function="default",
                block_idx=0):
        super().__init__()
        self.dim = out_features
        self.ffn_dim = ffn_dim
        self.num_heads = num_heads
        self.head_dim = out_features // num_heads
        self.qk_norm = qk_norm
        self.eps = eps
        self.attention_mode = attention_mode
        self.rope_func = rope_func
        self.block_idx = block_idx

        # layers
        # 第一层归一化，对输入的Token做LayerNorm，使特征分布更稳定
        self.norm1 = nn.LayerNorm(self.dim, eps, elementwise_affine=False)
        # 自注意力模块，负责在Token之间做全局信息交互
        self.self_attn = WanSelfAttention(in_features, out_features, num_heads, qk_norm, eps, self.attention_mode, rms_norm_function=rms_norm_function)

        # 跨注意力前的归一化层
        self.norm3 = nn.LayerNorm(out_features, eps, elementwise_affine=True)
        # 图像文本的自注意力模块，负责融合图像文本信息
        self.cross_attn = WanI2VCrossAttention(in_features, out_features, num_heads, qk_norm, eps, rms_norm_function=rms_norm_function,
                                                                          head_norm=False)
        # 归一化                                                                  
        self.norm2 = nn.LayerNorm(self.dim, eps, elementwise_affine=False)
        # 前馈网络，在注意力之后，做非线性变换
        self.ffn = nn.Sequential(
            nn.Linear(in_features, ffn_dim), nn.GELU(approximate='tanh'),
            nn.Linear(ffn2_dim, out_features))
      
        # 时间条件的参数
        self.modulation = nn.Parameter(torch.randn(1, 6, out_features) / in_features**0.5)
     
        self.seg_idx = None
      

    def get_mod(self, e, modulation):
        if e.dim() == 3:
            if e.shape[-1] == 512:
                e = self.modulation(e)
                return e.unsqueeze(2).chunk(6, dim=-1)
            return (modulation + e).chunk(6, dim=1) # 1, 6, dim
        elif e.dim() == 4:
            e_mod = modulation.unsqueeze(2) + e
            return [ei.squeeze(1) for ei in e_mod.unbind(dim=1)]


    def modulate(self, norm_x, shift_msa, scale_msa, seg_idx=None):
        """
        Modulate x with shift and scale. If seg_idx is provided, apply segmented modulation.
        """
        if seg_idx is not None:
            parts = []
            for i in range(2):
                part = torch.addcmul(
                    shift_msa[:, i:i + 1],
                    norm_x[:, seg_idx[i]:seg_idx[i + 1]],
                    1 + scale_msa[:, i:i + 1]
                )
                parts.append(part)
            norm_x = torch.cat(parts, dim=1)
            return norm_x
        else:
            return torch.addcmul(shift_msa, norm_x, 1 + scale_msa)
    
    def ffn_chunked(self, x, shift_mlp, scale_mlp, num_chunks=4):
        modulated_input = torch.addcmul(shift_mlp, self.norm2(x.to(shift_mlp.dtype)), 1 + scale_mlp).to(x.dtype)
        
        result = torch.empty_like(x)
        seq_len = modulated_input.shape[1]
        
        chunk_sizes = [seq_len // num_chunks + (1 if i < seq_len % num_chunks else 0) 
                    for i in range(num_chunks)]
        
        start_idx = 0
        for size in chunk_sizes:
            end_idx = start_idx + size
            chunk = modulated_input[:, start_idx:end_idx, :]
            result[:, start_idx:end_idx, :] = self.ffn(chunk)
            start_idx = end_idx
        
        return result

    #region attention forward
    def forward(
        self, x, e, seq_lens, grid_sizes, freqs, context,
        clip_embed=None,
        audio_scale=1.0, #fantasytalking
        nag_params={}, nag_context=None, #normalized attention guidance
        is_uncond=False,
        multitalk_audio_embedding=None, human_num=0, #multitalk
    ):
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.get_mod(e.to(x.device), self.modulation)
        del e
        input_dtype = x.dtype
       
        input_x = self.modulate(self.norm1(x.to(shift_msa.dtype)), shift_msa, scale_msa, seg_idx=self.seg_idx).to(input_dtype)

        del shift_msa, scale_msa

        # self-attention
        x_ref_attn_map = None

        q, k, v = self.self_attn.qkv_fn(input_x)
        q, k = apply_rope_comfy(q, k, freqs)
          
        y = self.self_attn.forward(q, k, v, seq_lens, lynx_ref_feature=None, lynx_ref_scale=1.0)
        
        x = x.addcmul(y, gate_msa)
        del y, gate_msa

        # cross-attention & ffn function
        if context is not None:
            x = x + self.cross_attn(self.norm3(x.to(self.norm3.weight.dtype)).to(input_dtype), context, clip_embed=clip_embed, 
                             nag_params=nag_params, nag_context=nag_context, is_uncond=is_uncond,
                                rope_func=self.rope_func)
            x = x.to(input_dtype)
                # MultiTalk
            if multitalk_audio_embedding is not None:
                x_audio = self.audio_cross_attn(self.norm_x(x.to(self.norm_x.weight.dtype)).to(input_dtype), encoder_hidden_states=multitalk_audio_embedding,
                                            shape=grid_sizes[0], x_ref_attn_map=x_ref_attn_map, human_num=human_num)
                x = x.add(x_audio, alpha=audio_scale)
       
        mod_x = torch.addcmul(shift_mlp, self.norm2(x.to(shift_mlp.dtype)), 1 + scale_mlp)
        x_ffn = self.ffn(mod_x.to(input_dtype))
        del shift_mlp, scale_mlp
        
        x = x.addcmul(x_ffn.to(gate_mlp.dtype), gate_mlp).to(input_dtype)
        del gate_mlp

        return x, None, None, None
    
    @torch.compiler.disable()
    def split_cross_attn_ffn(self, x, context, shift_mlp, scale_mlp, gate_mlp, clip_embed=None, grid_sizes=None):
        # Get number of prompts
        num_prompts = context.shape[0]
        num_clip_embeds = 0 if clip_embed is None else clip_embed.shape[0]
        num_segments = max(num_prompts, num_clip_embeds)
        
        # Extract spatial dimensions
        frames, height, width = grid_sizes[0]  # Assuming batch size 1
        tokens_per_frame = height * width
        
        # Distribute frames across prompts
        frames_per_segment = max(1, frames // num_segments)
        
        # Process each prompt segment
        x_combined = torch.zeros_like(x)
        
        for i in range(num_segments):
            # Calculate frame boundaries for this segment
            start_frame = i * frames_per_segment
            end_frame = min((i+1) * frames_per_segment, frames) if i < num_segments-1 else frames
            
            # Convert frame indices to token indices
            start_idx = start_frame * tokens_per_frame
            end_idx = end_frame * tokens_per_frame
            segment_indices = torch.arange(start_idx, end_idx, device=x.device, dtype=torch.long)
            
            # Get prompt segment (cycle through available prompts if needed)
            prompt_idx = i % num_prompts
            segment_context = context[prompt_idx:prompt_idx+1]
            
            # Handle clip_embed for this segment (cycle through available embeddings)
            segment_clip_embed = None
            if clip_embed is not None:
                clip_idx = i % num_clip_embeds
                segment_clip_embed = clip_embed[clip_idx:clip_idx+1]
            
            # Get tensor segment
            x_segment = x[:, segment_indices, :].to(self.norm3.weight.dtype)
            
            # Process segment with its prompt and clip embedding
            processed_segment = self.cross_attn(self.norm3(x_segment), segment_context, clip_embed=segment_clip_embed)
            processed_segment = processed_segment.to(x.dtype)
            
            # Add to combined result
            x_combined[:, segment_indices, :] = processed_segment
        
        # Continue with FFN
        x = x + x_combined
        y = self.ffn_chunked(x, shift_mlp, scale_mlp)
        x = x.addcmul(y, gate_mlp)
        return x

class Head(nn.Module):

    def __init__(self, dim, out_dim, patch_size, eps=1e-6):
        super().__init__()
        self.dim = dim
        self.out_dim = out_dim
        self.patch_size = patch_size
        self.eps = eps

        # layers
        out_dim = math.prod(patch_size) * out_dim
        # 归一化
        self.norm = nn.LayerNorm(dim, eps, elementwise_affine=False)
        # 映射到out_dim维度，out_dim已经是一个patch的大小 
        self.head = nn.Linear(dim, out_dim)

        # modulation
        # 时间参数
        self.modulation = nn.Parameter(torch.randn(1, 2, dim) / dim**0.5)

    def get_mod(self, e):
        if e.dim() == 2:
            return (self.modulation + e.unsqueeze(1)).chunk(2, dim=1)
        elif e.dim() == 3:
            e = (self.modulation.unsqueeze(2) + e.unsqueeze(1)).chunk(2, dim=1)
            return [ei.squeeze(1) for ei in e]

    def forward(self, x, e, **kwargs):
        r"""
        Args:
            x(Tensor): Shape [B, L1, C]
            e(Tensor): Shape [B, C]
        """

        e = self.get_mod(e.to(x.device))
        x = self.head(self.norm(x.float()).to(x.dtype).mul_(1 + e[1]).add_(e[0]))
        return x

class MLPProj(torch.nn.Module):

    def __init__(self, in_dim, out_dim):
        super().__init__()

        self.proj = torch.nn.Sequential(
            torch.nn.LayerNorm(in_dim), torch.nn.Linear(in_dim, in_dim),
            torch.nn.GELU(), torch.nn.Linear(in_dim, out_dim),
            torch.nn.LayerNorm(out_dim))

    def forward(self, image_embeds):
        if hasattr(self, 'emb_pos'):
            image_embeds = image_embeds + self.emb_pos.to(image_embeds.device)
        clip_extra_context_tokens = self.proj(image_embeds)
        return clip_extra_context_tokens

class WanModel(torch.nn.Module):
    def __init__(self,
                model_type='t2v',
                patch_size=(1, 2, 2),
                text_len=512,
                in_dim=16,
                dim=2048,
                in_features=5120,
                out_features=5120,
                ffn_dim=8192,
                ffn2_dim=8192,
                freq_dim=256,
                text_dim=4096,
                out_dim=16,
                num_heads=16,
                num_layers=32,
                qk_norm=True,
                eps=1e-6,
                attention_mode='sdpa',
                rope_func='comfy',
                rms_norm_function='default',
                main_device=torch.device('cuda'),
                offload_device=torch.device('cpu'),
                dtype=torch.float16,
                ):
        super().__init__()

        self.model_type = model_type

        self.patch_size = patch_size
        self.text_len = text_len
        self.in_dim = in_dim
        self.dim = dim
        self.in_features = in_features
        self.out_features = out_features
        self.ffn_dim = ffn_dim
        self.ffn2_dim = ffn2_dim
        self.freq_dim = freq_dim
        self.text_dim = text_dim
        self.out_dim = out_dim
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.qk_norm = qk_norm
        self.eps = eps
        self.attention_mode = attention_mode
        self.rope_func = rope_func
        
        self.main_device = main_device
        self.offload_device = offload_device
        self.device = main_device

        self.blocks_to_swap = -1
        # self.offload_txt_emb = False
        # self.offload_img_emb = False


        self.use_non_blocking = False
        self.prefetch_blocks = 0

        self.base_dtype = dtype


        # embeddings
        # 将输入的latent，在三维上按照patch_size进行切分，并通过一个3D卷积层映射到dim维度上，in_dim -> dim
        self.patch_embedding = nn.Conv3d(in_dim, dim, kernel_size=patch_size, stride=patch_size)
        
        # 将输入的文本，通过一个两层的全连接网络映射到dim维度上, text_dim -> dim
        # 第一层Linear将文本从text_dim映射到dim，激活函数GELU非线性增强，第二层Linear
        # Linear是全连接线性层，做仿射变换
        self.text_embedding = nn.Sequential(
            nn.Linear(text_dim, dim), nn.GELU(approximate='tanh'),
            nn.Linear(dim, dim))

        # 将时间步长信息映射到dim维度上，也是一个两层的全连接网络。freq_dim -> dim
        # 第一层Linear将时间步长从freq_dim映射到dim，激活函数SiLU非线性增强，第二层Linear
        self.time_embedding = nn.Sequential(nn.Linear(freq_dim, dim), nn.SiLU(), nn.Linear(dim, dim))
        # 进一步将时间嵌入映射到dim * 6维度上，用于后续的调制操作
        self.time_projection = nn.Sequential(nn.SiLU(), nn.Linear(dim, dim * 6))

        self.blocks = nn.ModuleList([
            WanAttentionBlock(self.in_features, self.out_features, ffn_dim, ffn2_dim, num_heads,
                                qk_norm, eps,
                                attention_mode=self.attention_mode, rope_func=self.rope_func, rms_norm_function=rms_norm_function, 
                                block_idx=i)
            for i in range(num_layers)
        ])

        # head
        # 将transform的输出还原成patch
        self.head = Head(dim, out_dim, patch_size, eps)

        d = self.dim // self.num_heads
        # RoPE，频率生成器，用于在自注意力机制中引入位置信息
        # 旋转位置编码
        self.rope_embedder = EmbedND_RifleX(
            d, 
            10000.0, 
            [d - 4 * (d // 6), 2 * (d // 6), 2 * (d // 6)],
            num_frames=None,
            k=None,
            )
        
        
        # buffers (don't use register_buffer otherwise dtype will be changed in to())
        assert (dim % num_heads) == 0 and (dim // num_heads) % 2 == 0
        
        # 将输入的图片条件，通过一个MLP映射到dim维度上，用于后续的条件融合
        self.img_emb = MLPProj(1280, dim)
       

    def rope_encode_comfy(self, t, h, w, ntk_alphas=[1,1,1], device=None, dtype=None):
        patch_size = self.patch_size
        t_len = ((t + (patch_size[0] // 2)) // patch_size[0]) # 21
        h_len = ((h + (patch_size[1] // 2)) // patch_size[1]) # 30
        w_len = ((w + (patch_size[2] // 2)) // patch_size[2]) # 30

        # if steps_t is None:
        steps_t = t_len
        # if steps_h is None:
        steps_h = h_len
        # if steps_w is None:
        steps_w = w_len

        img_ids = torch.zeros((steps_t, steps_h, steps_w, 3), device=device, dtype=dtype) # [21, 30, 30, 3]
        img_ids[:, :, :, 0] = img_ids[:, :, :, 0] + torch.linspace(0, t_len - 1, steps=steps_t, device=device, dtype=dtype).reshape(-1, 1, 1) # 
        img_ids[:, :, :, 1] = img_ids[:, :, :, 1] + torch.linspace(0, h_len - 1, steps=steps_h, device=device, dtype=dtype).reshape(1, -1, 1) #
        img_ids[:, :, :, 2] = img_ids[:, :, :, 2] + torch.linspace(0, w_len - 1, steps=steps_w, device=device, dtype=dtype).reshape(1, 1, -1) #
        img_ids = img_ids.reshape(1, -1, img_ids.shape[-1]) # [1, 18900, 3]
        # if attn_cond_shape is not None:
        #     F_cond, H_cond, W_cond = attn_cond_shape[2], attn_cond_shape[3], attn_cond_shape[4]
        #     cond_f_len = ((F_cond + (self.patch_size[0] // 2)) // self.patch_size[0])
        #     cond_h_len = ((H_cond + (self.patch_size[1] // 2)) // self.patch_size[1])
        #     cond_w_len = ((W_cond + (self.patch_size[2] // 2)) // self.patch_size[2])
        #     cond_img_ids = torch.zeros((cond_f_len, cond_h_len, cond_w_len, 3), device=device, dtype=dtype)
            
        #     #shift
        #     shift_f_size = 81 # Default value
        #     shift_f = False
        #     if shift_f:
        #         cond_img_ids[:, :, :, 0] = cond_img_ids[:, :, :, 0] + torch.linspace(shift_f_size, shift_f_size + cond_f_len - 1,steps=cond_f_len, device=device, dtype=dtype).reshape(-1, 1, 1)
        #     else:
        #         cond_img_ids[:, :, :, 0] = cond_img_ids[:, :, :, 0] + torch.linspace(0, cond_f_len - 1, steps=cond_f_len, device=device, dtype=dtype).reshape(-1, 1, 1)
        #     cond_img_ids[:, :, :, 1] = cond_img_ids[:, :, :, 1] + torch.linspace(h_len, h_len + cond_h_len - 1, steps=cond_h_len, device=device, dtype=dtype).reshape(1, -1, 1)
        #     cond_img_ids[:, :, :, 2] = cond_img_ids[:, :, :, 2] + torch.linspace(w_len, w_len + cond_w_len - 1, steps=cond_w_len, device=device, dtype=dtype).reshape(1, 1, -1)

        #     # Combine original and conditional position ids
        #     #img_ids = repeat(img_ids, "t h w c -> b (t h w) c", b=1)
        #     #cond_img_ids = repeat(cond_img_ids, "t h w c -> b (t h w) c", b=1)
        #     cond_img_ids = cond_img_ids.reshape(1, -1, cond_img_ids.shape[-1])
        #     combined_img_ids = torch.cat([img_ids, cond_img_ids], dim=1)
            
        #     # Generate RoPE frequencies for the combined positions
        #     freqs = self.rope_embedder(combined_img_ids, ntk_alphas).movedim(1, 2)
        # else:
        freqs = self.rope_embedder(img_ids, ntk_alphas).movedim(1, 2)
        return freqs

    def forward(
        self, x, t, context, seq_len,
        is_uncond=False,
        clip_fea=None,
        y=None,
        device=torch.device('cuda'),
        audio_scale=1.0,
        nag_params={}, nag_context=None,
        multitalk_audio=None,
        ntk_alphas = [1.0, 1.0, 1.0],
    ):
        device = self.main_device
        # x的shape[16,21,60,60]
        _, F, H, W = x[0].shape
        # y的shape[20,21,60,60]
        if y is not None:
            x = [torch.cat([u, v], dim=0) for u, v in zip(x, y)] # 合并x和y，变成[36,21,60,60]  
        
        self.patch_embedding.to(self.main_device) # 将输入的latent(noise+img_cond)切分成patch，先转成f32计算，再结果还原
        x = [self.patch_embedding(u.unsqueeze(0).to(torch.float32)).to(x[0].dtype) for u in x] # x的shape[1,5120,21,30,30]


        # grid sizes and seq len 记录patch的shape，也就是[21,30,30]
        grid_sizes = torch.stack([torch.tensor(u.shape[2:], device=device, dtype=torch.long) for u in x])
        original_grid_sizes = grid_sizes.clone()
        x = [u.flatten(2).transpose(1, 2) for u in x] # x的shape经过flatten[1, 5120, 18900], 在经过transpose[1, 18900, 5120], 把patch变成token
        self.original_seq_len = x[0].shape[1] # 第一个token的长度
        seq_lens = torch.tensor([u.size(1) for u in x], dtype=torch.int32) # 收集每个token的长度
        assert seq_lens.max() <= seq_len

        # attn_cond_shape = None
        # 补齐每个token，如果某个token的长度小于seq_len，则补零，x的shape[18900,5120]，for循环解开了第一维
        x = torch.cat([torch.cat([u, u.new_zeros(1, seq_len - u.size(1), u.size(2))], dim=1) for u in x])

        if "comfy" in self.rope_func: #comfy rope
            freqs = self.rope_encode_comfy(F, H, W, ntk_alphas=ntk_alphas, device=x.device, dtype=x.dtype)

        # if hasattr(self, "time_projection"):
        time_embed_dtype = self.time_embedding[0].weight.dtype
        if time_embed_dtype not in [torch.float16, torch.bfloat16, torch.float32]:
            time_embed_dtype = self.base_dtype
        e = self.time_embedding(sinusoidal_embedding_1d(self.freq_dim, t.flatten()).to(time_embed_dtype))  # b, dim
        e0 = self.time_projection(e).unflatten(1, (6, self.dim))  # b, 6, dim

        # clip vision embedding
        clip_embed = None
        if clip_fea is not None and hasattr(self, "img_emb"):
            clip_fea = clip_fea.to(self.main_device)
            # if self.offload_img_emb:
            #     self.img_emb.to(self.main_device)
            clip_embed = self.img_emb(clip_fea)  # bs x 257 x dim
            #context = torch.concat([context_clip, context], dim=1)
            # if self.offload_img_emb:
            #     self.img_emb.to(self.offload_device, non_blocking=self.use_non_blocking)

        #context (text embedding)
        if context != []:
            text_embed_dtype = self.text_embedding[0].weight.dtype
            if text_embed_dtype not in [torch.float16, torch.bfloat16, torch.float32]:
                text_embed_dtype = self.base_dtype
           
            context = torch.stack([torch.cat([u, u.new_zeros(self.text_len - u.size(0), u.size(1))]) for u in context]).to(text_embed_dtype)
           
            context = self.text_embedding(context)

        # MultiTalk
        if multitalk_audio is not None:
            self.multitalk_audio_proj.to(self.main_device)
            audio_cond = multitalk_audio.to(device=x.device, dtype=self.base_dtype) # [1,81,5,12,768]
            first_frame_audio_emb_s = audio_cond[:, :1, ...] # [1, 1, 5, 12, 768]
            latter_frame_audio_emb = audio_cond[:, 1:, ...] # [1, 80, 5, 12, 768]
            latter_frame_audio_emb = rearrange(latter_frame_audio_emb, "b (n_t n) w s c -> b n_t n w s c", n=4) # [1, 20, 4, 5, 12, 768]
            middle_index = self.multitalk_audio_proj.seq_len // 2 # 窗口大小5，中间的位置索引是2
            latter_first_frame_audio_emb = latter_frame_audio_emb[:, :, :1, :middle_index+1, ...] #[1, 20, 1, 3, 12, 768]
            latter_first_frame_audio_emb = rearrange(latter_first_frame_audio_emb, "b n_t n w s c -> b n_t (n w) s c") #[1, 20, 3, 12, 768]
            latter_last_frame_audio_emb = latter_frame_audio_emb[:, :, -1:, middle_index:, ...]  #[1, 20, 1, 3, 12, 768]
            latter_last_frame_audio_emb = rearrange(latter_last_frame_audio_emb, "b n_t n w s c -> b n_t (n w) s c") #[1, 20, 3, 12, 768]
            latter_middle_frame_audio_emb = latter_frame_audio_emb[:, :, 1:-1, middle_index:middle_index+1, ...] #[1, 20, 2, 1，12, 768]
            latter_middle_frame_audio_emb = rearrange(latter_middle_frame_audio_emb, "b n_t n w s c -> b n_t (n w) s c") #[1, 20, 2，12, 768]
            latter_frame_audio_emb_s = torch.concat([latter_first_frame_audio_emb, latter_middle_frame_audio_emb, latter_last_frame_audio_emb], dim=2) #[1, 20, 8, 12,768]
            multitalk_audio_embedding = self.multitalk_audio_proj(first_frame_audio_emb_s, latter_frame_audio_emb_s) #[1, 21, 32, 768]
            multitalk_audio_embedding = torch.concat(multitalk_audio_embedding.split(1), dim=2).to(self.base_dtype) #[1, 21, 32, 768]
            self.multitalk_audio_proj.to(self.offload_device)

        should_calc = True

        x = x.to(self.base_dtype)
        if isinstance(e0, list):
            e0 = [item.to(self.base_dtype) if torch.is_tensor(item) else item for item in e0]
        else:
            e0 = e0.to(self.base_dtype)

        if should_calc:
            # arguments
            kwargs = dict(
                e=e0,
                seq_lens=seq_lens,
                grid_sizes=grid_sizes,
                freqs=freqs,
                context=context,
                clip_embed=clip_embed,
                audio_scale=audio_scale,
                nag_params=nag_params, nag_context=nag_context,
                is_uncond = is_uncond,
                multitalk_audio_embedding=multitalk_audio_embedding,
            )

            # Asynchronous block offloading with CUDA streams and events
            if torch.cuda.is_available():
                cuda_stream = None #torch.cuda.Stream(device=device, priority=0) # todo causes issues on some systems
                events = [torch.cuda.Event() for _ in self.blocks]
                swap_start_idx = len(self.blocks) - self.blocks_to_swap if self.blocks_to_swap > 0 else len(self.blocks)
           
            for b, block in enumerate(self.blocks):
                # Prefetch blocks if enabled
                if self.prefetch_blocks > 0:
                    for prefetch_offset in range(1, self.prefetch_blocks + 1):
                        prefetch_idx = b + prefetch_offset
                        if prefetch_idx < len(self.blocks) and self.blocks_to_swap > 0 and prefetch_idx >= swap_start_idx:
                            context_mgr = torch.cuda.stream(cuda_stream) if torch.cuda.is_available() else nullcontext()
                            with context_mgr:
                                self.blocks[prefetch_idx].to(self.main_device, non_blocking=self.use_non_blocking)
                                if events is not None:
                                    events[prefetch_idx].record(cuda_stream)
                # Wait for block to be ready
                if b >= swap_start_idx and self.blocks_to_swap > 0:
                    if self.prefetch_blocks > 0 and events is not None:
                        if not events[b].query():
                            events[b].synchronize()
                    block.to(self.main_device)
               
                x, x_ip, lynx_ref_feature, x_ovi = block(x, **kwargs) #run block [1, 18900, 5120]
                
                if b >= swap_start_idx and self.blocks_to_swap > 0:
                    block.to(self.offload_device, non_blocking=self.use_non_blocking)
        
        x = x[:, :self.original_seq_len] # [1, 18900, 5120]

        x = self.head(x, e.to(x.device), temp_length=F) # [1, 18900, 64]

        x = self.unpatchify(x, original_grid_sizes) # type: ignore[arg-type] [16, 21, 60, 60]
        x = [u.float() for u in x]
        return (x, None, None)

    def unpatchify(self, x, grid_sizes):
        c = self.out_dim
        out = []
        for u, v in zip(x, grid_sizes.tolist()):
            u = u[: math.prod(v)].view(*v, *self.patch_size, c)
            u = torch.einsum("fhwpqrc->cfphqwr", u)
            u = u.reshape(c, *[i * j for i, j in zip(v, self.patch_size)])
            out.append(u)
        return out
