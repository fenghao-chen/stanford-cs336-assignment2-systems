import torch
from einops import rearrange, einsum
import math
from cs336_systems.flash_attention_fwd import flash_fwd_kernel

class FlashAttention(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, is_causal: bool = False) -> torch.Tensor:
        # Make sure tensors are contiguous
        q = q.contiguous()
        k = k.contiguous()
        v = v.contiguous()

        batch_size, seq_len, d_model = q.shape

        Q_TILE_SIZE = 16
        K_TILE_SIZE = 16
        grid = (math.ceil(seq_len / Q_TILE_SIZE), batch_size)

        o_full = torch.zeros_like(q, dtype=torch.float32)
        l_full = torch.full((batch_size, seq_len), float('-inf'), dtype=torch.float32, device='cuda')
        scale = 1.0 / math.sqrt(d_model)

        flash_fwd_kernel[grid](
            q, k, v, o_full, l_full,
            q.stride(0), q.stride(1), q.stride(2),
            k.stride(0), k.stride(1), k.stride(2),
            v.stride(0), v.stride(1), v.stride(2),
            o_full.stride(0), o_full.stride(1), o_full.stride(2),
            l_full.stride(0), l_full.stride(1),
            seq_len, seq_len, scale,
            d_model, Q_TILE_SIZE, K_TILE_SIZE, is_causal
        )

        ctx.save_for_backward(l_full, is_causal)
        return o_full

    @staticmethod
    def backward(ctx, grad_out):
        raise NotImplementedError