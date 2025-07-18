import torch
from einops import rearrange, einsum
import math
from cs336_systems.flash_attention_fwd import flash_fwd_kernel
from cs336_systems.flash_attention_backward import flash_backward

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
            seq_len, seq_len, scale, # assuming N_QUERIES == N_KEYS
            d_model, Q_TILE_SIZE, K_TILE_SIZE, is_causal
        )

        ctx.save_for_backward(l_full)
        ctx.is_causal = is_causal
        return o_full

    @staticmethod
    def backward(ctx, grad_out):
        Q, K, V, O, L = ctx.saved_tensors
        flash_backward_compiled = torch.compile(flash_backward)
        dQ, dK, dV = flash_backward_compiled(Q=Q, K=K, V=V, O=O, dO=grad_out, L=L)
        return dQ, dK, dV, None  # None for is_causal gradient because it's not differentiable