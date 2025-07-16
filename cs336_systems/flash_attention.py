import torch
from einops import rearrange, einsum
import math
from cs336_systems.flash_attention_fwd import flash_fwd_kernel

class FlashAttention(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, is_causal: bool = False) -> torch.Tensor:
        batch_size, n_q, d_model = q.shape
        _, n_k, _ = k.shape
        b_q = b_k = 16 # tile sizes
        t_q = math.ceil(n_q / b_q)
        O = torch.zeros_like(q)
        L = torch.full((batch_size, n_q), float('-inf'))
        scale = 1 / math.sqrt(d_model)

        flash_fwd_kernel[(t_q, batch_size)](q, k, v,
                         O, L,
                         q.stride(0), q.stride(1), q.stride(2),
                         k.stride(0), k.stride(1), k.stride(2),
                         v.stride(0), v.stride(1), v.stride(2),
                         O.stride(0), O.stride(1), O.stride(2),
                         L.stride(0), L.stride(1), L.stride(2),
                         N_QUERIES=n_q, N_KEYS=n_k,
                         scale=scale,
                         D=d_model,
                         Q_TILE_SIZE=b_q,
                         K_TILE_SIZE=b_k)

        ctx.save_for_backward(L)
        return O

    @staticmethod
    def backward(ctx, grad_out):
        raise NotImplementedError