import torch
from einops import rearrange, einsum
import math
from cs336_systems.flash_attention_backward import flash_backward

class FlashAttentionPyTorch(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, is_causal: bool = False) -> torch.Tensor:
        batch_size, n_q, d_model = q.shape
        _, n_k, _ = k.shape
        b_q = b_k = 16 # tile sizes
        t_q = math.ceil(n_q / b_q)
        t_k = math.ceil(n_k / b_k)
        O = torch.zeros_like(q)
        L = torch.full((batch_size, n_q), float('-inf'))

        mask = torch.tril(torch.ones(n_q, n_k)).bool()

        for i in range(t_q):
            q_start = i * b_q
            q_end = min(q_start + b_q, n_q)
            q_i = q[:, q_start: q_end, :]  # (batch_size, tile_size, d_model)

            m = torch.full((batch_size, q_end - q_start), float('-inf'))
            l = torch.zeros((batch_size, q_end - q_start))
            o = torch.zeros_like(q_i)

            for j in range(t_k):
                k_start = j * b_k
                k_end = min(k_start + b_k, n_k)

                k_j = k[:, k_start : k_end, :] # (batch_size, tile_size, d_model)
                v_j = v[:, k_start : k_end, :] # (batch_size, tile_size, d_model)

                attn_scores = einsum(q_i, k_j, '... queries d_k,  ... keys d_k -> ... queries keys') / math.sqrt(d_model)
                if is_causal:
                    mask_new = mask[q_start: q_end, k_start: k_end]
                    attn_scores = attn_scores.masked_fill(~mask_new, -1e6)

                m_new = torch.maximum(m, attn_scores.max(dim=-1)[0])

                scale = torch.exp(m - m_new)
                l = l * scale
                o = o * scale.unsqueeze(-1)

                p_tilde = torch.exp(attn_scores - m_new.unsqueeze(-1))
                l += p_tilde.sum(dim=-1)
                o += p_tilde @ v_j

                m = m_new

            o_i = o / l.unsqueeze(-1)
            l_i = m + torch.log(l)

            O[:, q_start : q_end, :] = o_i
            L[:, q_start : q_end] = l_i

        ctx.save_for_backward(q, k, v, O, L)
        ctx.is_causal = is_causal
        return O

    @staticmethod
    def backward(ctx, grad_out):
        Q, K, V, O, L = ctx.saved_tensors
        is_causal = ctx.is_causal
        flash_backward_compiled = torch.compile(flash_backward)
        dQ, dK, dV = flash_backward_compiled(Q=Q, K=K, V=V, O=O, dO=grad_out, L=L, is_causal=is_causal)
        return dQ, dK, dV, None # None for is_causal gradient because it's not differentiable