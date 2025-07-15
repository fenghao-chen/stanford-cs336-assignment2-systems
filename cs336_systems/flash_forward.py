import torch
from einops import rearrange, einsum
import math

class FlashForward(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, is_causal: bool = False) -> torch.Tensor:
        batch_size = q.shape[0]
        # Reshape them into 2D tensors
        q = rearrange(q, "... d -> (...) d")
        k = rearrange(k, "... d -> (...) d")
        v = rearrange(v, "... d -> (...) d")

        n_q, d_model = q.shape
        n_k, _ = k.shape
        b_q = b_k = 16 # tile sizes
        t_q = math.ceil(n_q / b_q)
        t_k = math.ceil(n_k / b_k)
        O = torch.zeros_like(q)
        L = torch.zeros(n_q)

        for i in range(t_q):
            q_start = i * b_q
            q_end = min(q_start + b_q, n_q)
            q_i = q[q_start: q_end]

            m = torch.empty(b_q).fill_(float("-inf")) # todo maybe incorrect ?
            l = torch.zeros(b_q)
            o = torch.zeros(b_q, d_model)

            for j in range(t_k):
                k_start = j * b_k
                k_end = min(k_start + b_k, n_k)

                k_j = k[k_start : k_end]
                v_j = v[k_start : k_end]

                attn_scores = einsum(q_i, k_j, 'queries d_k,  keys d_k -> queries keys') / math.sqrt(d_model)
                m_new = torch.max(m, attn_scores.max(dim=1)[0])

                scale = torch.exp(m - m_new)
                l = l * scale
                o = o * scale[:, None]

                p_tilde = torch.exp(attn_scores - m_new[:, None])
                l += p_tilde.sum(dim=1)
                o += p_tilde @ v_j

                m = m_new

            o_i = o / l[:, None]
            l_i = m + torch.log(l)

            O[q_start : q_end] = o_i
            L[q_start : q_end] = l_i

        O = rearrange(O, "(batch_size queries) d -> batch_size queries d", batch_size=batch_size)
        L = rearrange(L, "(batch_size queries) -> batch_size queries", batch_size=batch_size)
        ctx.save_for_backward(L)
        return O

    @staticmethod
    def backward(ctx, grad_out):
        raise NotImplementedError