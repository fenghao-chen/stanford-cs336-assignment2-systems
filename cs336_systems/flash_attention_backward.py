import torch
from einops import rearrange, einsum
import math
from jaxtyping import Float, Int, jaxtyped, Bool
from torch import Tensor


def flash_backward(Q: Float[Tensor, " ... queries d_model"],
                   K: Float[Tensor, " ... keys d_model"],
                   V: Float[Tensor, " ... values d_model"],
                   O: Float[Tensor, " ... queries d_model"],
                   dO: Float[Tensor, " ... queries d_model"],
                   L: Float[Tensor, " ... queries"]) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    d_model = Q.shape[-1]
    S: Tensor = einsum(Q, K, '... queries d_model,  ... keys d_model -> ... queries keys') / math.sqrt(d_model)
    P: Tensor = torch.exp(S - L.unsqueeze(-1)) # (batch_size, queries, keys)
    dV: Tensor = einsum(P, dO, '... queries keys,  ... queries d_model -> ... keys d_model')
    dP = einsum(dO, V, '... queries d_model,  ... values d_model -> ... queries values')
    D: Tensor = (O * dO).sum(dim=-1) # (batch_size, queries)
    dS = P * (dP - D.unsqueeze(-1)) # (batch_size, queries, keys)
    dQ = einsum(dS, K, '... queries keys, ... keys d_model -> ... queries d_model') / math.sqrt(d_model)
    dK = einsum(dS, Q, '... queries keys, ... queries d_model -> ... keys d_model') / math.sqrt(d_model)
    return dQ, dK, dV

