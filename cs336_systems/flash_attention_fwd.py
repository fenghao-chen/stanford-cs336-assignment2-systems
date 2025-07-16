import triton
import triton.language as tl

@triton.jit
def flash_fwd_kernel(
    Q_ptr, K_ptr, V_ptr,
    O_ptr, L_ptr,
    stride_qb, stride_qq, stride_qd,
    stride_kb, stride_kk, stride_kd,
    stride_vb, stride_vk, stride_vd,
    stride_ob, stride_oq, stride_od,
    stride_lb, stride_lq,
    N_QUERIES, N_KEYS,
    scale,
    D: tl.constexpr,
    Q_TILE_SIZE: tl.constexpr,
    K_TILE_SIZE: tl.constexpr,
):
    # Program indices
    query_tile_index = tl.program_id(0)
    batch_index = tl.program_id(1)

    # Early return if out of bounds
    if query_tile_index * Q_TILE_SIZE >= N_QUERIES:
        return

    # Offset each pointer with the corresponding batch index
    # multiplied with the batch stride for each tensor
    Q_block_ptr = tl.make_block_ptr(
      Q_ptr + batch_index * stride_qb,
      shape=(N_QUERIES, D),
      strides=(stride_qq, stride_qd),
      offsets=(query_tile_index * Q_TILE_SIZE, 0),
      block_shape=(Q_TILE_SIZE, D),
      order=(1, 0),
    )

    K_block_ptr = tl.make_block_ptr(
        K_ptr + batch_index * stride_kb,
        shape=(N_KEYS, D),
        strides=(stride_kk, stride_kd),
        offsets=(0, 0), # TODO not sure
        block_shape=(K_TILE_SIZE, D),
        order=(1, 0),
    )

    V_block_ptr = tl.make_block_ptr(
        V_ptr + batch_index * stride_vb,
        shape=(N_KEYS, D),
        strides=(stride_vk, stride_vd),
        offsets=(0, 0), # TODO not sure
        block_shape=(K_TILE_SIZE, D),
        order=(1, 0)
    )

    O_block_ptr = tl.make_block_ptr(
        O_ptr + batch_index * stride_ob,
        shape=(N_QUERIES, D),
        strides=(stride_oq, stride_od),
        offsets=(query_tile_index * Q_TILE_SIZE, 0), # Should be the same as q
        block_shape=(Q_TILE_SIZE, D), # Should be the same as q
        order=(1, 0),
    )

    L_block_ptr = tl.make_block_ptr(
        L_ptr + batch_index * stride_lb,
        shape=(N_QUERIES, ),
        strides=(stride_lq,),
        offsets=(query_tile_index * Q_TILE_SIZE,), # TODO not sure
        block_shape=(Q_TILE_SIZE,),
        order=(0,),
    )
    q_i = tl.load(Q_block_ptr, boundary_check=(0, 1), padding_option="zero")  # (tile_size, d_model)

    m = tl.full((Q_TILE_SIZE,), float('-inf'), dtype=tl.float32)
    l = tl.zeros((Q_TILE_SIZE,), dtype=tl.float32)
    o = tl.zeros((Q_TILE_SIZE, D), dtype=tl.float32)

    num_k_tiles = tl.cdiv(N_KEYS, K_TILE_SIZE)
    for j in range(num_k_tiles):
        k_j = tl.load(K_block_ptr, boundary_check=(0, 1), padding_option="zero")  # (tile_size, d_model)
        v_j = tl.load(V_block_ptr, boundary_check=(0, 1), padding_option="zero")  # (tile_size, d_model)

        attn_scores = tl.dot(q_i, tl.trans(k_j)) * scale
        row_max = tl.max(attn_scores, axis=1)
        m_new = tl.maximum(m, row_max)

        s = tl.exp(m - m_new)
        l = l * s
        o = o * s[:, None]

        p_tilde = tl.exp(attn_scores - m_new[:, None]).to(v_j.dtype)
        l += tl.sum(p_tilde, axis=1)
        o += tl.dot(p_tilde, v_j)

        m = m_new

        if j < num_k_tiles - 1:
            K_block_ptr = K_block_ptr.advance((K_TILE_SIZE, 0))  # Move by K_TILE_SIZE
            V_block_ptr = V_block_ptr.advance((K_TILE_SIZE, 0))  # Move by K_TILE_SIZE

    o_i = (o / l[:, None]).to(O_block_ptr.type.element_ty)
    l_i = (m + tl.log(l)).to(L_block_ptr.type.element_ty)

    tl.store(O_block_ptr, o_i, boundary_check=(0, 1))
    tl.store(L_block_ptr, l_i, boundary_check=(0,))