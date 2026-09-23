"""A Triton FlashAttention-2 forward, as the second baseline.

vLLM and SGLang do not use torch SDPA on ROCm; they ship their own Triton
attention kernels.  Neither package can be installed in this pod (no network),
so this is a faithful stand-in written against the same Triton that is
installed here: the standard FA-2 forward with an online softmax, autotuned over
the block sizes and wave counts that matter on RDNA3.

It is a baseline, not a contribution.  If the HipKittens kernel cannot beat a
few dozen lines of Triton then it has no reason to exist.
"""

import torch
import triton
import triton.language as tl


def _configs():
    # The full 3x3x3x2 grid is 54 configs, and since the autotune key includes
    # N_CTX it is re-tuned for every sequence length -- on the H3 shapes that is
    # dominated by compile time and does not finish inside an hour.  num_stages
    # buys nothing on gfx11 (no async global->LDS copy to pipeline), num_warps=2
    # never wins at HEAD_DIM=128, and BLOCK_N=128 runs out of LDS, so the grid
    # below is the part of the original that was ever selected.
    return [triton.Config({"BLOCK_M": bm, "BLOCK_N": bn}, num_warps=w, num_stages=1)
            for bm in (32, 64, 128) for bn in (32, 64) for w in (4, 8)]


@triton.autotune(configs=_configs(), key=["N_CTX", "HEAD_DIM", "IS_CAUSAL"],
                 warmup=5, rep=20)
@triton.jit
def _attn_fwd(Q, K, V, Out,
              sqb, sqh, sqn,
              skb, skh, skn,
              svb, svh, svn,
              sob, soh, son,
              H, N_CTX, sm_scale,
              HEAD_DIM: tl.constexpr, IS_CAUSAL: tl.constexpr,
              BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr):
    pid_m = tl.program_id(0)
    bh = tl.program_id(1)
    b = bh // H
    h = bh % H

    qo = b * sqb + h * sqh
    ko = b * skb + h * skh
    vo = b * svb + h * svh

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_d = tl.arange(0, HEAD_DIM)

    q = tl.load(Q + qo + offs_m[:, None] * sqn + offs_d[None, :],
                mask=offs_m[:, None] < N_CTX, other=0.0)

    m_i = tl.full([BLOCK_M], float("-inf"), dtype=tl.float32)
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32)
    acc = tl.zeros([BLOCK_M, HEAD_DIM], dtype=tl.float32)

    # log2(e) folded into the scale so the exponential is exp2, which is one
    # hardware instruction.
    qk_scale = sm_scale * 1.44269504089

    hi = tl.minimum(N_CTX, (pid_m + 1) * BLOCK_M) if IS_CAUSAL else N_CTX
    for start_n in range(0, hi, BLOCK_N):
        start_n = tl.multiple_of(start_n, BLOCK_N)
        offs_n = start_n + tl.arange(0, BLOCK_N)
        kmask = offs_n < N_CTX

        k = tl.load(K + ko + offs_n[:, None] * skn + offs_d[None, :],
                    mask=kmask[:, None], other=0.0)
        qk = tl.dot(q, tl.trans(k)) * qk_scale
        qk = tl.where(kmask[None, :], qk, float("-inf"))
        if IS_CAUSAL:
            qk = tl.where(offs_m[:, None] >= offs_n[None, :], qk, float("-inf"))

        m_new = tl.maximum(m_i, tl.max(qk, 1))
        # A row whose every entry is masked has m_new = -inf; the subtraction
        # would be inf-inf.  Clamping to 0 leaves p all-zero, which is what the
        # running sum wants.
        m_safe = tl.where(m_new == float("-inf"), 0.0, m_new)
        p = tl.math.exp2(qk - m_safe[:, None])
        alpha = tl.math.exp2(m_i - m_safe)
        alpha = tl.where(m_i == float("-inf"), 0.0, alpha)

        l_i = l_i * alpha + tl.sum(p, 1)
        acc = acc * alpha[:, None]
        v = tl.load(V + vo + offs_n[:, None] * svn + offs_d[None, :],
                    mask=kmask[:, None], other=0.0)
        acc = tl.dot(p.to(v.dtype), v, acc)
        m_i = m_new

    acc = acc / tl.where(l_i == 0.0, 1.0, l_i)[:, None]
    tl.store(Out + b * sob + h * soh + offs_m[:, None] * son + offs_d[None, :],
             acc.to(Out.dtype.element_ty), mask=offs_m[:, None] < N_CTX)


def triton_attention(q, k, v, causal=False, scale=None):
    """(B, H, N, D) in and out, matching F.scaled_dot_product_attention."""
    b, h, n, d = q.shape
    assert k.shape == v.shape == q.shape, "this baseline covers MHA only"
    assert d in (64, 128), f"unsupported head_dim {d}"
    q, k, v = (t.contiguous() if not t.is_contiguous() else t for t in (q, k, v))
    o = torch.empty_like(q)
    if scale is None:
        scale = d ** -0.5

    grid = lambda META: (triton.cdiv(n, META["BLOCK_M"]), b * h)
    _attn_fwd[grid](
        q, k, v, o,
        q.stride(0), q.stride(1), q.stride(2),
        k.stride(0), k.stride(1), k.stride(2),
        v.stride(0), v.stride(1), v.stride(2),
        o.stride(0), o.stride(1), o.stride(2),
        h, n, scale,
        HEAD_DIM=d, IS_CAUSAL=causal,
    )
    return o
