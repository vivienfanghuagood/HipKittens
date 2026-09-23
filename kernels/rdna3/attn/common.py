"""Shapes, timing and the fp32 reference shared by every script in this directory.

The shapes are MiniMax H3's.  H3 is a video+audio diffusion transformer, not an
LLM: 56 heads, head_dim 128, no `num_key_value_heads` (so MHA, not GQA), and
bidirectional attention.  Sequence length is the whole story -- it is the number
of latent tokens in the clip, and softmax attention is >85% of the transformer's
runtime at the lengths a real clip produces.

Token counts come from the pipeline geometry: the VAE is 8x spatial and 4x
temporal, then the transformer's patch_size [1,2,2] halves h and w again.  So a
`f`-frame `H x W` clip is `t*h*w` tokens with t=(f-1)/4+1, h=H/16, w=W/16.
"""

import time

import torch

D_HEAD = 128
N_HEADS = 56


def h3_tokens(frames, height, width):
    """Latent token count for a clip, from H3's VAE + patch geometry."""
    t = (frames - 1) // 4 + 1
    return t * (height // 16) * (width // 16)


# (name, batch, heads, seqlen).  Batch 2 is classifier-free guidance.
SHAPES = [
    ("N=4096",        1, N_HEADS,  4096),
    ("N=8192",        1, N_HEADS,  8192),
    ("N=16384",       1, N_HEADS, 16384),
    ("N=32768",       1, N_HEADS, 32768),
    # 832x480, 125 frames (~5 s @ 25 fps) -> t=32, h=30, w=52
    ("480p/5s",       1, N_HEADS, h3_tokens(125, 480, 832)),
    ("N=65536",       1, N_HEADS, 65536),
    ("N=16384 cfg",   2, N_HEADS, 16384),
]

# Run separately, not in the sweep: aotriton needs ~18 s for a single layer here.
BIG_SHAPES = [
    # 1280x720, 125 frames -> t=32, h=45, w=80
    ("720p/5s",       1, N_HEADS, h3_tokens(125, 720, 1280)),
]

# Shapes whose seqlen is not a multiple of any plausible KV block, for the
# correctness gates only.
RAGGED = [4097, 5000, 12345, 1, 17, 255]


def flops(b, h, n, causal=False, d=D_HEAD):
    """Useful FLOPs in one attention forward: two b*h*n*n*d matmuls.

    Halved for causal, which is the convention every FA paper reports in: only
    the lower triangle is computed.  It slightly flatters any implementation
    that still computes whole blocks on the diagonal, and every implementation
    compared here does, so it is at least an even flattery.
    """
    f = 4 * b * h * n * n * d
    return f / 2 if causal else f


def bench(fn, iters=10, warmup=3):
    """Milliseconds per call.  Callers must keep competing backends inside one
    process and one run: clock and power state drift between runs on this node,
    so a number measured in a different process is not comparable."""
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / iters * 1e3


def make_qkv(b, h, n, d=D_HEAD, dtype=torch.bfloat16, seed=0, h_kv=None):
    """q, k, v for one attention call.  `h_kv` fewer than `h` gives GQA."""
    g = torch.Generator(device="cuda").manual_seed(seed)
    hk = h if h_kv is None else h_kv
    return (torch.randn(b, h,  n, d, device="cuda", dtype=dtype, generator=g),
            torch.randn(b, hk, n, d, device="cuda", dtype=dtype, generator=g),
            torch.randn(b, hk, n, d, device="cuda", dtype=dtype, generator=g))


def reference(q, k, v, causal=False, scale=None, q_chunk=512, heads=None):
    """Chunked fp32 attention, computed one (batch, head, q-block) at a time.

    Chunked because the whole point of these shapes is that the score matrix
    does not fit: at n=49920 a single head's fp32 scores are 10 GB.  `heads`
    restricts the computation to a subset, for spot checks on shapes where
    doing all 56 would take minutes.

    GQA is inferred from k's head count rather than passed: a q head reads the
    kv head it maps to, which is the whole of what GQA is.
    """
    b, h, n, d = q.shape
    nk = k.shape[2]
    q_per_kv = h // k.shape[1]
    if scale is None:
        scale = d ** -0.5
    hs = list(range(h)) if heads is None else list(heads)
    out = torch.zeros(b, len(hs), n, d, device=q.device, dtype=torch.float32)
    for bi in range(b):
        for oi, hi in enumerate(hs):
            kk = k[bi, hi // q_per_kv].float()
            vv = v[bi, hi // q_per_kv].float()
            for i in range(0, n, q_chunk):
                j = min(i + q_chunk, n)
                s = (q[bi, hi, i:j].float() @ kk.transpose(-1, -2)) * scale
                if causal:
                    pos_q = torch.arange(i, j, device=q.device)[:, None]
                    pos_k = torch.arange(nk, device=q.device)[None, :]
                    s.masked_fill_(pos_k > pos_q + (nk - n), float("-inf"))
                out[bi, oi, i:j] = torch.softmax(s, dim=-1) @ vv
                del s
    return out


def check(name, got, ref, rtol=5e-2):
    """Elementwise comparison against the fp32 reference.

    rtol 5e-2: bf16 inputs are rounded twice and then summed over thousands of
    terms, which lands around 1.2e-2; 4x of that is still 20x below the O(100%)
    error a wrong tile or a wrong mask produces.  The magnitude check is there
    because an all-zero `got` compared against an all-zero `ref` passes every
    relative test -- that failure mode has happened before.
    """
    got = got.float()
    ref = ref.float()
    if ref.abs().max().item() < 1e-6:
        return False, f"{name}: FAIL reference has no magnitude"
    rel = (got - ref).abs().max().item() / ref.abs().max().item()
    ok = rel < rtol and torch.isfinite(got).all().item()
    nan = "" if torch.isfinite(got).all().item() else "  (non-finite output)"
    return ok, f"{name}: {'ok  ' if ok else 'FAIL'} rel {rel:.2e}{nan}"


def clocks():
    """sclk / power, so a table says which state the chip was in."""
    import subprocess
    try:
        out = subprocess.run(["rocm-smi", "--showgpuclocks", "--showpower"],
                             capture_output=True, text=True, timeout=20).stdout
        want = [l.strip() for l in out.splitlines()
                if "sclk" in l.lower() or "Power" in l]
        return " | ".join(want[:4]) or "unavailable"
    except Exception as e:
        return f"unavailable ({type(e).__name__})"
