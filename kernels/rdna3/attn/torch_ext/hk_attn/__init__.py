"""A HipKittens attention forward for RDNA3, shaped like torch's SDPA.

On Radeon, every framework that runs attention -- diffusers, comfyUI, vLLM,
SGLang -- reaches ``F.scaled_dot_product_attention``, and on ROCm that lands in
aotriton's ``attn_fwd``.  Measured on a W7900D (gfx1100, 96 CU) at MiniMax H3's
shape (B=1, H=56, D=128, bf16, non-causal), aotriton is flat at 20-21 TFLOPs
from N=4K to N=32K, which is 21% of the machine's measured WMMA ceiling.  This
module is the same operation at 54-57 TFLOPs.

Usage -- the signature is ``F.scaled_dot_product_attention``'s, argument for
argument, so a framework can be redirected in one line::

    import hk_attn
    import torch.nn.functional as F
    F.scaled_dot_product_attention = hk_attn.scaled_dot_product_attention

or, without the monkeypatch, call ``hk_attn.scaled_dot_product_attention``
wherever the framework calls torch's.

**What runs on the kernel and what falls back.**  The kernel takes bf16
(B, H, N, D) with D of 64 or 128, N at least the Q tile, MHA or GQA, causal or
not, and the same sequence length for q, k and v.  Everything else -- fp16, an
attn_mask, dropout, a grad-requiring input, cross attention with n_kv != n_q --
is forwarded to ``F.scaled_dot_product_attention`` unchanged.  Falling back is
silent by design; set ``HK_ATTN_VERBOSE=1`` to have each fallback print its
reason once.
"""

import os
import pathlib
import warnings

import torch
import torch.nn.functional as F

__all__ = ["scaled_dot_product_attention", "attention", "supported", "loaded"]

_VERBOSE = os.environ.get("HK_ATTN_VERBOSE", "") not in ("", "0")
_SEEN = set()

_SO = pathlib.Path(__file__).resolve().parent.parent / "hk_attn_ext.so"
try:
    torch.ops.load_library(str(_SO))
    _LOADED = True
except Exception as e:  # noqa: BLE001 -- any load failure means "use torch"
    warnings.warn(f"hk_attn: could not load {_SO} ({e}); every call falls back "
                  f"to F.scaled_dot_product_attention")
    _LOADED = False


def loaded():
    """True if the kernel is available at all."""
    return _LOADED


def _why(q, k, v, attn_mask, dropout_p, is_causal, enable_gqa):
    """The reason this call cannot run on the kernel, or '' if it can."""
    if not _LOADED:
        return "extension not loaded"
    if attn_mask is not None:
        return "attn_mask is not supported"
    if dropout_p != 0.0:
        return "dropout is not supported (this is an inference kernel)"
    if torch.is_grad_enabled() and any(t.requires_grad for t in (q, k, v)):
        return "no backward: an input requires grad"
    if q.dtype != torch.bfloat16 or k.dtype != q.dtype or v.dtype != q.dtype:
        return f"bf16 only, got {q.dtype}/{k.dtype}/{v.dtype}"
    if not (q.is_cuda and k.is_cuda and v.is_cuda):
        return "device tensors only"
    if q.dim() != 4 or k.dim() != 4 or v.dim() != 4:
        return f"expects (B, H, N, D), got {q.dim()} dims"
    if k.shape != v.shape:
        return f"k and v must match: {tuple(k.shape)}/{tuple(v.shape)}"
    if k.shape[0] != q.shape[0] or k.shape[3] != q.shape[3]:
        return f"batch and head_dim must match: {tuple(q.shape)}/{tuple(k.shape)}"
    if k.shape[2] != q.shape[2]:
        # Cross attention.  The kv loop is bounded by q's length, and for causal
        # there would additionally be two conventions for the diagonal.
        return f"n_kv must equal n_q, got {k.shape[2]} vs {q.shape[2]}"
    if k.shape[1] != q.shape[1] and not enable_gqa:
        # torch itself rejects this, so falling back keeps its error message
        # rather than inventing one.
        return "head counts differ but enable_gqa is False"
    return torch.ops.hk_attn.why_unsupported(q.shape[2], q.shape[3], bool(is_causal),
                                             q.shape[1], k.shape[1])


def supported(q, k, v, attn_mask=None, dropout_p=0.0, is_causal=False,
              enable_gqa=False):
    """Would this exact call run on the kernel?"""
    return _why(q, k, v, attn_mask, dropout_p, is_causal, enable_gqa) == ""


def attention(q, k, v, scale=None, is_causal=False):
    """The kernel with no fallback: raises if the shape is not supported.

    Use this when a wrong dispatch should be loud -- benchmarks, tests, and any
    integration where silently running torch's kernel would be mistaken for a
    win.
    """
    return torch.ops.hk_attn.fwd(q.contiguous(), k.contiguous(), v.contiguous(),
                                 0.0 if scale is None else float(scale),
                                 bool(is_causal))


def scaled_dot_product_attention(query, key, value, attn_mask=None,
                                 dropout_p=0.0, is_causal=False, scale=None,
                                 enable_gqa=False):
    """``F.scaled_dot_product_attention``, on the HipKittens kernel where it fits.

    Signature-identical to torch's, including argument order and defaults, so it
    can replace it by assignment.  Anything the kernel does not cover is
    forwarded to torch unchanged.
    """
    why = _why(query, key, value, attn_mask, dropout_p, is_causal, enable_gqa)
    if why:
        if _VERBOSE and why not in _SEEN:
            _SEEN.add(why)
            print(f"hk_attn: falling back to torch -- {why}")
        return F.scaled_dot_product_attention(
            query, key, value, attn_mask=attn_mask, dropout_p=dropout_p,
            is_causal=is_causal, scale=scale, enable_gqa=enable_gqa)
    # The globals are plain row-major (B, H, N, D) with no stride support, and
    # the op refuses a non-contiguous input rather than reading it wrong, so the
    # copy happens here.  A framework that hands us the usual
    # `.view(B, N, H, D).transpose(1, 2)` pays it; one that keeps (B, H, N, D)
    # contiguous does not, because .contiguous() is a no-op then.
    return torch.ops.hk_attn.fwd(query.contiguous(), key.contiguous(),
                                 value.contiguous(),
                                 0.0 if scale is None else float(scale),
                                 bool(is_causal))
