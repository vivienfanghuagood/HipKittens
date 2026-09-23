"""Route SGLang's RowParallelLinear through the fused GEMM->AllReduce kernel.

    import sglang_patch
    sglang_patch.apply(max_tokens=8192, hidden_size=5120)   # before model load

NOT VERIFIED END TO END, for the same reason as vllm_patch.py: no SGLang build
for RDNA3 was available. The kernel underneath is verified (see
../torch_ext/test_torch_ext.py); this file's reading of SGLang's internals is
not. Check it against your copy of sglang/srt/layers/linear.py.

SGLang's RowParallelLinear is a close relative of vLLM's -- same weight layout,
same tp_rank-0 bias rule, same reduce_results flag -- so the patch is nearly the
same and the differences are called out where they matter.
"""

import os
import sys

import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, "..", "torch_ext"))
import hk_dist   # noqa: E402

_original_forward = None
_stats = {"fused": 0, "fallback": 0}


def _eligible(layer, x):
    # See the long version of this list in vllm_patch.py; the clauses and the
    # reasons are the same. SGLang names the quant method class the same thing
    # and keeps `weight` in the same (output_size, input_size_per_partition)
    # layout, which is why nothing here transposes.
    if not hk_dist.is_initialized():
        return False
    if not getattr(layer, "reduce_results", False) or getattr(layer, "tp_size", 1) <= 1:
        return False
    if not getattr(layer, "input_is_parallel", False):
        return False
    qm = getattr(layer, "quant_method", None)
    if qm is None or type(qm).__name__ != "UnquantizedLinearMethod":
        return False
    w = getattr(layer, "weight", None)
    if w is None or w.dtype != torch.bfloat16 or w.dim() != 2:
        return False
    if x.dtype != torch.bfloat16 or x.shape[-1] != w.shape[1]:
        return False
    M = int(x.numel() // x.shape[-1])
    return hk_dist.supported(M, w.shape[0], w.shape[1])


def _forward(self, input_, can_fuse_mlp_allreduce=False, **kwargs):
    # can_fuse_mlp_allreduce is SGLang's own flag for handing the all-reduce to
    # a later fusion pass; when it is set the layer is *not* supposed to reduce
    # here, so this kernel -- which always reduces -- must not run.
    if can_fuse_mlp_allreduce or kwargs or not _eligible(self, input_):
        _stats["fallback"] += 1
        return _original_forward(self, input_,
                                 can_fuse_mlp_allreduce=can_fuse_mlp_allreduce,
                                 **kwargs)

    _stats["fused"] += 1
    output = hk_dist.linear_allreduce(input_, self.weight)

    # Bias on tp_rank 0 only: the reduction would otherwise add it tp_size
    # times. Same rule as upstream, applied after the fused reduction rather
    # than folded into the GEMM.
    if getattr(self, "skip_bias_add", False):
        output_bias = self.bias
    else:
        output_bias = None
        if self.bias is not None and getattr(self, "tp_rank", 0) == 0:
            output = output + self.bias

    if not getattr(self, "return_bias", True):
        return output
    return output, output_bias


def apply(max_tokens, hidden_size, group=None):
    """Install the patch and build the heaps.

    max_tokens   the largest flattened M the server will pass, i.e. SGLang's
                 chunked-prefill size (or max_total_tokens if unchunked). The
                 ops refuse anything larger and fall back, which is safe but
                 slow; size it from the server config, not from a guess.
    hidden_size  the model's un-sharded hidden size.
    """
    global _original_forward
    from sglang.srt.layers.linear import RowParallelLinear

    if _original_forward is None:
        _original_forward = RowParallelLinear.forward
        RowParallelLinear.forward = _forward
    if not hk_dist.is_initialized():
        hk_dist.init(max_tokens=max_tokens, hidden_size=hidden_size, group=group)
    return _stats


def revert():
    global _original_forward
    if _original_forward is None:
        return
    from sglang.srt.layers.linear import RowParallelLinear
    RowParallelLinear.forward = _original_forward
    _original_forward = None
    hk_dist.shutdown()


def stats():
    return dict(_stats)
