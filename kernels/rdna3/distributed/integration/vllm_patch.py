"""Route vLLM's RowParallelLinear through the fused GEMM->AllReduce kernel.

    import vllm_patch
    vllm_patch.apply(max_tokens=8192)      # before the model is built

NOT VERIFIED END TO END. There is no vLLM build for RDNA3 on the machine this
was developed on, so this file has never been run inside a vLLM server. What is
verified is everything underneath it: torch.ops.hk_dist matches
``F.linear + dist.all_reduce`` elementwise on the Qwen3 TP=2 shapes, under
torchrun, at prefill and decode both (see ../torch_ext/test_torch_ext.py). What
is unverified is this file's reading of vLLM's internals -- the attribute names,
the quant-method check, and whether `forward` still looks like this in whatever
version you have. Read the guard below against your copy of
vllm/model_executor/layers/linear.py before trusting it.

Design: wrap `forward` rather than reimplement it. The eligible path is short
and self-contained, and everything else calls straight through to the original
bound method, so a vLLM that has moved on breaks by falling back -- not by
computing something subtly wrong.
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
    """Everything that has to be true before the fused kernel is correct here.

    Each clause is a real requirement, not defensive padding:

      reduce_results / tp_size   the kernel *is* the all-reduce. With no
                                 all-reduce to do there is nothing to fuse.
      input_is_parallel          otherwise forward() first splits the input
                                 along the last dim, and the fused path would
                                 have to redo that; not worth the surface.
      UnquantizedLinearMethod    the kernel reads a plain bf16 (N, K_local)
                                 weight. Any quantised method stores something
                                 else entirely, and the shape check would not
                                 catch all of them.
      bf16                       the only dtype the RDNA3 GEMM implements.
      supported(...)             the shape constraints, asked of the extension
                                 rather than duplicated here so the two cannot
                                 drift: N % 128, (N/tp) % 64, K_local % 32, and
                                 M within the bound init() was given.
    """
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


def _forward(self, input_):
    if not _eligible(self, input_):
        _stats["fallback"] += 1
        return _original_forward(self, input_)

    _stats["fused"] += 1
    output = hk_dist.linear_allreduce(input_, self.weight)

    # vLLM adds the bias on tp_rank 0 only, so that an all-reduce over tp_size
    # ranks does not add it tp_size times. The fused kernel *is* the all-reduce,
    # so the same rule applies and for the same reason -- except that here the
    # add has to happen after the reduction rather than being folded into the
    # GEMM, which is where vLLM puts it.
    if getattr(self, "skip_bias_add", False):
        output_bias = self.bias
    else:
        output_bias = None
        if self.bias is not None and getattr(self, "tp_rank", 0) == 0:
            output = output + self.bias

    if not getattr(self, "return_bias", True):
        return output
    return output, output_bias


def apply(max_tokens, hidden_size=None, group=None):
    """Install the patch and build the heaps.

    max_tokens    the largest flattened M the server will ever pass, i.e.
                  max_num_batched_tokens. The kernel refuses anything larger,
                  falling back rather than corrupting, but every refusal is a
                  layer running the slow path -- size this correctly.
    hidden_size   the model's un-sharded hidden size (N). Defaults to
                  max_tokens' own config lookup being unavailable here, so pass
                  it; required.
    """
    global _original_forward
    from vllm.model_executor.layers.linear import RowParallelLinear

    if hidden_size is None:
        raise ValueError("hidden_size is required: the heaps are sized (max_tokens, "
                         "hidden_size) and cannot be grown later")
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
    from vllm.model_executor.layers.linear import RowParallelLinear
    RowParallelLinear.forward = _original_forward
    _original_forward = None
    hk_dist.shutdown()


def stats():
    """How many forwards took each path. A fallback count that is not zero after
    warmup means a shape check is failing; ask why_unsupported() which one."""
    return dict(_stats)
