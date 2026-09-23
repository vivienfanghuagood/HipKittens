"""Fused GEMM -> collective for RDNA3, as a drop-in tensor-parallel linear.

A row-parallel linear in vLLM and SGLang is a matmul over this rank's K-slice
followed by an all-reduce, and the two are strictly sequential: the collective
cannot start until the whole partial exists. The kernels behind this module put
the collective in the GEMM's epilogue instead -- each output column block is
pushed into its owner's inbox as soon as its accumulator is finished -- so the
transfer overlaps the rest of the matmul.

Usage::

    import hk_dist
    hk_dist.init(max_tokens=8192, hidden_size=5120)       # after dist.init
    y = hk_dist.linear_allreduce(x, layer.weight)         # x: (M, K_local)

or, replacing a module::

    layer = hk_dist.HKRowParallelLinear.from_linear(layer)

`init` must be called by every rank in the group, with identical arguments, and
before any of the ops. It allocates fine-grained IPC-exportable heaps sized for
`max_tokens` rows, so `max_tokens` has to cover the largest prefill batch the
server will see; the ops refuse anything larger rather than silently corrupting.

What this is and is not measured on: two W7900D over PCIe, gfx1100, TP=2, Qwen3
shapes. Prefill wins 1.14-1.38x over hipBLASLt + RCCL. Decode is 1.05-1.11x and
the honest reading of that is that the collective is *not* hidden at decode
sizes -- the fused path runs at about 2.1x the pure-GEMM floor, because two
cross-rank barriers at ~6 us each plus a combine kernel are a large fraction of
a 41-98 us GEMM. `bench.py` prints the ratio so this stays visible.
"""

import os
import pathlib
import secrets

import torch
import torch.distributed as dist
from torch import nn

__all__ = [
    "init", "shutdown", "is_initialized", "supported", "why_unsupported",
    "linear_allreduce", "linear_reducescatter", "HKRowParallelLinear",
]

_LIB = pathlib.Path(__file__).resolve().parent.parent / "hk_dist_ext.so"
_loaded = False
_state = None


def _load():
    global _loaded
    if _loaded:
        return
    if not _LIB.exists():
        raise RuntimeError(
            f"{_LIB} is missing. Build it with `make` in {_LIB.parent}."
        )
    torch.ops.load_library(str(_LIB))
    _loaded = True


class _State:
    def __init__(self, rank, world, max_tokens, hidden_size, run_id):
        self.rank = rank
        self.world = world
        self.max_tokens = max_tokens
        self.hidden_size = hidden_size
        self.run_id = run_id


def is_initialized():
    return _state is not None


def init(max_tokens, hidden_size, group=None, run_id=None):
    """Build the symmetric heaps. Collective: every rank in `group` must call it.

    max_tokens   the largest M (tokens x batch, flattened) any call will pass.
                 Rounded up to a 16-row tile.
    hidden_size  the largest N, i.e. the un-sharded output width.
    group        a torch.distributed process group, or None for the default.
                 Used only for the rendezvous; the data path never touches it.
    run_id       normally left None. The heaps rendezvous through
                 /tmp/hk_dist_<run_id> and a POSIX shm segment of the same name,
                 so every rank needs the same string and every *run* needs a
                 different one -- a stale directory from a crashed run would
                 otherwise be mistaken for this one's. Rank 0 picks it and
                 broadcasts.
    """
    global _state
    if _state is not None:
        raise RuntimeError("hk_dist.init() called twice")
    _load()

    if dist.is_available() and dist.is_initialized():
        rank = dist.get_rank(group)
        world = dist.get_world_size(group)
        if run_id is None:
            # broadcast_object_list rather than each rank reading a shared env
            # var: under torchrun the ranks are separate processes that inherit
            # the launcher's environment, so an env var is only the same string
            # by luck, and two runs in the same shell would collide.
            box = [secrets.token_hex(8) if rank == 0 else None]
            dist.broadcast_object_list(box, src=0, group=group)
            run_id = box[0]
    else:
        rank, world = 0, 1
        run_id = run_id or secrets.token_hex(8)

    os.environ["HK_DIST_RUN_ID"] = run_id
    torch.ops.hk_dist.init(rank, world, max_tokens, hidden_size)
    _state = _State(rank, world, max_tokens, hidden_size, run_id)
    return _state


def shutdown():
    global _state
    if _state is None:
        return
    torch.ops.hk_dist.shutdown()
    _state = None


def supported(M, N, K_local, which="all_reduce"):
    """Would the fused path take this shape? Cheap; call it to decide a fallback."""
    _load()
    return bool(torch.ops.hk_dist.supported(int(M), int(N), int(K_local), which))


def why_unsupported(M, N, K_local, which="all_reduce"):
    _load()
    return torch.ops.hk_dist.why_unsupported(int(M), int(N), int(K_local), which)


def linear_allreduce(x, weight):
    """x @ weight.T, reduced across the group. x: (..., K_local), w: (N, K_local).

    Equivalent to ``F.linear(x, weight)`` followed by ``dist.all_reduce``, to
    within bf16 summation order.
    """
    flat = x.reshape(-1, x.shape[-1])
    out = torch.ops.hk_dist.linear_allreduce(flat.contiguous(), weight)
    return out.reshape(*x.shape[:-1], weight.shape[0])


def linear_reducescatter(x, weight):
    """Same product, but each rank keeps only its row shard: (M/world, N).

    This is the sequence-parallel form. It cannot serve decode -- M=1 has no
    rows to shard -- so `supported(..., "reduce_scatter")` rejects small M.
    """
    flat = x.reshape(-1, x.shape[-1])
    return torch.ops.hk_dist.linear_reducescatter(flat.contiguous(), weight)


class HKRowParallelLinear(nn.Module):
    """A RowParallelLinear whose matmul and all-reduce are one kernel.

    The weight layout is vLLM's exactly -- (output_size, input_size_per_partition),
    row major -- so `from_linear` is a rebind, not a copy or a transpose.

    Falls back to F.linear + all_reduce on any shape the fused epilogue does not
    accept, which is what makes this safe to drop in: the constraints (N a
    multiple of 128, N/world a multiple of 64, K_local a multiple of 32, M below
    the init() bound) hold for Qwen3-class models but are not universal.
    """

    def __init__(self, weight, bias=None, group=None, reduce_results=True):
        super().__init__()
        self.weight = weight
        self.bias = bias
        self.group = group
        self.reduce_results = reduce_results
        self._fused_calls = 0
        self._fallback_calls = 0

    @classmethod
    def from_linear(cls, layer, group=None):
        """Rebind an existing vLLM/SGLang RowParallelLinear or nn.Linear."""
        weight = layer.weight
        bias = getattr(layer, "bias", None)
        reduce_results = getattr(layer, "reduce_results", True)
        return cls(weight, bias, group, reduce_results)

    def forward(self, x):
        N, K_local = self.weight.shape
        M = int(x.numel() // x.shape[-1])
        use_fused = (
            _state is not None
            and self.reduce_results
            and x.dtype == torch.bfloat16
            and self.weight.dtype == torch.bfloat16
            and supported(M, N, K_local)
        )
        if use_fused:
            self._fused_calls += 1
            y = linear_allreduce(x, self.weight)
        else:
            self._fallback_calls += 1
            y = torch.nn.functional.linear(x, self.weight)
            if self.reduce_results and dist.is_initialized():
                dist.all_reduce(y, group=self.group)
        if self.bias is not None:
            y = y + self.bias
        return y

    def extra_repr(self):
        N, K = self.weight.shape
        return (f"out_features={N}, in_features_per_partition={K}, "
                f"fused={self._fused_calls}, fallback={self._fallback_calls}")
