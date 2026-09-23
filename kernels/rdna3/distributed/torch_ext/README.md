# `torch.ops.hk_dist` — fused GEMM → collective as a torch operator

A row-parallel linear in vLLM or SGLang is a matmul over this rank's K-slice
followed by an all-reduce, and the two are strictly sequential: the collective
cannot begin until the whole partial output exists. The kernels here put the
collective in the GEMM's epilogue — each output column block is pushed into its
owner's inbox the moment its accumulator is finished — so the transfer happens
while the rest of the matmul is still running.

```
make                                   # -> hk_dist_ext.so
HIP_VISIBLE_DEVICES=0,3 torchrun --nproc_per_node=2 test_torch_ext.py
```

```python
import hk_dist
hk_dist.init(max_tokens=8192, hidden_size=5120)     # after dist.init_process_group
y = hk_dist.linear_allreduce(x, layer.weight)       # x: (..., K_local)
```

`weight` is vLLM's `RowParallelLinear.weight` unchanged — `(output_size,
input_size_per_partition)`, row major. Nothing transposes anywhere.

## Measured

Two W7900D (gfx1100) over PCIe, TP=2, Qwen3-27B shapes, torch 2.10 + ROCm 7.2.4,
25 GB/s peer bandwidth. Every row was checked elementwise against `F.linear +
dist.all_reduce` before it was timed. `ref` is that reference on RCCL; `gemm` is
`F.linear` alone with no collective at all, i.e. hipBLASLt's time for the same
matmul.

| phase | op | M | ref (ms) | fused (ms) | speedup |
|---|---|---:|---:|---:|---:|
| decode | attn_out | 1–32 | 0.160–0.181 | 0.162–0.176 | 0.92–1.06x |
| decode | mlp_down | 1–32 | 0.280–0.299 | 0.216–0.231 | 1.21–1.38x |
| prefill | attn_out | 2048–8192 | 1.96–7.19 | 1.49–5.52 | 1.30–1.31x |
| prefill | mlp_down | 2048–8192 | 3.82–13.79 | 2.98–11.32 | 1.22–1.28x |

Reduce-scatter (the sequence-parallel form, against `F.linear +
dist.reduce_scatter_tensor`) does better, because its output is local and it
skips the copy out of the symmetric heap: **1.31–1.58x** across the same prefill
shapes.

### What the decode numbers mean

`attn_out` decode is a wash — 0.92x at M=1, 1.06x at M=16. Two things are
stacked there and it is worth keeping them apart:

* The collective is **not** hidden at decode sizes. Against our own GEMM
  (measured by `../gemm_ar_mp.hip`, which has that floor available) the fused
  path runs at about 2.1x the pure-GEMM time. Two cross-rank barriers at ~6 µs
  each plus a combine kernel is a large fraction of a 41–98 µs GEMM. There is no
  overlap to be had when the thing to overlap with is that short.
* Our decode GEMM beats hipBLASLt on `mlp_down` (K_local = 8704) and loses to it
  on `attn_out` (K_local = 3072, where we reach ~65% of its bandwidth). That is
  why one row wins and the other does not, and it is a GEMM problem, not a
  collective one.

## Two costs this design has

**Two host syncs per call.** On gfx1100 a peer's write into my HBM does not
invalidate my L2, and nothing a shader can execute does either; the only thing
that does is the command processor's system-scope acquire, which the runtime
emits on the first dispatch after a host sync. So the op brackets its barrier
with `hipStreamSynchronize` on each side. `HK_DIST_UNCACHED_INBOX=1` removes the
need for them by keeping the inbox out of L2 entirely, and was measured slower
end to end (2.57 vs 2.09 ms on prefill attn_out). The default is the faster one.

**The all-reduce output is copied.** `combine_and_gather` must write into a
*symmetric* buffer — every peer pushes its finished column shard into mine — and
torch's caching allocator cannot hand out fine-grained IPC-exportable memory. So
the op runs into the heap and then copies out, about 0.21 ms at M=8192 (4% of
the fused attn_out, 2% of mlp_down). Removing it means a pluggable torch
allocator backed by the symmetric heap, not a kernel change. The reduce-scatter
form does not pay it: its output is local, so the epilogue writes the torch
tensor directly.

## Shape constraints

Asked of the extension rather than duplicated by callers —
`hk_dist.why_unsupported(M, N, K_local, which)` returns the reason as a string,
and `HKRowParallelLinear` falls back to `F.linear + all_reduce` on anything it
rejects.

All-reduce (`which="all_reduce"`, the default):

* `N % world == 0` and `(N / world) % 64 == 0` — a warp tile is at most 64
  columns wide and must not straddle two owners. The epilogue decides ownership
  once per warp tile and *not* per 16-column chunk, because doing it per chunk
  spills registers and silently corrupts the last accumulator; see the comment
  on the `FUSED_AR` branch in `../../gemm/bf16fp32/gemm.cpp`.
* `N % 128 == 0`, `K_local % 32 == 0` — the GEMM's own tiling.
* `M` up to the `max_tokens` given to `init()`. Otherwise free, including M=1:
  this is why all-reduce shards along N rather than M, and it is what lets
  decode use this path at all.

Reduce-scatter additionally needs `M % world == 0`, `M >= 128`, `M % 32 == 0`
and `(M / world) % 32 == 0`. It cannot serve decode — M=1 has no rows to shard.

Qwen3-27B at TP=2/4/8 satisfies the all-reduce set (N=5120 gives shards of
2560/1280/640, all multiples of 64).

## Files

| | |
|---|---|
| `hk_dist_ext.hip` | the operator: heaps, barriers, the two fused paths |
| `hk_dist/__init__.py` | `init`, `linear_allreduce`, `HKRowParallelLinear` |
| `test_torch_ext.py` | correctness against `F.linear + dist.all_reduce`, and the table above |
| `Makefile` | plain `hipcc` → `.so`; `make resources` runs the spill check |
| `../fused.cuh` | the device core shared with the standalone benchmarks |
| `../integration/` | vLLM and SGLang patches — **unverified**, see their headers |

Built with plain `hipcc` and loaded via `torch.ops.load_library`, not
`torch.utils.cpp_extension.load`: on ROCm the latter runs hipify over the
source, and this source is already HIP.
