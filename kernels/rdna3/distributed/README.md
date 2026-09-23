# Fusing the collective into the GEMM, on a PCIe workstation GPU

A row-parallel linear in vLLM or SGLang is two things in a row:

```python
out = F.linear(x, self.weight)              # this rank's slice of K
out = tensor_model_parallel_all_reduce(out) # every rank's slice, summed
```

and they are strictly sequential. The all-reduce cannot start until the whole
partial product exists, because a collective has no way to know which parts of
its input are finished. So on a machine where the interconnect is PCIe rather
than xGMI or NVLink, a measurable fraction of every TP layer is the GPU sitting
idle with a finished matrix, waiting for a bus.

The kernels here delete that wait by moving the collective *inside* the GEMM's
epilogue. Every output tile, at the moment its accumulator is complete and still
in registers, is written either to local memory or straight into its owner's
inbox on another GPU. By the time the last tile is computed, everything else is
already on the wire.

This is the RDNA3 half of that idea — gfx1100, one process per rank, IPC-mapped
peer memory, W7900D over PCIe Gen4. It is verified end to end as a torch
operator; what it is *not* verified inside is a real inference server, and
[§10](#10-what-this-is-not) says so precisely.

```
                  measured on 2 x W7900D, TP=2, Qwen3-27B shapes
                  against F.linear + RCCL, on the same hardware in the same run

  prefill  all-reduce        1.22 - 1.31x
  prefill  reduce-scatter    1.31 - 1.59x
  decode   mlp_down          1.21 - 1.38x
  decode   attn_out          0.92 - 1.06x     <- a wash, and §9 explains why
```

---

## Contents

1. [Why the upstream kernel could not be used](#1-why-the-upstream-kernel-could-not-be-used)
2. [The substrate: a symmetric heap across processes](#2-the-substrate-a-symmetric-heap-across-processes)
3. [Three things gfx1100 does that break the obvious design](#3-three-things-gfx1100-does-that-break-the-obvious-design)
4. [The fused epilogue](#4-the-fused-epilogue)
5. [Sharding along N, and why all-reduce gets to choose](#5-sharding-along-n-and-why-all-reduce-gets-to-choose)
6. [Making M arbitrary: prefill remainders and decode](#6-making-m-arbitrary-prefill-remainders-and-decode)
7. [The torch operator](#7-the-torch-operator)
8. [Framework integration](#8-framework-integration)
9. [Results](#9-results)
10. [What this is not](#10-what-this-is-not)
11. [Reproducing](#11-reproducing)
12. [Notes for anyone continuing this](#12-notes-for-anyone-continuing-this)

---

## 1. Why the upstream kernel could not be used

HipKittens upstream has a distributed story built on Iris, and a first fused
reduce-scatter here (`gemm_rs.hip`) reached 1.22–1.61x on real Qwen3 prefill
shapes almost immediately. It was also unusable by any framework, for four
independent reasons:

| gap | why it blocks integration |
|---|---|
| **One process drives N GPUs.** `symmem.cuh` allocates every rank's buffer with `hipMalloc` and reaches peers through `hipDeviceEnablePeerAccess`. | vLLM and SGLang are one process per rank. Peer memory there can only be IPC-mapped, and the entire allocation model differs. |
| **`M % 128 != 0` is unimplemented.** The GEMM had no M remainder path at all. | Decode is M ∈ [1, 32]. Prefill batches are not multiples of 128. |
| **Exactly two ranks.** The epilogue held a single `_gl_C c_peer`. | `kittens::gl` has no default constructor and a `__host__`-only one, so a kernel argument can hold neither an array of them nor one built on the device. TP=4/8 was not expressible. |
| **Reduce-scatter only.** | The default TP path in both frameworks is all-reduce. And reduce-scatter is not merely slower for decode — at M=1 it is *meaningless*, there is no row axis to shard. |

Each of those is addressed below, in that order.

## 2. The substrate: a symmetric heap across processes

[`ipc_heap.cuh`](ipc_heap.cuh) is `symmem.cuh` rebuilt for one-process-per-rank.
Each rank allocates its own slab with `hipExtMallocWithFlags`, exports it with
`hipIpcGetMemHandle`, and maps everyone else's with `hipIpcOpenMemHandle`:

```cpp
HKD_CHECK(hipExtMallocWithFlags(&mine_, bytes_, hipDeviceMallocFinegrained));
hipIpcMemHandle_t h;
HKD_CHECK(hipIpcGetMemHandle(&h, mine_));
std::vector<hipIpcMemHandle_t> all(bs_.world());
bs_.allgather(&h, all.data(), sizeof h);      // bootstrap, not a GPU collective
for (int r = 0; r < bs_.world(); r++) { ...
    HKD_CHECK(hipIpcOpenMemHandle(&p, all[r], hipIpcMemLazyEnablePeerAccess));
    view_.heap_bases_[r] = reinterpret_cast<uintptr_t>(p);
}
```

The device side is reused **verbatim**. `sym_view` is a trivially copyable table
of heap bases with

```cpp
template <typename T> __device__ T *translate(T *p, int owner) const;
```

which turns my address for a symmetric object into that rank's address by a
single subtract-and-add. A kernel genuinely cannot tell whether a base came from
`hipMalloc` on another device or from `hipIpcOpenMemHandle` on this one — an
offset into it is the same offset either way. Nothing written against
`symmem.cuh` had to change.

Two things do change, and both are about losing a guarantee that used to hold by
construction:

**Symmetry is now a convention, so it is checked.** `sym_heap::allocate_all` in
the single-process version could *enforce* that every rank allocated the same
sizes in the same order — that is what makes `translate()` a subtraction. Across
processes nothing enforces it. So `allocate()` accumulates an FNV-1a hash over
the requested sizes in order, and `check_symmetry()` allgathers `{used, hash}`
and aborts on a mismatch. Hashing the *sequence* rather than comparing the total
matters: two ranks that allocate the same bytes in a different order produce
valid-looking pointers into the wrong object, which is exactly the failure that
would otherwise be found by staring at numerically plausible garbage.

**Handle exchange is abstracted.** `bootstrap` is two virtuals — an allgather of
fixed-size blobs and a barrier. The standalone benchmarks implement it over
files (tmp + rename, no dependency on torch); the torch extension gets one
backed by the process group. There is deliberately no MPI: the node has none and
each implementation is a dozen lines.

### The bandwidth result that corrected a wrong conclusion

The whole cross-process port was gated on a benchmark, because the expectation
was that IPC mapping would cost bandwidth: an early probe read **14.5 GB/s**
through an IPC mapping against **27.0 GB/s** through `hipDeviceEnablePeerAccess`,
and the gap survived controlling for heap type, store width, buffer size,
consumer spin, and eager peer access. The mapping looked like the only variable
left.

It was not. **This node's peer bandwidth is bimodal per process launch** — about
15.5 GB/s or about 24–27 GB/s, with nothing in between, stable to three decimals
*within* a process and re-rolled at every launch. Eight consecutive runs on the
same GPU pair: `{15.5, 27.1, 15.5, 15.5, 15.5, 27.1, 27.0, 24.0}`. Ruled out:
iteration count, heap base alignment, other tenants, the GPU pair (all five
pairs are bimodal, including two completely idle ones), and PCIe link state
(both ends negotiate Gen4 x16 either way). `ipc_heap_test` reaches 24.0 GB/s
through IPC in both uni and ring modes.

So **there is no penalty for the deployment process model**, and the "IPC is
1.9x slower" conclusion was two measurements landing in different modes.

What it costs instead is the right to compare across runs, and that is a
structural constraint on every benchmark in this directory rather than noise to
average away: **a fused time from one process and a collective time from another
describe different hardware.** Everything compared here is measured in one
process, in one run, and each run prints which mode it landed in.

## 3. Three things gfx1100 does that break the obvious design

These are the three that cost the most time, and all three are measured
behaviours rather than documented ones.

### 3.1 A peer's write does not invalidate my L2, and no shader instruction can

This is the big one. Fine-grained memory is coherent at the right granularity
for cross-kernel visibility, so the natural design — peer pushes into my inbox,
barrier, my kernel reads the inbox — should work. It does not. Rank 1 read
`rel 0.6` garbage while every after-the-fact comparison of the same bytes from
the host read `rel 0`, because by then the lines had aged out.

The peer's store lands in my HBM without invalidating my L2. The zeros I
`hipMemset` into my own inbox are still sitting in that L2, and an ordinary load
returns *those*. And nothing a kernel can execute fixes it:

* `__threadfence_system()` compiles to `s_waitcnt_vscnt` plus `buffer_gl0_inv` /
  `buffer_gl1_inv` — L0 and L1, and it stops there. No writeback, no L2.
* A system-scope acquire load compiles to `global_load_b32 glc` plus the same
  two invalidates. Again L0 and L1.
* `slc`/`dlc` on a load are an L2 *policy* hint (evict-first), not a forced
  miss. A nontemporal load does compile to `flat_load_u16 ... slc dlc` and does
  **not** fix the read.

The only thing that invalidates L2 is the command processor's system-scope
acquire at dispatch, which the runtime emits on the first dispatch **after a
host sync**. That was confirmed directly: a bare `hipStreamSynchronize` between
the push and the consumer kernel makes the read correct.

So the fused path brackets its barrier with two syncs — a release before (which
pushes my own dirty lines out to the fabric) and an acquire after:

```cpp
dispatch_any(g);                                       // GEMM + peer pushes
if (st.inbox_cached) HKD_CHECK(hipStreamSynchronize(S));   // release
sync_ranks(S);                                             // device barrier
if (st.inbox_cached) HKD_CHECK(hipStreamSynchronize(S));   // acquire
hipLaunchKernelGGL(hk_dist::combine_and_gather, ...);
```

The symmetric fix is `hipDeviceMallocUncached`, which keeps the buffer out of L2
in the first place and needs no syncs at all. It is available
(`HK_DIST_UNCACHED_INBOX=1`) and it is **slower**: 2.57 vs 2.09 ms on prefill
attn_out. It is also a per-*heap* choice, not a global one — with the whole heap
uncached the fused GEMM collapses from 1.28x to 1.02x, because A and B are read
many times each. The inbox is the one buffer that wants it: written once by a
peer, streamed once by the combine, never reused. The default is the faster of
the two.

The write direction has the same disease and a different cure. A peer-aimed
store can sit dirty in *my* L2 long after the barrier has told the peer to read
it. `__builtin_nontemporal_store` is the one form that carries the bypass bits —
it emits `global_store_* ... glc slc dlc` — so every peer push in this tree goes
through `store_at_nt` / `peer_store`, and the inbox reads go through
`inbox_load`. **Those are correctness requirements, not hints.** No locality is
lost either way: each inbox element is read exactly once.

### 3.2 Device-memory flags do not work as barrier flags over PCIe

The obvious place for a barrier flag is the symmetric heap. In one process that
works. Across processes over PCIe it does not, and the failure is not subtle:
[`bar_probe.hip`](bar_probe.hip) measured a lockstep round trip through a
peer-mapped device flag at **seconds** — with every stream topology (one kernel,
two kernels, waiter on its own stream, event chain, host sync) and with the
signal as both a release store and an exchange.

Worse, the arrangements that *looked* fast were fast for a bad reason: their
signals ran ahead of their waits, so an early epoch's wait was satisfied by a
later epoch's store and no round trip ever happened. A barrier benchmark that
does not version its epochs will report success for a barrier that does not
barrier.

The flags therefore live in **host memory** — a POSIX shm segment named after
the run id, mmapped by every rank and pinned with `hipHostRegister`. It is
uncached by the GPU, both ranks reach the same physical page over PCIe, and no
`translate()` is needed because there is one array rather than one per rank.
This is what RCCL does for P2P flags on a PCIe fabric, for the same reason.

### 3.3 `clock64()` is 20 bits wide

Every in-kernel spin here has a deadline, because the alternative on a shared
node is a wedged GPU that takes the machine with it. The first version built
those deadlines on `clock64()` and they fired at random. On gfx1100 `clock64()`
is 20 bits and wraps in about 0.5 ms. `wall_clock64()` is the one to use:

```cpp
__global__ void dev_barrier(uint32_t *flags, uint32_t epoch, int world, int me,
                            long long budget, unsigned *timed_out) {
    const int r = threadIdx.x;
    if (r >= world) return;
    if (r == me)
        __hip_atomic_store(flags + me, epoch, __ATOMIC_RELEASE, __HIP_MEMORY_SCOPE_SYSTEM);
    const long long t0 = wall_clock64();
    while (__hip_atomic_load(flags + r, __ATOMIC_ACQUIRE, __HIP_MEMORY_SCOPE_SYSTEM) < epoch) {
        SPIN_TICK;
        if (wall_clock64() - t0 > budget) { atomicExch(timed_out, epoch); return; }
    }
}
```

One thread per rank, epoch-versioned (`<` not `==`, so a late arrival is never
satisfied by a stale value), budget set from
`hipDeviceAttributeWallClockRate * 2000` for a 2-second deadline. A timeout
latches in the extension and poisons the process group, because a peer that
missed a barrier makes every number after it void.

A host barrier would be correct and useless here: the bootstrap costs
milliseconds, the same order as the GEMMs being overlapped.

## 4. The fused epilogue

### 4.1 The one library change

The entire generalization to arbitrary TP rests on a single mechanical
refactor in [`include/rdna3/ops/warp/memory/tile/global_to_register.cuh`](../../../include/rdna3/ops/warp/memory/tile/global_to_register.cuh).
`store()` was doing two things: turning a `gl` and a `coord` into a base pointer
and a row stride, then vectorizing the tile out. Split at that seam:

```cpp
template<ducks::rt::all RT, typename U, bool NT = false>
__device__ inline static void store_at(U *dst_ptr, int row_stride, const RT &src);

template<int axis, ducks::rt::all RT, ducks::gl::all GL, ducks::coord::tile COORD>
__device__ inline static void store(const GL &dst, const RT &src, const COORD &idx) {
    store_at<RT, U>((U*)&dst[(idx.template unit_coord<axis, 3>())],
                    dst.template stride<axis>(), src);
}
```

The two share a body rather than duplicating the vectorization decision, and the
`gl` path is byte-identical to what it was. What this buys: a destination that
is a peer GPU's buffer is an ordinary global pointer but is *not* describable as
a `gl`. With `store_at`, a peer store is still a fully vectorized
`global_store_b128` that happens to land on another GPU — no per-element loop,
and the same code for every world size.

`store_at_nt` is the same with the bypass bits, per §3.1.

That is the only file under `include/rdna3` that this work touched.

### 4.2 No peer count anywhere

Combined with `sym_view`, the globals structs never mention a world size:

```cpp
struct micro_globals_ar {
    _gl_A a; _gl_B b;
    _gl_C c;          // (M, N) final output, symmetric
    sym_view v;       // the inbox heap's view
    bf16 *c_own;      // (M, shard_cols) my columns of my partial
    bf16 *c_sym;      // my slot of the inbox
    int shard_cols; int my_rank; float *ws; hipStream_t stream;
    static constexpr bool FUSED_RS = false;
    static constexpr bool FUSED_AR = true;
};
```

`v.translate(c_sym, owner)` is correct for any world. The two `constexpr` flags
exist on both structs so the shared epilogue selects with `if constexpr` and no
trait machinery; a plain `micro_globals` sets both false and compiles to exactly
the code it had before.

`c_sym` is already *my slot* — translating it re-points it at an owner without
changing which slot it is, so **no sender needs to know which slot it is
writing**, and two senders never collide. (The first version added
`my_rank * stride` here as well, offsetting twice: rank 1's tiles landed a slot
past where rank 0 looked for them, and at prefill M, a whole output past the end
of the inbox.)

### 4.3 The register-pressure trap

This one is worth reading even if you never touch this kernel, because the
symptom is so specific and so quiet.

The ownership test in the all-reduce epilogue is per **warp tile**, outside the
`N_SPLIT` unroll:

```cpp
const int owner = c_base / n_shard;
if (owner == g.my_rank) {
    bf16 *dst = g.c_own + off;
    #pragma unroll
    for (int s = 0; s < N_SPLIT; s++) store_at(dst + s * SPLIT_N, n_shard, C_accum[s]);
} else {
    bf16 *inbox = g.v.translate(g.c_sym, owner) + off;
    #pragma unroll
    for (int s = 0; s < N_SPLIT; s++) store_at_nt(inbox + s * SPLIT_N, n_shard, C_accum[s]);
}
```

Deciding per `N_SPLIT` chunk instead — which is the more natural way to write it
— puts a branch with two store bodies *inside* the loop. On `big_config` that
costs: **209 VGPRs and no spill becomes 256 VGPRs, 8 VGPR spills, 36 bytes/lane
of scratch.**

And this kernel does not survive spilling. Its LDS pipeline hand-manages
`s_waitcnt` counts — `lds_wait<N>()` deliberately leaves N operations
outstanding — and the spill/reload traffic the compiler interleaves is counted
by *the same hardware counters*. The waits stop meaning what they were written
to mean.

The symptom: every `s == N_SPLIT-1` tile, and only those, came out NaN. Prefill
shapes only, because the decode configs are far enough from the register ceiling
that they never spilled. Silently wrong, in one quarter of the output, on
exactly the shapes you would benchmark.

**Any change to this epilogue must be checked with
`-Rpass-analysis=kernel-resource-usage`, and `ScratchSize` must be 0.**
`make resources` in [`torch_ext/`](torch_ext/) runs it.

## 5. Sharding along N, and why all-reduce gets to choose

Reduce-scatter shards along **M**, because it has no choice: the framework owns
that axis — it splits by token — and the kernel has to agree with whatever the
sequence-parallel layout says. Every rank computes a full M×N partial over its
own K-slice, and each output tile is either mine to keep or the owner's to
receive:

```cpp
const int owner     = m_row / g.shard_rows;
const int local_row = m_row - owner * g.shard_rows;
```

All-reduce has no such obligation. Every rank ends up holding the whole M×N, so
which axis the *reduction* is sharded over is entirely an internal choice, and N
is the better one for two reasons:

**It makes M unconstrained.** The M-sharded path needs
`shard_rows % REG_BLOCK_M == 0`, which rejects most decode shapes outright — and
at M=1 there is nothing to shard at all. Sharding along N leaves M alone, so the
same epilogue serves prefill and decode. *This is the single decision that makes
decode possible.*

**For world > 2 it moves less data.** Reduce along N then gather back is
`2(w-1)/w` of C; having every rank push its whole partial to everyone is
`(w-1)`.

The constraint that replaces the M one is that a warp tile must not straddle two
owners. A warp tile is `REG_BLOCK_N` contiguous columns starting at a multiple
of `REG_BLOCK_N`, so it stays inside one shard as long as the shard width does
too. The host cannot see which config the dispatcher picked, so it checks the
loosest sufficient width — 64 — and a `static_assert` in the kernel keeps that
honest:

```cpp
static_assert(64 % REG_BLOCK_N == 0,
              "the fused all-reduce host check assumes a warp tile is "
              "at most 64 columns and divides 64");
```

For Qwen3 at N=5120 the shards are 2560 / 1280 / 640 at TP=2/4/8 — all multiples
of 64.

The gather is a second kernel, [`combine_and_gather`](fused.cuh), which
accumulates in fp32 (the partials are bf16 and summing `world-1` of them in bf16
would throw away most of what little mantissa there is; the pass is entirely
memory-bound, so the wider accumulator is free), then writes each finished
element into every rank's copy of the output through the same `translate()`:

```cpp
const int row = (int)(i / ns), lc = (int)(i % ns);
const size_t off = (size_t)row * N + (size_t)me * ns + lc;
for (int r = 0; r < world; r++) {
    bf16 *dst = v.translate(out, r) + off;
    if (r == me) *dst = val; else peer_store(dst, val);
}
```

`translate(p, me)` is the identity, so routing my own copy through it costs
nothing and keeps one code path.

## 6. Making M arbitrary: prefill remainders and decode

### 6.1 The last block backs up

The GEMM had no M remainder path. The fix is not masking — it is arranging that
there is never a partial tile:

```cpp
const int m_base = min(row * BLOCK_M, M - BLOCK_M);
```

The last block starts at `M - BLOCK_M` instead of where the grid says, so the
whole tile is in range and **neither the load nor the store needs a predicate**.
That matters more than it looks: `global_to_shared.cuh`'s `load()` is bare
pointer arithmetic with no bounds check at all, so a predicated design would have
meant touching the library's hot path. The overlapped rows are computed twice and
written with identical values — idempotent.

The grid's `m_blocks` becomes a `ceil` to match (a floor leaves the tail block
unlaunched while the kernel still expects it), and the addressing switches from
tile units to element units:

```cpp
store(g.c, C_accum[s], coord<>{0, 0, m_row, (col_tile + s) * SPLIT_N});
```

This needed no library change either: `ducks::coord::tile` already accepts
`default_type`, and `unit_coord()` is the identity for it.

Verified at M ∈ {129, 255, 1500, 2049, 3008, 4097, 7777}.

### 6.2 Decode is weight streaming, not a GEMV

The backup trick only works for M ≥ BLOCK_M. Below that, the other half:
`BLOCK_M = 16`.

The key observation is that at decode sizes this is not a compute problem at
all. The whole `mlp_down` WMMA is about **6 µs** against a **114 µs** floor set
by reading the weights. Padding M up to the WMMA's 16 rows therefore costs
nothing measurable — the padded rows ride in cache lines that had to be fetched
anyway — and *no GEMV kernel is needed*. The decode path is the same kernel with
a thin config:

```cpp
using thin_sk1_config = config<16, 128, 32, 16, 4, 1, 8, 2, 1, 2, 1>;
using thin_config     = config<16, 128, 32, 16, 4, 1, 8, 2, 1, 2, 4>;
using thin_sk8_config = config<16, 128, 32, 16, 4, 1, 8, 2, 1, 2, 8>;
using thin32_config   = config<32, 128, 32, 16, 4, 1, 8, 2, 1, 2, 1>;
```

All 8 warps spread along N (`WARP_ROWS=1`), because what matters is keeping B in
flight. `K_STEP=32` rather than 64 because at `BLOCK_N=256` a 64-deep B tile is
32 KB and two of those plus A overflow 64 KB of LDS.

**Split-K, and where it stops paying.** At N=5120 the M/N grid is 40 workgroups
however K is cut, so a K split is the only source of parallelism. But each slice
writes a full fp32 partial to a `[SPLIT_K][M][N]` workspace and reads it back,
and that traffic is pure overhead. Measured at M=8, N=5120 (GB/s of weight
traffic):

| | SPLIT_K=1 | 2 | 4 | 8 |
|---|---|---|---|---|
| attn_out K=3072 | 291 | 314 | 342 | **372** |
| mlp_down K=8704 | 588 | 631 | **633** | 577 |

Short K has too little work per block to cover latency and wants the deeper
split; long K already covers it and just pays. Hence 8 slices below ~128
K-tiles, 4 above.

At M=32 the table inverts completely:

| | SPLIT_K=1 | 2 | 4 | 8 |
|---|---|---|---|---|
| attn_out K=3072 | **329** | 300 | 316 | 258 |
| mlp_down K=8704 | **628** | 605 | 595 | 458 |

**The reason is that split-K's overhead scales with M and the weight traffic
does not.** At M=32, K=3072, SPLIT_K=8 the workspace is 10.5 MB against 31 MB of
weights — a 33% surcharge on a kernel that is purely bandwidth-bound. So the
32-row config does not split K, and `SPLIT_K > 1` is a decode-only tool.

The reduction is a fixed-order sum rather than fp32 atomics, because a framework
cannot accept a kernel whose output changes run to run. When the dispatcher
picks a split-K config the register epilogue cannot push to peers — it has fp32
partials, not an answer — so the scatter moves into the reduction pass
(`reduce_splitk_scatter_n`), which is the right place for it anyway: that pass
already touches every output element exactly once, so the push costs only the
difference between a local store and a peer store.

Decode results, against TunableOp-tuned hipBLASLt:

| op | M=1..16 | M=32 | vs hipBLASLt |
|---|---|---|---|
| mlp_down (K_local 8704) | 627–637 GB/s | 628 | ~95% of 665 GB/s |
| attn_out (K_local 3072) | 337–380 GB/s | 329 | ~65% of 549 GB/s |

The attn_out column is a known, unclosed gap, and it is the reason one row of
the final table does not win. See §9.

### 6.3 The dispatch hole at 17 ≤ M ≤ 31

Found while generalizing, and worth stating because it is an out-of-bounds
*write*, not a wrong answer:

```cpp
} else if (g.a.rows() < thin32_config::BLOCK_M) {
    // 17..31. The 32-row config would be the faster one, but the
    // last-block backup that makes M remainders work needs
    // M >= BLOCK_M: below that m_base pins to 0 and the kernel writes
    // a whole 32-row tile into an M-row buffer, past its end. The
    // 16-row config has two blocks here and keeps m_base in
    // {0, M-16}, both in range.
    launch<thin_sk1_config>(g);
}
```

Below one thin block (M < 16) there is nothing to tile with at all, so the
non-fused path stages through a zero-padded copy — under 150 KB at decode shapes
against ~30 MB of weight traffic, invisible in the measurement.

## 7. The torch operator

[`torch_ext/hk_dist_ext.hip`](torch_ext/hk_dist_ext.hip) registers:

```
hk_dist::init(rank, world, max_tokens, hidden)   build the IPC heaps
hk_dist::shutdown()
hk_dist::supported(M, N, K_local, which) -> bool
hk_dist::why_unsupported(...) -> str
hk_dist::linear_allreduce(x, w) -> (M, N)        x: (M, K_local), w: (N, K_local)
hk_dist::linear_reducescatter(x, w) -> (M/w, N)
```

`w` is exactly vLLM's `RowParallelLinear.weight` — `(output_size,
input_size_per_partition)`, row major. Nothing transposes anywhere, because
HipKittens' native layout (`A(m,k)`, `B(n,k)`, `C = A·Bᵀ`) is already what a
framework's row-parallel weight is stored as.

The shape constraints live in one function, `unsupported()`, so that python can
ask before committing and the ops refuse with the same words:

```cpp
if (N % st.world)              return "N does not split across ranks";
if (N % thin_config::BLOCK_N)  return "N does not tile";
if (K_local % big_config::K_STEP) return "K_local does not tile";
if (ar) {
    if ((N / st.world) % 64)   return "the column shard boundary falls inside a warp tile";
    ...
```

Four things about this layer are non-obvious:

**Build with plain `hipcc`, load with `torch.ops.load_library`.** Not
`torch.utils.cpp_extension.load` — on ROCm that runs hipify over the source, and
this source is already HIP, so the second pass mangles it. The Makefile needs
`-D_GLIBCXX_USE_CXX11_ABI=$(python -c "import torch;print(int(torch._C._GLIBCXX_USE_CXX11_ABI))")`
and `-I$(ROCM_PATH)/include/hip` (without the latter, `hip_bf16.h` is not found).

**`HIPGuard` will not accept the device torch hands you.**
`c10::hip::HIPGuard guard(x.device())` throws
`HIPGuardImpl initialized with non-HIP DeviceType: cuda` — torch on ROCm still
tags its devices `DeviceType::CUDA`, and `HIPGuardImpl` rejects that. Use the
`DeviceIndex` overload:

```cpp
const c10::hip::HIPGuard guard((c10::DeviceIndex)x.get_device());
```

**An op with no Tensor argument has no dispatch key.** `init`, `shutdown`,
`supported`, `why_unsupported` and `reset_inbox` must be registered under
`CompositeExplicitAutograd`; under `CUDA` the dispatcher has no device argument
to read and they are never called.

**The inbox slot stride is derived from the *padded* M.** The epilogue writes
whole `TILE_M`-row tiles, so it places slot `r` at `Inbox + r * M_pad * ns`,
while `combine_and_gather` computes `n = M * ns` from the M it is handed. Pass
the unpadded M and the two disagree for every shape below one tile — rank 1's
slot gets read up to 15 rows early. Symptom: every M < 16 at rel 0.55–0.84,
every M ≥ 16 clean. The padded rows are zeroed so their products are zero, and
the copy-out takes only the first M rows.

### The two costs this design has

**Two host syncs per call** — §3.1. The op blocks the calling thread twice per
layer.

**The all-reduce output is copied.** `combine_and_gather` must write into a
*symmetric* buffer, and torch's caching allocator cannot hand out fine-grained
IPC-exportable memory. So the op runs into the heap and copies out: about
0.21 ms at M=8192, 4% of the fused attn_out and 2% of mlp_down. Removing it
means a pluggable torch allocator backed by the symmetric heap — an allocator
change, not a kernel change. **The reduce-scatter form does not pay it**: its
output is local, so the epilogue writes the torch tensor directly. That is most
of why the RS numbers are better.

### The python side

```python
import hk_dist
hk_dist.init(max_tokens=8192, hidden_size=5120)   # after dist.init_process_group
y = hk_dist.linear_allreduce(x, layer.weight)     # x: (..., K_local)
```

plus `HKRowParallelLinear`, a drop-in `nn.Module` with a `from_linear()`
classmethod that checks `supported()` on every forward and falls back to
`F.linear + dist.all_reduce` on anything the epilogue rejects, counting both
paths so a silent fallback cannot masquerade as a fast one.

## 8. Framework integration

[`integration/vllm_patch.py`](integration/vllm_patch.py) and
[`integration/sglang_patch.py`](integration/sglang_patch.py) monkeypatch
`RowParallelLinear.forward` in each framework.

The design decision worth stating: **wrap `forward`, do not reimplement it.**
The eligible path is short and self-contained and everything else calls straight
through to the original bound method, so a framework that has moved on breaks by
falling back — not by computing something subtly wrong.

The gate is eight clauses, each a real requirement:

| clause | why |
|---|---|
| `hk_dist.is_initialized()` | no heaps, no fused path |
| `reduce_results` and `tp_size > 1` | the kernel *is* the all-reduce; with none to do there is nothing to fuse |
| `input_is_parallel` | otherwise `forward` first splits the input along the last dim |
| `type(quant_method).__name__ == "UnquantizedLinearMethod"` | the kernel reads a plain bf16 `(N, K_local)` weight; a quantised method stores something else entirely and a shape check would not catch all of them |
| bf16 weight, 2-D | the only dtype the RDNA3 GEMM implements |
| bf16 `x` with matching K | |
| `hk_dist.supported(M, N, K_local)` | asked of the extension rather than duplicated, so the two cannot drift |

Bias is added on `tp_rank == 0` only, for the same reason upstream does it — an
all-reduce over `tp_size` ranks would otherwise add it `tp_size` times — except
that here it happens after the fused reduction rather than being folded into the
GEMM. `skip_bias_add` and `return_bias` are honoured.

SGLang gets one extra clause: `can_fuse_mlp_allreduce`. When SGLang sets it, the
layer is deliberately *not* supposed to reduce — it is deferring the all-reduce
to a later fusion pass — so a kernel that always reduces must not run.

**Neither file has been run inside a server.** See §10.

## 9. Results

Two W7900D (gfx1100) over PCIe, TP=2, Qwen3-27B shapes, torch 2.10 + ROCm
7.2.4, 25 GB/s peer mode. `ref` is `F.linear` + `dist.all_reduce` on RCCL —
literally the body of vLLM's `RowParallelLinear.forward`. `gemm` is `F.linear`
alone with no collective at all: not a correct implementation, just the floor
the fused path is trying to reach.

**Every row was checked elementwise before it was timed.** 30 shapes, rel
1.7e-3 to 1.2e-2 against a 5e-2 tolerance (bf16 double rounding plus the
summation is ≈1.2e-2, so the tolerance is 4x headroom while still being 20x
below what a mis-routed tile would produce). Weights are rotated four ways at
decode sizes, because 96 MB of Infinity Cache against 31–89 MB of decode weights
will otherwise report bandwidths above the HBM peak.

### All-reduce

| phase | op | M | gemm | ref | fused | speedup | rel |
|---|---|---:|---:|---:|---:|---:|---:|
| decode | attn_out | 1–8 | 0.128–0.133 | 0.160–0.169 | 0.173–0.176 | 0.92–0.97x | ~2e-3 |
| decode | attn_out | 16 | 0.132 | 0.172 | 0.162 | 1.06x | 2.98e-3 |
| decode | attn_out | 32 | 0.136 | 0.181 | 0.176 | 1.03x | 1.16e-2 |
| decode | mlp_down | 1–8 | 0.246–0.247 | 0.280–0.289 | 0.228–0.231 | 1.21–1.26x | ~3e-3 |
| decode | mlp_down | 16 | 0.247 | 0.293 | 0.216 | **1.35x** | 6.58e-3 |
| decode | mlp_down | 32 | 0.248 | 0.299 | 0.218 | **1.38x** | 1.25e-2 |
| prefill | attn_out | 2048 | 1.057 | 1.958 | 1.493 | **1.31x** | 8.16e-3 |
| prefill | attn_out | 4096 | 1.895 | 3.660 | 2.804 | **1.31x** | 9.26e-3 |
| prefill | attn_out | 8192 | 3.679 | 7.186 | 5.519 | **1.30x** | 8.70e-3 |
| prefill | mlp_down | 2048 | 2.886 | 3.823 | 2.980 | **1.28x** | 1.12e-2 |
| prefill | mlp_down | 4096 | 5.183 | 7.036 | 5.703 | **1.23x** | 1.05e-2 |
| prefill | mlp_down | 8192 | 10.493 | 13.792 | 11.316 | **1.22x** | 1.10e-2 |

### Reduce-scatter (sequence-parallel form)

Against `F.linear` + `dist.reduce_scatter_tensor`. Better than all-reduce
throughout, because the output is local and there is no copy out of the
symmetric heap:

| op | M=2048 | 4096 | 8192 |
|---|---|---|---|
| attn_out | **1.50x** | **1.52x** | **1.59x** |
| mlp_down | **1.36x** | **1.31x** | **1.31x** |

### Reading the decode `attn_out` row honestly

It is a wash: 0.92x at M=1, 1.06x at M=16. Two separate things are stacked
there and they should not be reported as one number.

**(a) The collective is not hidden at decode sizes.** Against *our own* GEMM —
which the standalone benchmark can measure, and the torch table cannot — the
fused path runs at about **2.1x the pure-GEMM floor**. Two cross-rank barriers at
~6 µs each plus a combine kernel is a large fraction of a 41–98 µs GEMM. There
is simply nothing to overlap with when the thing to overlap with is that short.
The fused path still beats the *sequential* path (which pays the same collective
without hiding any of it), which is why the mlp_down rows win — but the margin
is set by the fixed overhead, not by overlap quality.

**(b) Our decode GEMM is not uniformly competitive.** It beats hipBLASLt on
mlp_down (K_local=8704, ~95% of a 665 GB/s reference and faster end to end) and
reaches only ~65% of it on attn_out (K_local=3072). That is a GEMM problem
inherited from §6.2, not a collective one, and it is why one row wins and the
other does not.

The `f/g` column in the test output (fused time over `F.linear` time) is
reported but easy to misread: `gemm` there is **hipBLASLt**, not our GEMM, so
`f/g < 1` at decode means our GEMM beat hipBLASLt — it does not mean the
collective vanished.

### The slow bandwidth mode

The node's slow mode (§2) could not be sampled directly — after a reboot,
fourteen consecutive launches all landed fast. The equivalent experiment is to
hold M and N and shrink K by 25.6/15.5 = 1.655, which leaves the transfer
unchanged and scales the compute down by exactly the factor the slow wire would
have added to `t_comm/t_gemm`:

| shape | K | gemm | seq | fused | speedup | f/g |
|---|---|---|---|---|---|---|
| attn_out | 6144 (real) | 3.38 | 5.22 | 3.94 | 1.33x | 1.16 |
| attn_out | 3712 (≈slow) | 2.06 | 3.87 | 3.39 | **1.14–1.21x** | **1.57–1.65** |
| mlp_down | 17408 (real) | 9.41 | 11.58 | 9.56 | 1.21x | 1.02 |
| mlp_down | 10496 (≈slow) | 5.77 | 7.51 | 6.03 | **1.25–1.32x** | 1.04–1.06 |

mlp_down does *better* on a slow wire — there is more to hide, so hiding it is
worth more — and its overlap quality does not degrade at all (f/g stays 1.02–1.06).
attn_out sits on the line, and the real signal is f/g jumping to 1.6: at
K_local=1856 there is no longer enough arithmetic to cover a 1.8 ms push.

**The honest general statement: on this PCIe fabric the fusion is reliably
worthwhile for high-arithmetic-intensity layers, and marginal for low-intensity
ones when the wire is slow.**

## 10. What this is not

* **The framework patches have never run inside a server.** There is no RDNA3
  vLLM or SGLang build on the development machine. What is verified is
  everything underneath them — `torch.ops.hk_dist` matches `F.linear +
  dist.all_reduce` elementwise on 30 Qwen3 TP=2 shapes under real `torchrun` —
  and the gate logic itself, which `test_torch_ext.py` exercises against a
  stand-in layer with five cases per patch. What is unverified is each file's
  reading of its framework's internals: the attribute names, the quant-method
  check, and whether `forward` still looks like that in your version.
* **TP > 2 is generalized, not measured.** The code has no peer count in it and
  TP=4/8 satisfy the shape constraints, but only two GPUs on this shared node
  were free. It is also expected to *lose* to RCCL there: the fabric's aggregate
  bandwidth is ~36 GB/s and flat, putting break-even at ~8600 FLOP/byte for
  TP=4 and ~17000 for TP=8, while Qwen3 at TP=8 has K_local=2176. Generalizing
  was about not hard-coding a world size, not about winning at one.
* **bf16 only.** No fp8, no quantised weights. The gate rejects them rather than
  guessing.
* **Column-parallel / all-gather is not fused.** Only the row-parallel
  (all-reduce) and sequence-parallel (reduce-scatter) forms.
* **MoE is not covered.** The epilogue assumes one dense weight per rank.
* **Two host syncs per call** are a real cost to a framework's CPU-side
  scheduling, and no attempt was made to hide them behind graph capture.

## 11. Reproducing

Standalone benchmarks, no torch:

```
make gemm_ar_mp gemm_rs_mp
HIP_VISIBLE_DEVICES=0,3 ./run_mp.sh 2 ./gemm_ar_mp
```

`run_mp.sh` is a torchrun-free launcher that wraps each rank in
`timeout --signal=KILL`. Do that, or the equivalent, on any shared machine: an
unbounded in-kernel spin on a wedged peer takes the node down, and a cgroup OOM
that interrupts a collective on this hardware takes the whole machine with it.

The torch operator:

```
cd torch_ext && make
HIP_VISIBLE_DEVICES=0,3 timeout --signal=KILL 1200 \
    torchrun --nproc_per_node=2 test_torch_ext.py
make resources     # the spill check -- ScratchSize must be 0
```

Probes that establish the facts in §2 and §3 rather than assuming them:
`ipc_heap_test.hip` (translate / handshake / bandwidth), `bar_probe.hip` (the
device-flag round trip, across five stream topologies), `symmem_test.hip`.

## 12. Notes for anyone continuing this

Things that cost time here, written down so they cost it only once. The
architecture-level ones are in [`RDNA.md`](../../../RDNA.md); these are the
distributed ones.

* **Nothing a kernel can execute invalidates gfx1100's L2.** Not
  `__threadfence_system()`, not a system-scope acquire, not `slc`/`dlc` on a
  load. Only the dispatch-time acquire after a host sync. Any design that has a
  kernel consume peer-written data without a host sync in between is wrong on
  this hardware, and wrong in a way that passes every after-the-fact check from
  the host because the lines age out.
* **Cross-rank flags go in host memory.** Device flags over PCIe measured a
  round trip in *seconds*, and the topologies that looked fast were satisfying
  each epoch's wait with the next epoch's store. Version your barrier epochs or
  your barrier benchmark will lie to you.
* **`clock64()` is 20 bits on gfx1100** and wraps in ~0.5 ms. Every timeout
  built on it fires at random. Use `wall_clock64()`.
* **Measure the wire in the same process as the compute.** This node's peer
  bandwidth is bimodal per launch with nothing in between, so any cross-run
  "communication vs computation" comparison has a coin-flip chance of comparing
  two different hardware states rather than two pieces of code.
* **Check `ScratchSize` after every epilogue change.** This GEMM hand-manages
  `s_waitcnt` and spill traffic uses the same counters; the failure mode is one
  accumulator tile per warp coming out NaN on prefill shapes only. See §4.3.
* **Rotate the weights in any decode benchmark.** 96 MB of Infinity Cache
  against 31–89 MB of weights will report above-HBM-peak bandwidth from a single
  reused copy.
* **Split-K overhead scales with M.** The `[SPLIT_K][M][N]` fp32 workspace is
  written and read while the weight traffic is M-independent, so a split that
  wins by 28% at M=8 loses by 22% at M=32.
* **On ROCm, `torch.utils.cpp_extension.load` hipifies HIP source.** Build with
  plain `hipcc` and load the `.so` with `torch.ops.load_library`. And
  `c10::hip::HIPGuard` needs the `DeviceIndex` overload, because torch on ROCm
  tags devices `DeviceType::CUDA` and `HIPGuardImpl` rejects exactly that.

## Files

| | |
|---|---|
| [`ipc_heap.cuh`](ipc_heap.cuh) | symmetric heap across processes, bootstrap, host flags |
| [`symmem.cuh`](symmem.cuh) | `sym_view`, `translate()` — the single-process original, device side reused verbatim |
| [`fused.cuh`](fused.cuh) | the device core: globals, peer access, combines, barrier |
| [`gemm_ar_mp.hip`](gemm_ar_mp.hip) | standalone all-reduce sweep, measures its own wire |
| [`gemm_rs_mp.hip`](gemm_rs_mp.hip) | standalone reduce-scatter sweep |
| [`torch_ext/`](torch_ext/) | the torch operator, `HKRowParallelLinear`, the end-to-end test |
| [`integration/`](integration/) | vLLM and SGLang patches — **unverified**, see their headers |
| [`bar_probe.hip`](bar_probe.hip), [`ipc_heap_test.hip`](ipc_heap_test.hip) | the probes behind §2 and §3 |
| [`baseline_torch.py`](baseline_torch.py), [`bench_rccl.py`](bench_rccl.py) | the hipBLASLt + RCCL references |

The fused epilogues themselves are in
[`../gemm/bf16fp32/gemm.cpp`](../gemm/bf16fp32/gemm.cpp), selected with
`if constexpr` on the globals type, so the fused and unfused kernels differ in
nothing but the epilogue — which is the only way the comparison means anything.
