# Flash attention on a Radeon, derived from the register file outwards

Every framework that runs attention on a Radeon — diffusers, comfyUI, vLLM,
SGLang — reaches `torch.nn.functional.scaled_dot_product_attention`, and on
ROCm/gfx1100 that lands in aotriton's `attn_fwd`. Confirmed with the profiler,
not inferred: both the `FLASH` and the `EFFICIENT` backend dispatch to the same
kernel.

On a W7900D that kernel is **flat at 20–21 TFLOPs** from N=4096 to N=65536. The
same chip does 77 TFLOPs on our own bf16 GEMM and has a measured WMMA ceiling
around 100. So the attention every Radeon user actually runs is at **21% of the
machine**, and it stays there no matter how long the sequence gets.

This directory is a single-GPU flash-attention forward that is not.

```
              B=1, H=56, D=128, bf16, non-causal, one process, one run
              W7900D (gfx1100, 96 CU), against the two things a Radeon
              can actually run today

       N        HipKittens      aotriton      triton FA-2
    4096          56.0 TF         21.2 TF        57.6 TF
   16384          59.7 TF         20.9 TF        58.6 TF
   49920          56.9 TF         20.1 TF        57.5 TF     <- 480p/5s of H3
```

The target is [MiniMax H3](#1-the-shape-that-drove-every-decision), a video+audio
diffusion transformer, where softmax attention is more than 85% of the
transformer's runtime at the lengths a real clip produces.

Almost nothing in this kernel is a transcription of the CUDA or CDNA flash
attention everyone knows. The register layouts of gfx11's WMMA are different
enough that the *shape of the algorithm* comes out different — S is computed
transposed, O is accumulated transposed, and V is transposed on the way into
LDS — and each of those falls out of a table in the library rather than out of a
choice. [§2](#2-the-derivation) is that derivation.

The most expensive thing found along the way was not in this kernel at all:
**`s_waitcnt` carries no register dependence**, so a `lds_wait<0>()` does not
stop the compiler from scheduling the consuming `v_wmma` above it. That is
[§4](#4-the-bug-that-was-not-in-this-kernel), it was silently corrupting the
GEMM in this repository too, and the fix is now library-wide.

---

## Contents

1. [The shape that drove every decision](#1-the-shape-that-drove-every-decision)
2. [The derivation](#2-the-derivation)
3. [Getting V into LDS, and getting it back out](#3-getting-v-into-lds-and-getting-it-back-out)
4. [The bug that was not in this kernel](#4-the-bug-that-was-not-in-this-kernel)
5. [Tiling: a comb, not a curve](#5-tiling-a-comb-not-a-curve)
6. [Where the time goes](#6-where-the-time-goes)
7. [Coverage: causal, GQA, head_dim 64](#7-coverage-causal-gqa-head_dim-64)
8. [The torch operator and the drop-in](#8-the-torch-operator-and-the-drop-in)
9. [Results](#9-results)
10. [What this is not](#10-what-this-is-not)
11. [Reproducing](#11-reproducing)
12. [Notes for anyone continuing this](#12-notes-for-anyone-continuing-this)

---

## 1. The shape that drove every decision

MiniMax H3 is not an LLM, which is the whole reason comfyUI and diffusers are on
the list of things to beat. From `transformer/config.json`:

| | |
|---|---|
| `num_attention_heads` | 56 |
| `attention_head_dim` | 128 |
| `hidden_size` | 5376 |
| layers | 50 (+2 refiner) |
| `num_key_value_heads` | **absent** — so MHA, not GQA |
| attention | **bidirectional** — no causal mask |
| direction | **forward only** — this is diffusion inference |

3D MM-RoPE is applied outside the attention call, so the kernel never sees it.

Sequence length is the entire story. It is the latent token count of the clip,
and it comes out of the pipeline geometry: the VAE is 8× spatial and 4×
temporal, then the transformer's `patch_size [1,2,2]` halves h and w again. An
`f`-frame `H×W` clip is `t·h·w` tokens with `t=(f-1)/4+1`, `h=H/16`, `w=W/16`.
So 832×480 at 125 frames (5 s at 25 fps) is 32·30·52 = **49 920 tokens**, and
720p is 115 200. At those lengths the N² term dominates everything else in the
transformer.

That fixed phase 1 of this work: **non-causal, D=128, MHA, bf16, forward**, and
nothing else until that was faster than every alternative. Causal, GQA and D=64
came afterwards, in [§7](#7-coverage-causal-gqa-head_dim-64), and are checked
not to have cost the H3 path anything.

### The two baselines, and why those two

There is no network on the benchmark pod, so diffusers, comfyUI, vLLM and SGLang
cannot be installed. The comparison is therefore **operator-level**, against the
two kernels those frameworks would have called:

* **aotriton**, via `F.scaled_dot_product_attention`. This is literally the code
  path diffusers and comfyUI take on Radeon — both of them just call SDPA.
* **a Triton FA-2 forward written here** (`baselines/triton_fa.py`), autotuned
  over `BLOCK_M`/`BLOCK_N`/`num_warps`/`num_stages`, standing in for the Triton
  attention vLLM and SGLang ship for ROCm.

The second one turned out to matter a great deal. aotriton is easy to beat by
2.7×; the hand-written Triton is not, and it is what the interesting part of the
tuning work was actually against.

---

## 2. The derivation

The conventional flash-attention inner loop is

```
S[q,kv]  = Q · Kᵀ
P        = softmax_online(S)          reductions along kv, i.e. along a row of S
O[q,d]  += P · V
```

On gfx11 that form is wrong in three separate places, and all three come out of
`include/rdna3/types/register/rt_base.cuh`:

```
rt_base<bf16, row>   A operand     e -> (row l%16,      col e),        e in 0..15
rt_base<bf16, col>   B operand     e -> (row e,         col l%16),     e in 0..15
rt_base<float, col>  accumulator   e -> (row 2e + l/16, col l%16),     e in 0..7
rt_base<float, row>  transposed    e -> (row l%16,      col 2e + l/16), e in 0..7
```

### 2.1 The softmax reduction wants S transposed

A WMMA accumulator on gfx11 is `col` layout, and for a `col`-layout tile the
**element axis is the row axis**. The reduction table at the top of
`ops/warp/register/tile/reductions.cuh` then says:

```
           row layout          col layout
 row_red   element (ortho)     lane    (align)
 col_red   lane    (align)     element (ortho)
```

A reduction along the element axis is eight register-local steps plus one
`permlanex16`. A reduction along the lane axis is a four-step 16-lane butterfly,
every step of which is a cross-lane instruction.

In the conventional form, the online softmax reduces `S[q,kv]` over `kv` — the
column axis of a `col`-layout accumulator — so every `max` and every `sum` is a
butterfly, once per KV block, forever.

Store it transposed instead, as **`Sᵀ[kv,q]`**, and `kv` becomes the element
axis. The max and the sum are now register-local, and the running statistics
(`m`, `l`, and the rescale factor `α`) are indexed by `q`, which is the lane
axis — **one VGPR each** at Q_BLOCK=16.

The CDNA4 kernel in this repository (`kernels/cdna4/attn/gqa/kernel.cpp`) uses
the same transposed form, arrived at from a different register file. That is
reassurance, not a source: nothing else about it survives the port.

### 2.2 The accumulator wants to be transposed too

`α` and the final `1/l` are per-query. On `Oᵀ[d,q]`, "per-query" is per-column,
which takes exactly the row-vector the softmax just produced, in exactly the
layout it produced it in. Keep `O` as `[q,d]` and the same statistic is needed in
the *other* vector layout, which means a conversion between layouts on every
single KV block.

Transposing the accumulator costs one transpose per Q block in the epilogue.
Not transposing it costs one layout conversion per KV block, of which there are
`N/KV_BLOCK` — 512 of them at N=16384. It is not close.

### 2.3 Which `mma_*` primitive, and why that is free

`ops/warp/memory/tile/shared_to_register.cuh` has the constraint that decides
everything else: **only a `row`-layout bf16 operand tile takes the vectorized
`ds_read_b128` path out of LDS.** A `col`-layout one degrades to 16 scalar
`ds_read_u16` per 16×16 base tile — eight times the instructions, paid by every
warp on every KV block.

So every operand that comes out of LDS must be `row` layout. That is a hard
constraint, and it picks the primitive for each matmul:

```
Sᵀ[kv,q]  = K · Qᵀ      mma_ABt_base(s_t, k, q, s_t)     k  row (LDS) ✓   q    row ✓
Oᵀ[d,q]  += Vᵀ · Pᵀ     mma_AB_base (o_t, vt, p_t, o_t)  vt row (LDS) ✓   p_t  col
```

`p_t` is `col`, but `p_t` never comes out of LDS — it is the softmax's own
accumulator, converted in registers. Both LDS operands are `row`. 

Choosing between `mma_AB` / `mma_ABt` / `mma_AtBt` costs nothing, incidentally:
a row/col operand pair is bit-identical on gfx11 (see the note above
`mma_ABt_base`), and all three lower to the same `v_wmma_f32_16x16x16_bf16`. The
choice is purely about which layout *tag* the tiles get to carry — which is to
say, purely about which of them is allowed to be read with `ds_read_b128`.

### 2.4 Two small things that follow

**`exp2`, not `exp`.** One hardware instruction instead of a sequence. `log2(e)`
is folded into the softmax scale. The scale is applied to the *fp32 scores*, not
to Q — Q is bf16, and multiplying it by a non-power-of-two would round the
inputs a second time. The cost is `KV_BLOCK/16 · 4` packed multiplies per KV
block against roughly 64 WMMAs.

**The tails are not symmetric.** The Q tail is handled by backing the last block
up to `N - Q_TILE` and letting the overlapping rows be computed twice: each
query row's output depends on nothing but that row, so the two computations are
identical and the two stores are idempotent. The KV tail is backed up the same
way but **must** then be masked, because a score counted twice would be counted
twice in the softmax denominator. That masking is free — a row of `Sᵀ` is a kv
index, which is the element axis, so it is register-local with no cross-lane
traffic. This is the same asymmetry as in the GEMM's M and K remainders, for the
same reason.

---

## 3. Getting V into LDS, and getting it back out

### 3.1 The transpose has to happen somewhere

`Vᵀ[d,kv]` is what the PV matmul reads, and V arrives from global as `[n,d]`.
There is no way around a transpose: the PV matmul's B operand must have `kv` on
the element axis, and a vectorized LDS read always puts the tile's row index on
the lane axis.

The question is only *where*. Leaving V row-major in LDS and reading `col`-layout
operand tiles out of it means 16 `ds_read_u16` per base tile, paid by every warp
on every KV block. Transposing it once, during staging, means
`transpose_base_data`'s 20 lane exchanges per base tile paid once per KV block
per **workgroup** — divided by `NUM_WARPS`, and then divided again because the
warps split the work.

So staging does it: a warp reads a `VT_D_CHUNK`-wide slice of 16 V rows (a
perfectly coalesced `[16, VT_D_CHUNK]` global read), transposes it in registers
with `transpose_sep`, and writes the result as a `[VT_D_CHUNK, 16]` row-major
block of `vt_smem` — which is a vectorized `ds_write_b128`, because the
transposed tile is still `row` layout.

The loop nest is written with the **kv tile as the outer, compile-time loop and
the head-dim chunk as the inner, runtime one**, which is backwards from how it
wants to be written. The reason is in §3.2: the kv tile index is the
destination's *column*, so it picks the granule; the head-dim chunk is the
destination's *row*, which is a runtime addend all eight writes share. Written
the natural way each write needs its own address, those addresses are invariant
across the KV loop, and LLVM hoists eight of them out and spills three.

### 3.2 128 LDS addresses collapse to nine

The library's `load()` forms one full VGPR address per `ds_read_b128`. That is
the right default for a tile or two. Here the QK and PV loops between them issue
**128 reads whose addresses are every one of them loop-invariant**, so LLVM
hoists 128 addresses out of the KV loop and then spills most of them. Measured,
on the version that used `load()`:

```
91 scratch_store, all in the loop preheader, none in the body
87 scratch_load,  all in the body
every spilled value is  v_add_nc_u32 v0, s6|s7, vN   -- an LDS address
```

372 bytes/lane of scratch. And in a kernel that hand-manages `s_waitcnt`, a
spill is not a slowdown, it is a **correctness bug** ([§4](#4-the-bug-that-was-not-in-this-kernel)).

Those 128 addresses collapse to nine, because `st::idx` is affine in almost all
of its arguments. Write `SB` for the tile's `swizzle_bytes` and `SUB = SB/2` for
its `subtile_cols`. Then for any 2-byte tile with `SB >= 64`:

```
idx(ptr, {R + l, c}) = idx(ptr, {l, 8*((c % SUB) / 8)})   <- lane-dependent, SB/16 of them
                     + (c / SUB) * rows * SB + SB * R     <- entirely compile-time
```

for `l` in 0..15 and `R` a multiple of 16. Two facts make it true:

* the swizzle key is bits 7..9 of the tile-relative offset, and both dropped
  terms are multiples of 1024, so neither perturbs the key — which is what lets
  them leave the XOR and become addends;
* the swizzle is a permutation of 16-byte granules *within* a row, so the
  lane-dependent part takes `SB/16` values, one per granule, however wide the
  tile is and however many fragments are read out of it.

`SB >= 64` is a real precondition and not an artifact: at `SB = 32` the row
stride is small enough that 16 rows reach bit 9, the key changes with `R`, and
the row term stops being constant. A 32-byte swizzle means a 16-column tile,
which nothing here stages.

The result is `SB/16` granule addresses computed once for the whole kernel, and
the entire `(R, c)` dependence moved into `ds_read_b128`'s 16-bit offset field,
where it costs no register at all. **Nine VGPRs of LDS addressing for the whole
kernel**, down from 128 addresses and 372 bytes/lane of scratch. Checked
exhaustively against `st::idx` over every `(rows, cols, R, l, c)` with `SB >= 64`
up to 256×256.

---

## 4. The bug that was not in this kernel

This is the most important thing in this document, and it is a library bug, not
a kernel bug. It was found here and it was also silently present in
`kernels/rdna3/gemm/bf16fp32/gemm.cpp`, which was shipping and passing its tests
by scheduling luck.

### The symptom

Some builds of the attention kernel produced wrong answers. Not slightly wrong —
`rel 1.0`, whole tiles of garbage. Whether a given build was wrong depended on
tiling parameters that should not have been able to change the answer, and
`ScratchSize` was 0 in every one of them, so it was not a spill.

### The mechanism

The kernel pipelines LDS reads by hand:

```cpp
lds_read_frag<R, C>(dst, gr);   // ds_read_b128, does not wait
lds_wait<0>();                  // s_waitcnt lgkmcnt(0)
mma_ABt_base(acc, dst, q, acc); // v_wmma
```

Three separate facts, each harmless alone:

1. `ds_read_b128_off` (and the library's `load_shared_vec4_async`) writes its
   destination through an inline-asm **output operand**. As far as LLVM is
   concerned, the value is ready the instant the asm statement ends.
2. `lds_wait<N>()` is a separate `asm volatile` that **shares no operands** with
   it. It has a memory clobber, so it cannot be reordered against other memory
   operations — but the destination register is not a memory operation.
3. `v_wmma` is a pure builtin with no memory effect at all.

So there is nothing in the IR connecting the WMMA to the wait. The scheduler is
free to hoist the WMMA above the `s_waitcnt`, and it does — nondeterministically,
as register pressure shifts, with `ScratchSize` at 0 throughout. The WMMA then
reads whatever was in those registers before the `ds_read` landed.

This is exactly the failure mode that is hardest to find: a correctness bug whose
trigger is a *performance* knob, in a kernel where the obvious suspect (spilling)
has been ruled out by the resource report.

### Finding it

The debug ladder in `fwd/attn.cpp` (`HK_DUMP`) writes one intermediate per level
into the output tensor. Levels 1–9 all go through `store()`, and every one of
them looked correct — because `store()` reads the same registers and the same
hoist applies to it, so the dump was consistent with the corruption rather than
revealing it. Levels **10 and 11** are a different shape on purpose: warp 0
writes its 16 raw accumulator VGPRs to `o[lane, slot]` with no transpose and no
store path in between. That is what made the reordering visible.

### The fix

`lds_bind`, in `include/rdna3/ops/warp/memory/util/util.cuh`:

```cpp
template<typename T> __device__ inline void lds_bind(T &x) {
    // for each 128-bit group of x:
    //     asm volatile("" : "+v"(v));
}
```

An empty `asm volatile` with the fragment as a read-write register operand. It
emits no instruction, so it is free; it re-establishes the data dependence that
`s_waitcnt` does not carry, so the WMMA cannot be scheduled above it. The offset
is a template parameter rather than a loop index so that every access is a
constant offset in the IR that SROA can see — an object reached through a runtime
offset would be forced to scratch, which in a hand-waited kernel is worse than
the bug being fixed.

The paired form is `lds_wait_for<N>(tiles...)`, which is what callers should use:

```cpp
lds_wait_for<0>(k_chunk.tiles[0][0], k_chunk.tiles[1][0]);
```

`shared_to_register.cuh`'s asynchronous `load<false>` now documents that the
caller owes the tile a `lds_wait_for`, not a bare `lds_wait`.

### The cost

None. The GEMM was A/B'd three runs each way at 4096³ and 8192×8192×4096: 72 and
77 TFLOPs, with the binds and without. All 24 GEMM shapes pass either way — the
unbound version was simply getting lucky, and it was one register-pressure change
away from not.

---

## 5. Tiling: a comb, not a curve

The tiling was picked by measurement, not by a register-pressure estimate.
`fwd/sweep.sh` builds one configuration at a time, greps
`-Rpass-analysis=kernel-resource-usage` for VGPRs / spill / scratch / occupancy,
and then runs `fwd/quickbench.py` — which **checks correctness before it times**,
because a spilling configuration is wrong rather than slow, and a config that
prints a good number and a wrong answer is the exact failure this whole harness
exists to catch.

```
config                                       | registers                  | TF @4096 / @16384
Q_BLOCK=16 KV_BLOCK=32  QK=2 PV=2            | vgpr=221 spill=0  scr=0  occ=6 | 54.3  56.8
Q_BLOCK=16 KV_BLOCK=64  QK=2 PV=2            | vgpr=256 spill=0  scr=0  occ=5 | 52.8  54.8
Q_BLOCK=16 KV_BLOCK=64  QK=4 PV=2            | vgpr=256 spill=0  scr=0  occ=5 | 51.5  53.5
Q_BLOCK=16 KV_BLOCK=128 QK=2 PV=2            | vgpr=256 spill=37 scr=152 occ=5 | 46.2 46.1
Q_BLOCK=16 KV_BLOCK=32  QK=1 PV=1            | vgpr=213 spill=0  scr=0  occ=7 | 54.3  56.7
Q_BLOCK=16 KV_BLOCK=32  QK=2 PV=4            | vgpr=217 spill=0  scr=0  occ=6 | 54.7  57.3
Q_BLOCK=16 KV_BLOCK=64  QK=4 PV=4            | vgpr=256 spill=0  scr=0  occ=5 | 52.0  54.0
Q_BLOCK=32 KV_BLOCK=32  QK=2 PV=2            | vgpr=221 spill=0  scr=0  occ=6 | WRONG (see below)
```

### The knob that mattered was the one that was not in the first sweep

`NUM_WARPS` is also the Q tile — `Q_TILE = Q_BLOCK · NUM_WARPS` — and every warp
in the workgroup reads the same staged K and Vᵀ. So it is the knob that
amortizes staging, which [§6](#6-where-the-time-goes) puts at 28% of the runtime:
global K/V traffic is `N / Q_TILE` complete passes over K and V, and a wider
workgroup divides that down.

The measured shape of it is not "wider is better". It is a comb:

```
  NUM_WARPS     8      10      12      14      16      24
  TFLOPs     56.8    51.2    59.7    43.5    49.5    56.2      (N=16384)
```

At 217–221 VGPRs the occupancy limit is 6 waves/SIMD, and the scheduling unit
here is the WGP — 4 SIMDs, and what
`hipOccupancyMaxActiveBlocksPerMultiprocessor` calls a multiprocessor — so the
budget is **24 waves per WGP**. What matters is whether `NUM_WARPS` divides 24.
8 and 12 do (3 and 2 workgroups per WGP); 10, 14 and 16 strand 4, 10 and 8 wave
slots, and the loss swamps the staging they save. Among the divisors, 12 wins
because it stages half as often as 8. 24 divides it too, but leaves one
workgroup per WGP with nothing to cover its barriers.

Shipped: `Q_BLOCK=16, KV_BLOCK=32, NUM_WARPS=12, VT_D_CHUNK=16, QK_TILES=2,
PV_TILES=4` → 217 VGPRs, 0 spill, 0 scratch, occupancy 6, **59.7 TF at N=16384**.

The cost is coverage at the bottom: `Q_TILE` is 192 rather than 128, so the
kernel needs N ≥ 192 and the drop-in falls back below that. For H3, where N is
tens of thousands, that is free.

### Two knobs that are not usable

* **`KV_BLOCK=128` spills** — 37 VGPRs, 152 bytes/lane of scratch. Slow, and
  after §4, unsafe.
* **`Q_BLOCK=32` is broken**, and this is a real defect rather than a tuning
  result. The `mma_*` calls hardcode `.tiles[r][0]` for `s_t` and `o_t`, so only
  the first 16 query columns are ever computed; every variant returns
  `rel 1.00e+00`. Making it work also needs Q moved to LDS, because `q` and
  `o_t` both double and 128+128 VGPRs does not fit. It is the largest unclaimed
  lever in the kernel — see [§12](#12-notes-for-anyone-continuing-this).

---

## 6. Where the time goes

`fwd/attn.cpp` has `ABLATE_*` switches that each delete one stage. Every one of
them makes the answer wrong on purpose; they exist to time one thing at a time.
`ABLATE_LDS_READ` is the interesting one — it keeps **both** matmuls and drops
only the LDS reads that feed them (the operand tiles are zeroed once and reused),
which is the one measurement that separates "LDS-bound" from "WMMA-bound". No
combination of the per-stage switches can do that, because each of those removes
a stage's reads and its math together.

At N=16384, as a share of runtime:

```
  staging (global -> LDS, incl. the Vᵀ transpose)   27.7%
  QK stage                                          32.1%
  PV stage                                          35.0%
  online softmax                                     8.4%
  ----
  all LDS reads, both stages                        18.3%
  => WMMA + fp32->bf16 conversion                  ~49%
```

The read side is already well hidden: the WMMAs are roughly half the runtime and
the machine's WMMA ceiling is ~100 TF, so at 59.7 TF the matmul stages are
running near peak. The largest remaining lever is **staging**, which is still
fully serialized with the math (`G::load` → `lds_wait<0>` → barrier). See §12.

---

## 7. Coverage: causal, GQA, head_dim 64

None of these are what H3 needs; all three are what makes the kernel usable by an
LLM serving stack as well. Each is a template parameter, and the non-causal
D=128 path was re-measured afterwards at the same **217 VGPRs, occupancy 6,
ScratchSize 0** — byte for byte the budget it had before.

### Causal

`CAUSAL` is a template parameter, not a branch on a flag, precisely so the
non-causal path keeps its codegen.

Both axes of `Sᵀ` turn out to be free here, and that is the second dividend of
storing it transposed. A *row* of `Sᵀ` is a kv index, which for a `col`-layout
fp32 accumulator is the element axis — unrolled, so the kv of every register is a
compile-time-shaped expression in the lane id. A *column* is a query index, which
is the lane axis, so `q` is `lane & 15` and does not vary within a register. The
causal predicate is therefore **one `v_cmp` per element, no cross-lane traffic,
no materialized mask tile**. The `[q, kv]` orientation would have needed
`make_causal`'s shuffles.

Two levels of skipping:

* **Workgroup level.** The last query the workgroup owns is
  `q_tile_start + Q_TILE - 1`, so every KV block past it is never staged at all.
  This bound has to be uniform across the workgroup, since staging is a group
  operation with barriers in it.
* **Wave level.** A wave whose own Q block is entirely below a KV block still
  stages and still hits both barriers, but skips the QK/softmax/PV work. On the
  diagonal block that is most of the workgroup. Skipping is exactly a no-op and
  not an approximation: an all `-inf` `Sᵀ` gives `m_new == m_old`, `α == 1` and a
  zero `Pᵀ`, so `l_run` and `o_t` would come out unchanged.

Measured, B=1 H=56 D=128:

```
     N     non-causal          causal          wall-clock
  4096    8.71 ms  55.2 TF   5.17 ms  46.5 TF     1.68x
  8192   32.69 ms  58.9 TF  18.59 ms  51.8 TF     1.76x
 16384  129.04 ms  59.6 TF  71.47 ms  53.8 TF     1.81x
```

(Causal TFLOPs use the usual halved-FLOP convention. 1.81× against an ideal 2× is
three things: the diagonal blocks, which are computed whole; the load imbalance
of a static grid where workgroup *i* does *i*+1 KV blocks; and the wave-level
skip above, which saves the *math* but not the *staging* — a skipping wave has
already paid for the K and Vᵀ it will not read, and `Q_TILE` is six `KV_BLOCK`s
wide, so there can be six such blocks per workgroup. That third one is the
largest and it is why causal comes out a few percent behind the Triton baseline
where non-causal is ahead; [§9](#causal--where-this-loses) measures it.)

### GQA

An addressing change and nothing else. The grid is over *query* heads; several of
them read the same K/V head:

```cpp
const int head_kv = head / (g.q.depth() / g.k.depth());
```

One scalar divide per workgroup, hoisted out of the KV loop, against thousands of
trips through it. There is no MHA/GQA specialization — when the head counts match
it is a divide by 1. Tested at ratios 1, 4 and 8, and combined with causal.

### head_dim 64

The same template at a different width. `HEAD_DIM` was already a compile-time
parameter of every tile type; the dispatcher now instantiates 64 as well as 128
and picks on `q.size(3)`.

D=64 is *cheaper per byte* than D=128, which is worth stating because it is the
opposite of the usual intuition: `q` and `o_t` are half the registers, so the
kernel drops to **145 VGPRs and occupancy 9** (167 / 9 for the causal variant)
against 217 / 6. The D=128 shape is the tight one.

---

## 8. The torch operator and the drop-in

Two layers, deliberately.

**`hk_attn_ext.hip`** registers three ops:

```
hk_attn::supported(int n, int d, bool causal, int h_q, int h_kv) -> bool
hk_attn::why_unsupported(int n, int d, bool causal, int h_q, int h_kv) -> str
hk_attn::fwd(Tensor q, Tensor k, Tensor v, float scale, bool causal) -> Tensor
```

The shape rule lives in `fwd/attn.cpp`, next to the tiling it comes from; the
extension adds only the head-count relation and turns a "no" into a sentence.
Callers are expected to *ask* rather than to reimplement the rule.

Contiguity is **checked, not fixed**, at this layer: the globals are plain
`(B, H, N, D)` row-major with no stride support, so a non-contiguous input would
be read wrong. The op refuses it, so a direct `torch.ops` caller cannot get
silent garbage.

**`hk_attn/__init__.py`** is the drop-in. Its signature is
`F.scaled_dot_product_attention`'s, argument for argument, so a framework can be
redirected in one line:

```python
import hk_attn
import torch.nn.functional as F
F.scaled_dot_product_attention = hk_attn.scaled_dot_product_attention
```

Anything the kernel cannot take — fp16, an `attn_mask`, dropout, a grad-requiring
input, cross attention with `n_kv != n_q`, `head_dim` other than 64 or 128, N
below the Q tile — is forwarded to torch unchanged. This layer *does* call
`.contiguous()` and eat the copy, because attention modules project to
`(B, N, H, D)` and transpose, and the result is never contiguous.

There is also `hk_attn.attention(...)`, which is the same kernel with **no
fallback**: it raises rather than silently running torch's. That is what
benchmarks and tests call, because a fallback timed in our column would be
aotriton's number printed under our name.

### How the gate is written

`torch_ext/test_torch_ext.py` asserts each fallback branch **twice**: that
`supported()` says no, *and* that the result still matches the fp32 reference. A
fallback that silently returned a wrong answer and a fallback that never happened
would both look like "fast and correct" in a naive test.

The kwargs in the shape table are shared between `make_qkv`, the reference and
the drop-in, so a case that is routed wrong is also checked wrong and cannot pass
by accident.

---

## 9. Results

W7900D (gfx1100, 96 CU / 48 WGPs), torch 2.10.0+rocm7.2.4, bf16, B=1, H=56,
D=128.
**All three backends in one process, in one run, interleaved** — clock and power
state drift between runs on this node, so a number from a different run is not
comparable to one from this one. The HipKittens column goes through
`hk_attn.attention`, the entry point that *raises* rather than falling back, so
a mis-dispatch cannot show up here as our number.

### Non-causal — the H3 path

```
 shape          B   H       N     hk ms   hk TF  aotri ms  aotri TF  triton ms  triton TF
 N=4096         1  56    4096     8.458    56.9    22.686      21.2      8.363       57.5
 N=8192         1  56    8192    32.535    59.1    92.376      20.8     32.948       58.4
 N=16384        1  56   16384   128.542    59.9   368.640      20.9    131.282       58.6
 N=32768        1  56   32768   518.621    59.4  1552.183      19.8    532.580       57.8
 480p/5s        1  56   49920  1207.556    59.2  3512.293      20.3   1243.151       57.5
 N=65536        1  56   65536  2086.834    59.0  6073.892      20.3   2150.806       57.3
 N=16384 cfg    2  56   16384   256.923    59.9   735.965      20.9    262.041       58.7
```

| | vs aotriton (what diffusers/comfyUI run) | vs the Triton FA-2 (the vLLM/SGLang stand-in) |
|---|---|---|
| N=4096 | 2.68× | **0.99×** |
| N=8192 | 2.84× | 1.01× |
| N=16384 | 2.87× | 1.02× |
| N=32768 | 2.99× | 1.03× |
| 480p/5s (49920) | 2.91× | 1.03× |
| N=65536 | 2.91× | 1.03× |
| N=16384 CFG | 2.86× | 1.02× |

Two things worth saying plainly about that table.

**aotriton does not scale, and this does.** aotriton is flat at 19.8–21.2 TF
across a 16× range of sequence length. Ours holds 59–60 TF from N=8192 to
N=65536, which on a machine whose measured WMMA ceiling is ~100 TF is about 59%
of peak — for an operation that is not a GEMM and has a softmax in the middle of
it. For reference, hipBLASLt's 4096³ bf16 GEMM on this chip is 67.6 TF and our
own GEMM is 77.2.

**The Triton baseline is genuinely good, and at N=4096 it is still 1% ahead.**
That is reported rather than rounded away. The gap closes and reverses by
N=8192 and stays reversed through N=65536, which is the regime H3 actually runs
in — but a 4096-token workload is a tie, not a win, and the honest summary is
"2.7–3.0× the backend a Radeon actually uses today, and level-to-slightly-ahead
of the best thing you could write in Triton."

### Causal — where this loses

Same harness, `--causal`, same discipline — one process, one run, interleaved.
TFLOPs count half the non-causal work for all three backends alike, which
slightly flatters everyone (the diagonal blocks are half empty and are counted
as empty), but flatters them identically.

```
 shape          B   H       N     hk ms   hk TF  aotri ms  aotri TF  triton ms  triton TF
 N=4096         1  56    4096     5.127    46.9    18.882      12.7      4.652       51.7
 N=8192         1  56    8192    18.381    52.3    69.886      13.8     17.781       54.1
 N=16384        1  56   16384    71.895    53.5   272.449      14.1     69.107       55.7
 N=32768        1  56   32768   281.281    54.7  1099.156      14.0    274.216       56.1
 480p/5s        1  56   49920   649.637    55.0  2547.582      14.0    641.538       55.7
 N=65536        1  56   65536  1136.448    54.2  4376.786      14.1   1112.219       55.4
 N=16384 cfg    2  56   16384   140.964    54.6   548.910      14.0    136.977       56.2
```

| | vs aotriton | vs the Triton FA-2 |
|---|---|---|
| N=4096 | 3.68× | 0.91× |
| N=8192 | 3.80× | 0.97× |
| N=16384 | 3.79× | 0.96× |
| N=32768 | 3.91× | 0.97× |
| 480p/5s (49920) | 3.92× | **0.99×** |
| N=65536 | 3.85× | 0.98× |
| N=16384 CFG | 3.89× | 0.97× |

**Against aotriton the causal margin is larger, and that is aotriton's doing,
not ours.** It falls from ~20.5 TF to ~14 TF when the mask is switched on — it
gets *less* efficient per surviving FLOP, so halving its work does not halve its
time.

**Against the Triton baseline, causal is a loss: 1–9%, consistently.** Non-causal
we are 1–3% ahead of it; causal we are 1–3% behind, and 9% behind at N=4096. The
kernel keeps 89% of its non-causal throughput (59.9 → 53.5 TF at N=16384) where
Triton keeps 95%, and the reason is the staging this shape does for blocks it
then throws away. `Q_TILE` is `NUM_WARPS × Q_BLOCK` = 192 queries, six times
`KV_BLOCK`; the KV loop bound has to be uniform across the workgroup, so it is
taken from the *last* query in the tile, and the waves holding earlier queries
stage and barrier through up to six KV blocks whose scores they will skip
entirely. With staging at 27.7% of the runtime (§6), that is roughly the size of
the gap. It is fixable — a smaller `Q_TILE` under causal, or a wave-major work
assignment that puts the whole workgroup on the same diagonal — and it has not
been done, because the target is non-causal.

So the one-line causal summary is: **3.7–3.9× the backend a Radeon actually
runs, and a few percent behind a good Triton FA-2.**

### Coverage

24 shapes pass elementwise against the chunked fp32 reference (`rtol` 5e-2, with
a magnitude guard so that an all-zero output cannot pass): H3's sizes, sequence
lengths that divide neither the Q tile nor the KV block (4097, 5000, 12345, 193),
B=2, exactly-one-Q-tile, causal at five lengths, GQA at ratios 1/4/8, D=64, and
the combinations. `ScratchSize` is 0 and `VGPRs Spill` is 0 in all four
instantiations.

---

## 10. What this is not

* **Not verified end to end.** The pod has no network, so diffusers, comfyUI,
  vLLM and SGLang cannot be installed, and H3's weights (~66 GB in bf16) are not
  there either. Every number here is operator-level. The claim is "faster than
  the attention kernel those frameworks would have called on this hardware",
  which was measured; it is *not* "an H3 clip renders N% faster", which was not.
* **No backward.** Forward only. The target is diffusion and LLM inference.
* **No fp8, no paged KV, no sliding window, no ALiBi, no soft-capping.**
* **No cross attention.** The KV loop is bounded by q's sequence length, so
  `n_kv != n_q` is rejected rather than read wrong.
* **bf16 in, bf16 out.** fp16 falls back to torch; it would be a second
  instantiation and nothing else, but it is not built.
* **`Q_BLOCK=32` is broken**, not merely untuned — see §5.
* **One GPU.** Nothing here is distributed; that is `../distributed/`.
* **gfx1100 only.** Everything in §2 is a consequence of gfx11's WMMA register
  layouts. On CDNA the derivation is different and the answer is different.

---

## 11. Reproducing

All of it needs a gfx1100 with ROCm 7.x and torch built against it. Measured on
a W7900D (96 CU, 48 GiB) with torch 2.10.0+rocm7.2.4 and triton 3.6.0+rocm7.2.4.

```bash
# correctness, all 24 shapes: H3 sizes, ragged N, causal, GQA, D=64
cd fwd && make && python3 test.py

# registers: VGPRs / spill / ScratchSize / occupancy, per instantiation
cd fwd && make 2>&1 | grep -E 'VGPRs|Scratch|Occupancy'
#   ScratchSize must be 0.  See §4 -- a spill here is wrong, not slow.

# the tiling sweep that picked the shipped config
cd fwd && ./sweep.sh

# per-stage attribution
cd fwd && make EXTRA_HIPFLAGS=-DABLATE_LDS_READ=1 && python3 quickbench.py --no-check

# the torch op and the drop-in: every shape, every fallback branch
cd torch_ext && make && python3 test_torch_ext.py

# the three-way table, one process, one run
cd torch_ext && make && cd ../baselines && python3 bench_baselines.py
cd baselines && python3 bench_baselines.py --causal        # causal
cd baselines && python3 bench_baselines.py --check         # + fp32 verification
```

The node used here is shared. Pin an idle GPU with `HIP_VISIBLE_DEVICES` and wrap
every launch in `timeout --signal=KILL`.

---

## 12. Notes for anyone continuing this

In rough order of how much is left on the table.

1. **Staging is 27.7% of the runtime and is fully serialized with the math.**
   `G::load` → `lds_wait<0>` → barrier, then all the WMMAs, then the next block.
   Double-buffering K and Vᵀ is 64 KB on a part with exactly 64 KB per workgroup,
   which drops occupancy to one workgroup per WGP — the prefetch that buys is
   worth less than the waves it costs, which is why it is not there. A **register
   prefetch** of the next KV block (the GEMM's `GPREFETCH` approach; gfx11 has no
   global→LDS DMA, so every byte goes through a VGPR anyway) is the version that
   might work, and it has not been tried.

2. **Fix `Q_BLOCK=32`.** It halves LDS traffic per FLOP, which is the only knob
   that does. It needs the `.tiles[r][0]` hardcoding removed *and* Q moved to
   LDS, because `q` and `o_t` both double.

3. **Causal: stop staging blocks the waves are going to skip.** This is the one
   place the kernel is measurably behind the Triton baseline (1–9%, §9), and it
   is one cause: the KV loop bound is uniform across the workgroup and therefore
   taken from the *last* query in a 192-query tile, so waves holding earlier
   queries stage up to six KV blocks they then skip. A smaller `Q_TILE` when
   `CAUSAL` (fewer waves per workgroup, so a tighter uniform bound) is the
   cheapest fix and needs no restructuring — it is a template parameter already.
   Separately, the static grid leaves workgroup *i* doing *i*+1 KV blocks;
   pairing block *i* with block *n-1-i* is the standard fix for that, and both
   together are worth most of the gap between the measured 1.81× and the ideal
   2×.

4. **fp16.** One more instantiation.

5. **Split-KV (flash-decoding).** Not needed for H3 — at N=16384/H=56 the grid is
   already thousands of workgroups on 48 WGPs — but it is what a batch-1 decode
   would need.

6. **Read §4 before touching any `lds_wait`.** Every `lds_wait<0>()` in this
   kernel is paired with `lds_bind` on the fragments it retires. That pairing is
   a correctness requirement, and removing it produces a kernel that is wrong
   sometimes, with `ScratchSize` still at 0.

---

## Files

```
common.py                   shapes, timing, the chunked fp32 reference, check()
fwd/attn.cpp                the kernel: config, micro_tk, launch, dispatch
fwd/test.py                 correctness gate, 24 shapes
fwd/quickbench.py           one check + two timings, small enough to sit in a sweep
fwd/sweep.sh                build+measure one tiling per line
fwd/dump.py                 numpy models of each HK_DUMP level
fwd/Makefile
baselines/triton_fa.py      autotuned Triton FA-2 forward, the vLLM/SGLang stand-in
baselines/bench_baselines.py  the three-way table
torch_ext/hk_attn_ext.hip   torch custom op
torch_ext/hk_attn/          the SDPA drop-in
torch_ext/test_torch_ext.py gate for the op and every fallback branch
torch_ext/Makefile
```
