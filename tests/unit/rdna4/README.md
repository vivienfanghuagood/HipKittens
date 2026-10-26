# HipKittens unit tests — RDNA4 (gfx1200 / gfx1201)

Unit tests for `include/rdna4`, for an RX 9060 XT / RX 9070 (XT) class card.

## Status: green on a gfx1201

Run on a Radeon AI PRO R9700 (gfx1201, 64 CU, ROCm 7.2.3), one leaf section
at a time (`TEST_DEFINES=-DTEST_WARP_REGISTER_TILE_MMA` and so on, so that a
fault in one section cannot hide the others): every section passes, 0
failures -- warp memory (tile and vector, global<->shared<->register), group
memory, register tile maps / reductions / conversions / mma (bf16, half, all
four fp8 pairings, int8), register vector maps / reductions / conversions, and
the shared tile and vector conversions.

What the first run found, before the fixes that make it green:

- **The accumulator layout was wrong.** The tree had carried gfx11's over:
  lane `l` register `r` holding row `2r + l/16`. On gfx12 it is row
  `r + 8*(l/16)` -- the upper half holds rows 8..15 -- measured with the
  `PROBE_GFX12_W32` variant of `tools/rdna-probes/wmma_layout.hip`. Every mma
  test failed (224 of 224). The operand layout (lane `l` holds
  `k = 8*(l/16) .. +7` of row `l%16`) was right. See the layout note in
  `include/rdna4/common/util.cuh`: every gfx12 fragment now has one shape.
- **A dynamically indexed register array went to LDS.** `rv<float, align>`'s
  loads indexed a lane's registers by a lane-dependent position; LLVM's
  promote-alloca moved the array into *static LDS* (64 KB of it at S=2), which
  with the harness's 32 KB of dynamic LDS is over the limit, and the dispatch
  failed with HSA_STATUS_ERROR_INVALID_ALLOCATION. The rdna3 tree's later
  `rv_align_held` / `rv_align_entry` loops (constant indices) fixed it.
- **One test copied what it meant to reference.** `warp/shared/vec/conversions`
  took `auto ref = subvec_inplace(...)`, a copy of the subvector in private
  memory; once the shared-vector accessors addressed LDS directly, they read
  that copy's private address as an LDS offset. Now `auto &ref`, in both trees.

The tree was also brought up to `include/rdna3` (register strips, int8 WMMA,
indexed and bounded loads, w4 dequantize, granule addressing, Philox), each
piece re-derived for the gfx12 layout rather than copied: a row-layout tile's
lane holds a contiguous run of its row, so every global<->register transfer is
one 8/16/32-byte run a lane, with no cross-half exchange.

One thing you will notice in a disassembly and should not chase: the test bodies
call `__builtin_amdgcn_s_waitcnt(0)`, which is the combined gfx11-style wait, so
this binary contains a few hundred `s_waitcnt vmcnt(0) expcnt(0) lgkmcnt(0)`.
That is not a leftover that will fault. GFX12 keeps the combined encoding
(`BF89`) alongside the split counters and the assembler accepts it for gfx1201
while rejecting genuinely gfx11-only forms like `s_waitcnt_vscnt` — checked both
ways. LLVM's own codegen never emits it, and it is redundant next to the
`s_wait_loadcnt` the compiler inserts anyway, but it is legal and harmless.
`include/rdna4` itself contains none of them.

## Requirements

- A gfx1200 or gfx1201 GPU (wave32, 64KB LDS per workgroup, 256 arch VGPRs)
- ROCm 6.4 or newer (this tree was built with 7.1)
- A C++20 compiler

## Building and running

```bash
make -j32
./unit_tests
```

`GPU_TARGET`, `COMP_LEVEL`, `TEST_INTENSITY` and `TEST_DEFINES` are set in the
Makefile here and default to `RDNA4`, `profile`, `2` and `-DTEST_ALL`. Raise
`TEST_INTENSITY` to sweep larger tiles; it costs compile time roughly
quadratically. To narrow the run, replace `-DTEST_ALL` with a subsection such as
`-DTEST_WARP_MEMORY` or a single test like `-DTEST_WARP_REGISTER_TILE_MMA`.

`./unit_tests printout` dumps failing tensors into `outputs/`, which you have to
create first.

## How this tree differs from `tests/unit/rdna3`

The test *sources* are identical — `diff -r` the two trees and only this file
and the Makefile come back. The arch difference lives entirely in
`include/rdna4`, and the Makefile's `GPU_TARGET` is the only functional edit.
That is deliberate: the tests are written against the tile API, so if the RDNA4
fragment inference is right the same tests pass unchanged, and if it is wrong,
the diff between the two trees is not where you should be looking.

The one thing genuinely specific to gfx12 — the fp8 section of
`warp/register/tile/mma.cu` — is behind `#ifdef KITTENS_RDNA4` and sits in both
trees rather than only this one, which is what keeps that property true. Keep it
that way if you add more.

The library differences that these tests actually exercise:

**Operands cost half the registers.** A gfx11 WMMA operand lane held the whole
K=16 vector and the two wave halves were mirrors of each other: 16 elements, 8
VGPRs, 2× redundant. gfx12 splits K between the halves: 8 elements, 4 VGPRs, no
redundancy. Accumulators are unchanged. This is the point of RDNA4 for this
library and the reason a gfx12 GEMM should carry a larger register block than
the gfx1100 one in `kernels/rdna3/`.

**Layout conversion got more expensive, not less.** On gfx11, converting an
accumulator to an operand needed a cross-half exchange but the reverse was a
free discard, because the operand's halves were mirrors and already held
everything. On gfx12 neither half holds everything, so *both* directions cross
the wave halves, at 8 `permlanex16` each. `include/rdna4/ops/warp/register/tile/
conversions.cuh` opens with the derivation; read it before touching
`test_copy` or the `rv_layout::align` vector tests.

**The mma tests count.** `mma_wrapper_2d` in `warp/register/tile/mma.cu` has its
own copy of the harness wrapper, and the cdna4 original it was forked from sets
`this_result` and then never pushes it, so an mma failure prints but is missing
from the tally. Both RDNA trees push it. On a tree that has never run, the tally
is the whole report.

**Reductions along the element axis always exchange.** Same cause: on gfx11 the
operand case was free. Here it is not, so `halves_interleave` is true for every
type and the exchange in `element_axis_reduce` is unconditional in practice.

## How this tree differs from `tests/unit/cdna4`

Inherited from the RDNA3 port, and still true here:

**One tile shape.** The CDNA trees parameterize tests over `ducks::rt_shape` and
`ducks::st_shape` because MFMA comes in several geometries. gfx12 has exactly
one matrix instruction, `v_wmma_*_16x16x16_*`, so `include/rdna4` declares one
member of each namespace and every sweep here passes `rt_16x16` / `st_16x16`.
The shape parameters remain in the test signatures — the harness in
`../common/testing_commons` is shared with the CDNA trees and threads them
through positionally — but they are not a degree of freedom on this
architecture.

**`swap_layout` means something else.** On CDNA the layout-swap test really
sweeps *shapes*, often with both layouts the same. Here the only thing
`swap_layout` can change is the layout, and the destination layout is pinned to
`transpose<L>`, so `test_swap_layout` sweeps row→col and col→row instead.

**fp8 exists here and does not exist there.** gfx12 has four fp8 WMMA opcodes
that gfx11 has none of, so `include/rdna4` grows `rt_fp8e4m3` / `rt_fp8e5m2` and
their shared counterparts, and `warp/register/tile/mma.cu` ends with a section
testing them. It is behind `#ifdef KITTENS_RDNA4` and is byte-identical in
`tests/unit/rdna3`, where it compiles away -- see the note on the fork below.

Two things about that section are worth knowing before you extend it. It is
hand-rolled rather than run through the sweep harness, because the harness types
the input buffer, the output buffer and the reference off a single type and here
A, B and C are three different ones. And it instantiates the full (e4m3, e5m2) x
(e4m3, e5m2) cross product on purpose: A's and B's encodings are independent,
there is no operand-select bit, so the four opcodes are four separate builtins
and the only way to know the dispatch reaches all of them is to ask for all of
them.

What it does *not* cover: fp8 as anything but a WMMA operand. `include/rdna4`
deliberately has no elementwise maps, reductions or vector ops on fp8 register
tiles, and `constants<fp8e4m3>` deliberately has no infinity, so a reduction
over an fp8 tile is a compile error rather than a silently wrong answer. If you
want fp8 arithmetic, convert to `float` first; that path is tested.

## Adding a test

Each `.cu` here wraps its scaffolding in an anonymous namespace. That is not
decoration: every file in the tree names its helpers the same way
(`test_generator`, `vec_load_store`, …), and as templates with external linkage
they collapse into one definition at link time, so whole sections silently run
some other file's tests. Keep the anonymous namespace, and keep
`<scope>::tests()` outside it.

The tree mirrors the layout of `include/rdna4`, which is how coverage gaps stay
visible.
