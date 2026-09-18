# HipKittens unit tests — RDNA4 (gfx1200 / gfx1201)

Unit tests for `include/rdna4`, for an RX 9060 XT / RX 9070 (XT) class card.

## Read this first: nothing here has ever run

No gfx12 part was available while `include/rdna4` was written. This tree is
**compile-verified only**. `make` succeeds for `--offload-arch=gfx1201`; the
resulting `./unit_tests` binary has never been executed, on any machine.

What that does and does not buy you:

- **Verified.** Every type, op and test in the tree type-checks and codegens for
  gfx1201. The WMMA builtin signatures were confirmed against the compiler
  (`wmma_f32_16x16x16_{f16,bf16}_w32_gfx12` take `v8h`/`v8s` operands and a
  `v8f` accumulator). The instruction selection is what it should be — the LDS
  fast path emits `ds_load_b128`, not `flat_load_b128`, and the MMA emits
  `v_wmma_f32_16x16x16_bf16`. The 64 KB per-workgroup LDS ceiling was confirmed
  by compiling a 65540-byte `__shared__` array and reading the error. The gfx12
  split wait counters (`s_wait_dscnt` / `s_wait_loadcnt` / `s_wait_storecnt`)
  are what the tree emits, checked against the disassembly. All four fp8 WMMA
  opcodes are reachable from the tile API and all four appear in this tree's
  object code (`v_wmma_f32_16x16x16_{fp8_fp8,fp8_bf8,bf8_fp8,bf8_bf8}`), and the
  fp8 LDS path emits `ds_load_b64` / `ds_store_b64` rather than the `b128` the
  2-byte types get.
- **Not verified.** Every claim about *where a value physically lives*. The
  operand fragment layout — that lane `l` holds `k = 8*(l/16) .. +7` of row
  `l%16` — is inferred from the builtin's `v8` operand width plus the shape of
  the gfx11 layout that was measured. It is the single assumption the whole tree
  rests on, and it is not a measurement.

So: when a gfx12 card turns up, `./unit_tests` is the first thing to try, and
if the fragment inference is wrong it is the thing that will say so. Expect
`warp/register/tile/{mma,conversions}` and `warp/memory/tile/shared_to_register`
to fail together if it is — they all derive from `rt_base_coord()`.

Before trusting a fix, re-run the layout probe rather than guessing: the
`PROBE_GFX12_W32` variant of `tools/rdna-probes/wmma_layout.hip` measures the
fragment directly, the same way the gfx1100 layout was established.

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
