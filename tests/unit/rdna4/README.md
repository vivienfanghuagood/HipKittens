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
  are what the tree emits, checked against the disassembly.
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
`PROBE_GFX12_W32` variant of `hk-rdna/probe/wmma_layout.hip` measures the
fragment directly, the same way the gfx1100 layout was established.

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

The test *sources* are nearly identical — this tree was forked from the RDNA3
one and the arch difference lives entirely in `include/rdna4`. The Makefile's
`GPU_TARGET` is the only edit. That is deliberate: the tests are written against
the tile API, so if the RDNA4 fragment inference is right, the same tests pass
unchanged, and if it is wrong, the diff between the two trees is not where you
should be looking.

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

**No fp8.** gfx12 does have `wmma_f32_16x16x16_fp8_fp8_w32_gfx12`, unlike gfx11,
but `include/rdna4` does not expose fp8 tiles, so there is nothing to test yet.

## Adding a test

Each `.cu` here wraps its scaffolding in an anonymous namespace. That is not
decoration: every file in the tree names its helpers the same way
(`test_generator`, `vec_load_store`, …), and as templates with external linkage
they collapse into one definition at link time, so whole sections silently run
some other file's tests. Keep the anonymous namespace, and keep
`<scope>::tests()` outside it.

The tree mirrors the layout of `include/rdna4`, which is how coverage gaps stay
visible.
