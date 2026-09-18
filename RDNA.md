# HipKittens on RDNA3 / RDNA4

A port of HipKittens to AMD's consumer / workstation architectures — gfx11
(RDNA3: RX 7900, W7900) and gfx12 (RDNA4: RX 9070, W9000-class). Upstream
supports CDNA3/4/5 only.

This is a fork. Upstream is [HazyResearch/HipKittens](https://github.com/HazyResearch/HipKittens);
the work here lives on the `rdna` branch, seven commits on top of `de0cddbd`.

| | RDNA3 (gfx1100) | RDNA4 (gfx1201) |
|---|---|---|
| core library | `include/rdna3`, 68 files | `include/rdna4`, 68 files |
| unit tests | **1659 passed, 0 failed** on a W7900D | compiles; **never executed** |
| bf16 GEMM | **71.5 TFLOPs** peak, 71% of the measured WMMA ceiling, 76-82% of rocBLAS across shapes | compiles; never executed |
| fp8 | not available in hardware | implemented, all four opcodes, untested |

**Read the second column as "written, not verified."** No gfx12 part was
available. Every gfx12 claim about *where a value physically lives* is inferred
from builtin signatures, not measured. See
[`tests/unit/rdna4/README.md`](tests/unit/rdna4/README.md) for exactly which
claims are which, and run `./unit_tests` there first if you have the hardware.

## Why this is a separate tree and not a `#ifdef`

RDNA's matrix unit is WMMA, not MFMA, and the fragment layout is different
enough that sharing code would have meant an `#ifdef` in every load, store and
conversion. The repo already splits by architecture (`include/cdna3`,
`cdna4`, `cdna5`), so this follows that.

Three hardware facts drove essentially every design decision, all three
established by probe rather than by documentation:

**1. gfx11 WMMA operands are mirrored across the wave halves.** On gfx1100
wave32, a `v_wmma_f32_16x16x16_bf16` operand is `v16bf16` — 8 VGPRs — and lanes
`l` and `l+16` must hold *identical* data. A lane holds the entire K=16 vector.
Compare CDNA3's MFMA, where a lane holds 4 halves, K is split across lane
groups, and there is no redundancy at all.

The consequence is counterintuitive and it is the single most important thing
about this port: **a bf16 operand tile costs the same registers as an fp32
accumulator tile.** On RDNA3 the operand registers bound the tile shape, which
is the reverse of CDNA. It is why the GEMM cannot grow past a 128×128 block, and
why software-pipelining the operand loads measures *slower*.

(This also cost a false start. The first layout probe put a one-hot value in a
single lane without mirroring it, violating the hardware constraint, and read
back undefined behaviour that looked like a coherent — and wrong — layout.)

**2. gfx12 splits K between the halves instead.** Operands are `v8bf16`, 4
VGPRs, no redundancy: lane `l` holds `k = 8·(l/16) … +7` of row `l%16`. Half the
operand registers for the same tile. That is the point of RDNA4 for this
library. It also makes accumulator↔operand conversion *more* expensive, not
less — on gfx11 one direction was a free discard because the operand halves
already held everything; on gfx12 neither half does, so both directions cross
the halves at 8 `permlanex16` each.

**3. There is no global→LDS DMA on either.** `vmem-to-lds-load-insts` is absent
on gfx11 and gfx12 alike. HipKittens' third pillar — "asynchronous loads/stores
using direct buffer loads to shared memory" — has no hardware to stand on here.
Every prefetched byte passes through VGPRs and into the wave's instruction
stream. This is why `include/rdna3` is derived from `cdna3` rather than `cdna4`:
cdna3's `global_to_shared.cuh` already staged through `float4` buffers, which is
the only path RDNA has. It also caps how much of the paper's ping-pong schedule
is reachable, and the GEMM numbers should be read with that in mind.

## Coverage

**Library** (`include/rdna3`, `include/rdna4` — 68 files each, the same shape as
`include/cdna3`):

- Types: `rt_base` / `rt` / `st` / `st_subtile` / `sv` / `rv` / `gl`, both
  layouts, all three `rv_layout`s, `shared_allocator`, `group<N>`.
- Ops: warp and group, register and shared and global, tile and vector — maps,
  reductions, conversions, `mma_{AB,ABt,AtB,AtBt}`, and every
  `{global,shared}_to_{shared,register}` transfer.
- gfx11 WMMA (`f16`, `bf16`), gfx12 WMMA (`f16`, `bf16`, and all four fp8
  opcodes: `fp8_fp8`, `fp8_bf8`, `bf8_fp8`, `bf8_bf8`).

**Tests** (`tests/unit/rdna3`, `tests/unit/rdna4` — 64 files each, ported from
`tests/unit/cdna4`). RDNA3 at `TEST_INTENSITY=2`, on a W7900D:

```
1659 tests passed
0 tests failed
509 tests skipped (invalid template parameters)
```

The two test trees are the *same sources* — `diff -r` returns only the Makefile
and the README. That is deliberate and worth preserving: it tells a reader that
if gfx12 fails, the bug is in `include/rdna4`, not in the tests.

**Kernels**: `kernels/rdna3/gemm/bf16fp32` (measured, see its README) and
`kernels/rdna4/gemm/bf16fp32` (same source, gfx1201 target, untuned).

### Not covered

- **Attention.** The library primitive it needs most — accumulator→operand
  layout conversion via `permlanex16` — is implemented and tested on RDNA3, so
  the road is open, but no attention kernel is written.
- **fp8 as anything but a WMMA operand.** No maps, reductions or vector ops on
  fp8 register tiles. `constants<fp8e4m3>` deliberately has no infinity, so a
  reduction over one is a compile error rather than a silently wrong answer.
- **Split-K**, which is what the small-shape GEMM numbers are missing.
- **Inner-loop software pipelining in the GEMM.** The global→LDS stage is
  double-buffered; the LDS→register stage is not. That is what caps the warp
  tile at 32×64 and leaves 20% on the table against rocBLAS.
- **Multi-GPU / distributed**, untouched.
- **RDNA4 on hardware**, at all.

## Performance

W7900D (gfx1100, 96 CU). bf16 GEMM, fp32 accumulate. Vendor columns are
best-of-{NN,NT,TN,TT}; HipKittens implements one layout.

| shape (M×N×K) | hipBLASLt | rocBLAS | HK | HK / best | HK / ceiling |
|---|---|---|---|---|---|
| 4096³ | 69.1 | **83.2** | 68.5 | 82% | 68% |
| 8192×8192×4096 | 65.0 | **88.6** | 69.5 | 78% | 69% |
| 4096×8192×2048 | 70.4 | **88.6** | 71.5 | 81% | 71% |
| 8192×4096×2048 | 70.0 | **88.6** | 71.4 | 81% | 71% |
| 2048×4096×4096 | 68.0 | **85.7** | 64.9 | 76% | 65% |
| 2048×2048×4096 | 63.7 | **75.1** | 57.4 | 76% | 57% |
| 2048³ | 60.3 | **69.5** | 55.3 | 80% | 55% |

**Two corrections to numbers this file previously carried**, both of which made
the kernel look better than it is:

*The baseline was the wrong library.* torch on ROCm defaults to hipBLASLt, and
on gfx1100 hipBLASLt is the slower of the two by up to 33% — rocBLAS wins every
shape here. Benchmarking against torch's default is benchmarking against the
library AMD has not tuned for this architecture. The real gap is 20-25%, not the
"wins 2 of 4" that the hipBLASLt-only table showed.

*The denominator was unreachable.* 122.6 TFLOPs is 96 CU × 512 FLOP/clk ×
2.495 GHz boost. This card does not run a GEMM at 2.495 GHz: `rocm-smi` sampled
under load shows 2.0-2.1 GHz against a 241 W limit. Back-to-back WMMA with no
memory at all measures **100.5 TFLOPs**
([`tools/rdna-probes/wmma_peak.hip`](tools/rdna-probes/wmma_peak.hip)) — 94% of
the architectural issue rate, at 2.18 GHz and only 101 W. The reachable peak is
~104, so percentages against 122.6 understate by about 18%.

The 20% behind rocBLAS is one identified thing: **LDS bytes per WMMA**, which is
arithmetic intensity, which is the warp tile. A 32×64 tile moves 1.5× the operand
bytes per WMMA that rocBLAS's 64×64 does — `(M+N)/(M·N)`, exactly — and that puts
this kernel at 48 bytes/clk/CU against a measured 64.8 ceiling, 74% utilised,
where rocBLAS sits at 32.

It is worth saying what it is *not*, because both alternatives look plausible
from the disassembly and both were checked. It is not instruction count: rocBLAS
issues 5.3× more LDS instructions (128 × `ds_load_u16` against 24 ×
`ds_read_b128` per 16 WMMAs) and still wins. And it is not bank conflicts:
[`tools/rdna-probes/lds_peak.hip`](tools/rdna-probes/lds_peak.hip) measures this
tree's access pattern at 64.8 bytes/clk/CU, tied with a hand-built conflict-free
reference and 4.1× the same read with the swizzle removed. The swizzle constant
is right.

Every attempt to adopt the wider tile measures slower, because this kernel hides
LDS latency with occupancy and the wider tile costs occupancy — rocBLAS hides it
with a software pipeline (`PLR1`) instead, which costs registers. The two changes
only pay together, and on gfx1100 they do not both fit: 64×64 needs 128 VGPRs of
accumulator plus 64 of operands, and the operands are already 2× larger than they
would be on CDNA because of the wave-half mirroring, so double-buffering them
overruns the 256-VGPR file. **That is the wall, and it is an architectural one.**
The full config sweep is in
[`kernels/rdna3/gemm/bf16fp32/README.md`](kernels/rdna3/gemm/bf16fp32/README.md).

Small shapes are a separate, fixed problem. They were bound by grid quantization
— 2048² is 256 workgroups against 192 concurrent slots, so the second pass ran
two thirds empty — and `dispatch_micro` now compiles two tilings and picks by
workgroup count, worth 16% at 2048³ and nothing above the crossover. Split-K is
still missing and is the next thing for that regime.

RDNA4 has not been run. What is known is a register count: the operand tiles
cost 24 VGPRs there against 48 on gfx1100, and the ported kernel sits at 128
VGPRs where 10 waves/SIMD holds to 144 — so there is headroom the gfx1100 shape
could not use, and double-buffering costs one occupancy step there instead of
two. fp8 will *not* help the math rate: gfx12's fp8 WMMA is 16×16×16, the same
shape as bf16, not CDNA's K=32. One WMMA is one WMMA.

## Notes for anyone continuing this

Things that cost time and are written down so they cost it only once:

- **`s_waitcnt` is split on gfx12** into `s_wait_dscnt` / `s_wait_loadcnt` /
  `s_wait_storecnt` / `s_wait_kmcnt`. The trap is that the combined legacy
  `s_waitcnt lgkmcnt(0)` still *assembles* for gfx1201 — LLVM just never emits
  it — so inline asm inherited from a CDNA tree looks fine and silently waits on
  the wrong thing.
- **Inline asm for LDS is mandatory, not stylistic.** A `shared_allocator` tile
  is a generic pointer, so `*(float4*)p` emits `flat_load_b128` rather than
  `ds_load_b128`. Right width, wrong pipe. Worth 7 TFLOPs in the GEMM.
- **The V# buffer descriptor word is `0x31004000`** for gfx11 *and* gfx12, not
  CDNA's `0x110000` — gfx10+ replaced `DATA_FORMAT`/`NUM_FORMAT` with `FORMAT`.
- **fp8 on gfx12 is OCP, on gfx942 it is fnuz**, and HIP gates them by target.
  The fnuz types are still declared on gfx12, so a typedef inherited from the
  CDNA tree compiles until the first thing instantiates it.
- **Benchmark against rocBLAS, not against torch's default.** torch on ROCm
  picks hipBLASLt, and on gfx1100 that is the library with the thinner Tensile
  tuning — rocBLAS is up to 33% faster on the same shape. Switch with
  `torch.backends.cuda.preferred_blas_library("cublas")`, and take best-of-four
  transposes while you are there; the spread across NN/NT/TN/TT reaches 40%.
- **Do not use the data-sheet peak as a denominator.** Measure the ceiling with
  `tools/rdna-probes/wmma_peak.hip` and sample `rocm-smi -c -P` during the run.
  This card is clock- and power-limited well below boost under any real load.
- **The layout probe is the ground truth**, not the ISA doc:
  [`tools/rdna-probes/`](tools/rdna-probes/), and `wmma_layout.hip` in
  particular, which has a `PROBE_GFX12_W32` variant already written for whenever
  a gfx12 card turns up. Mirror the one-hot across `l` and `l^16` on gfx11 or
  you will measure undefined behaviour.

Per-area detail lives next to the code: `include/rdna3/ops/warp/register/tile/
conversions.cuh` opens with the layout derivation the whole port rests on, and
each of the four READMEs (`tests/unit/rdna{3,4}`, `kernels/rdna{3,4}/gemm/
bf16fp32`) says what is verified and what is not.
