# bf16 GEMM, fp32 accumulate — RDNA3 (gfx1100)

C = A · Bᵀ, bf16 in, fp32 accumulate, bf16 out. B is passed pre-transposed as
`(n, k)`, which keeps both shared loads contiguous and makes `mma_ABt` the right
primitive.

This is the kernel the RDNA3 port was validated against. Unlike
`kernels/rdna4/`, everything below is measured.

## Measured

W7900D (gfx1100, 96 CU), ROCm 7.2.4. TFLOPs, HipKittens against **both** AMD
BLAS libraries, vendor columns taken as best-of-{NN,NT,TN,TT}:

| shape (M×N×K) | hipBLASLt | rocBLAS | HK | HK / best | HK / WMMA ceiling |
|---|---|---|---|---|---|
| 4096³ | 69.1 | **83.2** | 68.5 | 82% | 68% |
| 8192×8192×4096 | 65.0 | **88.6** | 69.5 | 78% | 69% |
| 4096×8192×2048 | 70.4 | **88.6** | 71.5 | 81% | 71% |
| 8192×4096×2048 | 70.0 | **88.6** | 71.4 | 81% | 71% |
| 2048×4096×4096 | 68.0 | **85.7** | 64.9 | 76% | 65% |
| 2048×2048×4096 | 63.7 | **75.1** | 57.4 | 76% | 57% |
| 2048³ | 60.3 | **69.5** | 55.3 | 80% | 55% |

**This kernel is 20-25% behind the best vendor library on every shape.** An
earlier version of this table compared only against torch's default backend and
claimed the kernel won two shapes of four. That was wrong twice over: torch on
ROCm defaults to hipBLASLt, and on gfx1100 hipBLASLt is the *slower* library by
up to 33% — its Tensile tuning evidently does not cover RDNA the way rocBLAS's
does. Run `python bench.py`, which now benchmarks both.

Percentages are against **100.5 TFLOPs**, the measured back-to-back WMMA issue
ceiling ([`tools/rdna-probes/wmma_peak.hip`](../../../../tools/rdna-probes/wmma_peak.hip)),
not against the 122.6 TFLOPs on the data sheet. 122.6 assumes 2.495 GHz boost;
sampling `rocm-smi` during a GEMM shows this card at 2.0-2.1 GHz against its
241 W limit. The reachable peak here is ~104 TFLOPs.

### Where the 20% goes, and why the obvious fixes do not work

It is LDS *bytes*, not LDS instructions and not bank conflicts. Both of those
were checked and neither is the answer.

Disassembling rocBLAS's kernel and histogramming the two inner loops, normalized
so both do 16 WMMAs:

| per 16 WMMAs | rocBLAS | this kernel |
|---|---|---|
| LDS read instructions | **128** × `ds_load_u16` (2 B) | 24 × `ds_read_b128` (16 B) |
| LDS bytes read per lane | **256** | **384** |
| non-WMMA VALU | **0** | 28 × `v_add_nc_u32` |
| total instructions in the loop | 188 | 122 |

rocBLAS issues 5.3× more LDS instructions and still wins, which invites a
bank-conflict story. There is not one.
[`tools/rdna-probes/lds_peak.hip`](../../../../tools/rdna-probes/lds_peak.hip)
measures the delivered bandwidth of each access pattern at equal instruction
count: the pattern this kernel uses reaches **64.8 bytes/clk/CU**, tied with a
hand-built conflict-free reference and 4.1× the same tile read with the swizzle
removed. rocBLAS's 2-byte pattern reaches 35.8. Both are conflict-free; the
`swizzle_bytes` constant in `st.cuh` does not need retuning.

What is left is arithmetic intensity. A warp tile of M×N needs `(M+N)·K` operand
elements for `(M/16)(N/16)` WMMAs, so the bytes per WMMA go as `(M+N)/(M·N)`:

| warp tile | intensity `(M·N)/(M+N)` | distinct bytes per 16 WMMAs |
|---|---|---|
| 32×64 (this kernel) | 21.3 | 6144 |
| 64×64 (rocBLAS) | **32.0** | **4096** |

1.5×, exactly. Converted to a rate: 16 WMMAs is 16 × 8192 FLOP, which at
512 FLOP/clk/CU is 256 CU-cycles, and a CU retires two wave-iterations in that
window — so this kernel asks LDS for 48 bytes/clk/CU against the 64.8 ceiling,
**74% utilised**, where rocBLAS asks for 32.

rocBLAS's Tensile kernel for this shape is
`Cijk_..._MT128x128x16_MI16x16x16x1_..._PGR1_PLR1_..._WG32_4_1`: the same 128×128
block, but **4 waves of 64×64 at 256 VGPRs** (6 waves/SIMD) instead of 8 waves of
32×64 at 128 VGPRs (9 waves/SIMD), with both global *and* local reads prefetched.

Every attempt to copy that shape here measures *slower*:

| config | VGPRs | occ | 4096³ / 8192×8192×4096 |
|---|---|---|---|
| 128×128, 8 warps, 32×64/warp (default) | 164 | 9 | **66 / 72** |
| 128×128, 4 warps, 64×64/warp | 238 | 6 | 52 / 57 |
| 256×128, 8 warps, 64×64/warp | 256 | 5 | 62 / 68 |
| 128×256, 8 warps, 64×64/warp | 256 | 5 | 62 / 69 |
| 256×256, 16 warps, 64×64/warp | 227 | 6 | 59 / 66 |

The 64×64 configurations do fit — 238 VGPRs, no spills, exactly rocBLAS's 6
waves/SIMD — they are just latency-bound. This kernel hides LDS latency by
*switching waves*, so every change that improves arithmetic intensity pays for it
in occupancy, and the occupancy loss is larger. rocBLAS escapes the trade by
hiding latency in a software pipeline (`PLR1`) instead, which costs registers
rather than occupancy.

**So the missing piece is inner-loop pipelining, and it has to come first.**
Double-buffering the operand tiles is what makes the 64×64 shape pay; the 64×64
shape is what makes the pipeline worth its registers. Adding either one alone
measures slower, which is how this kernel ended up at a local optimum — see the
first entry under "tried and did not work" below, which is that same experiment
run at 32×64 and correctly concluded, for that shape only.

### Small shapes get their own tiling

The inner loop above is not what bounds small shapes; grid quantization is. The
128×128 block uses 32 KB of LDS, so four workgroups fit per WGP and 48 WGPs hold
192 at once. At 2048×2048 the grid is 16×16 = 256 workgroups — one full pass plus
a second that is two thirds empty, with the machine idling through it.

So `dispatch_micro` compiles two tilings and picks by workgroup count:

| M,N | blocks | 128×128, 32×64/warp | 128×64, 32×32/warp |
|---|---|---|---|
| 1024² | 64 | 36 | **44** |
| 2048², K=2048 | 256 | 39 | **55** |
| 2048², K=4096 | 256 | 48 | **55** |
| 2048×4096 | 512 | **63** | 61 |
| 4096² | 1024 | **67** | 66 |
| 4096×8192 | 2048 | **72** | 69 |
| 8192² | 4096 | **71** | 59 |

The crossover at 256 blocks is sharp, and the rule is just `blocks <= 256`. The
small tiling gives up half the arithmetic intensity and gets back twice the
workgroups and a third more waves (103 VGPRs, 12 waves/SIMD against 164 and 9).
This is worth 16% at 2048³ and costs nothing above the crossover.

A third variant with `K_STEP=64` was measured too — it wins at small shapes for
the unrelated reason that it halves the number of K-tiles and therefore the
number of barriers, but it never beats the small tiling, so it is not carried.

Split-K is still missing and is the next thing for this regime.

How it got there, each step measured at 4096³ / 8192×8192×4096:

| | TFLOPs |
|---|---|
| first correct version | 50 / 54 |
| one barrier per K-tile | 53 / 58 |
| 16-byte LDS swizzle, 2× b128 (still `flat_load`) | 59 / 64 |
| `ds_read_b128` / `ds_write_b128` via inline asm | 66 / 72 |

(The last row and the first table are separate runs of the same build; ±1 TFLOP
run to run is normal on this part.)

The last row is worth reading twice. A shared tile reached through
`shared_allocator` is a *generic* pointer as far as the compiler is concerned,
so `*(float4*)p` compiles to `flat_load_b128` — right width, wrong pipe, and it
counts against `vmcnt` instead of `lgkmcnt`. Inline asm is not an optimization
here, it is the only way to name the instruction. Worth 7 TFLOPs on its own.

## Running it

```bash
make                 # builds tk_kernel.so
python test.py       # correctness against torch
python bench.py      # the table above
./sweep.sh           # tiling sweep, and the per-stage ablations
```

## Tuning

`BLOCK_M`, `BLOCK_N`, `K_STEP`, `DOT_SLICE`, `NUM_WARPS`, `WARP_ROWS`, `WGM` are
`-D`-overridable, but only take effect with `-DHK_MULTI_CONFIG=0`, which pins the
build to exactly that tiling instead of the two-way selection above. `sweep.sh`
passes it; without it a sweep would measure the selection rule rather than the
configuration it names. `ABLATE_GLOBAL` / `ABLATE_LDS_READ` / `ABLATE_MMA` delete one pipeline
stage each — they make the answer wrong on purpose, and they are what located
the cost above.

Two things that were tried and did *not* work, recorded so they are not retried
blind:

1. **Software-pipelining the LDS reads, at the 32×64 warp tile.**
   Double-buffering the operand tiles costs 48 VGPRs, occupancy goes 9 → 7
   waves/SIMD, and it measures 63/68 — slower. At 9 waves the SIMD was already
   covering LDS latency by switching waves.

   Read that as scoped to this tile shape, not as a verdict on pipelining. At
   32×64 the operand loads are only 1.33 per WMMA, so there is little for a
   pipeline to hide; at the 64×64 shape rocBLAS uses they are 2.0 per WMMA and
   the pipeline is what makes the shape affordable. The two changes have to be
   made together — see above. `load_async()` and `lds_wait<N>()` are in the
   library for exactly that.
2. **Bigger register blocks.** Every 4×4-base-tile shape hits the 256-VGPR
   ceiling and spills.

Both have the same root cause, and it is architectural: a gfx11 WMMA operand is
mirrored across the two wave halves, so a `bf16` `rt_base` costs the same 8
VGPRs as an `fp32` one. On RDNA3 it is the *operand* registers that bound the
tile, not the accumulators — the reverse of CDNA. gfx12 halves this, which is
why `kernels/rdna4/` has headroom this kernel does not.

## Structure

Follows `kernels/cdna4/gemm/bf16fp32`, with the schedule redesigned for three
RDNA3 facts:

- **wave32**, and the operand mirroring above.
- **No global→LDS DMA.** `vmem-to-lds-load-insts` does not exist on gfx11, so
  every prefetched byte passes through VGPRs. The "async copy" is
  `load_global_to_register_buffer` now and `store_register_buffer_to_shared` at
  the point the buffer is needed. This is the one HipKittens pillar that does
  not survive the port, and it caps how much of the paper's ping-pong schedule
  is reachable.
- **One XCD**, so there is no chiplet swizzle — only the L2 group swizzle.
