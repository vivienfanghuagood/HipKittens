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
| 4096³ | 69.5 | **83.0** | 74.5 | 90% | 74% |
| 8192×8192×4096 | 66.3 | **88.7** | 76.4 | 86% | 76% |
| 4096×8192×2048 | 70.5 | **88.6** | 75.4 | 85% | 75% |
| 8192×4096×2048 | 69.7 | **88.6** | 75.4 | 85% | 75% |
| 2048×4096×4096 | 67.9 | **85.3** | 73.4 | 86% | 73% |
| 2048×2048×4096 | 63.4 | **76.9** | 67.3 | 87% | 67% |
| 2048³ | 60.0 | **67.2** | 64.1 | 95% | 64% |

**This kernel beats hipBLASLt on every shape and is 5-15% behind rocBLAS.** Two
notes on reading that. First, an early version of this table compared only
against torch's default backend and claimed the kernel won two shapes of four.
That was wrong twice over: torch on ROCm defaults to hipBLASLt, and on gfx1100
hipBLASLt is the *slower* library by up to 33% — its Tensile tuning evidently
does not cover RDNA the way rocBLAS's does. Run `python bench.py`, which
benchmarks both. Second, the vendor columns get best-of-four layouts and HK is
measured in its one layout (NT), so this is if anything generous to them.

Percentages are against **100.5 TFLOPs**, the measured back-to-back WMMA issue
ceiling ([`tools/rdna-probes/wmma_peak.hip`](../../../../tools/rdna-probes/wmma_peak.hip)),
not against the 122.6 TFLOPs on the data sheet. 122.6 assumes 2.495 GHz boost;
sampling `rocm-smi` during a GEMM shows this card at 2.0-2.1 GHz against its
241 W limit. The reachable peak here is ~104 TFLOPs.

### Where the remaining gap goes

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
32×64, with both global *and* local reads prefetched.

Every attempt to copy that warp tile here measures *slower*:

| config | VGPRs | occ | 4096³ / 8192×8192×4096 |
|---|---|---|---|
| 128×128×64, 8 warps, 32×64/warp (default) | 208 | 7 | **73 / 78** |
| 128×128, 4 warps, 64×64/warp | 238 | 6 | 52 / 57 |
| 256×128, 8 warps, 64×64/warp | 256 | 5 | 62 / 69 |
| 128×256, 8 warps, 64×64/warp | 256 | 5 | 62 / 69 |
| 256×256, 16 warps, 64×64/warp | 249 | 5 | 64 / 71 |

The 64×64 configurations do fit — 238 VGPRs, no spills, exactly rocBLAS's 6
waves/SIMD — they are just latency-bound, and there is no room left to pipeline
them: 64×64 is 128 accumulator VGPRs plus 64 operand VGPRs plus staging, and a
second copy of the operands would need 64 more against a 256-VGPR file. There is
no intermediate tile either. Accumulator VGPRs are `M·N/32` regardless of shape,
so among powers of two the square tile is strictly optimal, and 48-row tiles
need `BLOCK_M=192`, which divides none of the benchmark shapes (the kernel has
no bounds predication). **That is an architectural wall**, and it is the reason
the last 10% is not reachable by tuning: a gfx11 WMMA operand is mirrored across
the two wave halves, so a `bf16` `rt_base` costs the same 8 VGPRs as an `fp32`
one. gfx12 halves this, which is why `kernels/rdna4/` has headroom this kernel
does not.

### The LDS reads run underneath the WMMAs, for free

Ablating one pipeline stage at a time (`sweep.sh` with `ABLATE_*`, at 4096³ /
8192×8192×4096) is how the cost was located:

| build | VGPRs | occ | TFLOPs |
|---|---|---|---|
| full | 208 | 7 | 73 / 78 |
| no global prefetch | 164 | 9 | 79 / 83 |
| no LDS reads | 118 | 12 | 88 / 92 |
| no WMMAs | 144 | 10 | 117 / 123 |
| WMMAs only | 92 | 16 | 95 / 101 |

Before the change described here the full build was 66/71 and the no-LDS-read
build was 84/95 — the read stage was the dominant cost, and perfectly overlapping
it was worth 30%.

The textbook fix is to double-buffer the operand tiles so slice `k+1`'s reads
issue before slice `k`'s WMMAs. That was tried, and it is *slower*: the second
set of tiles costs 48 VGPRs, occupancy goes 9 → 7 waves/SIMD, and at 9 waves the
SIMD was already covering some LDS latency by switching waves. The overlap does
not pay for the registers.

What does work is getting the overlap without allocating anything, and
`lds_wait<N>` is what makes that possible. It waits until *at most N* LDS
operations are still outstanding, and LDS retires in order, so a partial wait
retires a prefix of a batch and leaves the rest moving. Two changes follow:

1. **Split along N.** `B` and the accumulator are cut into `N_SPLIT` chunks of
   `SPLIT_N` columns. The whole slice's reads — `A` plus every `B` chunk — go in
   flight at once, then the waits step down: `lds_wait<(N_SPLIT-1-c)*CHUNK_READS>`
   before chunk `c` retires exactly what chunk `c` needs. Total registers are
   unchanged; this is a re-association, not a buffer. Worth 65 → 69 / 71 → 75.
2. **Rotate across slices** (`ROTATE=1`). `B_tile[c]` is dead the instant chunk
   `c`'s WMMAs have issued, so the *next* slice's chunk `c` goes in flight right
   there, a full chunk of math early. `A_tile` is dead only after the slice's
   last chunk, so `A` is issued at the boundary and *before* that chunk's `B`,
   which keeps the next slice's first chunk gated on `A` rather than on a read
   queued behind it. The steady state then has a uniform wait depth, and fifteen
   of a K-tile's sixteen chunks have reads underneath them. Worth another 69 → 70
   / 75 → 76.

Neither step moves the register count: 161 VGPRs before the split, after the
split, and after the rotation. The compiler does not rename around the
write-after-read on `B_tile[c]`, which was the thing most likely to go wrong.

**This also changed what the rest of the tuning wants**, which is the part worth
remembering. More K-slices means more rotation points, so a deeper K-tile is
suddenly worth its staging registers even at lower occupancy:

| K_STEP | slices | VGPRs | occ | 4096³ / 8192×8192×4096 |
|---|---|---|---|---|
| 32 | 2 | 161 | 9 | 70 / 76 |
| 64 | 4 | 208 | 7 | **73 / 78** |

`K_STEP=64` was measured before the rotation landed and lost (66/69 against
66/72); it wins now. `K_STEP=128` does not fit in 64 KB of LDS with two buffers.

### Small shapes get their own tiling

The inner loop above is not what bounds small shapes; grid quantization is. The
128×128×64 block uses 48 KB of LDS, so only one workgroup fits per WGP and 48
WGPs hold 48 at once. A grid that is a poor fit for the coarse tiling idles
through a mostly-empty final pass.

So `dispatch_micro` compiles two tilings and picks by workgroup count:

| M,N,K | blocks | 128×128×64, 32×64/warp | 128×64×64, 32×32/warp |
|---|---|---|---|
| 512×512×4096 | 16 | 19 | **33** |
| 1024×1024×4096 | 64 | 49 | **51** |
| 2048×1024×4096 | 128 | **63** | 58 |
| 2048×2048×2048 | 256 | **64** | 60 |
| 4096×1024×4096 | 256 | **65** | 60 |
| 4096³ | 1024 | **73** | — |
| 8192×8192×4096 | 4096 | **78** | — |

so the rule is `blocks <= 64`. **That threshold moved by 4× when the rotated
schedule landed** — it used to be 256. Overlapping the LDS reads made the coarse
tiling much better at mid-size grids, and it now wins everywhere it has enough
workgroups to fill the machine once. The small tiling gives up half the
arithmetic intensity and gets back twice the workgroups and three more
waves/SIMD (134 VGPRs and 10 waves against 208 and 7).

A third config exists purely for correctness: `K_STEP=64` requires K to be a
multiple of 64, and the kernel has no K remainder handling, so `dispatch_micro`
falls back to a `K_STEP=32` build when it is not.

At 512×512 even the small tiling only reaches 33 TFLOPs. That regime wants
split-K, which is still missing and is the next thing here.

How it got there, each step measured at 4096³ / 8192×8192×4096:

| | TFLOPs |
|---|---|
| first correct version | 50 / 54 |
| one barrier per K-tile | 53 / 58 |
| 16-byte LDS swizzle, 2× b128 (still `flat_load`) | 59 / 64 |
| `ds_read_b128` / `ds_write_b128` via inline asm | 66 / 72 |
| split B along N, step the waits down | 69 / 75 |
| rotate the next slice's reads into the freed slots | 70 / 76 |
| K_STEP 32 → 64, which the rotation made affordable | **73 / 78** |

(The last row and the first table are separate runs of the same build; ±1 TFLOP
run to run is normal on this part.)

The `ds_read_b128` row is worth reading twice. A shared tile reached through
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

`BLOCK_M`, `BLOCK_N`, `K_STEP`, `DOT_SLICE`, `NUM_WARPS`, `WARP_ROWS`, `WGM`,
`N_SPLIT`, `ROTATE` are
`-D`-overridable, but only take effect with `-DHK_MULTI_CONFIG=0`, which pins the
build to exactly that tiling instead of the two-way selection above. `sweep.sh`
passes it; without it a sweep would measure the selection rule rather than the
configuration it names. `ABLATE_GLOBAL` / `ABLATE_LDS_READ` / `ABLATE_MMA` delete one pipeline
stage each — they make the answer wrong on purpose, and they are what located
the cost above. Run them with `QB_ARGS=--no-check`, or the correctness check in
`quickbench.py` reports WRONG and no timing comes out.

Things that were tried and did *not* work, recorded so they are not retried
blind:

1. **Double-buffering the operand tiles.** 48 VGPRs, occupancy 9 → 7, measures
   63/68 against 66/72. The zero-register version above is what replaced it —
   see "The LDS reads run underneath the WMMAs".
2. **Bigger register blocks.** Every 4×4-base-tile shape hits the 256-VGPR
   ceiling and spills, and the ones that fit are latency-bound. See the warp
   tile table above.
3. **Forcing occupancy with `__launch_bounds__`.** The second argument is min
   waves per EU in HIP, not min blocks per CU. Asking for 8 at `K_STEP=64`
   spills 21 registers and drops to 66/69. 208 VGPRs is the genuine floor for
   this tile; there is no free occupancy to take.
4. **Tuning `WGM`.** 1 / 4 / 8 / 16 measure 72/75, 73/78, 72/78, 71/76. Within
   noise between 4 and 8.

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
