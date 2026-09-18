# bf16 GEMM, fp32 accumulate — RDNA3 (gfx1100)

C = A · Bᵀ, bf16 in, fp32 accumulate, bf16 out. B is passed pre-transposed as
`(n, k)`, which keeps both shared loads contiguous and makes `mma_ABt` the right
primitive.

This is the kernel the RDNA3 port was validated against. Unlike
`kernels/rdna4/`, everything below is measured.

## Measured

W7900D (gfx1100, 96 CU, 122.6 TFLOPs bf16 peak), ROCm 7.2.4. TFLOPs, HipKittens
vs torch on hipBLASLt, same shapes, same machine:

| shape (M×N×K) | HK | torch | peak reached |
|---|---|---|---|
| 4096³ | 67 | 69 | 55% |
| 8192×8192×4096 | 71 | 65 | 58% |
| 8192×4096×2048 | 71 | 66 | 58% |
| 2048³ | 46 | 55 | 38% |

Small shapes lose, and the reason is occupancy rather than the inner loop: 2048³
is 256 workgroups over 96 CUs, and there is no split-K here.

The ceiling for this schedule is 98/102 TFLOPs — that is `ABLATE_GLOBAL=1
ABLATE_LDS_READ=1`, i.e. the WMMA issue rate alone with both memory stages
deleted. So the gap to peak is roughly half instruction issue and half LDS read.

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

`BLOCK_M`, `BLOCK_N`, `K_STEP`, `DOT_SLICE`, `NUM_WARPS`, `WARP_ROWS` are
`-D`-overridable; the defaults (128, 128, 32, 16, 8, 4) are what the sweep
picked. `ABLATE_GLOBAL` / `ABLATE_LDS_READ` / `ABLATE_MMA` delete one pipeline
stage each — they make the answer wrong on purpose, and they are what located
the cost above.

Two things that were tried and did *not* work, recorded so they are not retried
blind:

1. **Software-pipelining the LDS reads.** Double-buffering the operand tiles
   costs 48 VGPRs, occupancy goes 9 → 7 waves/SIMD, and it measures 63/68 —
   slower. At 9 waves the SIMD was already covering LDS latency by switching
   waves. `load_async()` and `lds_wait<N>()` are kept in the library because
   they are the right primitives for a low-occupancy schedule, where the trade
   goes the other way.
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
