# bf16 GEMM, fp32 accumulate — RDNA4 (gfx1200 / gfx1201)

## This has never been run

No gfx12 part was available when this was written. What exists is a kernel that
**compiles** for `--offload-arch=gfx1201`. It has not produced a correct result,
because it has not produced any result.

What is verified, because it is a property of the generated code:

| | gfx1100 (`../../../rdna3/gemm/bf16fp32`) | gfx1201 (here) |
|---|---|---|
| VGPRs | 164 | 128 |
| Occupancy | 9 waves/SIMD | 10 waves/SIMD |
| Spills / scratch | none | none |

The 36-VGPR drop is the RDNA4 operand fragment doing what it is supposed to: a
bf16 `rt_base` costs 4 VGPRs on gfx12 against 8 on gfx11, because the wave
halves split K instead of mirroring it.

What is not verified: everything else. Correctness rests on the operand layout
inferred in `include/rdna4/types/register/rt_base.cuh`, which was never measured
on hardware.

## Running it, when there is hardware

```bash
make
python3 test.py      # correctness against torch
python3 bench.py     # TFLOPs vs torch/hipBLASLt, and % of peak
```

`bench.py`'s `PEAK_TFLOPS` defaults to the RX 9070 XT's 194.6; change it for
another SKU.

Do the correctness run first and take a failure as evidence about the fragment
layout, not about the schedule. If `test.py` fails, run `tests/unit/rdna4` — it
localizes the same bug far better than a GEMM does — and re-measure the layout
with the `PROBE_GFX12_W32` variant of `tools/rdna-probes/wmma_layout.hip` before
changing anything here.

## Tuning

The tiling constants (`BLOCK_M`, `BLOCK_N`, `K_STEP`, `DOT_SLICE`, `NUM_WARPS`,
`WARP_ROWS`) are inherited verbatim from the gfx1100 kernel, where they were
swept. They are not tuned for gfx12. `sweep.sh` is the same sweep harness and
should be the first thing run after `test.py` passes.

Two specific things to look at, both of which the gfx12 fragment change should
move:

1. **A bigger register block.** The operand tiles cost 24 VGPRs here where they
   cost 48 on gfx1100, and the kernel sits at 128 VGPRs where 10 waves/SIMD
   holds to 144. There is 16 VGPRs of free headroom the gfx1100 shape could not
   use.
2. **Software-pipelining the inner loop.** On gfx1100 double-buffering the
   operand tiles measured *slower* (63/68 TFLOPs against 66/72) because it cost
   two occupancy steps, 9 waves to 7. Here the same change costs 24 VGPRs
   instead of 48, which is one step, 10 to 9. The reasoning is written out at
   the `dot_tile` lambda in `gemm.cpp`. This is the single most likely place for
   RDNA4 to beat a straight port.

Not on that list, and worth being explicit about: **fp8 will not make this
kernel faster.** `include/rdna4` exposes fp8 operand tiles and all four gfx12
fp8 WMMA opcodes, but the shape is 16x16x16 — the same as bf16, unlike CDNA
where the fp8 MFMA doubles K. One WMMA is one WMMA, so the math rate is
identical and the only wins are 12 VGPRs per operand tile instead of 24 and half
the LDS bytes. That can buy an occupancy step or a bigger block, which is a real
effect, but it is a second-order one; do not expect the 2x a CDNA fp8 kernel
gets. And the LDS read does not halve either: the 16-byte swizzle granule is
tuned for a `ds_load_b128`, and fp8's `ds_load_b64` reaches the same 4-cycle
floor for half the data. The derivation is at the top of
`include/rdna4/ops/warp/memory/tile/shared_to_register.cuh`.
