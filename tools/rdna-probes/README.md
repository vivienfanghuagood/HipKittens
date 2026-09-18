# RDNA hardware probes

Standalone HIP programs that measure what the RDNA3 / RDNA4 port is built on.
They do not use the library (except where noted) and they do not need it to be
correct — that is the point. When something in `include/rdna3` or
`include/rdna4` disagrees with the hardware, these are what tell you which side
is wrong.

```bash
./build.sh wmma_layout.hip gfx1100        # compile
./wmma_layout                              # run on the card
./build.sh mma_gate.hip gfx1201 --asm      # or just look at the ISA
./run_features.sh gfx1201                  # compile-probe the feature matrix
```

`build.sh <file.hip> [arch] [--asm]` — `arch` is `gfx1100` or `gfx1201` and sets
`-DKITTENS_RDNA3` / `-DKITTENS_RDNA4`. `--asm` stops at assembly, which is how
the gfx12 claims were checked without a gfx12 card.

## What each one establishes

| | |
|---|---|
| `wmma_layout.hip` | **The foundational measurement.** One-hot probes plus gauge analysis, recovering the physical register layout of every WMMA operand and accumulator. Everything in `rt_base_coord()` comes from here. Has a `PROBE_GFX12_W32` variant, written but never run. |
| `features.hip` + `run_features.sh` | The hardware feature matrix — `vmem-to-lds`, `s_setprio`, `sched_group_barrier`, `permlane*`, WMMA variants, SWMMAC, and the maximum static LDS allocation, found by bisecting the compiler error. |
| `mma_gate.hip` | Phase 3 gate: does the `rt_base` layout derivation agree with real WMMA output, for a single 16×16×16 and for blocked GEMMs. |
| `mem_gate.hip` | Phase 4 gate: every memory path round-tripped — global↔shared, global↔register, shared↔register, tiles and vectors, both layouts. |
| `red_gate.hip`, `conv_gate.hip` | Phase 5 gates: tile reductions, and the layout conversions that have to move data across the wave halves. |
| `srsrc.hip`, `srsrc2.hip`, `srsrc3.hip` | How the gfx10+ buffer resource descriptor actually behaves. CDNA's `0x110000` config word is a gfx9 format; these sweep word 3 against in-bounds and out-of-bounds accesses and land on `0x31004000`, valid for gfx11 and gfx12 both. |
| `empty.hip` | The minimal "does the toolchain work for this target" file. |

## One warning, learned the hard way

A gfx11 WMMA operand is **mirrored across the wave halves** — lanes `l` and
`l+16` must hold identical data. The first version of `wmma_layout.hip` set a
one-hot in a single lane without mirroring it to `l^16`. That violates the
hardware constraint, so what came back was undefined behaviour, and it did not
look like garbage: it looked like a coherent layout in which half the lanes
contributed nothing. The probe now sets `l` and `l^16` together, and feeds
operands from global memory so that dynamic `insertelement` codegen cannot
confuse the picture either.

If a probe result surprises you, suspect the probe first.
