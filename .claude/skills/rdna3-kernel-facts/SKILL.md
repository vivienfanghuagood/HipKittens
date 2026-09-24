---
name: rdna3-kernel-facts
description: >
  Hardware facts for writing HipKittens kernels on AMD RDNA3 / RDNA4 (gfx1100
  W7900D, gfx1201): the s_waitcnt scheduling hazard that silently reorders WMMA,
  the 24-register VGPR allocation granule, why only row-layout bf16 operands get
  vectorised ds_read_b128, the 64 KB LDS ceiling that makes CDNA tilings
  unportable, dead SQ performance counters, bimodal peer bandwidth, and the
  operating rules for the shared wx-ms-w7900d-0043 node. Use when writing,
  porting, or debugging a kernel for gfx1100/gfx1201, when a hand-scheduled
  kernel is intermittently wrong, when a CDNA kernel will not port, or before
  running anything on that node.
  Usage: /rdna3-kernel-facts
allowed-tools: Read Bash Grep Glob
---

# RDNA3 / gfx11 Kernel Facts

Architecture-specific. **Everything here is measured on our hardware**, not read
off a datasheet — where a datasheet disagrees, the datasheet is wrong for this
part. The *method* for kernel work is architecture-independent and lives in
`/kernel-bringup`; this file is the layer underneath it.

## Pick the right skill first

| Question | Skill |
|---|---|
| What does this hardware actually do? | **this skill** |
| How do I approach the kernel at all? | `/kernel-bringup` |
| Did my change spill or cost occupancy? | `/kernel-resource-check` |
| Is variant A faster than B? | `/kernel-ab-bench` |
| Where does the time go? | `/kernel-attribution` |

## Load the reference you need

| File | Load it when |
|---|---|
| `references/correctness.md` | A kernel is intermittently wrong, produces NaN, hand-manages `s_waitcnt`, spills, or does a cross-rank handshake. **Read before debugging anything numerical.** |
| `references/performance.md` | Choosing a tiling or LDS layout, porting a CDNA kernel, deciding an operand layout, or setting up a benchmark. |
| `references/node-discipline.md` | **Before running anything on `wx-ms-w7900d-0043`.** Non-optional. |

## The five that change a design

If you read nothing else:

1. **`s_waitcnt` carries no register dependence.** The compiler will hoist a
   WMMA that consumes LDS data above the wait that guarantees it arrived.
   Symptom: randomly wrong results, `ScratchSize=0`, no diagnostic.
   → `references/correctness.md`
2. **The VGPR allocation granule is 24.** Occupancy 6 waves/SIMD needs **≤240**,
   not ≤256; 241 silently gives 5. Verified against LLVM across 42 compiled
   points. → `/kernel-resource-check explain`
3. **Only `row`-layout bf16 operands get `ds_read_b128`.** `col` degrades to 16
   scalar `ds_read_u16` — 8× the instructions. This single fact determines the
   LDS layout, and hence the whole tiling.
   → `references/performance.md`
4. **64 KB LDS per workgroup**, not CDNA4's 160. Any CDNA tiling must be
   re-derived rather than ported; this usually forces a different algorithm
   shape, not a different constant. → `references/performance.md`
5. **Hardware counters are dead.** Every `SQ` counter but `SQ_WAVES` reads 0
   under `rocprofv3`. Attribution has to be done by ablation.
   → `/kernel-attribution`

## Adding a fact here

Only if it is measured, and record how. A fact in this file is trusted without
re-checking, so an inferred one does more damage than none. State the probe or
the sweep that established it, the way the granule entry states 42 points and 6
discriminating cases.
