---
name: kernel-resource-check
description: >
  Gate or diff per-kernel GPU resource usage (VGPRs, SGPRs, register spills,
  ScratchSize, LDS, occupancy) for a hipcc build, across EVERY template
  instantiation, using its `hip_resources.py` helper. Compile-only: no GPU, no
  profiler, no benchmark, runs in seconds. Applies the gfx1100 24-register
  allocation granule so occupancy is charged on the rounded count, not the raw
  one. Use after any tiling, unroll, epilogue or template change, when asked
  whether a change spilled or cost occupancy, before spending a benchmark run,
  or when a hand-scheduled kernel produces silently wrong results.
  Usage: /kernel-resource-check [<build-dir-or-log>]
allowed-tools: Read Write Edit Bash Grep Glob
---

# Kernel Resource Check

Read the compiler's own resource remarks for every kernel a build produced, and
turn them into a pass/fail rather than a number that scrolls past.

## Pick the right skill first

| Question | Skill |
|---|---|
| Did my change spill, cost occupancy, or grow LDS? | **this skill** (compile-only, seconds, no GPU) |
| Is variant A faster than variant B? | `/kernel-ab-bench` (needs a GPU, one process) |
| Which pipeline stage is the time in? | `/kernel-attribution` (ablation switches) |
| What does this hardware actually do? | `/rdna3-kernel-facts` |
| How do I approach a new kernel at all? | `/kernel-bringup` (the method these four serve) |

This skill measures **resources, not time**. A clean result does not mean the
kernel got no slower; it means register/LDS/spill pressure did not move. A
*dirty* result is a cheap, strong signal worth acting on before profiling.

## Why not `grep -E 'VGPRs|Scratch'`

Three things grep gets wrong, each of which has cost real time here:

- **It does not tell you which instantiation a line belongs to.** A templated
  kernel emits one block per instantiation. Reading only the shipped shape's
  block is how a build that was never timed sat 5% off.
- **`ScratchSize != 0` is a correctness failure, not a slowdown**, in any kernel
  that hand-manages `s_waitcnt` — which is all of `kernels/rdna3`. The compiler's
  inserted reloads interleave with the manual waits and the result is silently
  wrong. It needs to fail a gate, not print a number.
- **Occupancy is charged on the granule-rounded VGPR count.** On gfx1100 the
  granule is 24: **240 VGPRs buys 6 waves/SIMD, 241 buys 5.** A budget read off
  the raw count is wrong at exactly the boundary you are tuning against. The
  model is verified against LLVM, not asserted — see Pitfalls.

## Workflow

### Step 1 — Capture a build log

The remark is already on for `BUILD_MODE=pyext` (see `kernels/common.mk`); it is
`-Rpass-analysis=kernel-resource-usage`. Capture a **full** rebuild:

```bash
cd kernels/rdna3/attn/fwd
make clean && make 2>&1 | tee /tmp/res-after.log
```

`make clean` is **required, not optional** — an up-to-date target emits no
remarks at all, which is indistinguishable from a clean build. The tool reports
that case as exit 2 rather than a pass.

### Step 2 — Gate it

```bash
python3 ${CLAUDE_SKILL_DIR}/scripts/hip_resources.py check /tmp/res-after.log \
    --arch gfx1100 --min-occupancy 6 --expect-kernels 4
```

`ScratchSize` and spills are gated unconditionally. `--max-vgprs`,
`--min-occupancy`, `--max-lds` are opt-in. `--expect-kernels N` reports
**NOT TRUSTWORTHY** rather than a pass when the instantiation count is not what
you expect, which catches a partial rebuild.

### Step 3 — Diff against the previous build, when tuning

```bash
git stash && make clean && make 2>&1 | tee /tmp/res-before.log && git stash pop
python3 ${CLAUDE_SKILL_DIR}/scripts/hip_resources.py diff /tmp/res-before.log /tmp/res-after.log
```

Either side may be a log or a `--json` snapshot, so a baseline is captured once
and reused:

```bash
… hip_resources.py capture /tmp/res-before.log --json /tmp/base.json
… hip_resources.py diff /tmp/base.json /tmp/res-after.log
```

### Step 4 — Read the occupancy arithmetic before choosing a tile

```bash
python3 ${CLAUDE_SKILL_DIR}/scripts/hip_resources.py explain --arch gfx1100 --vgprs 240
```

Prints the waves/SIMD ladder and how many registers are left before the next
step down. Use it when sizing a tile, not after.

## Reading the output

Real output, the shipped attention kernel's four instantiations:

```
kernel                                             vgpr   occ   scr spill     lds  hdrm
-----------------------------------------------------------------------------------------
~128, 16, 32, 12, 16, 2, 2>, true, mi...(micro_globals)     240     6     0     0     0     0
~128, 16, 32, 12, 16, 2, 2>, false, m...(micro_globals) 227->240     6     0     0     0    13
~64, 16, 32, 12, 16, 2, 2>, true, mic...(micro_globals) 170->192     8     0     0     0    22
~64, 16, 32, 12, 16, 2, 2>, false, mi...(micro_globals) 155->168     9     0     0     0    13

  ~ = void micro_tk<config<
```

Read that: the `HEAD_DIM=128, CAUSAL=true` build sits at **exactly** 240 with
`hdrm` 0 — one more register costs a wave. The non-causal build is charged the
same 240 despite using 227, so it has 13 free at no occupancy cost. Neither
fact is visible in the raw numbers.

Names are elided at their **common prefix**, not a fixed window, because
truncating a fixed window renders two different instantiations as the same
string — which would defeat the purpose of the tool.

| Column | What it is |
|---|---|
| `vgpr` | `a->b` means raw `a`, charged as `b` after rounding to the granule. **Occupancy is decided by `b`.** |
| `occ` | The compiler's number. **Register-only** — LDS can bind below it. |
| `scr` | `ScratchSize` bytes/lane. Non-zero is a correctness failure here, not a slowdown. |
| `spill` | VGPR + SGPR spills. |
| `lds` | Static LDS bytes/block. gfx1100 gives 64 KB per workgroup. |
| `hdrm` | VGPRs addable before occupancy drops a step. `0` means you are exactly at the edge. |

### Exit codes

| Code | stdout | Meaning | What to do |
|---|---|---|---|
| `0` | `RESULT: OK` | Nothing triggered | Proceed |
| `1` | `RESULT: REGRESSION` | A claim about the code | Act on it — this is a real finding |
| `2` | `RESULT: NOT TRUSTWORTHY` | A claim about the tool's confidence | Fix the inputs and rerun. **Do not report a clean result.** |

Exit 2 covers: no remarks in the log, a truncated log, a missing file, a
snapshot captured for a different arch, a kernel present on only one side of a
diff, and an instantiation count that misses `--expect-kernels`. Everything that
would leave the answer partial reports 2 and never 1.

The **last** stdout line is always the verdict and matches the exit code; script
against that line, not a fixed line count.

## Acting on a result

| What moved | Usual cause |
|---|---|
| `scr` / `spill` off zero | The tile crossed the register budget. In a hand-scheduled kernel, **stop — the kernel is now wrong**, not slow. |
| `vgpr` up across a granule boundary | More live values: a bigger tile, deeper prefetch, an unrolled branch. A branch moved inside an unroll once took 209 → 256 + 8 spills. |
| `lds` up across an occupancy step | Bigger staged tiles or added double buffering. At 12 warps on gfx1100 there are two workgroups per WGP; losing that is usually worse than what the extra buffer bought. |
| `occ` down, everything else flat | Granule rounding. Check `hdrm` — you may be one register over a step. |

If resources regressed but the kernel is not measurably slower, **say that
rather than "fixing" it**: these are proxies for occupancy, not timings.
Confirm with `/kernel-ab-bench` before reworking anything.

## Pitfalls

These silently produce a *confident wrong answer*:

- **A partial rebuild emits remarks only for what recompiled.** Always
  `make clean`. The tool cannot distinguish "this kernel is fine" from "this
  kernel was not rebuilt" except through `--expect-kernels`.
- **The reported occupancy is register-only.** A kernel at 49 KB LDS gets one
  workgroup per 64 KB allocation regardless of what the `occ` column says. Read
  the `lds` column alongside it; the GEMM's binding limit was LDS, not registers.
- **The granule is per-architecture and is measured, not read off a datasheet.**
  `gfx1100` is the only entry in the tool's `ARCH` table, verified against LLVM
  across 42 compiled points including 6 that discriminate granule 24 from
  granule 8 — LLVM agreed with 24 at all of them. An unknown arch degrades to
  reporting the compiler's own occupancy and skipping the arithmetic; it does
  not guess. Adding an arch means running that sweep, not editing the table.
- **`lds` is static LDS only.** Dynamically-sized shared memory reports 0.
- **Demangling is best-effort.** Without `llvm-cxxfilt`/`c++filt` on PATH the
  names stay mangled; the numbers are unaffected.
