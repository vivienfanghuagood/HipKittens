---
name: kernel-attribution
description: >
  Find where a GPU kernel's time actually goes when hardware counters are
  unavailable or dead, by building compile-time ABLATE_* switches that delete
  one pipeline stage each, timing them all in one process, and reading the
  result as shares. Includes the switch that separates memory bandwidth from
  math, and the rule for converting a share into an expected speedup — which is
  where this measurement is normally misread. Use when a kernel is slower than
  its ceiling and the bottleneck is unknown, when rocprofv3 SQ counters read
  zero, or before committing to an optimisation that a share made look large.
  Usage: /kernel-attribution
allowed-tools: Read Write Edit Bash Grep Glob
---

# Kernel Attribution

## Pick the right skill first

| Question | Skill |
|---|---|
| Which pipeline stage owns the time? | **this skill** |
| Is my change actually faster? | `/kernel-ab-bench` |
| Did it spill or cost occupancy? | `/kernel-resource-check` |
| Is the remaining gap reachable at all? | `/kernel-bringup` Phase 0, ceiling probe |

## When to use this instead of a profiler

On gfx1100 under `rocprofv3`, **every `SQ` counter except `SQ_WAVES` reads 0** —
`SQ_INSTS_VALU`, `SQ_INSTS_LDS`, `LDSBankConflict`, `ALUStalledByLDS` included.
There is no instruction-level trace to read. Ablation is not a second-best
technique here; it is the only one.

## Workflow

### Step 1 — One `-D` per pipeline stage

Each switch deletes that stage and makes the output wrong on purpose:

```cpp
#ifndef ABLATE_SOFTMAX
#define ABLATE_SOFTMAX 0
#endif
…
#if !ABLATE_SOFTMAX
    col_max(m_new, s_t, m_old);
    sub_col(s_t, s_t, m_new);
    exp2(s_t, s_t);
    …
#endif
```

Delete *work*, not correctness checks. A switch that also removes a `s_waitcnt`
or changes the loop trip count is measuring something else.

### Step 2 — Add the one switch that separates bandwidth from math

Keep both matmuls, drop only the LDS reads that feed them: zero the operand
tiles once and reuse them. **No combination of per-stage switches can do this**,
because each removes a stage's reads *and* its math together.

This is the switch that settled whether attention's inner loop was LDS-bound or
WMMA-bound — stripping the LDS reads from both matmuls left ~55%, so it was
WMMA and conversion, not bandwidth. The tiling decisions downstream of that
answer were all different from what the per-stage table alone suggested.

### Step 3 — Build them all, time them all in one process

```bash
for v in base softmax staging qk pv ldsreads; do
  make TARGET=abl_$v EXTRA_HIPFLAGS="-DABLATE_${v^^}=1 -DTK_MODULE_NAME=abl_$v"
done
AB_CHECK=0 python3 ab.py abl_base abl_softmax abl_staging abl_qk abl_pv abl_ldsreads
```

`AB_CHECK=0` is correct **here and nowhere else** — these builds are wrong by
construction. The shares are ratios between variants, and ratios measured in
separate processes are the exact mistake `/kernel-ab-bench` exists to prevent,
so they still have to be interleaved in one process.

### Step 4 — Read the table

## The rule that makes this useful

> **A share measures work deleted, not time recoverable.**

Before acting on any row, ask: **what covers this stage today?**

This has been read wrong twice, at real cost:

- Staging ablated at **18.3%**. Halving the staged bytes (`NUM_WARPS=24`,
  doubling the Q tile) returned **~1%** — at 12 warps there are two workgroups
  per WGP, and the co-resident one was already covering staging. A week for 1%.
- The standing fix for causal's wasted staging, a narrower tile
  (`NUM_WARPS=8`), measured **−4.6%**. It had been assumed correct across three
  sessions on the strength of that same 18.3%.

The rows that *do* pay out are the ones you can prove are **identities**. The
softmax row's 8.4% was mostly a 64-register accumulator rescale that is exactly
a no-op whenever the running max did not grow. Guarding it on an exact equality
— not a tolerance — returned ~2% with **bit-identical output**. That is the
shape of a share worth chasing: not "this stage is big" but "this stage is
provably redundant under a condition that usually holds".

A useful discipline: write the expected speedup down *before* building the
optimisation, then compare. The gap between predicted and measured is the thing
you actually learn.

## Pitfalls

- **Shares do not sum to 100% and should not be normalised as if they do.**
  Stages overlap in a pipelined kernel; that overlap is the point.
- **An ablation that changes occupancy is measuring two things.** Deleting a
  stage frees registers, which can raise waves/SIMD and flatter the row. Run
  `/kernel-resource-check` on the ablated builds and note any that moved.
- **Deleting a stage can let the compiler delete more.** If the ablated output
  is never consumed, dead-code elimination removes the feeding loads too. Keep a
  dependency alive (write the tile to a dummy output) if the row looks too good.
- **Record the negative results in the README.** Each disproven lever, with its
  mechanism, so it is not retried blind — including by you, six weeks later. See
  `/kernel-writeup`, section 9.
