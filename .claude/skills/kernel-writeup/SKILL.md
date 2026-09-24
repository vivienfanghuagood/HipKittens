---
name: kernel-writeup
description: >
  Write or review the README for a high-performance GPU kernel, using the
  twelve-section structure the HipKittens RDNA3 kernels share: derivation from
  hardware constraints, measured facts, the bug that took longest, the
  register/LDS budget, the attribution table with its caveat, coverage, the
  integration surface, results with baseline versions and ceiling percentages,
  levers that were measured and are NOT levers, explicit scope limits,
  reproduction commands, and a ranked list for whoever continues. Use when
  documenting a finished or in-progress kernel, reviewing a kernel README for
  what it is missing, or recording a negative result so it is not retried blind.
  Usage: /kernel-writeup [<kernel-dir>]
allowed-tools: Read Write Edit Bash Grep Glob
---

# Kernel Write-up

## Pick the right skill first

| Question | Skill |
|---|---|
| How do I document this kernel? | **this skill** |
| What should the results table be? | `/kernel-ab-bench`, `/kernel-resource-check` |
| What should the "where the time goes" table be? | `/kernel-attribution` |
| What goes in the hardware-facts section for this hardware? | `/rdna3-kernel-facts` |

The three kernels in `kernels/rdna3/` (GEMM, fused GEMM+collective, attention)
share one README structure. It is not a template for its own sake: each section
is there because leaving it out cost something.

Write it **as you go**, not at the end. The sections that pay for themselves —
the negative results and the bug — are exactly the ones you cannot reconstruct
later, because by then you remember the conclusion and not the measurement.

## The sections, in order

**1. Derivation — why the kernel has the shape it has.**
Start from the constraint, not the code. "The accumulator is `col` layout, so
reductions over kv land on the element axis, so S must be stored transposed" is
a derivation; "we store S transposed" is a note. The next person needs to know
which choices are forced by hardware (cannot be revisited) and which are
arbitrary (can). Say explicitly why the reference kernel for the *other*
architecture could not be ported — that is the first question anyone asks.

**2. Hardware facts.**
The measured numbers the design rests on: real ceiling vs datasheet, LDS per
workgroup, register file and allocation granule, what the probes returned.
Include the probe source path. These are the facts that would have to change for
the design to change.

**3. The bug.**
One section, in full, for the failure that took the longest. Not an apology —
the mechanism. The RDNA3 attention one: `s_waitcnt` carries no register
dependence, so the compiler hoisted WMMA above the wait; results were randomly
wrong with `ScratchSize=0` and no diagnostic anywhere. The distributed one: a
branch moved inside an unrolled loop pushed 209 VGPRs → 256 + 8 spills, and
every `s == N_SPLIT-1` tile came out NaN.

These sections exist because the same bug class recurs across kernels on the
same architecture, and the second occurrence is recognised in minutes instead of
days — but only if the mechanism was written down, not just the fix.

**4. Tiling and the register/LDS budget.**
The table: every resident tile, its shape, its base-tile count, its VGPR cost,
and the total. Then the measured `ScratchSize` and occupancy for **every**
template instantiation. A budget that only adds up for the shipped shape is
a budget that will break when someone adds a shape.

**5. Where the time goes.**
The ablation table, with the caveat attached to it in the same section: these
are *shares of work deleted*, not *time recoverable*. Without the caveat in the
table's own section, the next reader will treat the largest row as headroom.
See `/kernel-attribution`.

**6. Coverage.**
What shapes, dtypes, layouts, and flags are actually tested — and the shapes
that divide nothing, which is where correctness bugs live.

**7. The integration surface.**
The torch op / library entry point, its exact signature, and what it falls back
to. If it is a drop-in, state which signature it matches and which argument
combinations are forwarded to the original rather than handled.

**8. Results.**
Every baseline, its version, JIT-or-AOT, one process, one run. Speedups **and**
percentage of the measured ceiling. The baseline's run-to-run spread. Where the
vendor library has freedom you do not, both columns.

**9. Levers that were measured and are not levers.**
The highest-value section in the document and the one most often skipped. Each
entry: the lever, what it was expected to do, what it measured, and *why* — the
mechanism, so a reader can tell whether their situation differs. Examples worth
the shape they take:

- "occupancy 6 → 8: no change; the kernel is WMMA-issue-bound, not
  latency-bound, so more waves have nothing to hide."
- "`NUM_WARPS=24` halves staged bytes and returns ~1%, not the 18% the ablation
  suggested, because the co-resident workgroup already covers staging."
- "`NUM_WARPS=8` under causal: **−4.6%**" — killing a fix that had been assumed
  correct across three sessions.
- "`Q_BLOCK=32` at `HEAD_DIM=128` is architecturally dead" with the LDS budget
  that proves it, not just a slower number.

Without this section every lever gets retried, blind, by the next person —
including you, six weeks later.

**10. What this is not.**
Explicit scope limits, stated as flatly as the results. No backward pass. No
paged KV. No fp8. No end-to-end pipeline validation, and why (the pod has no
network, so no framework could be installed). The purpose is that nobody builds
on a capability that was never measured. A reader who discovers a limit
themselves stops trusting the results section too.

**11. Reproducing.**
Exact commands, including the A/B recipes for each knob, so a claim in section 8 or 9
can be re-run rather than believed.

**12. Notes for anyone continuing.**
The ranked list of what is left, each with what is known about it — including
the ideas that are *structural* rather than knobs, so they are not mistaken for
quick wins. Mark the ones already disproven with a pointer to section 9.

## Two rules for the prose

- **Every number carries its conditions.** Shape, dtype, causal flag, process,
  date. A number without them will be quoted later against a different setup.
- **Distinguish measured from inferred, in the sentence.** "Confirmed with the
  profiler, not inferred" is worth writing every time it is true, because the
  reader cannot otherwise tell, and the one inferred claim in a document of
  measured ones is what eventually breaks.
