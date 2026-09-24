# gfx11 / gfx12 gotchas

Architecture-specific. Read only if the target is RDNA3 (gfx1100) or RDNA4
(gfx1201); the phases in `SKILL.md` are architecture-independent, this file is
not. Everything here was paid for once.

## Correctness

**`s_waitcnt` carries no register dependence.** It is a scalar instruction with
no operands, so the compiler is free to hoist a WMMA that consumes LDS data
*above* the wait that guarantees the data arrived. Symptom: results randomly
wrong, `ScratchSize=0`, no diagnostic, no pattern. Fix: after the wait, an empty
`volatile` asm that names the fragment registers as operands, which gives the
scheduler the dependence the wait does not carry.

**`ScratchSize` must be 0 whenever `s_waitcnt` is hand-managed.** A spill is not
a slowdown, it is silent corruption — the compiler's inserted reloads interact
with the manual waits. Check `-Rpass-analysis=kernel-resource-usage` after
*every* edit, and read **every** template instantiation, not the one you are
benchmarking. In the attention kernel only one of four instantiations sat at the
edge (240 VGPRs) and it was not the default one.

**VGPR allocation granule is 24 on gfx1100.** Occupancy 6 waves/SIMD needs
**≤240**, not ≤256. 241 VGPRs silently gives 5 waves. Budget to the granule.

**gfx12 splits `s_waitcnt`.** `vmcnt`/`lgkmcnt` become separate instructions;
code that packs both into one `s_waitcnt` will not assemble. Guard by target.

**fp8 is OCP on gfx12, fnuz on CDNA.** Same name, different bias and NaN
encoding. Converting with the wrong one produces plausible, wrong numbers.

**Device-side flags are unreliable across ranks.** A peer's write does not
invalidate your L2, and no shader instruction flushes L2 on gfx1100 — only the
first dispatch after a host-side sync sees it. Cross-rank handshakes must go
through host flags. Related: `clock64()` is only 20 bits here, so any in-kernel
timeout built on it is fiction.

## Performance

**Only `row`-layout bf16 operands get the vectorised `ds_read_b128`** (2
instructions per base tile). `col` layout degrades to 16 scalar `ds_read_u16` —
8× the instruction count. This single fact determines the tiling: arrange LDS so
every matmul operand is `row`, including by staging V transposed rather than
transposing after the load.

**`swap_layout` on gfx11 moves data** (~20 lane swaps per base tile), it is not
a relabel. "Load then transpose" is not free; stage it in the layout you need.

**Inline asm is mandatory for LDS access** in the hand-scheduled inner loop —
the intrinsics do not let you place the waits. The buffer resource descriptor
word that works here is `0x31004000`.

**No global→LDS DMA on gfx11.** Every staged byte goes through VGPRs, so a
"double-buffered LDS" pipeline is really a register-prefetch pipeline; budget
registers for it up front.

**WMMA requires operand mirroring across lanes `l` and `l^16`** (wave32,
wave-half mirroring). Violating it does not fault; it returns undefined data
that can look self-consistent. Any layout probe must mirror, or it will
confidently report a wrong layout.

**64 KB LDS per workgroup**, not 160 like CDNA4. Any CDNA kernel's tiling has to
be re-derived, not ported — this is usually the constraint that forces a
different algorithm shape rather than a different constant.

**Hardware counters are dead.** Under `rocprofv3` on gfx1100 every `SQ` counter
except `SQ_WAVES` reads 0 — `SQ_INSTS_VALU`, `SQ_INSTS_LDS`, `LDSBankConflict`,
`ALUStalledByLDS` included. Attribution must come from ablation switches
(`measurement.md` §3).

**The part is clock- and power-limited under any load**: 122.6 TFLOPs
architectural, ~100.5 measured back-to-back WMMA at 2.18 GHz / 101 W. Use the
measured number as the denominator.

**Peer bandwidth is bimodal per process launch** (~15.5 or ~24–27 GB/s, nothing
between, stable within a process). Communication and compute must be timed in
the same process or the comparison describes two different machines.

## Benchmarking on this hardware

**Compare against rocBLAS/hipBLASLt tuned, not torch's default path.** Torch's
default is not the vendor's best and beating it proves nothing.

**Cross-process A/B is invalid on `wx-ms-w7900d-0043`** — it manufactures 13–16%
differences. See `measurement.md` §1.

**Infinity Cache will fake a decode benchmark.** Re-using one weight buffer
measures cache bandwidth and reports figures above the card's memory peak.
Rotate buffers.

## Node discipline (`wx-ms-w7900d-0043`)

The node is shared and has been taken down once by a kernel.

- **GPUs 1/2/4 belong to other tenants. Use GPU 0 or 3 only**, pinned with
  `HIP_VISIBLE_DEVICES`.
- Compute the memory budget **before** spawning ranks. A cgroup OOM that
  interrupts a collective takes the whole machine down, not the process.
- Wrap every launch in an external `timeout --signal=KILL`, and bound every
  in-kernel wait (not with `clock64()`, see above).
- Never rebuild a `.so` while a process has it mapped.
- Serialise benchmark jobs; a co-tenant's load invalidates the numbers.
