# gfx11 correctness traps

Architecture-specific facts for gfx1100 (RDNA3) / gfx1201 (RDNA4), each paid for once on real hardware.
Not applicable to CDNA or NVIDIA; the method in `/kernel-bringup` is.

These do not fault, do not warn, and do not show up as a slowdown. They produce wrong
numbers, sometimes intermittently.

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
