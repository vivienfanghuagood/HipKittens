# gfx11 performance facts

Architecture-specific facts for gfx1100 (RDNA3) / gfx1201 (RDNA4), each paid for once on real hardware.
Not applicable to CDNA or NVIDIA; the method in `/kernel-bringup` is.

These decide the shape of the tiling, not just its constants.

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
