---
name: kernel-bringup
description: Method for writing a high-performance GPU kernel that has to beat a vendor library or a framework's existing backend — GEMM, attention/SDPA, fused collectives, or anything else compute-bound. Use when starting a new kernel, porting one to an architecture it was not written for, tuning one that is close but losing to a baseline, or deciding whether a remaining gap is reachable at all. Covers: picking the target shape and the honest baseline, probing hardware for the facts that change the algorithm's shape, correctness and resource gates that must fire before any timing, per-stage attribution when hardware counters are dead, single-process A/B measurement, and the negative-results discipline that stops a lever being retried blind. Triggers on kernel, tile, WMMA/MFMA/tensor core, occupancy, VGPR/register pressure, LDS/shared memory, bank conflict, roofline, TFLOPs, warp/wave/workgroup, HIP/CUDA/Triton kernel tuning, hipBLASLt/rocBLAS/cuBLAS/aotriton comparison, flash attention, collective fusion.
---

# Bringing up a high-performance kernel

Distilled from three kernels that shipped on gfx1100 (W7900D) in this repo — a
bf16 GEMM at 77% of the measured WMMA ceiling, a fused GEMM→all-reduce/
reduce-scatter at 1.22–1.59× hipBLASLt+RCCL, and an SDPA forward at 2.8–3.3×
the backend every Radeon framework actually dispatches. All three converged on
the same sequence, and every rule below is here because breaking it cost real
time on one of them.

**The organising idea: at every phase, the expensive failure is not being slow,
it is being confidently wrong.** A wrong baseline, a wrong denominator, a
cross-process A/B, an ablation share read as recoverable time, a sweep that
timed the wrong template instantiation — each of these produced a number that
looked authoritative and pointed the next week of work in the wrong direction.
The gates exist to make wrongness loud and early.

## Phases

Each phase has a gate. Do not start the next one until it fires.

### Phase 0 — Aim. Three numbers before a line of kernel code.

1. **Target shapes, from the consumer's real config.** Not round squares. Read
   the model's `config.json` / the serving stack's launch shapes and derive the
   actual tensor dimensions, including the awkward ones. For MiniMax H3 that
   was `num_attention_heads=56, attention_head_dim=128`, no
   `num_key_value_heads` (so MHA, not GQA), bidirectional (so non-causal), and
   49 920 tokens for a 480p/5s clip. Those four facts deleted half the design
   space before anything was written. Ask also: what fraction of end-to-end time
   is this op? (>85% for attention at those lengths — that is the whole
   justification for the work.)

2. **The baseline, verified by profiler, not assumed.** Find out what the user's
   stack *actually dispatches*, then check it. `F.scaled_dot_product_attention`
   on gfx1100 lands in aotriton `attn_fwd` for both the FLASH and the EFFICIENT
   backend — confirmed with the profiler; a warning in the log claimed flash was
   disabled and was a red herring. torch on ROCm defaults to hipBLASLt, and on
   gfx1100 hipBLASLt is up to **33% slower than rocBLAS**: benchmarking against
   torch's default there is benchmarking the library the vendor has not tuned.
   Take the strongest honest baseline, and where the vendor has freedom you do
   not (e.g. it picks any of NN/NT/TN/TT and you implement one), report **both**
   columns — its best, and it restricted to your problem.

3. **The ceiling, measured.** Never the datasheet. 96 CU × 512 FLOP/clk ×
   2.495 GHz boost = 122.6 TFLOPs; back-to-back WMMA with no memory at all
   measures **100.5** at 2.18 GHz and 101 W, because the part is clock- and
   power-limited under any real load. Percentages against the datasheet
   understated by 18%. Write the probe (`tools/rdna-probes/wmma_peak.hip`),
   sample `rocm-smi -c -P` during the run, and measure the secondary ceilings
   too — LDS bytes/clk decided where the GEMM's wall was.

**Gate:** you can state target shape, baseline TFLOPs, and reachable ceiling,
each with the command that produced it. A speedup claim without all three is not
yet a claim.

### Phase 1 — Probe. Find the facts that change the algorithm's *shape*.

Documentation and ported code are not ground truth; a probe is. Look
specifically for facts that invalidate the reference implementation you were
going to transcribe. On gfx11 there were three, and each one restructured the
kernel rather than adjusting a constant:

- WMMA operands are **mirrored across wave halves**, so a bf16 operand tile
  costs the same registers as an fp32 accumulator — the reverse of CDNA, where
  accumulators dominate. This is why the GEMM cannot grow past 128×128 and why
  textbook operand double-buffering measures *slower*.
- **No global→LDS DMA** on gfx11/gfx12 at all. Every prefetched byte passes
  through a VGPR and into the wave's instruction stream, so the async-copy
  pillar of the upstream design has no hardware under it.
- Only **`row`-layout bf16 operands** reach vectorized `ds_read_b128`; the `col`
  fallback costs 8× the instructions. That single table entry is why V is
  transposed on the way into LDS.

**A probe can be wrong, and a wrong probe reads as a coherent answer.** The
first layout probe wrote a one-hot into a single lane without mirroring it
across `l` and `l^16`, violating the hardware's own constraint, and read back
undefined behaviour that looked like a clean, self-consistent — and completely
wrong — layout. Design probes so a violated precondition fails loudly.

**Gate:** a written list of 3–5 facts, each with the probe that established it,
and for each one a sentence on what it forbids.

### Phase 2 — Derive, don't port.

Let the layout and reduction tables pick the algorithm. In the attention kernel
this was mechanical once the tables were in front of me: the fp32 accumulator is
`col` layout, so in `Sᵀ` the kv axis is the cheap *element* axis and q is the
lane axis, so online-softmax's max and sum over kv become eight in-lane steps
plus one `permlanex16` instead of a four-step butterfly — therefore **S is
computed transposed**, therefore **O accumulates transposed**, therefore V must
be `[D, KV]` in LDS. Three structural decisions, none of them a choice, all
falling out of two tables.

Write the derivation down *before* coding. It is the part that does not survive
in the source, it is what makes the kernel explicable later, and if it cannot
be written the design is not yet understood.

Budget registers and shared memory on paper in the same step. At
`Q_BLOCK=16, D=128`: accumulator 64 VGPRs + Q 64 + scores 32 + operands 32 ≈
224–250 of 256 — which said at design time that the shape was viable and that
`Q_BLOCK=32` never would be.

**Gate:** a derivation someone else could follow, and a register/LDS budget that
closes.

### Phase 3 — Correct before fast. Gates that must fire on every build.

These are cheap and they are the difference between a bug found in an hour and
one found in a week.

1. **An independent reference with a magnitude guard.** fp32, chunked so long
   sequences fit. Relative tolerance sized from the arithmetic (bf16 double
   rounding + long summation → 5e-2), **plus a check that the reference itself
   has magnitude**, so an all-zero output cannot pass.

2. **Shapes that divide nothing.** 4097, 5000, 12345, 193 alongside the round
   ones, plus batch, plus every template instantiation and every fallback
   branch. 24 shapes for attention, 30 for the distributed kernel — every one
   checked elementwise *before* it was timed.

3. **`ScratchSize == 0`, as a hard gate.** This is not a performance check. A
   kernel that hand-manages `s_waitcnt` does not survive spilling: the
   spill/reload traffic is counted by *the same hardware counters* the waits
   use, so the waits stop meaning what they were written to mean. The
   distributed epilogue's symptom was every `s == N_SPLIT-1` tile coming out
   NaN — one quarter of the output, silently, on exactly the shapes you would
   benchmark. Run `-Rpass-analysis=kernel-resource-usage` on every build.

4. **Read the occupancy line of every instantiation, not the one you are
   timing.** Template parameters (`CAUSAL`, `HEAD_DIM`) produce several kernels
   per `make`. A tiling sweep that only ever timed non-causal left the causal
   build one VGPR granule over an occupancy cliff and cost 5% invisibly. Know
   your allocation granule: on gfx1100 it is 24, so 6 waves/SIMD needs ≤ 240
   registers, not ≤ 256, and 241 costs a wave.

**Gate:** all shapes pass, `ScratchSize` 0 and spill 0 in *every* instantiation.

### Phase 4 — Attribute before tuning.

Hardware counters may not exist. On this part every `SQ` counter except
`SQ_WAVES` reads 0 under `rocprofv3` — `SQ_INSTS_VALU`, `LDSBankConflict`,
`ALUStalledByLDS`, all zero. So build the attribution in:

- **`ABLATE_*` compile-time switches, one per pipeline stage**, each deleting a
  stage and making the answer wrong on purpose.
- Include one ablation that separates *bandwidth* from *math* — e.g. keep both
  matmuls but drop only the LDS reads feeding them (zero the operand tiles once
  and reuse). No combination of per-stage switches can do this, because each
  removes a stage's reads and its math together. This is the measurement that
  said the attention inner loop was WMMA-bound, not LDS-bound.

**The caveat that cost two wrong predictions — read it before acting on any
ablation table: a share measures work *deleted*, not time *recoverable*.**
Staging ablated at 18.3% of runtime; halving the staged bytes returned ~1%,
because at that occupancy the co-resident workgroup on the same WGP was already
covering it. Deleting a stage tells you what executing it costs. It does not
tell you anything was waiting on it. Before you spend a week on a stage, ask
what is covering it today.

The ablation entries that *do* pay out are the ones you can prove are
identities. Same table, softmax line, 8.4%: most of it is one rescale multiply
over all 64 accumulator registers on every KV block, and whenever the running
max did not grow, `m_new == m_old` bitwise, `alpha` is exactly 1, and all 64
multiplies are a guaranteed no-op. Guarding it with an **exact equality** (not a
tolerance — the output stays bit-identical) returned ~2%.

### Phase 5 — Tune, with a measurement protocol you trust more than the numbers.

**Never compare across processes.** Build A → time A → rebuild as B → time B is
two processes, and clock/power state drifts between them. On this node that
manufactured a **13–16% "gain"** for a change that was really worth 5%, and
separately hid a real 5% behind an apparent no-change. Instead:

```
make TARGET=v_a EXTRA_HIPFLAGS='-DKNOB=1 -DTK_MODULE_NAME=v_a'
make TARGET=v_b EXTRA_HIPFLAGS='-DKNOB=2 -DTK_MODULE_NAME=v_b'
python ab.py v_a v_b      # imports both, interleaves round-robin, min over rounds
```

(`TARGET` names the output file; `TK_MODULE_NAME` names the pybind module —
setting only the first gives `ImportError: does not define PyInit_…`.)

Rules that go with it:

- **Put the current shipped version in the comparison as a control.** If it does
  not reproduce its historical number, the harness is wrong, not the kernel.
  This is what caught a "+21%" that was really +5%.
- **Correctness gate inside the harness, before the timing loop** — a variant
  that spills is wrong, not slow, and a wrong variant that happens to be fast is
  the trap. Provide an explicit opt-out for the deliberately-wrong `ABLATE_*`
  builds and for nothing else.
- **Record the baseline's own run-to-run spread.** The Triton FA-2 here moves
  ±3% between processes even with all backends interleaved *within* each. One
  run would have justified "12–22% ahead"; two runs agreed on 4–14%, so 4–14%
  is what got published.
- **Print versions and machine state in the banner** — of the baseline too. A
  JIT-compiled baseline's number is only comparable to another from the same
  compiler version; an AOT one in the framework wheel is not affected at all,
  and the banner should say which is which.
- **Re-run the sweep after any structural change.** A scheduling change in the
  GEMM moved two other optima: the best K-tile depth doubled (because more
  K-slices means more rotation points) and the small-shape crossover moved 4×.
  Optima do not compose.

Expect the knob landscape to be a **comb, not a curve**. Warps 8/10/12/14/16
measured 56.8/51.2/59.7/43.5/49.5 — what matters is whether the count divides
the occupancy limit, and the loss from stranded wave slots swamps whatever the
knob was supposed to buy.

### Phase 6 — Ship into what the consumer already calls, then write it up.

- **Integration surface = the signature the framework already uses.** A drop-in
  whose signature matches `F.scaled_dot_product_attention` *exactly*, forwarding
  to torch for everything it cannot take (masks, dropout, fp16, backward), is a
  one-line integration for any framework. A framework patch is not. Same for
  `torch.ops.hk_dist` behind a drop-in `HKRowParallelLinear`.
- **Write the README to the template in `references/writeup.md`.** The three
  here share a structure on purpose, and the two sections that pay for
  themselves are *Levers that were measured and are not levers* and *What this
  is not*.

## The cross-cutting traps

| Trap | Symptom | Guard |
|---|---|---|
| Cross-process A/B | Double-digit gains that vanish | One process, interleaved, old version as control |
| Datasheet denominator | "% of peak" understated ~18% | Measure the ceiling with a probe |
| Wrong baseline library | Beating the untuned one | Check what the stack dispatches, with a profiler |
| Sweep timed one instantiation | 5% lost on a build you never timed | Read every kernel's occupancy line |
| Register spill | Silently wrong, not slow | `ScratchSize == 0` on every build |
| Ablation share read as headroom | A week for 1% | "What covers this stage today?" |
| Probe violating a hw precondition | Coherent, wrong answer | Make preconditions fail loudly |
| Optima assumed to compose | Stale tuning after a refactor | Re-sweep after structural changes |

## References

- `references/measurement.md` — the A/B harness, ceiling and layout probes,
  ablation switch design, what to print in a results banner.
- `references/writeup.md` — the README structure the three kernels share, and
  why each section is there.
- `references/gfx11-gotchas.md` — the hardware-specific catalogue for
  gfx1100/gfx1201. Read only if the target is RDNA3/RDNA4; the phases above are
  architecture-independent, this file is not.
