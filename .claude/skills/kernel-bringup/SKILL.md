---
name: kernel-bringup
description: >
  The end-to-end method for bringing up a high-performance GPU kernel that has
  to beat a vendor library or a framework's existing backend — GEMM, attention /
  SDPA, fused collectives, anything compute-bound. Seven gated phases: pick the
  target shape and an honest baseline, probe for the hardware facts that change
  the algorithm's shape, derive rather than port, pass correctness and resource
  gates before any timing, attribute the time, tune against a trusted
  measurement protocol, then ship into the signature the consumer already calls.
  Use when starting a new kernel, porting one to an architecture it was not
  written for, tuning one that is close but losing to a baseline, or deciding
  whether a remaining gap is reachable at all. This skill is the map; the
  measurement steps delegate to /kernel-resource-check, /kernel-ab-bench,
  /kernel-attribution, /kernel-writeup and /rdna3-kernel-facts.
  Usage: /kernel-bringup
allowed-tools: Read Write Edit Bash Grep Glob Agent
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
timed the wrong template instantiation — each produced a number that looked
authoritative and pointed the next week of work in the wrong direction. The
gates exist to make wrongness loud and early.

## Pick the right skill first

This skill is the map. Each measurement step below has a skill that does it,
with a tool and an exit-code contract:

| Question | Skill |
|---|---|
| How do I approach this kernel at all? | **this skill** |
| Did my change spill / cost occupancy? | `/kernel-resource-check` — compile-only, seconds, no GPU. Run it first, it is free. |
| Is variant A faster than variant B? | `/kernel-ab-bench` — one process, interleaved, control included |
| Which pipeline stage owns the time? | `/kernel-attribution` — ablation switches, for when counters are dead |
| What does this hardware actually do? | `/rdna3-kernel-facts` — gfx1100/gfx1201 only |
| How do I document it? | `/kernel-writeup` |

## Phases

Each phase has a gate. Do not start the next one until it fires.

### Phase 0 — Aim. Three numbers before a line of kernel code.

1. **Target shapes, from the consumer's real config.** Not round squares. Read
   the model's `config.json` / the serving stack's launch shapes and derive the
   actual dimensions, including the awkward ones. For MiniMax H3 that was
   `num_attention_heads=56, attention_head_dim=128`, no `num_key_value_heads`
   (so MHA, not GQA), bidirectional (so non-causal), and 49 920 tokens for a
   480p/5s clip. Those four facts deleted half the design space before anything
   was written. Ask also: what fraction of end-to-end time is this op? (>85% for
   attention at those lengths — that is the whole justification for the work.)

2. **The baseline, verified by profiler, not assumed.** Find out what the user's
   stack *actually dispatches*, then check it.
   `F.scaled_dot_product_attention` on gfx1100 lands in aotriton `attn_fwd` for
   both the FLASH and the EFFICIENT backend — confirmed with the profiler; a log
   warning claiming flash was disabled was a red herring. torch on ROCm defaults
   to hipBLASLt, and on gfx1100 hipBLASLt is up to **33% slower than rocBLAS**:
   benchmarking against torch's default there is benchmarking the library the
   vendor has not tuned. Take the strongest honest baseline, and where the vendor
   has freedom you do not (it picks any of NN/NT/TN/TT, you implement one),
   report **both** columns — its best, and it restricted to your problem.

3. **The ceiling, measured.** Never the datasheet. 96 CU × 512 FLOP/clk ×
   2.495 GHz boost = 122.6 TFLOPs; back-to-back WMMA with no memory at all
   measures **100.5** at 2.18 GHz and 101 W, because the part is clock- and
   power-limited under any real load. Percentages against the datasheet
   understated by 18%. Write the probe (`tools/rdna-probes/wmma_peak.hip`),
   sample `rocm-smi -c -P` during the run, and measure the secondary ceilings
   too — LDS bytes/clk decided where the GEMM's wall was. Details:
   `references/probes-and-reporting.md`.

**Gate:** you can state target shape, baseline TFLOPs, and reachable ceiling,
each with the command that produced it. A speedup claim without all three is not
yet a claim.

### Phase 1 — Probe. Find the facts that change the algorithm's *shape*.

Documentation and ported code are not ground truth; a probe is. Look
specifically for facts that invalidate the reference implementation you were
going to transcribe. On gfx11 there were three, and each restructured the kernel
rather than adjusting a constant — WMMA operand mirroring across wave halves, no
global→LDS DMA at all, and `ds_read_b128` only for `row`-layout bf16. The full
catalogue is `/rdna3-kernel-facts`; how to write a probe that cannot lie to you
is `references/probes-and-reporting.md`.

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
in the source, it is what makes the kernel explicable later, and if it cannot be
written the design is not yet understood.

Budget registers and LDS on paper in the same step. At `Q_BLOCK=16, D=128`:
accumulator 64 VGPRs + Q 64 + scores 32 + operands 32 ≈ 224–250 of 256 — which
said at design time that the shape was viable and `Q_BLOCK=32` never would be.
Check the budget against the occupancy ladder before committing:

```bash
/kernel-resource-check          # explain --arch gfx1100 --vgprs 240
```

**Gate:** a derivation someone else could follow, and a register/LDS budget that
closes on the ladder, not on 256.

### Phase 3 — Correct before fast. Gates that fire on every build.

1. **An independent reference with a magnitude guard.** fp32, chunked so long
   sequences fit. Relative tolerance sized from the arithmetic (bf16 double
   rounding + long summation → 5e-2), **plus a check that the reference itself
   has magnitude**, so an all-zero output cannot pass.

2. **Shapes that divide nothing.** 4097, 5000, 12345, 193 alongside the round
   ones, plus batch, plus every template instantiation and every fallback
   branch. 24 shapes for attention, 30 for the distributed kernel — every one
   checked elementwise *before* it was timed.

3. **`ScratchSize == 0` and every instantiation's occupancy**, as a hard gate,
   on every build. This is not a performance check: a kernel that hand-manages
   `s_waitcnt` does not survive spilling, and the distributed epilogue's symptom
   was every `s == N_SPLIT-1` tile coming out NaN — a quarter of the output,
   silently, on exactly the shapes you would benchmark.

```bash
/kernel-resource-check          # gates scratch + spills; --expect-kernels N
```

**Gate:** all shapes pass elementwise; `RESULT: OK` from the resource check with
`--expect-kernels` set to the full instantiation count.

### Phase 4 — Attribute before tuning.

Hardware counters may not exist — on this part every `SQ` counter except
`SQ_WAVES` reads 0. Build the attribution in with `ABLATE_*` switches, including
one that separates bandwidth from math.

```bash
/kernel-attribution
```

**Read the caveat in that skill before acting on any row.** A share measures
work *deleted*, not time *recoverable*; reading it as headroom twice cost a week
for ~1% and a −4.6% "fix". The rows that pay out are the ones you can prove are
identities.

**Gate:** you can name the bottleneck stage and say what is covering it today.

### Phase 5 — Tune, with a protocol you trust more than the numbers.

```bash
/kernel-ab-bench                # never compare across processes
```

Two things that belong here rather than in the harness:

- **Expect the knob landscape to be a comb, not a curve.** Warps 8/10/12/14/16
  measured 56.8/51.2/59.7/43.5/49.5 TF. What matters is whether the count
  divides the occupancy limit; the loss from stranded wave slots swamps whatever
  the knob was supposed to buy. Sweep every value, do not bisect.
- **Optima do not compose.** A scheduling change in the GEMM moved two unrelated
  optima: the best K-tile depth doubled, and the small-shape crossover moved 4×.
  Re-sweep after any structural change; a table from before a rewrite is stale,
  not a starting point.

**Gate:** the win survives with the shipped build as control, and is larger than
the worst round-to-round spread.

### Phase 6 — Ship into what the consumer already calls, then write it up.

- **Integration surface = the signature the framework already uses.** A drop-in
  whose signature matches `F.scaled_dot_product_attention` *exactly*, forwarding
  to torch for everything it cannot take (masks, dropout, fp16, backward), is a
  one-line integration for any framework. A framework patch is not. Same for
  `torch.ops.hk_dist` behind a drop-in `HKRowParallelLinear`.
- **Write it up**: `/kernel-writeup`. The two sections that pay for themselves
  are *Levers that were measured and are not levers* and *What this is not*.

**Gate:** the drop-in passes against the original on every shape *and* every
fallback branch, and the README's numbers each carry their conditions.

## The cross-cutting traps

| Trap | Symptom | Guard |
|---|---|---|
| Cross-process A/B | Double-digit gains that vanish | `/kernel-ab-bench`: one process, interleaved, shipped build as control |
| Datasheet denominator | "% of peak" understated ~18% | Measure the ceiling with a probe |
| Wrong baseline library | Beating the untuned one | Check what the stack dispatches, with a profiler |
| Sweep timed one instantiation | 5% lost on a build you never timed | `/kernel-resource-check --expect-kernels` |
| Register spill | Silently wrong, not slow | `ScratchSize == 0` on every build |
| Ablation share read as headroom | A week for 1% | "What covers this stage today?" |
| Probe violating a hw precondition | Coherent, wrong answer | Make preconditions fail loudly |
| Optima assumed to compose | Stale tuning after a refactor | Re-sweep after structural changes |

## References

| File | Load it when |
|---|---|
| `references/probes-and-reporting.md` | Writing a ceiling or layout probe, or deciding what a results banner and a published number must carry. |
