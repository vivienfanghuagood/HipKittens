# Measurement

Everything in this file exists because a number lied once.

## 1. The A/B harness

The failure it prevents: build A → time A → rebuild as B → time B. That is two
processes, and clock/power state drifts between them. Measured cost of getting
this wrong on `wx-ms-w7900d-0043`: a `PV_TILES` change read as **+13–16%** when
it was worth **+5%**, and separately a change read as **no difference** when it
was worth **+5%**. Drift direction is not predictable; the node is shared.

Absolute numbers are not comparable across processes. **Ratios between variants
inside one process are** — drift hits every variant equally.

Build each variant as a separately-named module:

```make
# TARGET names the .so; TK_MODULE_NAME names the pybind module.
# Setting only TARGET gives: ImportError: dynamic module does not define
# module export function (PyInit_<target>)
make TARGET=v_a EXTRA_HIPFLAGS='-DKNOB=1 -DTK_MODULE_NAME=v_a'
```

Then interleave round-robin over several rounds, taking min per variant:

```python
mods = [(n, importlib.import_module(n)) for n in sys.argv[1:]]

# Correctness first, every variant: a tiling that spills is WRONG, not slow,
# and a wrong-but-fast variant is the trap this gate exists for.
if CHECK:
    for n, m in mods:
        ok, msg = check(n, run(m, *small_inputs), reference)
        if not ok: return fail(n, msg)

best = {n: inf for n, _ in mods}
for _ in range(ROUNDS):            # interleave, do not batch per variant
    for n, m in mods:
        best[n] = min(best[n], bench(lambda: run(m, *inputs)))
```

Reference implementation: `kernels/rdna3/attn/fwd/ab.py`.

**Always include the current shipped build as a control.** If it does not
reproduce its historical number, the harness is wrong, not the kernel. This is
what caught a claimed +21% that was actually +5%: the "baseline" module in the
comparison was not what had shipped.

`AB_CHECK=0` exists only for `ABLATE_*` builds, which are wrong on purpose. Any
other use defeats the gate.

## 2. Probes

### Ceiling probe

Never use the datasheet. `tools/rdna-probes/wmma_peak.hip` runs back-to-back
WMMA with no memory traffic: 122.6 TFLOPs architectural vs **100.5 measured**,
because the part is clock- and power-limited (2.18 GHz, 101 W) under any load.
Sample `rocm-smi -c -P` during a 20 s steady-state run and confirm the tuned
kernel and the probe ran at the same clock — otherwise the denominator is
hiding a thermal story.

Measure the **secondary** ceilings too. `lds_peak.hip` reproducing the kernel's
exact addressing gave 64.8 lane-bytes/clk/CU = 129.6/WGP = the hardware's
128 B/clk/WGP, which is how the GEMM's remaining gap was accounted for exactly
(LDS reads 75% of the WMMA window, +writes 87%) instead of estimated. That is
the number that says *stop*: there is no room under 87% for 77% to rise much.

### Layout probe

The ISA doc is not ground truth; the probe is. Write a one-hot into a known
position, run the instruction, read back where it landed.

**Design the probe so a violated hardware precondition fails loudly.** The first
attempt here wrote a one-hot into a single lane without mirroring it across `l`
and `l^16`, which gfx11 WMMA requires, and read back undefined behaviour that
looked like a clean and self-consistent layout. It was wrong, and nothing about
the output said so.

## 3. Ablation switches

Use when hardware counters are unavailable — on gfx1100 every `SQ` counter
except `SQ_WAVES` reads 0 under `rocprofv3`, including `SQ_INSTS_VALU`,
`SQ_INSTS_LDS`, `LDSBankConflict` and `ALUStalledByLDS`.

One `-D` per pipeline stage, each deleting that stage and making the result
wrong on purpose. Build them all, interleave them all in one process (§1), read
the shares as ratios.

Include **one switch that separates bandwidth from math**: keep both matmuls but
drop only the LDS reads that feed them, zeroing the operand tiles once and
reusing them. No combination of per-stage switches can do this, since each
removes a stage's reads *and* its math together. This is the measurement that
showed attention's inner loop was WMMA-bound rather than LDS-bound (~55% WMMA +
conversion after stripping LDS reads from both matmuls).

### Reading the table

**A share measures work deleted, not time recoverable.** Twice this was read as
headroom and twice it was wrong:

- staging ablated at 18.3%; halving staged bytes (`NUM_WARPS=24`, doubling the
  Q tile) returned ~1%, because at 12 warps there are two workgroups per WGP and
  the co-resident one already covers staging.
- the standing fix for causal's wasted staging, a narrower tile (`NUM_WARPS=8`),
  measured **−4.6%**.

Before acting on a line, ask **what covers this stage today**. Entries that do
pay out are ones you can prove are identities — the softmax line's 8.4% was
mostly a 64-register rescale that is exactly a no-op whenever the running max
did not grow, and an exact-equality guard (output bit-identical) returned ~2%.

## 4. The results banner

Print, every run:

- device + CU/WGP count, and the driver/runtime version
- **every baseline's version**, and whether it is JIT or AOT. A JIT baseline
  (Triton) is only comparable to another from the same compiler version; an AOT
  one compiled into the framework wheel (aotriton) is unaffected by the
  installed compiler entirely. Say which is which in the banner.
- clocks/power sampled at run time
- for multi-GPU: which bandwidth mode the process landed in. Peer bandwidth on
  this node is **bimodal per process launch** — ~15.5 or ~24–27 GB/s, nothing
  between, stable to three decimals within a process, re-rolled every launch.
  Eight consecutive runs: `{15.5, 27.1, 15.5, 15.5, 15.5, 27.1, 27.0, 24.0}`.
  A fused time from one process and a collective time from another describe
  different hardware.

Confirm the kernel actually dispatched is the one you think: print the
profiler's kernel name for each backend before the table. `F.sdpa`'s FLASH and
EFFICIENT backends both land in aotriton `attn_fwd` on gfx1100, and a log
warning claiming flash was disabled was a red herring.

## 5. Reporting

- Report the **baseline's own run-to-run spread**, and publish the conservative
  end. One run put the Triton baseline at 53.4–56.2 TF (→ "12–22% ahead"); two
  runs agreed on 56.7–58.6 (→ "4–14%"). 4–14% is what shipped.
- Where the vendor has freedom you do not (it picks any of NN/NT/TN/TT; you
  implement one), publish **both** columns — its best, and it restricted to your
  problem. Neither is "the real one"; they answer different questions.
- Give the ceiling percentage alongside the speedup. "65% of the measured WMMA
  ceiling, within 4% of this machine's best bf16 GEMM" is the sentence that says
  how much is left; "3.2× aotriton" is not.
