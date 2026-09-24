# Probes, banners and published numbers

The two ends of Phase 0 and Phase 6: establishing the numbers you will divide
by, and not misreporting the ones you end up with.

The other halves of measurement live in their own skills — `/kernel-ab-bench`
for the single-process A/B protocol, `/kernel-attribution` for ablation switch
design, `/kernel-resource-check` for the compile-only resource gate.

## 1. The ceiling probe

Never use the datasheet. `tools/rdna-probes/wmma_peak.hip` runs back-to-back
WMMA with no memory traffic at all: 122.6 TFLOPs architectural vs **100.5
measured**, because the part is clock- and power-limited (2.18 GHz, 101 W) under
any load. Using the architectural figure understated every percentage by 18%.

Sample `rocm-smi -c -P` during a 20 s steady-state run, and confirm the tuned
kernel and the probe ran at the **same clock** — otherwise the denominator is
hiding a thermal story rather than describing the hardware.

**Measure the secondary ceilings too.** `lds_peak.hip`, reproducing the kernel's
exact addressing, gave 64.8 lane-bytes/clk/CU = 129.6/WGP against the hardware's
128 B/clk/WGP. That is what let the GEMM's remaining gap be *accounted for*
rather than estimated: LDS reads occupy 75% of the WMMA issue window, reads plus
writes 87%. It is also the number that says **stop** — there is no room under
87% for 77% to rise much, and knowing that ended the search instead of
extending it.

A ceiling probe is worth writing even when you are sure of the answer, because
its output is what every later percentage is divided by.

## 2. The layout probe

The ISA document is not ground truth; the probe is. Write a one-hot into a known
position, run the instruction, read back where it landed.

**Design the probe so a violated hardware precondition fails loudly.** The first
attempt here wrote a one-hot into a single lane without mirroring it across `l`
and `l^16`, which gfx11 WMMA requires. It read back undefined behaviour that
looked like a clean, self-consistent layout. It was wrong, and nothing in the
output said so.

Concretely: assert the precondition in the probe, and include a case whose
correct answer you already know, so a probe that has stopped measuring anything
fails on the known case instead of returning plausible garbage for the unknown
one.

## 3. The results banner

Print, every run:

- device, CU/WGP count, driver/runtime version
- **every baseline's version, and whether it is JIT or AOT.** A JIT baseline
  (Triton) is comparable only to another from the same compiler version; an AOT
  one compiled into the framework wheel (aotriton) is unaffected by the
  installed compiler entirely. The banner should say which is which — otherwise
  someone will later "explain" a difference with a version change that could not
  have caused it.
- clocks and power sampled at run time
- for multi-GPU, **which bandwidth mode the process landed in**. Peer bandwidth
  on this node is bimodal per process launch — ~15.5 or ~24–27 GB/s, nothing
  between, stable to three decimals within a process, re-rolled on every launch.
  Eight consecutive runs: `{15.5, 27.1, 15.5, 15.5, 15.5, 27.1, 27.0, 24.0}`. A
  fused time from one process and a collective time from another describe
  different hardware.

**Confirm the kernel that actually dispatched is the one you think.** Print the
profiler's kernel name for each backend before the table. `F.sdpa`'s FLASH and
EFFICIENT backends both land in aotriton `attn_fwd` on gfx1100, and a log
warning claiming flash was disabled was a red herring — the profiler settled it,
the enum and the log did not.

## 4. Publishing a number

- **Report the baseline's own run-to-run spread, and publish the conservative
  end.** One run put the Triton baseline at 53.4–56.2 TF, which would have
  justified "12–22% ahead". Two further runs agreed on 56.7–58.6 → "4–14%".
  4–14% is what shipped. A margin quoted off one run is a margin you will have
  to retract.
- **Where the vendor has freedom you do not**, publish both columns: its best,
  and it restricted to your problem. Neither is "the real one"; they answer
  different questions, and picking one silently is the choice a reader would
  object to if they saw it.
- **Give the ceiling percentage alongside the speedup.** "65% of the measured
  WMMA ceiling, within 4% of this machine's best bf16 GEMM" says how much is
  left. "3.2× aotriton" does not, and invites work that cannot pay out.
- **Every number carries its conditions** — shape, dtype, causal flag, process,
  date. A number without them gets quoted later against a different setup.
- **Say which numbers are measured and which are inferred, in the sentence.**
  "Confirmed with the profiler, not inferred" is worth writing every time it is
  true: the reader cannot otherwise tell, and one inferred claim in a document
  of measured ones is what eventually breaks.
