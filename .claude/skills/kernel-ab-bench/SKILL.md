---
name: kernel-ab-bench
description: >
  Measure whether one build of a GPU kernel is faster than another, correctly:
  build each variant as a separately-named extension module, import them all
  into ONE process, interleave them round-robin, gate correctness before timing,
  and report ratios against the currently shipped build as control. Uses its
  `ab_harness.py` helper. Use when comparing tilings, knob settings, or a
  candidate against the shipped kernel, when a measured speedup looks too large
  to believe, or when deciding whether a result is above the noise floor.
  Usage: /kernel-ab-bench <variantA> <variantB> [...]
allowed-tools: Read Write Edit Bash Grep Glob
---

# Kernel A/B Bench

## Pick the right skill first

| Question | Skill |
|---|---|
| Is variant A faster than variant B? | **this skill** |
| Did the change spill or cost occupancy? | `/kernel-resource-check` (compile-only, run this FIRST — it is free) |
| Which pipeline stage owns the time? | `/kernel-attribution` |
| Is the *baseline* I am beating the right one? | `/kernel-bringup` Phase 0 |

## The rule

**Never compare timings from two processes.** Build A → time A → rebuild as
B → time B is the natural workflow and it is wrong: clock and power state drift
between process launches. On `wx-ms-w7900d-0043` that drift manufactured a
**+13–16%** result for a change actually worth **+5%**, and separately reported a
real +5% as "no difference". The direction is not predictable, and the node is
shared with other tenants.

Absolute numbers across processes are not comparable. **Ratios between variants
inside one process are** — the drift is charged to all of them equally.

## Workflow

### Step 1 — Build each variant as a separately-named module

```bash
cd kernels/rdna3/attn/fwd
make TARGET=v_base EXTRA_HIPFLAGS='-DTK_MODULE_NAME=v_base'
make TARGET=v_kv64 EXTRA_HIPFLAGS='-DKV_BLOCK=64 -DTK_MODULE_NAME=v_kv64'
```

`TARGET` names the `.so`; `TK_MODULE_NAME` names the pybind module. Setting only
`TARGET` gives, at import:

```
ImportError: dynamic module does not define module export function (PyInit_v_kv64)
```

**Never rebuild a `.so` while a process has it mapped.**

### Step 2 — Check resources before spending a GPU run

```bash
/kernel-resource-check     # ScratchSize must be 0 for every variant
```

A spilling variant is *wrong*, not slow. Timing it wastes the run.

### Step 3 — Run the harness

```bash
python3 ${CLAUDE_SKILL_DIR}/scripts/ab_harness.py spec_attn.py \
    v_base v_kv64 --control v_base --rounds 5
```

The spec module supplies the kernel-specific half — shapes, input construction,
the call, FLOPs, and a reference. See `## The spec module` below, and
`kernels/rdna3/attn/fwd/ab.py` for a worked one.

### Step 4 — Include the shipped build as the control

`--control` should name the **currently shipped** build, and it should be in
every comparison. If the control does not reproduce its historical number, the
harness is wrong, not the kernel. Enforce it:

```bash
… ab_harness.py spec_attn.py v_shipped v_new --control v_shipped --expect v_shipped=63.9
```

This is not paranoia: a claimed +21% turned out to be +5% because the module
labelled "baseline" in the comparison was not what had shipped.

## The spec module

```python
SHAPES = [(1, 56, 4096), (1, 56, 16384)]
CHECK_SHAPE = (1, 4, 4097)        # optional; pick a shape that divides nothing

def make_inputs(shape): ...       # -> whatever run() takes
def run(mod, inputs): ...         # -> output tensor
def flops(shape): ...             # -> float, for the TFLOP/s column
def reference(inputs): ...        # -> fp32 ground truth
def check(name, out, ref): ...    # optional; default is rel-error < 5e-2
def bench(fn): ...                # optional; default is torch cuda events, min of 10
```

The default `check` **fails when the reference has no magnitude**, rather than
passing: an all-zero output would otherwise sail through a relative test.

## Reading the output

```
(1, 56, 16384)                 ms    TFLOP/s   vs ctrl   spread
  v_shipped               59.930       63.9     1.000x      0.4%
  v_kv64                  62.410       61.4     0.960x      0.6%
```

`vs ctrl` is the ratio that survives drift and the only column worth quoting.
`spread` is the variant's own round-to-round range. If the largest gap between
variants is smaller than the worst spread, the harness says so:

```
  ! largest gap (0.4%) is under the worst round-to-round spread (1.1%):
    raise --rounds before believing this row.
```

### Exit codes

| Code | Meaning | What to do |
|---|---|---|
| `0` `RESULT: OK` | Ratios printed, control behaved | Read the `vs ctrl` column |
| `1` `RESULT: REGRESSION` | A variant is numerically wrong, or the control missed `--expect` | Act — the ratios are not usable until this is fixed |
| `2` `RESULT: NOT TRUSTWORTHY` | The comparison cannot be made | Fix inputs and rerun. **Do not report a result.** |

Exit 2 covers: a module that will not import, fewer than two variants (a lone
absolute number is exactly what this harness exists to stop being quoted), a
spec missing a required function, `--no-check` with no `reference`, and a
variant that raises during the gate.

Validate the harness itself with `ab_harness.py --selftest` — no GPU needed.

## Pitfalls

- **`--no-check` is for `ABLATE_*` builds and nothing else.** Those delete a
  pipeline stage and are wrong on purpose. Any other use defeats the gate that
  exists because a spilled tiling is wrong rather than slow.
- **Batching, not interleaving, reintroduces the bug in miniature.** Running all
  rounds of A then all rounds of B lets a thermal ramp land entirely on B. The
  harness interleaves; do not "optimise" that away.
- **Taking the mean rather than the min** imports the co-tenant's load into your
  number. Min over rounds is the right estimator on a shared node.
- **A GPU run is long; do not poll it with `sleep`.** Launch it in the
  background and wait for the completion notification. Use `python3 -u` when
  interim output matters — Python buffers through a pipe and the job looks hung.
- **Serialise benchmark jobs.** The node is shared; a second job of your own
  running concurrently invalidates both.
- **Pin an idle GPU.** On `wx-ms-w7900d-0043`, GPUs 1/2/4 belong to other
  tenants — use `HIP_VISIBLE_DEVICES=0` or `3`, and wrap every launch in
  `timeout --signal=KILL`.
- **Re-sweep after a structural change.** Knob optima do not compose; a tuning
  table from before a staging-loop rewrite is stale, not a starting point.
