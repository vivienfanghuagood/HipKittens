#!/usr/bin/env python3
"""Single-process, interleaved A/B benchmark of several builds of one kernel.

The failure this prevents: build A, time it, rebuild as B, time that. Those are
two processes, clock and power state drift between them, and on
`wx-ms-w7900d-0043` that drift manufactured a **+13-16%** result for a change
worth **+5%** -- and, separately, hid a real +5% as "no difference". Drift
direction is not predictable and the node is shared.

Absolute numbers are not comparable across processes. Ratios between variants
inside one process are, because the drift is charged to all of them equally.
So: import every variant into one interpreter, interleave them round-robin, and
report ratios against a named control.

Usage
  ab_harness.py <spec.py> <modA> <modB> [<modC> ...] [options]
  ab_harness.py --selftest                       # no GPU, validates this file

The spec module supplies the kernel-specific half:

  SHAPES = [(1, 56, 4096), ...]          # whatever tuple make_inputs takes
  def make_inputs(shape) -> object       # whatever run() takes
  def run(mod, inputs) -> output         # call the extension
  def flops(shape) -> float              # for the TFLOPs column
  def reference(inputs) -> output        # OPTIONAL but see --no-check
  def check(name, out, ref) -> (bool,str)# OPTIONAL, defaults to rel-error
  def bench(fn) -> float_ms              # OPTIONAL, defaults to torch cuda events
  CHECK_SHAPE = (1, 4, 4097)             # OPTIONAL, gate shape; divides nothing

Exit codes
  0  RESULT: OK               ratios printed, control behaved
  1  RESULT: REGRESSION       a variant is wrong, or the control missed --expect
  2  RESULT: NOT TRUSTWORTHY  the comparison cannot be made -- fix and rerun
"""

import argparse
import importlib
import importlib.util
import os
import sys


class NotTrustworthy(Exception):
    pass


def verdict(code):
    print({0: "RESULT: OK", 1: "RESULT: REGRESSION",
           2: "RESULT: NOT TRUSTWORTHY"}[code])
    return code


def load_spec(path):
    if not os.path.exists(path):
        raise NotTrustworthy(f"{path}: no such spec module")
    sys.path.insert(0, os.path.dirname(os.path.abspath(path)) or ".")
    name = os.path.splitext(os.path.basename(path))[0]
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    for required in ("SHAPES", "make_inputs", "run", "flops"):
        if not hasattr(mod, required):
            raise NotTrustworthy(f"{path} does not define {required}")
    return mod


def default_bench(fn, iters=10, warmup=3):
    import torch
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    ts = []
    for _ in range(iters):
        s, e = torch.cuda.Event(True), torch.cuda.Event(True)
        s.record()
        fn()
        e.record()
        torch.cuda.synchronize()
        ts.append(s.elapsed_time(e))
    return min(ts)


def default_check(name, out, ref):
    import torch
    o, r = out.float(), ref.float()
    scale = r.abs().max().item()
    # A reference with no magnitude cannot validate anything: an all-zero output
    # would pass a relative test against it. Fail loudly instead of passing.
    if scale < 1e-6:
        return False, "reference has no magnitude -- the gate cannot see anything"
    rel = (o - r).abs().max().item() / scale
    return rel < 5e-2, f"rel={rel:.3e}"


def run_ab(a, spec):
    mods = []
    for n in a.modules:
        try:
            mods.append((n, importlib.import_module(n)))
        except Exception as e:
            raise NotTrustworthy(f"cannot import {n}: {type(e).__name__}: {e}")
    if len(mods) < 2 and not a.allow_single:
        raise NotTrustworthy(
            "fewer than two variants: there is nothing to compare, and a lone "
            "absolute number from one process is exactly what this harness "
            "exists to stop being quoted. Pass --allow-single to override.")

    control = a.control or mods[0][0]
    if control not in dict(mods):
        raise NotTrustworthy(f"--control {control} is not among {list(dict(mods))}")
    if a.control is None and not a.quiet:
        print(f"# control = {control} (first module). Include the CURRENTLY SHIPPED\n"
              f"# build as the control: if it does not reproduce its historical\n"
              f"# number, the harness is wrong, not the kernel.\n")

    bench = getattr(spec, "bench", default_bench)
    check = getattr(spec, "check", default_check)

    # --- correctness gate, every variant, before any timing -------------------
    if a.check:
        if not hasattr(spec, "reference"):
            raise NotTrustworthy(
                f"{a.spec} has no reference(); pass --no-check only for ABLATE_* "
                "builds, which are wrong on purpose.")
        shape = getattr(spec, "CHECK_SHAPE", spec.SHAPES[0])
        inp = spec.make_inputs(shape)
        ref = spec.reference(inp)
        bad = []
        for n, m in mods:
            try:
                ok, msg = check(n, spec.run(m, inp), ref)
            except Exception as e:
                raise NotTrustworthy(f"{n} raised during the gate: {type(e).__name__}: {e}")
            print(f"  {'ok  ' if ok else 'WRONG'} {n:<24} {msg}")
            if not ok:
                bad.append(n)
        del inp, ref
        if bad:
            print(f"\n{len(bad)} variant(s) wrong: {bad}")
            print("A tiling that spills is WRONG, not slow. Do not time these.")
            return 1
        print()
    else:
        print("# --no-check: correctness gate SKIPPED. Valid only for ABLATE_* builds.\n")

    # --- timing, interleaved --------------------------------------------------
    rc = 0
    first_best = None
    for shape in spec.SHAPES:
        inp = spec.make_inputs(shape)
        samples = {n: [] for n, _ in mods}
        # Round-robin, not variant-at-a-time: any drift that survives is then
        # spread evenly over the variants instead of landing on whichever one
        # happened to run while the clock was high.
        for _ in range(a.rounds):
            for n, m in mods:
                samples[n].append(bench(lambda: spec.run(m, inp)))
        best = {n: min(v) for n, v in samples.items()}
        spread = {n: (max(v) - min(v)) / min(v) for n, v in samples.items()}
        if first_best is None:
            first_best = best

        fl = spec.flops(shape)
        print(f"{str(shape):<22} {'ms':>10} {'TFLOP/s':>10} {'vs ctrl':>9} {'spread':>8}")
        for n, _ in mods:
            ratio = best[control] / best[n]
            print(f"  {n:<20} {best[n]:>10.3f} {fl / best[n] * 1e-9:>10.1f} "
                  f"{ratio:>8.3f}x {spread[n] * 100:>7.1f}%")
        # A variant whose own round-to-round spread exceeds the gap being
        # claimed is not a measurement, it is noise with a name.
        gaps = [abs(best[control] / best[n] - 1) for n, _ in mods if n != control]
        if gaps and max(gaps) < max(spread.values()):
            print(f"  ! largest gap ({max(gaps)*100:.1f}%) is under the worst "
                  f"round-to-round spread ({max(spread.values())*100:.1f}%): "
                  f"raise --rounds before believing this row.")
        print()
        del inp

    if a.expect:
        name, val = a.expect.split("=")
        want = float(val)
        if name not in best:
            raise NotTrustworthy(f"--expect names {name}, which was not benchmarked")
        # Checked on the first shape, which is the one a historical number is
        # normally quoted for.
        got = spec.flops(spec.SHAPES[0]) / first_best[name] * 1e-9
        if abs(got - want) / want > a.expect_tol:
            print(f"control {name} at {got:.1f} TF, expected {want:.1f} "
                  f"(+-{a.expect_tol*100:.0f}%) -- the harness or the build is not "
                  f"what you think it is; the ratios above are suspect.")
            rc = 1
    return rc


# --- self-test ---------------------------------------------------------------

SELFTEST_SPEC = '''
# Generated by ab_harness.py --selftest. No GPU, no torch: validates the
# harness's own interleaving, gate, ratio and exit-code logic only.
import time
SHAPES = [(1, 1, 16)]
CHECK_SHAPE = (1, 1, 16)
def make_inputs(shape): return shape
def reference(inp): return [1.0] * inp[2]
def run(mod, inp): return mod.go(inp)
def flops(shape): return 2.0e9
def check(name, out, ref): return (out == ref), ("match" if out == ref else "differs")
def bench(fn):
    t = time.perf_counter(); fn(); return (time.perf_counter() - t) * 1e3
'''


def selftest():
    import tempfile
    import types
    ok = True
    d = tempfile.mkdtemp()
    sp = os.path.join(d, "spec_selftest.py")
    with open(sp, "w") as f:
        f.write(SELFTEST_SPEC)

    def fake(name, delay, correct=True):
        m = types.ModuleType(name)
        m.go = lambda inp: (time.sleep(delay), [1.0 if correct else 9.0] * inp[2])[1]
        sys.modules[name] = m
    import time
    fake("fast_v", 0.002)
    fake("slow_v", 0.004)
    fake("wrong_v", 0.001, correct=False)

    spec = load_spec(sp)
    base = dict(spec=sp, rounds=3, check=True, control=None, quiet=True,
                allow_single=False, expect=None, expect_tol=0.05)

    rc = run_ab(argparse.Namespace(modules=["fast_v", "slow_v"], **base), spec)
    print(f"  [selftest] two good variants -> {rc} (want 0)")
    ok &= rc == 0

    rc = run_ab(argparse.Namespace(modules=["fast_v", "wrong_v"], **base), spec)
    print(f"  [selftest] one wrong variant -> {rc} (want 1)")
    ok &= rc == 1

    for mods, why in ([["fast_v"], "single variant"],
                      [["fast_v", "no_such_module"], "unimportable variant"]):
        try:
            run_ab(argparse.Namespace(modules=mods, **base), spec)
            print(f"  [selftest] {why} -> no exception (want NotTrustworthy)")
            ok = False
        except NotTrustworthy:
            print(f"  [selftest] {why} -> NOT TRUSTWORTHY (want that)")

    print()
    return verdict(0 if ok else 1)


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("spec", nargs="?", help="path to the spec module")
    p.add_argument("modules", nargs="*", help="already-built extension module names")
    p.add_argument("--rounds", type=int, default=int(os.environ.get("AB_ROUNDS", 5)))
    p.add_argument("--control", help="variant to take ratios against; "
                                     "should be the currently shipped build")
    p.add_argument("--expect", metavar="NAME=TFLOPS",
                   help="fail if the control misses its historical number")
    p.add_argument("--expect-tol", type=float, default=0.05)
    p.add_argument("--no-check", dest="check", action="store_false",
                   help="ABLATE_* builds only -- they are wrong on purpose")
    p.add_argument("--allow-single", action="store_true")
    p.add_argument("-q", "--quiet", action="store_true")
    p.add_argument("--selftest", action="store_true")
    a = p.parse_args()

    if a.selftest:
        return selftest()
    try:
        if not a.spec or not a.modules:
            raise NotTrustworthy("need a spec module and at least two variants")
        return verdict(run_ab(a, load_spec(a.spec)))
    except NotTrustworthy as e:
        print(f"\ncannot answer: {e}", file=sys.stderr)
        return verdict(2)


if __name__ == "__main__":
    sys.exit(main())
