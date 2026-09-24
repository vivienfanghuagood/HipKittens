#!/usr/bin/env python3
"""Per-kernel GPU resource gate and regression diff for hipcc builds.

Parses `-Rpass-analysis=kernel-resource-usage` remarks -- VGPRs, SGPRs, spills,
scratch, LDS, occupancy -- for *every* kernel instantiation in a build, then
either gates them or diffs two builds.

Compile-only: no GPU, no profiler, no benchmark. Seconds, not minutes.

Why this exists rather than `grep -E 'VGPRs|Scratch'`:

  * grep shows the numbers but not which of the N template instantiations each
    one belongs to, so the instantiation you are not benchmarking gets no
    attention. Reading only the shipped shape's line is how a 5% loss hid in a
    build that was never timed.
  * on gfx1100 the VGPR allocation granule is 24, so 240 registers gives 6
    waves/SIMD and 241 gives 5. A budget read off the raw count is wrong at
    exactly the boundary that matters.
  * `ScratchSize != 0` is a correctness failure, not a slowdown, in any kernel
    that hand-manages `s_waitcnt` -- which is all of kernels/rdna3. A gate has
    to fail the build, not print a number that scrolls past.

Usage
  hip_resources.py capture  <log|-> [--json OUT]        parse a build log
  hip_resources.py check    <log|json> [gates...]       gate one build
  hip_resources.py diff     <before> <after> [-q]       compare two builds
  hip_resources.py explain  --arch gfx1100 --vgprs N    occupancy arithmetic

Exit codes (identical across subcommands; the last stdout line is the verdict)
  0  RESULT: OK               nothing to report -- proceed
  1  RESULT: REGRESSION       a claim about the code under test -- act on it
  2  RESULT: NOT TRUSTWORTHY  a claim about this tool's own confidence.
                              Fix the inputs and rerun. Do NOT report a clean
                              result on a 2.
"""

import argparse
import json
import os
import re
import shutil
import subprocess
import sys

# --- architecture facts ------------------------------------------------------
#
# Only what has been verified on hardware we have. An unknown arch degrades to
# reporting the compiler's own occupancy number and skipping the granule
# arithmetic -- it does not guess.
#
#   vgpr_granule  allocation quantum; a kernel is charged the rounded-up count
#   vgprs_per_simd  total VGPR file available to all resident waves on a SIMD
#   max_waves     hardware cap on waves/SIMD
#   lds_per_wg    LDS addressable by one workgroup
ARCH = {
    # gfx1100 granule verified empirically: 240 VGPRs -> 6 waves, 241 -> 5.
    "gfx1100": dict(vgpr_granule=24, vgprs_per_simd=1536, max_waves=16,
                    lds_per_wg=65536, wave=32),
}

FIELDS = {
    "TotalSGPRs": "sgpr",
    "VGPRs": "vgpr",
    "ScratchSize [bytes/lane]": "scratch",
    "Occupancy [waves/SIMD]": "occupancy",
    "SGPRs Spill": "sgpr_spill",
    "VGPRs Spill": "vgpr_spill",
    "LDS Size [bytes/block]": "lds",
}
# Increases in these are regressions; occupancy is handled separately (down is
# the bad direction).
TRIGGERS = ["vgpr", "scratch", "vgpr_spill", "sgpr_spill", "lds"]

NAME_RE = re.compile(r"remark:\s*Function Name:\s*(\S+)")
FIELD_RE = re.compile(r"remark:\s+(.+?):\s*(-?\d+|True|False)\s*\[")


class NotTrustworthy(Exception):
    """The tool cannot answer. Never downgrade one of these to a clean result."""


def demangle(names):
    """Best effort. A missing c++filt costs readability, not correctness."""
    exe = shutil.which("llvm-cxxfilt") or shutil.which("c++filt")
    if not exe or not names:
        return {n: n for n in names}
    try:
        out = subprocess.run([exe], input="\n".join(names), capture_output=True,
                             text=True, timeout=30).stdout.splitlines()
        if len(out) == len(names):
            return dict(zip(names, out))
    except Exception:
        pass
    return {n: n for n in names}


def parse(text, arch=None):
    """Build log -> {kernel name: resource dict}.

    Raises NotTrustworthy when the log cannot support an answer, which is a
    different thing from a log that reports bad numbers.
    """
    kernels, cur = {}, None
    for line in text.splitlines():
        m = NAME_RE.search(line)
        if m:
            cur = m.group(1)
            kernels.setdefault(cur, {})
            continue
        m = FIELD_RE.search(line)
        if m and cur is not None:
            key = FIELDS.get(m.group(1).strip())
            if key:
                val = m.group(2)
                v = {"True": 1, "False": 0}.get(val, None)
                kernels[cur][key] = int(val) if v is None else v

    if not kernels:
        raise NotTrustworthy(
            "no kernel-resource-usage remarks in the input.\n"
            "  The build must pass -Rpass-analysis=kernel-resource-usage AND must\n"
            "  actually recompile: an up-to-date target emits nothing at all, which\n"
            "  is indistinguishable from a clean build. Touch the source or make clean.")

    incomplete = [k for k, v in kernels.items() if "vgpr" not in v]
    if incomplete:
        raise NotTrustworthy(
            f"{len(incomplete)} kernel(s) with a Function Name but no VGPR line "
            f"(truncated log?): {incomplete[:3]}")

    pretty = demangle(list(kernels))
    facts = ARCH.get(arch or "")
    for name, r in kernels.items():
        r["name"] = pretty[name]
        r["mangled"] = name
        if facts:
            r.update(derive(r, facts))
    return kernels


def derive(r, facts):
    """Granule-rounded VGPR cost and the occupancy it actually buys."""
    g = facts["vgpr_granule"]
    rounded = -(-r["vgpr"] // g) * g                       # ceil to granule
    by_reg = min(facts["max_waves"], facts["vgprs_per_simd"] // rounded) if rounded else facts["max_waves"]
    # Registers you may add before the next occupancy step down, i.e. before
    # by_reg becomes by_reg-1. Reported because "how much room is left" is the
    # question every tiling change asks.
    headroom = 0
    if by_reg > 1:
        max_for_level = (facts["vgprs_per_simd"] // by_reg) // g * g
        headroom = max_for_level - r["vgpr"]
    return dict(vgpr_rounded=rounded, occ_by_reg=by_reg, vgpr_headroom=headroom)


def blocks_by_lds(r, facts):
    if not facts or not r.get("lds"):
        return None
    return facts["lds_per_wg"] // r["lds"]


# --- reporting ---------------------------------------------------------------

def verdict(code):
    print({0: "RESULT: OK", 1: "RESULT: REGRESSION",
           2: "RESULT: NOT TRUSTWORTHY"}[code])
    return code


def short(name, width=58, elide=""):
    """Template instantiations are long and differ only in the middle.

    Truncating a fixed window renders two different instantiations as the same
    string -- which is exactly the failure this tool exists to prevent, since
    the whole point is telling instantiations apart. So the caller passes the
    run of characters that is common to every name (see `common_run`) and that
    is what gets elided, leaving what differs.
    """
    if elide and elide in name and len(name) > width:
        name = name.replace(elide, "~", 1)
    if len(name) <= width:
        return name
    return name[: width - 21] + "..." + name[-18:]


def common_run(names):
    """Longest common prefix of a set of names, minus the trailing partial
    token, so eliding it cannot hide a distinguishing character."""
    names = [n for n in names if n]
    if len(names) < 2:
        return ""
    lo, hi = min(names), max(names)
    i = 0
    while i < len(lo) and i < len(hi) and lo[i] == hi[i]:
        i += 1
    pre = lo[:i]
    cut = max(pre.rfind(c) for c in "<,( ")
    pre = pre[: cut + 1] if cut > 0 else pre
    return pre if len(pre) > 12 else ""


def table(kernels, arch):
    facts = ARCH.get(arch or "")
    elide = common_run([r["name"] for r in kernels.values()])
    hdr = f"{'kernel':<58} {'vgpr':>10} {'occ':>5} {'scr':>5} {'spill':>5} {'lds':>7}"
    if facts:
        hdr += f" {'hdrm':>5}"
    print(hdr)
    print("-" * len(hdr))
    for _, r in sorted(kernels.items(), key=lambda kv: -kv[1].get("vgpr", 0)):
        vg = str(r["vgpr"])
        if facts and r["vgpr_rounded"] != r["vgpr"]:
            vg = f"{r['vgpr']}->{r['vgpr_rounded']}"
        spill = r.get("vgpr_spill", 0) + r.get("sgpr_spill", 0)
        row = (f"{short(r['name'], elide=elide):<58} {vg:>10} {r.get('occupancy', '?'):>5} "
               f"{r.get('scratch', 0):>5} {spill:>5} {r.get('lds', 0):>7}")
        if facts:
            row += f" {r['vgpr_headroom']:>5}"
        print(row)
    if elide:
        print(f"\n  ~ = {elide}")
    if facts:
        print(f"  vgpr a->b: raw -> rounded up to the {facts['vgpr_granule']}-register "
              f"allocation granule; occupancy is charged on b, not a.")
        print("  hdrm: VGPRs addable before occupancy drops a step.")
    print("  occ is the compiler's REGISTER-ONLY number. LDS can bind below it; "
          "see the lds column.")


# --- subcommands -------------------------------------------------------------

def load(path, arch):
    """Accept a build log or a previously captured .json, either way."""
    if path == "-":
        return parse(sys.stdin.read(), arch)
    if not os.path.exists(path):
        raise NotTrustworthy(f"{path}: no such file")
    if path.endswith(".json"):
        with open(path) as f:
            d = json.load(f)
        if not d.get("kernels"):
            raise NotTrustworthy(f"{path}: snapshot contains no kernels")
        if arch and d.get("arch") and d["arch"] != arch:
            raise NotTrustworthy(
                f"{path} was captured for {d['arch']}, --arch says {arch}")
        return d["kernels"]
    with open(path) as f:
        return parse(f.read(), arch)


def save(kernels, arch, path):
    with open(path, "w") as f:
        json.dump({"arch": arch, "kernels": kernels}, f, indent=1, sort_keys=True)


def cmd_capture(a):
    ks = load(a.input, a.arch)
    table(ks, a.arch)
    if a.json:
        save(ks, a.arch, a.json)
    print(f"\n{len(ks)} kernel(s)")
    return verdict(0)


def cmd_check(a):
    ks = load(a.input, a.arch)
    if not a.quiet:
        table(ks, a.arch)
    if a.json:
        save(ks, a.arch, a.json)

    fails = []
    for _, r in ks.items():
        n = short(r["name"], 40)
        # Scratch first: in a hand-scheduled kernel this is silent corruption,
        # not a slowdown, so it outranks every performance gate below it.
        if r.get("scratch", 0) and not a.allow_scratch:
            fails.append(f"{n}: ScratchSize={r['scratch']} B/lane (spilled to memory)")
        if r.get("vgpr_spill", 0) or r.get("sgpr_spill", 0):
            fails.append(f"{n}: spills vgpr={r.get('vgpr_spill',0)} sgpr={r.get('sgpr_spill',0)}")
        if a.max_vgprs and r["vgpr"] > a.max_vgprs:
            fails.append(f"{n}: vgpr={r['vgpr']} > --max-vgprs {a.max_vgprs}")
        occ = r.get("occ_by_reg", r.get("occupancy"))
        if a.min_occupancy and occ is not None and occ < a.min_occupancy:
            fails.append(f"{n}: occupancy={occ} < --min-occupancy {a.min_occupancy}")
        if a.max_lds and r.get("lds", 0) > a.max_lds:
            fails.append(f"{n}: lds={r['lds']} > --max-lds {a.max_lds}")

    print()
    if a.expect_kernels and len(ks) != a.expect_kernels:
        # Not a code regression -- the build is not the one being gated.
        raise NotTrustworthy(
            f"--expect-kernels {a.expect_kernels} but the log has {len(ks)}. "
            "A partial rebuild leaves instantiations out; make clean and rerun.")
    for f in fails:
        print(f"FAIL  {f}")
    print(f"{len(ks)} kernel(s) checked; {len(fails)} failure(s)")
    return verdict(1 if fails else 0)


def cmd_diff(a):
    before, after = load(a.before, a.arch), load(a.after, a.arch)
    only_b = set(before) - set(after)
    only_a = set(after) - set(before)
    if only_b or only_a:
        # A kernel on one side only means the two builds are not the same build.
        # Reporting "no regression" here would be a confident wrong answer.
        raise NotTrustworthy(
            f"kernel sets differ: {len(only_b)} only in before, {len(only_a)} "
            f"only in after.\n  e.g. {[short(before[k]['name'],40) for k in list(only_b)[:2]]}"
            f"{[short(after[k]['name'],40) for k in list(only_a)[:2]]}\n"
            "  Both sides must be a full rebuild of the same instantiation set.")

    rows, worse, better = [], 0, 0
    for k in before:
        b, af, deltas = before[k], after[k], []
        for f in TRIGGERS:
            if b.get(f, 0) != af.get(f, 0):
                d = af.get(f, 0) - b.get(f, 0)
                deltas.append((f, b.get(f, 0), af.get(f, 0), d))
                worse += d > 0
                better += d < 0
        occ_b = b.get("occ_by_reg", b.get("occupancy"))
        occ_a = af.get("occ_by_reg", af.get("occupancy"))
        if occ_b != occ_a:
            deltas.append(("occupancy", occ_b, occ_a, (occ_a or 0) - (occ_b or 0)))
            worse += (occ_a or 0) < (occ_b or 0)
            better += (occ_a or 0) > (occ_b or 0)
        if deltas:
            rows.append((b["name"], deltas))

    if not a.quiet:
        for name, deltas in rows:
            print(short(name))
            for f, x, y, d in deltas:
                mark = "*" if (d > 0) != (f == "occupancy") else " "
                print(f"  {mark} {f:<12} {x} -> {y} ({d:+d})")
        if rows:
            print("\n  * = worse. occupancy is register-only; a drop there is a "
                  "prediction, a spill is a fact.")
    if a.json:
        save({"before": before, "after": after}, a.arch, a.json)

    print(f"\ncompared {len(before)} kernel(s); {len(before)-len(rows)} unchanged; "
          f"{len(rows)} changed; worsened: {worse}; improved: {better}")
    return verdict(1 if worse else 0)


def cmd_explain(a):
    facts = ARCH.get(a.arch)
    if not facts:
        raise NotTrustworthy(
            f"no verified granule data for {a.arch}; known: {list(ARCH)}. "
            "Adding an arch here means measuring it, not reading a datasheet.")
    g, f = facts["vgpr_granule"], facts["vgprs_per_simd"]
    print(f"{a.arch}: {f} VGPRs/SIMD, granule {g}, max {facts['max_waves']} waves/SIMD\n")
    print(f"{'waves/SIMD':>10} {'max VGPRs':>10}")
    for w in range(facts["max_waves"], 0, -1):
        cap = (f // w) // g * g
        if cap:
            print(f"{w:>10} {cap:>10}" + ("   <-- you are here" if a.vgprs and
                  derive({'vgpr': a.vgprs}, facts)["occ_by_reg"] == w else ""))
    if a.vgprs:
        d = derive({"vgpr": a.vgprs}, facts)
        print(f"\n{a.vgprs} VGPRs -> charged as {d['vgpr_rounded']} -> "
              f"{d['occ_by_reg']} waves/SIMD; {d['vgpr_headroom']} more before the next step down.")
    return verdict(0)


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    # Accepted on either side of the subcommand: `--arch X check log` and
    # `check log --arch X` both read naturally and both get typed.
    archflag = dict(default=None,
                    help="enables granule/occupancy arithmetic; default gfx1100")
    p.add_argument("--arch", **archflag)
    sub = p.add_subparsers(dest="cmd", required=True)

    c = sub.add_parser("capture", help="parse a build log into a table/snapshot")
    c.add_argument("input", help="build log, or - for stdin")
    c.add_argument("--json")
    c.set_defaults(fn=cmd_capture)

    c = sub.add_parser("check", help="gate one build")
    c.add_argument("input")
    c.add_argument("--json")
    c.add_argument("-q", "--quiet", action="store_true")
    c.add_argument("--max-vgprs", type=int)
    c.add_argument("--min-occupancy", type=int)
    c.add_argument("--max-lds", type=int)
    c.add_argument("--expect-kernels", type=int,
                   help="fail as NOT TRUSTWORTHY if the count differs")
    c.add_argument("--allow-scratch", action="store_true",
                   help="only for kernels that do NOT hand-manage s_waitcnt")
    c.set_defaults(fn=cmd_check)

    c = sub.add_parser("diff", help="compare two builds")
    c.add_argument("before")
    c.add_argument("after")
    c.add_argument("--json")
    c.add_argument("-q", "--quiet", action="store_true")
    c.set_defaults(fn=cmd_diff)

    c = sub.add_parser("explain", help="occupancy arithmetic for an arch")
    c.add_argument("--vgprs", type=int)
    c.set_defaults(fn=cmd_explain)

    for s in sub.choices.values():
        s.add_argument("--arch", dest="sub_arch", **archflag)

    a = p.parse_args()
    a.arch = a.arch or a.sub_arch or os.environ.get("HIP_RES_ARCH", "gfx1100")
    try:
        return a.fn(a)
    except NotTrustworthy as e:
        print(f"\ncannot answer: {e}", file=sys.stderr)
        return verdict(2)


if __name__ == "__main__":
    sys.exit(main())
