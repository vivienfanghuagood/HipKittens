"""Bisect the attention kernel by dumping one intermediate at a time.

Build with -DHK_DUMP=n and run this with the same n.  Each level writes one
stage's result into the output tensor and returns from the kernel, so a
mismatch localises the defect to exactly one stage:

    1  K read back out of LDS      -- G::load + lds_read_frag
    2  V^T read back out of LDS    -- load + transpose_sep + lds_write_frag
                                      + lds_read_frag
    3  S^T after the QK matmul     -- the Q global load, mma_ABt_base, and the
                                      epilogue transpose/store path
    9  S^T after the scale fold
    4  P^T after the online softmax
    7  q  global -> register -> global, no LDS
    8  k  the same, one 16-row tile per warp
    5  K in LDS via the library load(); 6 the same for V^T
   10  warp 0's raw accumulator VGPRs before the scale fold
   11  the same after it

Levels 1, 2, 5 and 6 only make sense for the first KV block, so the comparison
uses k[:KV_BLOCK] / v[:KV_BLOCK].  Levels 3/4/9 cover the first Q tile against
the first KV block; 10/11 cover warp 0's 16 q rows.

10 and 11 bypass transpose() and store() entirely, writing one register per
(lane, slot).  That is the level to reach for when a store-path level passes but
something downstream disagrees with it -- a scheduling race between an in-flight
ds_read and the WMMA that consumes it is invisible through the store path.

    python3 dump.py <level> [--kv 32] [--q-tile 128] [--n 256]
"""

import argparse
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common import D_HEAD, make_qkv  # noqa: E402

import tk_kernel  # noqa: E402


def report(name, got, want):
    got = got.float()
    want = want.float()
    err = (got - want).abs()
    den = want.abs().clamp_min(1e-6)
    rel = (err / den).max().item()
    bad = (err > 1e-2 * den.clamp_min(1.0)).float()
    print(f"{name}: max|err| {err.max().item():.4e}  max rel {rel:.4e}  "
          f"mismatched {int(bad.sum().item())}/{bad.numel()}")
    if bad.sum() > 0:
        idx = bad.flatten().nonzero()[:8].flatten().tolist()
        flat_g, flat_w = got.flatten(), want.flatten()
        cols = want.shape[-1]
        for i in idx:
            print(f"    [{i // cols:>4},{i % cols:>4}]  got {flat_g[i]:>12.5f}"
                  f"   want {flat_w[i]:>12.5f}")
        # Which rows/columns are wrong tells us which fragment is wrong.
        rows = bad.any(dim=-1).nonzero().flatten().tolist()
        cs = bad.any(dim=0).nonzero().flatten().tolist()
        print(f"    bad rows {rows[:24]}{' ...' if len(rows) > 24 else ''}")
        print(f"    bad cols {cs[:24]}{' ...' if len(cs) > 24 else ''}")
    return bad.sum().item() == 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("level", type=int)
    ap.add_argument("--kv", type=int, default=32)
    ap.add_argument("--q-tile", type=int, default=128)
    ap.add_argument("--n", type=int, default=256)
    ap.add_argument("--ramp", choices=["row", "col"], default=None,
                    help="replace q/k/v with an exactly-representable row-index "
                         "or column-index ramp and print the raw output, so a "
                         "wrong layout reads off directly")
    args = ap.parse_args()

    torch.manual_seed(0)
    b, h, n = 1, 1, args.n
    q, k, v = make_qkv(b, h, n)
    if args.ramp:
        n_, d_ = q.shape[2], q.shape[3]
        r = torch.arange(n_, device="cuda").float()[:, None]
        c = torch.arange(d_, device="cuda").float()[None, :]
        q[0, 0] = (r * 0 + c if args.ramp == "col" else r + c * 0).bfloat16()
        k[0, 0] = q[0, 0]
        v[0, 0] = q[0, 0]
    o = torch.zeros_like(q)
    tk_kernel.dispatch_micro(q, k, v, o, 0.0, False)
    torch.cuda.synchronize()

    if args.ramp:
        torch.set_printoptions(precision=4, sci_mode=False, linewidth=200)
        qh = q[0, 0].view(torch.uint16)
        oh = o[0, 0].view(torch.uint16)
        for rr in (0, 1, 2, 15, 16, 17, 64, 80, 112):
            print(f"r{rr:>4} q {[hex(x) for x in qh[rr, :6].tolist()]}"
                  f"   o {[hex(x) for x in oh[rr, :6].tolist()]}"
                  f"  o[{rr},120:126] {[hex(x) for x in oh[rr, 120:126].tolist()]}")
        of = o[0, 0].float()
        print(f"ramp={args.ramp}  o shape {tuple(of.shape)}")
        print("o[:,0]   ", of[:, 0].tolist())
        print("o[:,1]   ", of[:, 1].tolist())
        print("o[:,8]   ", of[:, 8].tolist())
        print("o[0,:]   ", of[0].tolist())
        print("o[1,:]   ", of[1].tolist())
        print("o[16,:]  ", of[16].tolist())
        return 0

    kv, qt, d = args.kv, args.q_tile, D_HEAD
    if args.level == 7:
        ok = report("q identity", o[0, 0, :qt, :d], q[0, 0, :qt, :d])
    elif args.level == 8:
        ok = report("k identity", o[0, 0, :qt, :d], k[0, 0, :qt, :d])
    elif args.level in (1, 5):
        ok = report("K in LDS", o[0, 0, :kv, :d], k[0, 0, :kv, :d])
    elif args.level in (2, 6):
        ok = report("V^T in LDS", o[0, 0, :d, :kv], v[0, 0, :kv, :d].T)
    elif args.level in (3, 4, 9):
        raw = q[0, 0, :qt].float() @ k[0, 0, :kv].float().T
        x = raw * (d ** -0.5) * 1.4426950408889634
        m = x.max(dim=-1, keepdim=True).values
        want = {3: raw, 9: x, 4: torch.exp2(x - m)}[args.level]
        name = {3: "S^T raw", 9: "S^T scaled", 4: "P^T"}[args.level]
        ok = report(name, o[0, 0, :qt, :kv], want)
    elif args.level in (10, 11):
        # Raw per-lane accumulator registers.  Under the gfx11 fp32 col layout
        # slot s of lane l holds S^T[2*s + (l >> 4), l & 15], i.e. kv = 2s + half
        # and q = l % 16 -- so the whole 16x32 block should reconstruct exactly.
        raw = q[0, 0, :16].float() @ k[0, 0, :kv].float().T
        want = raw if args.level == 10 else raw * (d ** -0.5) * 1.4426950408889634
        got = o[0, 0, :32, :16].float()
        rebuilt = torch.empty_like(want)
        for lane in range(32):
            for slot in range(16):
                rebuilt[lane & 15, 2 * slot + (lane >> 4)] = got[lane, slot]
        ok = report("S^T raw regs" if args.level == 10 else "S^T scaled regs",
                    rebuilt, want)
        print(f"  skip/kv_start/N = {o[0, 0, 0, 16:19].float().tolist()}")
    else:
        raise SystemExit(f"no dump level {args.level}")

    print("OK" if ok else "MISMATCH")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
