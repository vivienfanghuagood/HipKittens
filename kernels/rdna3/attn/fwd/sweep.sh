#!/bin/bash
# Build attn.cpp once per tiling configuration and report register usage plus
# TFLOPs, so the tile shape is picked from measurements rather than from a
# register-pressure estimate. Each line is one configuration.
#
#   ./sweep.sh "Q_BLOCK=16 KV_BLOCK=64 QK_TILES=2 PV_TILES=2" ...
#
# With no arguments it runs the default grid below.
#
# Two things the table is for beyond the TFLOPs. `spill` must be 0: this kernel
# hand-manages s_waitcnt, so a compiler-inserted spill does not slow it down, it
# silently corrupts the result -- which is why quickbench.py checks before it
# times, and why a config can come back WRONG rather than merely slow. And
# `occ` is the number the whole tiling argument turns on, because the only thing
# hiding LDS latency here is other waves.
set -u
QB_ARGS="${QB_ARGS:-}"

run() {
    local desc="$1"
    local flags=""
    for kv in $desc; do flags="$flags -D${kv}"; done
    make clean >/dev/null 2>&1
    if ! make EXTRA_HIPFLAGS="$flags" >/tmp/attn_build.log 2>&1; then
        printf '%-64s | BUILD FAILED: %s\n' "$desc" \
            "$(grep -m1 -E 'error|static_assert|Error' /tmp/attn_build.log | cut -c1-90)"
        return
    fi
    local vgpr spill scratch occ lds out
    # The remark lines are prefixed with file:line:col, so anchor on the label.
    vgpr=$(sed -n 's/.*remark: *VGPRs: \([0-9]*\).*/\1/p'        /tmp/attn_build.log | head -1)
    spill=$(sed -n 's/.*remark: *VGPRs Spill: \([0-9]*\).*/\1/p' /tmp/attn_build.log | head -1)
    scratch=$(sed -n 's/.*remark: *ScratchSize \[bytes\/lane\]: \([0-9]*\).*/\1/p' /tmp/attn_build.log | head -1)
    occ=$(sed -n 's/.*remark: *Occupancy \[waves\/SIMD\]: \([0-9]*\).*/\1/p' /tmp/attn_build.log | head -1)
    lds=$(sed -n 's/.*remark: *LDS Size \[bytes\/block\]: \([0-9]*\).*/\1/p' /tmp/attn_build.log | head -1)
    out=$(HIP_VISIBLE_DEVICES=${HIP_VISIBLE_DEVICES:-0} timeout --signal=KILL 900 \
          python3 quickbench.py $QB_ARGS 2>&1 | tail -1)
    printf '%-64s | vgpr=%-3s spill=%-2s scr=%-3s occ=%-2s lds=%-5s | %s\n' \
        "$desc" "$vgpr" "$spill" "$scratch" "$occ" "$lds" "$out"
}

if [ $# -gt 0 ]; then
    for cfg in "$@"; do run "$cfg"; done
    exit 0
fi

echo "config                                                           | registers                          | TFLOPS N=4096 / N=16384"

# --- the KV block, at the shipped Q_BLOCK ---------------------------------
# Wider KV means fewer trips round the softmax and fewer barriers per unit of
# math, and costs 16 VGPRs per extra 16 rows in s_t plus 8 in p_t.
run "Q_BLOCK=16 KV_BLOCK=32  QK_TILES=2 PV_TILES=2"
run "Q_BLOCK=16 KV_BLOCK=64  QK_TILES=2 PV_TILES=2"
run "Q_BLOCK=16 KV_BLOCK=64  QK_TILES=4 PV_TILES=2"
run "Q_BLOCK=16 KV_BLOCK=128 QK_TILES=2 PV_TILES=2"

# --- the Q block ----------------------------------------------------------
# Q_BLOCK is the only knob that amortizes the LDS traffic: every warp reads the
# whole K and V^T block whatever its Q is, so doubling Q_BLOCK halves LDS reads
# per FLOP. It costs 64 VGPRs (q and o_t both grow), which is why it is the one
# most likely to spill.
run "Q_BLOCK=32 KV_BLOCK=32  QK_TILES=2 PV_TILES=2"
run "Q_BLOCK=32 KV_BLOCK=32  QK_TILES=1 PV_TILES=1"
run "Q_BLOCK=32 KV_BLOCK=64  QK_TILES=1 PV_TILES=1"

# --- the read batch size --------------------------------------------------
# How many fragments go in flight before the s_waitcnt. Bigger hides more LDS
# latency under the WMMAs and costs 8 VGPRs per fragment.
run "Q_BLOCK=16 KV_BLOCK=32  QK_TILES=1 PV_TILES=1"
run "Q_BLOCK=16 KV_BLOCK=32  QK_TILES=2 PV_TILES=4"
run "Q_BLOCK=16 KV_BLOCK=32  QK_TILES=4 PV_TILES=4"
run "Q_BLOCK=16 KV_BLOCK=64  QK_TILES=4 PV_TILES=4"

# --- warps per workgroup --------------------------------------------------
# Fewer warps is a smaller Q tile, so more workgroups and finer grid
# quantization, but each warp's share of the staging grows.
run "Q_BLOCK=16 KV_BLOCK=32  QK_TILES=2 PV_TILES=2 NUM_WARPS=4 VT_D_CHUNK=32"
run "Q_BLOCK=32 KV_BLOCK=32  QK_TILES=2 PV_TILES=2 NUM_WARPS=4 VT_D_CHUNK=32"
run "Q_BLOCK=16 KV_BLOCK=64  QK_TILES=2 PV_TILES=2 NUM_WARPS=4 VT_D_CHUNK=32"

# --- occupancy floor ------------------------------------------------------
# Asking for 2 workgroups per CU caps VGPRs at 128 and will spill at these
# tilings; it is here to show what that costs rather than because it can win.
run "Q_BLOCK=16 KV_BLOCK=32  QK_TILES=2 PV_TILES=2 MIN_BLOCKS_PER_CU=2"
