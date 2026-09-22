#!/usr/bin/env bash
# Launch one process per rank. torchrun without torch.
#
#   ./run_mp.sh 2 ./ipc_heap_test 64
#
# HIP_VISIBLE_DEVICES is the caller's business and every rank inherits the same
# one: rank r drives visible device r, but it has to be able to *see* the others
# because hipIpcOpenMemHandle needs to reach the exporting device. The node is
# shared, so pin it -- HIP_VISIBLE_DEVICES=0,3 ./run_mp.sh 2 ...
#
# Every rank runs under a timeout. A fused kernel waits on a peer flag, and the
# failure mode of a peer that died is the survivor spinning in a kernel that
# nothing will ever satisfy; on a shared node the cure for that is a device
# reset, so the bound is not optional.
set -u

if [ $# -lt 2 ]; then
    echo "usage: $0 <world> <program> [args...]" >&2
    exit 2
fi

WORLD=$1; shift
: "${HK_MP_TIMEOUT:=900}"

# Shared by every rank, different for every run. A stale id would let a previous
# run's IPC handles be read as this run's -- they would open successfully and
# name freed memory, which is the one bootstrap failure that is silent.
export HK_DIST_RUN_ID="$$-$(date +%s)"
export HK_WORLD=$WORLD

pids=()
for ((r = 0; r < WORLD; r++)); do
    HK_RANK=$r timeout --signal=KILL "$HK_MP_TIMEOUT" "$@" &
    pids+=($!)
done

rc=0
for p in "${pids[@]}"; do
    wait "$p" || rc=$?
done

rm -rf "/tmp/hk_dist_${HK_DIST_RUN_ID}"
exit $rc
