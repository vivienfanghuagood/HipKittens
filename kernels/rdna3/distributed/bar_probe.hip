// Why does an N-way device barrier over IPC cost seconds?
//
// gemm_rs_mp needs a cross-rank barrier that rides the stream, because the host
// bootstrap costs milliseconds and the GEMMs being timed cost about the same.
// The obvious barrier -- publish my epoch to every peer, then spin until every
// peer's epoch reaches mine -- measured seconds each against a 1.7 ms GEMM.
//
//                          --- what was measured ---
//
// Arrangements of the same barrier over a peer-mapped *device* flag, 2 ranks,
// nothing else on the GPUs (ms per barrier):
//
//   A  one kernel does both, on the work stream        987 - 8823
//   B  split into two kernels, same stream            5074 - 9507
//   C  split, waiter on its own stream, enqueued first   0.08
//   D  C, 20 back to back                                1.07
//   E  D + event chain (work stream waits on waiter)  4880 - 5323
//   F  E, signalling with an exchange not a store     5447 - 5765
//   G  E, host stream-sync instead of the event      13691
//   H  lockstep ping-pong, work stream idle           > 800 s (killed)
//
// C and D are the trap. Their signals are enqueued independently of their
// waits, so all 20 signals run immediately and an early epoch's wait is
// satisfied by a *later* epoch's store -- no round trip ever happens, and the
// number measures nothing. Every arrangement in which my signal for epoch i
// cannot issue until my wait for epoch i-1 has returned -- which is what a
// barrier is -- costs seconds. H removes every other variable (one kernel, one
// stream, work stream idle, no events, no host) and is the worst of all.
//
// So it is not the stream topology, not the split, not the event, and not the
// release store. It is the medium: a store from one GPU into another's
// fine-grained device memory is not seen by a spinning reader on any timescale
// a kernel can wait for. The ISA says why -- a system-scope acquire load is
// `global_load_b32 glc` + `buffer_gl1_inv` + `buffer_gl0_inv`, which invalidate
// L0 and L1 and nothing below -- so the line sits until something evicts it.
// Arrangements A-G differ only in how long that takes.
//
// I and J are the fix: the same lockstep round trip, and then a real N-way
// barrier, with the flags in host memory (ipc_heap.cuh's host_flags), which is
// uncached by the GPU and reached over PCIe by every rank.
//
// Also note: clock64() on gfx1100 is `s_getreg_b32 hwreg(HW_REG_SHADER_CYCLES,
// 0, 20)` -- a *20-bit* counter, wrapping every ~0.5 ms, with the high word
// forced to zero. Every deadline built on it is meaningless, which is why the
// runs above reported "completed" after five seconds against a two-second
// budget. wall_clock64() is the one to use: on this target it is
// `s_sendmsg_rtn_b64 MSG_RTN_GET_REALTIME`, a real 64-bit constant-rate
// counter, whose rate is hipDeviceAttributeWallClockRate.
//
//   HIP_VISIBLE_DEVICES=0,3 ./run_mp.sh 2 ./bar_probe
//   HK_BAR_ALL=1 HIP_VISIBLE_DEVICES=0,3 ./run_mp.sh 2 ./bar_probe   # + A,B,E,F
#include "ipc_heap.cuh"

#include <chrono>
#include <cstdio>
#include <cstdlib>

using namespace hk_dist;

// clock64() is not volatile, so the compiler is free to hoist it out of a spin
// loop -- and it does. An early run of this probe reported "completed" for a
// barrier that took 5.6 s against a 2 s budget, which is impossible unless the
// deadline check had been folded to a constant. Every spin loop below therefore
// ends in a memory clobber, which costs nothing and keeps the reload.
#define SPIN_TICK __asm__ __volatile__("" ::: "memory")

__global__ void sig(sym_view v, uint32_t *flags, uint32_t epoch, int world,
                    int me) {
    const int r = threadIdx.x;
    if (r >= world) return;
    __hip_atomic_store(v.translate(flags + me, r), epoch, __ATOMIC_RELEASE,
                       __HIP_MEMORY_SCOPE_SYSTEM);
}

__global__ void wai(const uint32_t *flags, uint32_t epoch, int world,
                    long long budget, unsigned *tmo) {
    const int r = threadIdx.x;
    if (r >= world) return;
    const long long t0 = wall_clock64();
    while (__hip_atomic_load(flags + r, __ATOMIC_ACQUIRE,
                             __HIP_MEMORY_SCOPE_SYSTEM) < epoch) {
        SPIN_TICK;
        if (wall_clock64() - t0 > budget) { atomicExch(tmo, epoch); return; }
    }
}

// Both halves in one launch.
__global__ void sigwai(sym_view v, uint32_t *flags, uint32_t epoch, int world,
                       int me, long long budget, unsigned *tmo) {
    const int r = threadIdx.x;
    if (r >= world) return;
    __hip_atomic_store(v.translate(flags + me, r), epoch, __ATOMIC_RELEASE,
                       __HIP_MEMORY_SCOPE_SYSTEM);
    const long long t0 = wall_clock64();
    while (__hip_atomic_load(flags + r, __ATOMIC_ACQUIRE,
                             __HIP_MEMORY_SCOPE_SYSTEM) < epoch) {
        SPIN_TICK;
        if (wall_clock64() - t0 > budget) { atomicExch(tmo, epoch); return; }
    }
}

// The same signal as an RMW rather than a store: an exchange cannot sit in a
// write-combine buffer, it has to go to memory and come back.
__global__ void sig_rmw(sym_view v, uint32_t *flags, uint32_t epoch, int world,
                        int me) {
    const int r = threadIdx.x;
    if (r >= world) return;
    __hip_atomic_exchange(v.translate(flags + me, r), epoch, __ATOMIC_ACQ_REL,
                          __HIP_MEMORY_SCOPE_SYSTEM);
}

// True lockstep, one kernel, one stream, work stream idle: rank 0 stores then
// waits, every other rank waits then stores. Nothing here can be satisfied by a
// later epoch racing ahead, so this is the round-trip latency of a flag with
// every other variable removed.
__global__ void pingpong_dev(sym_view v, uint32_t *flags, uint32_t epoch,
                             int world, int me, long long budget,
                             unsigned *tmo) {
    const int r = threadIdx.x;
    if (r >= world || r == me) return;   // my own slot is mine to write
    auto send = [&]() {
        __hip_atomic_store(v.translate(flags + me, r), epoch, __ATOMIC_RELEASE,
                           __HIP_MEMORY_SCOPE_SYSTEM);
    };
    auto recv = [&]() {
        const long long t0 = wall_clock64();
        while (__hip_atomic_load(flags + r, __ATOMIC_ACQUIRE,
                                 __HIP_MEMORY_SCOPE_SYSTEM) < epoch) {
            SPIN_TICK;
            if (wall_clock64() - t0 > budget) { atomicExch(tmo, epoch); return; }
        }
    };
    if (me == 0) { send(); recv(); } else { recv(); send(); }
}

// The same thing with the flags in host memory. No translate(): there is one
// physical array, mapped by every rank, so flags[i] means the same word
// everywhere and a rank simply writes its own slot.
__global__ void pingpong_host(uint32_t *flags, uint32_t epoch, int world,
                              int me, long long budget, unsigned *tmo) {
    const int r = threadIdx.x;
    if (r >= world || r == me) return;   // my own slot is mine to write
    auto send = [&]() {
        __hip_atomic_store(flags + me, epoch, __ATOMIC_RELEASE,
                           __HIP_MEMORY_SCOPE_SYSTEM);
    };
    auto recv = [&]() {
        const long long t0 = wall_clock64();
        while (__hip_atomic_load(flags + r, __ATOMIC_ACQUIRE,
                                 __HIP_MEMORY_SCOPE_SYSTEM) < epoch) {
            SPIN_TICK;
            if (wall_clock64() - t0 > budget) { atomicExch(tmo, epoch); return; }
        }
    };
    if (me == 0) { send(); recv(); } else { recv(); send(); }
}

// A real N-way barrier on host flags: thread r publishes my epoch if r is me,
// and every thread waits for rank r's. This is what the fused loop would call.
__global__ void barrier_host(uint32_t *flags, uint32_t epoch, int world, int me,
                             long long budget, unsigned *tmo) {
    const int r = threadIdx.x;
    if (r >= world) return;
    if (r == me)
        __hip_atomic_store(flags + me, epoch, __ATOMIC_RELEASE,
                           __HIP_MEMORY_SCOPE_SYSTEM);
    const long long t0 = wall_clock64();
    while (__hip_atomic_load(flags + r, __ATOMIC_ACQUIRE,
                             __HIP_MEMORY_SCOPE_SYSTEM) < epoch) {
        SPIN_TICK;
        if (wall_clock64() - t0 > budget) { atomicExch(tmo, epoch); return; }
    }
}

// The device-flag twin of barrier_host, so the two media are compared through
// the same code path rather than through two different arrangements.
__global__ void barrier_dev(sym_view v, uint32_t *flags, uint32_t epoch,
                            int world, int me, long long budget,
                            unsigned *tmo) {
    const int r = threadIdx.x;
    if (r >= world) return;
    __hip_atomic_store(v.translate(flags + me, r), epoch, __ATOMIC_RELEASE,
                       __HIP_MEMORY_SCOPE_SYSTEM);
    const long long t0 = wall_clock64();
    while (__hip_atomic_load(flags + r, __ATOMIC_ACQUIRE,
                             __HIP_MEMORY_SCOPE_SYSTEM) < epoch) {
        SPIN_TICK;
        if (wall_clock64() - t0 > budget) { atomicExch(tmo, epoch); return; }
    }
}

int main() {
    const int rank = atoi(getenv("HK_RANK")), world = atoi(getenv("HK_WORLD"));
    const bool all = getenv("HK_BAR_ALL") != nullptr;
    file_bootstrap bs(rank, world);
    ipc_heap heap(64u << 20, rank, bs);
    uint32_t *flags = heap.allocate<uint32_t>(MAX_RANKS);
    unsigned *tmo = heap.allocate<unsigned>(1);
    heap.check_symmetry();
    host_flags hf(MAX_RANKS, bs);
    uint32_t *const hflags = hf.device();
    const sym_view V = heap.view();
    const hipStream_t S = heap.stream();
    hipStream_t W; HKD_CHECK(hipStreamCreate(&W));
    setvbuf(stdout, nullptr, _IONBF, 0);

    // A budget in wall ticks, not shader cycles. 200 ms is ~2000x anything that
    // could be called a working barrier, so a timeout here is a real failure
    // and not a tight bound being unlucky.
    int khz = 0;
    HKD_CHECK(hipDeviceGetAttribute(&khz, hipDeviceAttributeWallClockRate,
                                    heap.device()));
    const long long budget = (long long)khz * 200;
    if (rank == 0)
        printf("wall clock %d kHz, budget %lld ticks (200 ms)\n\n", khz, budget);

    uint32_t epoch = 0;
    auto report = [&](const char *what, double ms_each) {
        unsigned t = 0;
        HKD_CHECK(hipMemcpy(&t, tmo, 4, hipMemcpyDeviceToHost));
        HKD_CHECK(hipMemset(tmo, 0, 4));
        if (rank == 0)
            printf("%-38s %11.3f ms each   %s\n", what, ms_each,
                   t ? "TIMED OUT" : "completed");
    };

    auto timed = [&](int reps, auto &&one) {
        heap.barrier();
        const auto t0 = std::chrono::steady_clock::now();
        for (int i = 0; i < reps; i++) one();
        HKD_CHECK(hipStreamSynchronize(S));
        HKD_CHECK(hipStreamSynchronize(W));
        const auto t1 = std::chrono::steady_clock::now();
        heap.barrier();
        return std::chrono::duration<double, std::milli>(t1 - t0).count() / reps;
    };

    if (all) {
        // A: one kernel does both.
        double ms = timed(3, [&]() {
            hipLaunchKernelGGL(sigwai, dim3(1), dim3(MAX_RANKS), 0, S, V, flags,
                               ++epoch, world, rank, budget, tmo);
        });
        report("A  one kernel, one stream", ms);

        // B: split, both halves on the same stream.
        ms = timed(3, [&]() {
            ++epoch;
            hipLaunchKernelGGL(sig, dim3(1), dim3(MAX_RANKS), 0, S, V, flags,
                               epoch, world, rank);
            hipLaunchKernelGGL(wai, dim3(1), dim3(MAX_RANKS), 0, S, flags, epoch,
                               world, budget, tmo);
        });
        report("B  split, same stream", ms);
    }

    // C: split, waiter on its own stream, launched first. Fast for the wrong
    // reason -- kept because that is the trap this file exists to document.
    double ms = timed(20, [&]() {
        ++epoch;
        hipLaunchKernelGGL(wai, dim3(1), dim3(MAX_RANKS), 0, W, flags, epoch,
                           world, budget, tmo);
        hipLaunchKernelGGL(sig, dim3(1), dim3(MAX_RANKS), 0, S, V, flags, epoch,
                           world, rank);
    });
    report("C  split, waiter own stream (no r/t)", ms);

    if (all) {
        static constexpr int NEV = 64;
        hipEvent_t WE[NEV];
        for (auto &e : WE)
            HKD_CHECK(hipEventCreateWithFlags(&e, hipEventDisableTiming));
        // E: C plus the event chain that makes it an actual barrier.
        ms = timed(3, [&]() {
            ++epoch;
            hipLaunchKernelGGL(wai, dim3(1), dim3(MAX_RANKS), 0, W, flags, epoch,
                               world, budget, tmo);
            hipLaunchKernelGGL(sig, dim3(1), dim3(MAX_RANKS), 0, S, V, flags,
                               epoch, world, rank);
            hipEvent_t &e = WE[epoch % NEV];
            HKD_CHECK(hipEventRecord(e, W));
            HKD_CHECK(hipStreamWaitEvent(S, e, 0));
        });
        report("E  C + event chain S<-W", ms);

        // F: E, signalling with an exchange.
        ms = timed(3, [&]() {
            ++epoch;
            hipLaunchKernelGGL(wai, dim3(1), dim3(MAX_RANKS), 0, W, flags, epoch,
                               world, budget, tmo);
            hipLaunchKernelGGL(sig_rmw, dim3(1), dim3(MAX_RANKS), 0, S, V, flags,
                               epoch, world, rank);
            hipEvent_t &e = WE[epoch % NEV];
            HKD_CHECK(hipEventRecord(e, W));
            HKD_CHECK(hipStreamWaitEvent(S, e, 0));
        });
        report("F  E, signal by exchange", ms);
        for (auto &e : WE) (void)hipEventDestroy(e);
    }

    // H: lockstep round trip through a peer device flag. Bounded now, so it
    // reports a timeout in 200 ms instead of running for the wall clock.
    ms = timed(3, [&]() {
        hipLaunchKernelGGL(pingpong_dev, dim3(1), dim3(MAX_RANKS), 0, W, V,
                           flags, ++epoch, world, rank, budget, tmo);
    });
    report("H  lockstep r/t, device flag", ms);

    // I: the same round trip through a host flag. Epoch restarts because the
    // host flags are a different array and start at zero.
    epoch = 0;
    ms = timed(20, [&]() {
        hipLaunchKernelGGL(pingpong_host, dim3(1), dim3(MAX_RANKS), 0, W, hflags,
                           ++epoch, world, rank, budget, tmo);
    });
    report("I  lockstep r/t, host flag", ms);

    // J: an N-way barrier, on each medium, through identical code. Back to back
    // on one stream, so barrier i+1's store cannot happen until barrier i has
    // returned -- a real chain of round trips, not C's race-ahead.
    ms = timed(20, [&]() {
        hipLaunchKernelGGL(barrier_host, dim3(1), dim3(MAX_RANKS), 0, W, hflags,
                           ++epoch, world, rank, budget, tmo);
    });
    report("J  N-way barrier, host flag", ms);

    ms = timed(20, [&]() {
        hipLaunchKernelGGL(barrier_dev, dim3(1), dim3(MAX_RANKS), 0, W, V, flags,
                           ++epoch, world, rank, budget, tmo);
    });
    report("K  N-way barrier, device flag", ms);

    heap.barrier();
    HKD_CHECK(hipStreamDestroy(W));
    return 0;
}
