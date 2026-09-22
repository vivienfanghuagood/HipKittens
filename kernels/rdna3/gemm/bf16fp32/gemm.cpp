// bf16 GEMM with fp32 accumulate for gfx1100 (RDNA3).
//
// Structurally this follows kernels/cdna4/gemm/bf16fp32, but the schedule is
// redesigned for RDNA3, where three things are different:
//
//   * wave32, and a WMMA operand is mirrored across the two wave halves, so a
//     bf16 rt_base costs the same 8 VGPRs as an fp32 one. Operand registers,
//     not accumulator registers, are what bounds the tile here.
//   * no global->LDS DMA. Every byte of a prefetch passes through VGPRs, so a
//     "async copy" is load_global_to_register_buffer now and
//     store_register_buffer_to_shared at the point the buffer is needed.
//   * one XCD, so there is no chiplet swizzle -- only the L2 group swizzle.
//
// B is passed pre-transposed as (n, k), which keeps both shared loads
// contiguous and makes mma_ABt the right primitive.

#include <utility>

#include "kittens.cuh"
// pyutils pulls in pybind11 and the Python headers. The distributed kernel
// includes this file for the tuned inner loop and has its own host harness, so
// both the include and the module below are opt-out -- otherwise a build with
// no Python development headers could not use the GEMM at all.
#ifndef HK_GEMM_NO_PYBIND
#include "pyutils/pyutils.cuh"
#endif
using namespace kittens;

// Tiling knobs. The defaults are the shape this file is named for; the sweep
// script overrides them on the command line.
#ifndef BLOCK_M
#define BLOCK_M 128
#endif
#ifndef BLOCK_N
#define BLOCK_N 128
#endif
#ifndef K_STEP
#define K_STEP 32
#endif
#ifndef DOT_SLICE
#define DOT_SLICE 16
#endif
#ifndef NUM_WARPS
#define NUM_WARPS 8
#endif
#ifndef WARP_ROWS
#define WARP_ROWS 4
#endif
// How many N-direction chunks the warp's operand tile and accumulator are split
// into. See the wait schedule in dot_tile(); 1 reproduces the unsplit loop.
#ifndef N_SPLIT
#define N_SPLIT 2
#endif
// How many K-tiles of global data are kept in flight in the staging registers.
// 1 is the classic one-tile-deep pipeline: the loads for tile t+2 are issued
// just before tile t's math and waited on at the top of the next iteration, so
// the distance they get to cover is one K-tile of WMMAs. Deeper costs
// GPREFETCH * (stage_a + stage_b) float4s and buys a proportionally longer
// shadow, which is what a short K-tile needs.
#ifndef GPREFETCH
#define GPREFETCH 1
#endif
// Where in the iteration the staged K-tile gets written to LDS.
//   0  before the math, which is where the obvious loop structure puts it
//   1  after the math, so the tile's ds_reads are not queued behind the writes
//   2  inside the math, at the point where the last ds_read has been issued:
//      late enough not to delay a read, early enough that the last K-slice's
//      WMMAs cover the write latency instead of the barrier doing it
// 1 is the default because it measures fastest (79 vs 78 TFLOPs at 8192^2x4096).
// 2 is the more interesting schedule and it is correct, but it buys nothing: the
// writes are only 1/8 of the tile's LDS traffic, so hiding them under the last
// K-slice's WMMAs just moves the same cycles around inside a pipe that is
// already the bottleneck.
#ifndef WRITE_POS
#define WRITE_POS 1
#endif
// Rotate each chunk's next-slice reads into the slot the chunk just freed.
#ifndef ROTATE
#define ROTATE 1
#endif
#ifndef MIN_BLOCKS_PER_CU
#define MIN_BLOCKS_PER_CU 1
#endif
// How many block-rows the L2 swizzle walks before moving on. Only affects which
// workgroups are co-resident, not what they compute.
#ifndef WGM
#define WGM 8
#endif

// Ablation switches for the inner loop. Setting any of these makes the result
// numerically wrong on purpose -- they exist to time one stage of the pipeline
// at a time, see sweep.sh.
#ifndef ABLATE_GLOBAL
#define ABLATE_GLOBAL 0
#endif
#ifndef ABLATE_LDS_READ
#define ABLATE_LDS_READ 0
#endif
#ifndef ABLATE_MMA
#define ABLATE_MMA 0
#endif
#ifndef ABLATE_LDS_WRITE
#define ABLATE_LDS_WRITE 0
#endif
#ifndef ABLATE_BARRIER
#define ABLATE_BARRIER 0
#endif

// A constexpr for loop, so the loop index can be a template argument. lds_wait<N>
// needs one; #pragma unroll does not give you that.
template<int N, typename F> __device__ inline void static_for(F &&f) {
    [&]<int... I>(std::integer_sequence<int, I...>) {
        (f(std::integral_constant<int, I>{}), ...);
    }(std::make_integer_sequence<int, N>{});
}

// How many LDS reads are still in flight once chunk (k, c)'s operands have
// landed, under the rotated issue schedule in dot_tile(). Reads retire in
// order, so this is just "how many were issued after the last one this chunk
// needs". Deriving it here rather than writing lds_wait<0> everywhere is the
// entire point of the schedule: a nonzero value is math running on top of LDS.
constexpr int reads_in_flight(int k, int c, int n_split, int num_slices, int chunk_reads) {
    const bool has_next = (k + 1 < num_slices);
    // First slice: its own B chunks were all issued up front, so the tail of
    // them is still moving, plus whatever has been rotated in behind them.
    if (k == 0) return (n_split - 1 - c + (has_next ? c : 0)) * chunk_reads;
    // Later slices: A was issued after B chunks 0..n-2 and before B chunk n-1,
    // so for every chunk but the last, A is the read that gates.
    if (c < n_split - 1) return ((has_next ? c : 0) + 1) * chunk_reads;
    return (has_next ? (n_split - 1) : 0) * chunk_reads;
}

// Pin a register tile's values as live without emitting an instruction. The
// ablations below delete whichever stage they are measuring, and without this
// the compiler would helpfully delete the stage feeding it too, so e.g.
// ABLATE_MMA would also drop every ds_read and measure nothing.
template<typename RT> __device__ inline void keep_live(RT &t) {
    #pragma unroll
    for (int i = 0; i < RT::height; i++)
        #pragma unroll
        for (int j = 0; j < RT::width; j++)
            #pragma unroll
            for (int k = 0; k < RT::packed_per_tile; k++)
                asm volatile("" :: "v"(t.tiles[i][j].data[k]));
}

using _gl_A = gl<bf16, -1, -1, -1, -1>;
using _gl_B = gl<bf16, -1, -1, -1, -1>;
using _gl_C = gl<bf16, -1, -1, -1, -1>;

struct micro_globals {
    _gl_A a;
    _gl_B b;
    _gl_C c;
    hipStream_t stream;   // a: (m, k), b: (n, k), c: (m, n)

    // The epilogue is selected on this rather than on a template parameter of
    // the kernel, so that a caller with a different output policy supplies a
    // different globals type and nothing else changes. See
    // kernels/rdna3/distributed/gemm_rs.hip, whose globals set it true to turn
    // the store into a reduce-scatter push.
    static constexpr bool FUSED_RS = false;
};

// The command line sets the tunables as macros, and the config struct below
// wants the same names as members, so capture the values and then get the macros
// out of the way.
namespace cli {
constexpr int block_m = BLOCK_M, block_n = BLOCK_N, k_step = K_STEP,
              dot_slice = DOT_SLICE, num_warps = NUM_WARPS,
              warp_rows = WARP_ROWS, wgm = WGM, n_split = N_SPLIT,
              gprefetch = GPREFETCH, write_pos = WRITE_POS;
}
#undef WRITE_POS
#undef GPREFETCH
#undef N_SPLIT
#undef BLOCK_M
#undef BLOCK_N
#undef K_STEP
#undef DOT_SLICE
#undef NUM_WARPS
#undef WARP_ROWS
#undef WGM

// One tiling. The kernel is templated on this rather than compiled from the
// macros directly, because no single tiling wins across shapes -- see the table
// in README.md and the selection rule in dispatch_micro below.
template<int _BLOCK_M, int _BLOCK_N, int _K_STEP, int _DOT_SLICE,
         int _NUM_WARPS, int _WARP_ROWS, int _WGM, int _N_SPLIT = 2,
         int _GPREFETCH = 1, int _WRITE_POS = 2>
struct config {
    static constexpr int BLOCK_M = _BLOCK_M, BLOCK_N = _BLOCK_N;
    static constexpr int K_STEP  = _K_STEP,  DOT_SLICE = _DOT_SLICE;
    static constexpr int NUM_WARPS = _NUM_WARPS, WARP_ROWS = _WARP_ROWS;
    static constexpr int WGM = _WGM, N_SPLIT = _N_SPLIT;
    static constexpr int GPREFETCH = _GPREFETCH, WRITE_POS = _WRITE_POS;

    static constexpr int NUM_THREADS = kittens::WARP_THREADS * NUM_WARPS;
    // Warps tile the block WARP_ROWS (M) by WARP_COLS (N).
    static constexpr int WARP_COLS = NUM_WARPS / WARP_ROWS;
    static_assert(WARP_ROWS * WARP_COLS == NUM_WARPS, "warp grid must cover the block");
    static constexpr int REG_BLOCK_M = BLOCK_M / WARP_ROWS;
    static constexpr int REG_BLOCK_N = BLOCK_N / WARP_COLS;
    static constexpr int SPLIT_N = REG_BLOCK_N / N_SPLIT;
    static_assert(SPLIT_N % 16 == 0 && SPLIT_N * N_SPLIT == REG_BLOCK_N,
                  "N_SPLIT must divide the warp tile into whole 16-column chunks");
    // ds_read_b128 count for one chunk: two per 16x16 base tile.
    static constexpr int CHUNK_READS = (SPLIT_N / 16) * (DOT_SLICE / 16) * 2;

    using st_a = st_bf<BLOCK_M, K_STEP>;
    using st_b = st_bf<BLOCK_N, K_STEP>;
    using G    = kittens::group<NUM_WARPS>;

    // 2 buffers each of A and B.
    static constexpr size_t SHARED_BYTES = 2 * (sizeof(st_a) + sizeof(st_b));
    static_assert(SHARED_BYTES <= MAX_SHARED_MEMORY, "block does not fit in 64KB of LDS");
};

// What -DBLOCK_M=... on the command line selects. sweep.sh drives this one.
using macro_config = config<cli::block_m, cli::block_n, cli::k_step, cli::dot_slice,
                            cli::num_warps, cli::warp_rows, cli::wgm, cli::n_split,
                            cli::gprefetch, cli::write_pos>;

template<typename C, typename GL = micro_globals>
__global__ __launch_bounds__(C::NUM_THREADS, MIN_BLOCKS_PER_CU)
void micro_tk(const GL g) {
    constexpr int BLOCK_M = C::BLOCK_M, BLOCK_N = C::BLOCK_N;
    constexpr int K_STEP = C::K_STEP, DOT_SLICE = C::DOT_SLICE;
    constexpr int WARP_ROWS = C::WARP_ROWS, WARP_COLS = C::WARP_COLS;
    constexpr int REG_BLOCK_M = C::REG_BLOCK_M, REG_BLOCK_N = C::REG_BLOCK_N;
    constexpr int WGM = C::WGM;
    constexpr int N_SPLIT = C::N_SPLIT, SPLIT_N = C::SPLIT_N, CHUNK_READS = C::CHUNK_READS;
    using st_a = typename C::st_a;
    using st_b = typename C::st_b;
    using G    = typename C::G;

    extern __shared__ alignment_dummy __shm[];
    shared_allocator al((int*)&__shm[0]);
    st_a (&As)[2] = al.allocate<st_a, 2>();
    st_b (&Bs)[2] = al.allocate<st_b, 2>();

    constexpr int NUM_SLICES = K_STEP / DOT_SLICE;
    // B and the accumulator are cut into N_SPLIT chunks along N. Same total
    // registers as one wide pair -- this is a re-association, not a buffer.
    rt_bf<REG_BLOCK_M, DOT_SLICE, row_l> A_tile;
    rt_bf<SPLIT_N, DOT_SLICE, row_l> B_tile[N_SPLIT];
    rt_fl<REG_BLOCK_M, SPLIT_N, col_l> C_accum[N_SPLIT];
    #pragma unroll
    for (int s = 0; s < N_SPLIT; s++) zero(C_accum[s]);
#if ABLATE_LDS_READ
    zero(A_tile);                 // the ds_reads that would fill these are gone
    #pragma unroll
    for (int s = 0; s < N_SPLIT; s++) zero(B_tile[s]);
#endif

    const int m_blocks = g.a.rows() / BLOCK_M;
    const int n_blocks = g.b.rows() / BLOCK_N;

    // L2 swizzle: walk WGM block-rows at a time so that the B panel a group of
    // workgroups reads stays resident. No chiplet term -- NUM_XCDS is 1 here.
    int wgid = blockIdx.x;
    const int wgs_per_group = WGM * n_blocks;
    const int group_id      = wgid / wgs_per_group;
    const int first_pid_m   = group_id * WGM;
    const int group_size_m  = min(m_blocks - first_pid_m, WGM);
    const int row = first_pid_m + ((wgid % wgs_per_group) % group_size_m);
    const int col = (wgid % wgs_per_group) / group_size_m;

    const int warp_id  = kittens::warpid();
    const int warp_row = warp_id / WARP_COLS;
    const int warp_col = warp_id % WARP_COLS;
    const int num_tiles = g.a.cols() / K_STEP;

    // Prefetch staging. There is no global->LDS DMA on RDNA, so the "async
    // copy" is: buffer_load into these VGPRs on one iteration, ds_write them on
    // a later one. GPREFETCH K-tiles of math sit between the two halves.
    constexpr int GP = C::GPREFETCH;
    constexpr int stage_a = G::template stage_calls<st_a>;
    constexpr int stage_b = G::template stage_calls<st_b>;
    float4 buf_a[GP][stage_a];
    float4 buf_b[GP][stage_b];
    // Loads still in flight once the oldest batch has landed. Buffer loads
    // return in vmcnt order, so this is the vmcnt analogue of the lgkmcnt
    // schedule in dot_tile(): wait for batch p and leave the GP-1 newer
    // batches moving.
    constexpr int VM_KEEP = (GP - 1) * (stage_a + stage_b);
    static_assert(VM_KEEP <= 63, "vmcnt is 6 bits; GPREFETCH is too deep");

    // One K-tile of math: read a DOT_SLICE-wide pair of operand tiles out of
    // LDS, multiply, repeat.
    //
    // Double-buffering the operand tiles so slice k+1's reads overlap slice k's
    // WMMAs was tried and is *slower*: 63/68 TFLOPs against 66/72, because the
    // second set of tiles costs 48 VGPRs and takes occupancy from 9 waves/SIMD
    // to 7. So the overlap here is arranged to cost zero registers instead.
    //
    // The whole slice's reads -- A plus every B chunk -- go in flight at once,
    // and then the waits step down: lds_wait<N> means "at most N still
    // outstanding", and LDS retires in order, so waiting for
    // (N_SPLIT-1-s)*CHUNK_READS retires exactly A and B chunks 0..s while
    // chunks s+1.. are still moving. Chunk s's WMMAs then issue on top of them.
    // No tile is duplicated: the reads that overlap the math are reads this
    // slice had to do anyway, just not fenced in front of it.
    auto load_A = [&](int buf, int k) {
        load<false>(A_tile, subtile_inplace<REG_BLOCK_M, DOT_SLICE>(As[buf], {warp_row, k}));
    };
    auto load_B = [&](int buf, int k, int c) {
        load<false>(B_tile[c], subtile_inplace<SPLIT_N, DOT_SLICE>(
                                   Bs[buf], {warp_col * N_SPLIT + c, k}));
    };
    auto mma_chunk = [&](int c) {
#if !ABLATE_MMA
        __builtin_amdgcn_s_setprio(1);
        mma_ABt(C_accum[c], A_tile, B_tile[c], C_accum[c]);
        __builtin_amdgcn_s_setprio(0);
#else
        keep_live(A_tile); keep_live(B_tile[c]);
#endif
    };

    // `has_tail` has to be a compile-time argument rather than derived from
    // WRITE_POS alone: the epilogue calls dot_tile with an empty tail, and an
    // lgkmcnt immediate that is too *large* does not over-wait, it fails to
    // wait at all. Counting writes that were never issued there would let the
    // last tile's WMMAs run on operands still in flight.
    auto dot_tile = [&](int buf, auto has_tail, auto &&tail) {
        // LDS ops the interleaved store adds to the queue. Writes issued after
        // a batch of reads retire after them -- lgkmcnt is in order -- so every
        // wait downstream of the store has to carry them, or it would retire
        // more reads than the schedule intends.
        constexpr int TAIL_OPS =
            (has_tail() && C::WRITE_POS == 2) ? (stage_a + stage_b) : 0;
#if ABLATE_LDS_READ
        tail();
        #pragma unroll
        for (int k = 0; k < NUM_SLICES; k++)
            static_for<N_SPLIT>([&](auto c) { mma_chunk(c); });
#elif !ROTATE
        tail();
        #pragma unroll
        for (int k = 0; k < NUM_SLICES; k++) {
            load_A(buf, k);
            static_for<N_SPLIT>([&](auto c) { load_B(buf, k, c); });
            static_for<N_SPLIT>([&](auto c) {
                lds_wait<(N_SPLIT - 1 - c) * CHUNK_READS>();
                mma_chunk(c);
            });
        }
#else
        // Slice 0 up front; after that every read is issued by the chunk whose
        // register slot it is about to reuse. B_tile[c] is dead the instant
        // chunk c's WMMAs have issued, so slice k+1's chunk c goes in flight
        // there, one full chunk of math early, at no register cost. A_tile is
        // dead only after the slice's last chunk, so A goes out at the boundary
        // -- and *before* that chunk's B, so that the next slice's first chunk
        // is gated on A rather than on a read issued behind it.
        load_A(buf, 0);
        static_for<N_SPLIT>([&](auto c) { load_B(buf, 0, c); });
        // Nothing left to issue for this tile when there is only one slice.
        if constexpr (NUM_SLICES == 1) tail();

        static_for<NUM_SLICES>([&](auto k) {
            static_for<N_SPLIT>([&](auto c) {
                // Slice NUM_SLICES-1's reads were all issued before the store,
                // so its waits are the ones that have to count the writes.
                constexpr int extra = (k == NUM_SLICES - 1) ? TAIL_OPS : 0;
                lds_wait<reads_in_flight(k, c, N_SPLIT, NUM_SLICES, CHUNK_READS) + extra>();
                mma_chunk(c);
                if constexpr (k + 1 < NUM_SLICES) {
                    if constexpr (c == N_SPLIT - 1) load_A(buf, k + 1);
                    load_B(buf, k + 1, c);
                    // Last read of the tile is now in flight: hand the LDS
                    // pipe to the store, and let this slice's math cover it.
                    if constexpr (k + 2 == NUM_SLICES && c == N_SPLIT - 1) tail();
                }
            });
        });
#endif
    };

    int tic = 0, toc = 1;

    // Loop invariant at the top of iteration `tile`:
    //   * LDS buffer `tic` holds K-tile `tile` and every warp has passed a
    //     barrier since it was written;
    //   * buf_a/buf_b hold K-tile `tile+1`, possibly still in flight;
    //   * buffer `toc` was last read during iteration `tile-1`, before that
    //     same barrier, so it is free to overwrite right now.
    //
    // That last clause is what lets this run on one barrier per K-tile instead
    // of the two the CDNA kernel uses: the write-after-read hazard on `toc` is
    // already covered by the previous iteration's barrier, so only the
    // write-before-read hazard needs a fresh one, at the bottom.
    // The K index is clamped rather than predicated. Past the end of the K
    // loop this re-reads the last tile into staging registers that are never
    // stored, which is a handful of L2-hot loads per workgroup -- and it keeps
    // exactly GP batches in flight at all times, which is what makes VM_KEEP a
    // compile-time immediate. Predicating the issue instead would make the
    // right vmcnt a runtime value, and there is no such instruction.
    auto gload = [&](int d, int t) {
        const int tt = t < num_tiles ? t : num_tiles - 1;
        G::load_global_to_register_buffer(buf_a[d], stage_a, g.a, coord<st_a>{0, 0, row, tt}, As[0]);
        G::load_global_to_register_buffer(buf_b[d], stage_b, g.b, coord<st_b>{0, 0, col, tt}, Bs[0]);
    };

    G::load(As[tic], g.a, {0, 0, row, 0});
    G::load(Bs[tic], g.b, {0, 0, col, 0});
#if !ABLATE_GLOBAL
    static_for<GP>([&](auto d) { gload(d, d + 1); });
#endif
    __builtin_amdgcn_s_barrier();

    // Unrolled by GP so that the staging slot is a compile-time index; a
    // runtime one would put buf_a/buf_b in scratch.
    const int n_main = num_tiles - 1;
    for (int base = 0; base < n_main; base += GP) {
        static_for<GP>([&](auto dd) {
            constexpr int d = dd;
            const int tile = base + d;
            if (tile < n_main) {
                // Land batch d -- issued GP iterations ago -- then immediately
                // refill that slot, so the staging registers are never idle.
                auto store_tile = [&] {
#if !ABLATE_GLOBAL
                    vm_wait<VM_KEEP>();
#if !ABLATE_LDS_WRITE
                    // No drain: dot_tile's own lgkmcnt schedule retires these,
                    // and its last chunk waits to zero, which is what the
                    // barrier below needs.
                    G::template store_register_buffer_to_shared<false>(As[toc], buf_a[d], stage_a);
                    G::template store_register_buffer_to_shared<false>(Bs[toc], buf_b[d], stage_b);
#endif
                    gload(d, tile + 1 + GP);
#endif
                };
                auto nop = [] {};
                if constexpr (C::WRITE_POS == 0) {
                    store_tile();
                    dot_tile(tic, std::false_type{}, nop);
                } else if constexpr (C::WRITE_POS == 1) {
                    dot_tile(tic, std::false_type{}, nop);
                    store_tile();
                } else {
                    dot_tile(tic, std::true_type{}, store_tile);
                    // dot_tile's last chunk stops at lgkmcnt(TAIL_OPS), so the
                    // writes are still moving. s_barrier does not order memory:
                    // without this the next tile's ds_reads race them. It is
                    // nearly free here -- a whole K-slice of WMMAs has run
                    // since the writes were issued.
                    lds_wait<0>();
                }
#if !ABLATE_BARRIER
                __builtin_amdgcn_s_barrier();
#endif
                tic ^= 1;
                toc ^= 1;
            }
        });
    }

    // Epilogue: last tile is already resident, and there is nothing left to
    // stage -- hence false_type, which zeroes TAIL_OPS.
    dot_tile(tic, std::false_type{}, [] {});

    // Column coords are in units of the tile width, which is now SPLIT_N.
    const int row_tile = row * WARP_ROWS + warp_row;   // in REG_BLOCK_M units
    const int col_tile = (col * WARP_COLS + warp_col) * N_SPLIT;

    if constexpr (!GL::FUSED_RS) {
        #pragma unroll
        for (int s = 0; s < N_SPLIT; s++)
            store(g.c, C_accum[s], {0, 0, row_tile, col_tile + s});
    } else {
        // Fused reduce-scatter. Every rank computes a full M x N partial
        // product over its own slice of K, and the final C is the sum of those
        // partials, row-sharded across ranks. So a tile is either mine to keep
        // or the owner's to receive, and the cheapest moment to decide is here:
        // the accumulator is already in registers, and the alternative is to
        // write the whole M x N locally and move half of it again afterwards.
        //
        // The shard boundary is in units of REG_BLOCK_M and the host has
        // checked it divides, so a warp tile never straddles two owners and no
        // tile has to be split.
        //
        // g.c_peer is the peer's inbox, already translated to the peer's heap
        // on the host, so the peer store is an ordinary vectorized global store
        // to an address that happens to live on another GPU -- no device-side
        // address arithmetic, and no per-element loop. Being able to do this is
        // the whole reason symmem.cuh keeps translate() public where Iris does
        // not.
        //
        // One peer, hence one gl: kittens::gl has no default constructor and
        // its only constructor is __host__, so an array of them cannot be a
        // member and a device-side one cannot be built. Supporting more ranks
        // means passing world-1 named gls or giving gl a device constructor;
        // two is the size the fabric measurements say is worth fusing anyway.
        // shard_rows is in rows, not tiles, because the tile height is a
        // property of the config the dispatcher picked and the host does not
        // know which one that was. One integer division at the very end of the
        // kernel costs nothing.
        const int shard_tiles = g.shard_rows / C::REG_BLOCK_M;
        const int owner = row_tile / shard_tiles;
        const int local_row_tile = row_tile - owner * shard_tiles;
        if (owner == g.my_rank) {
            #pragma unroll
            for (int s = 0; s < N_SPLIT; s++)
                store(g.c, C_accum[s], {0, 0, local_row_tile, col_tile + s});
        } else {
            #pragma unroll
            for (int s = 0; s < N_SPLIT; s++)
                store(g.c_peer, C_accum[s], {0, 0, local_row_tile, col_tile + s});
        }
    }
}

template<typename C, typename GL = micro_globals>
static void launch(const GL &g) {
    const unsigned long mem_size = C::SHARED_BYTES;
    hipFuncSetAttribute((void*)micro_tk<C, GL>, hipFuncAttributeMaxDynamicSharedMemorySize, mem_size);
    // The compiler's occupancy remark counts registers only. LDS is the other
    // limit and on this kernel it is usually the binding one, so ask the
    // runtime what actually fits. HK_OCC=1 to see it.
    if (getenv("HK_OCC")) {
        static bool once = false;
        if (!once) {
            once = true;
            int blocks = 0;
            hipOccupancyMaxActiveBlocksPerMultiprocessor(&blocks, (void*)micro_tk<C, GL>,
                                                         C::NUM_THREADS, mem_size);
            fprintf(stderr, "occ: %d blocks/WGP, %d waves/WGP, %.1f waves/SIMD"
                            " (LDS %lu B, %d threads)\n",
                    blocks, blocks * C::NUM_WARPS, blocks * C::NUM_WARPS / 4.0,
                    mem_size, C::NUM_THREADS);
        }
    }
    micro_tk<C, GL><<<dim3((g.a.rows() / C::BLOCK_M) * (g.b.rows() / C::BLOCK_N)),
                  dim3(C::NUM_THREADS), mem_size, g.stream>>>(g);
}

// Three tilings. Two are picked on the number of workgroups the M/N grid
// produces; the third only exists to cover a K the other two cannot tile.
//
//   big    128x128x64, 8 warps, 32x64 per warp, N_SPLIT=4. 208 VGPRs, 7
//          waves/SIMD, 48 KB of LDS. The deepest K-tile that fits, which is
//          what the rotated schedule in dot_tile() wants: with four K-slices
//          of four chunks each, fifteen of the sixteen chunks have LDS reads
//          in flight underneath them.
//   small  128x64x64, 8 warps, 32x32 per warp, N_SPLIT=2. 134 VGPRs, 10
//          waves/SIMD, 24 KB. Half the intensity, but twice the workgroups.
//   k32    big with K_STEP=32, for a K that is not a multiple of 64. The
//          kernel has no K remainder handling, so this is correctness, not
//          tuning.
//
// The big/small crossover is a tail-quantization effect, not an inner-loop one.
// `big` uses 48 KB of LDS, so only one of its workgroups fits per WGP and 48
// WGPs hold 48 at once; `small` at 24 KB holds 96. A grid that is a poor fit
// for the coarser tiling idles through a mostly-empty final pass.
//
// Measured, TFLOPs (blocks counted at 128x128):
//
//   M,N,K                blocks | big | small
//   512x512x4096             16 |  19 |  33
//   1024x1024x4096           64 |  49 |  51
//   2048x1024x4096          128 |  63 |  58
//   2048x2048x2048          256 |  64 |  60
//   4096x1024x4096          256 |  65 |  60
//   4096x4096x4096         1024 |  73 |   -
//   8192x8192x4096         4096 |  78 |   -
//
// so the rule is `blocks <= 64`. Note this threshold moved by 4x when the
// rotated schedule landed: overlapping the LDS reads made `big` much better at
// mid-size grids, and it now wins everywhere it has enough workgroups to fill
// the machine once.
using big_config   = config<128, 128, 64, 16, 8, 4, 8, 4>;
using small_config = config<128,  64, 64, 16, 8, 4, 8, 2>;
using k32_config   = config<128, 128, 32, 16, 8, 4, 8, 4>;

#ifndef HK_MULTI_CONFIG
#define HK_MULTI_CONFIG 1
#endif

// The selection rule, templated on the globals type so that a caller with a
// different output policy gets the same tiling decisions. The fused kernel used
// to pin itself to `big`, which measures the wrong thing on the small-M shapes
// a decode or short-prefill batch produces -- the point of comparing against an
// unfused baseline is that the two differ only in the epilogue.
template<typename GL>
static void dispatch_any(const GL &g) {
    // num_tiles is a plain division, so a K-tile that does not divide K would
    // silently drop the tail.
    if (g.a.cols() % big_config::K_STEP != 0) {
        launch<k32_config>(g);
        return;
    }
    const int blocks = (g.a.rows() / big_config::BLOCK_M) * (g.b.rows() / big_config::BLOCK_N);
    // Fall back to `big` on shapes `small` cannot tile; its BLOCK_N is the
    // smaller of the two, so this only triggers on N not divisible by 128.
    if (blocks <= 64 && g.b.rows() % small_config::BLOCK_N == 0) {
        launch<small_config>(g);
        return;
    }
    launch<big_config>(g);
}

void dispatch_micro(micro_globals g) {
#if HK_MULTI_CONFIG
    dispatch_any(g);
#else
    launch<macro_config>(g);   // -DBLOCK_M=... from sweep.sh
#endif
}

// The distributed kernel includes this file for the tuned inner loop and
// supplies its own host harness, so the python module is opt-out.
#ifndef HK_GEMM_NO_PYBIND
PYBIND11_MODULE(tk_kernel, m) {
    m.doc() = "tk_kernel python module";
    py::bind_function<dispatch_micro>(m, "dispatch_micro", &micro_globals::a, &micro_globals::b, &micro_globals::c);
}
#endif
