// bf16 GEMM with fp32 accumulate for gfx1200/gfx1201 (RDNA4).
//
// !! NOT HARDWARE-VERIFIED.  This compiles for gfx1201 and nothing more: no
// gfx12 part was available, so it has never been run, its output has never been
// compared against torch, and the tiling constants below have never been swept.
// See tests/unit/rdna4/README.md for what compile-verification does and does
// not establish. !!
//
// This is the gfx1100 kernel (kernels/rdna3/gemm/bf16fp32) with the schedule
// left alone and the arch notes rewritten.  Three things are different from
// RDNA3:
//
//   * wave32 still, but a WMMA operand is no longer mirrored across the wave
//     halves -- they split K instead -- so a bf16 rt_base costs 4 VGPRs where
//     an fp32 one costs 8.  On RDNA3 both cost 8, and operand registers were
//     what bounded the tile.  Here they are half as expensive and the
//     accumulator dominates: at the default shape the operand tiles are 24
//     VGPRs against the accumulator's 64.  The compiler agrees -- at the
//     default shape this kernel allocates 128 VGPRs for gfx1201 where the
//     gfx1100 one allocates 164, and reports 10 waves/SIMD against 9 (no
//     spills, no scratch, either way).  That much *is* verified; it is a
//     property of the generated code, not of a run.
//   * no global->LDS DMA, same as RDNA3.  Every byte of a prefetch passes
//     through VGPRs, so an "async copy" is load_global_to_register_buffer now
//     and store_register_buffer_to_shared at the point the buffer is needed.
//   * one XCD, so there is no chiplet swizzle -- only the L2 group swizzle.
//     gfx1200 (Navi 44) and gfx1201 (Navi 48) are both monolithic.
//
// B is passed pre-transposed as (n, k), which keeps both shared loads
// contiguous and makes mma_ABt the right primitive.

#include "kittens.cuh"
#include "pyutils/pyutils.cuh"
using namespace kittens;

// Tiling knobs.  These are inherited verbatim from the gfx1100 kernel, where
// they were swept; they are *not* tuned for gfx12 and there is no reason to
// think they are optimal here.  The halved operand cost above frees roughly 24
// VGPRs per wave at the default shape, which is the budget a bigger REG_BLOCK_N
// or a deeper K_STEP would spend -- run sweep.sh on real hardware before
// believing any of these numbers.
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
#ifndef MIN_BLOCKS_PER_CU
#define MIN_BLOCKS_PER_CU 1
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

#define NUM_THREADS (kittens::WARP_THREADS * NUM_WARPS)

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

// Warps tile the block WARP_ROWS (M) by WARP_COLS (N).
constexpr int WARP_COLS = NUM_WARPS / WARP_ROWS;
static_assert(WARP_ROWS * WARP_COLS == NUM_WARPS, "warp grid must cover the block");
constexpr int REG_BLOCK_M = BLOCK_M / WARP_ROWS;  // 32
constexpr int REG_BLOCK_N = BLOCK_N / WARP_COLS;  // 64

using st_a = st_bf<BLOCK_M, K_STEP>;
using st_b = st_bf<BLOCK_N, K_STEP>;

using _gl_A = gl<bf16, -1, -1, -1, -1>;
using _gl_B = gl<bf16, -1, -1, -1, -1>;
using _gl_C = gl<bf16, -1, -1, -1, -1>;

using G = kittens::group<NUM_WARPS>;

// 2 buffers each of A and B.
constexpr size_t SHARED_BYTES = 2 * (sizeof(st_a) + sizeof(st_b));
static_assert(SHARED_BYTES <= MAX_SHARED_MEMORY, "block does not fit in 64KB of LDS");

struct micro_globals {
    _gl_A a;
    _gl_B b;
    _gl_C c;
    hipStream_t stream;
    // a: (m, k), b: (n, k), c: (m, n)
    dim3 grid()  { return dim3((a.rows() / BLOCK_M) * (b.rows() / BLOCK_N)); }
    dim3 block() { return dim3(NUM_THREADS); }
    size_t dynamic_shared_memory() { return SHARED_BYTES; }
};

__global__ __launch_bounds__(NUM_THREADS, MIN_BLOCKS_PER_CU)
void micro_tk(const micro_globals g) {
    extern __shared__ alignment_dummy __shm[];
    shared_allocator al((int*)&__shm[0]);
    st_a (&As)[2] = al.allocate<st_a, 2>();
    st_b (&Bs)[2] = al.allocate<st_b, 2>();

    constexpr int NUM_SLICES = K_STEP / DOT_SLICE;
    rt_bf<REG_BLOCK_M, DOT_SLICE, row_l> A_tile;
    rt_bf<REG_BLOCK_N, DOT_SLICE, row_l> B_tile;
    rt_fl<REG_BLOCK_M, REG_BLOCK_N, col_l> C_accum;
    zero(C_accum);
#if ABLATE_LDS_READ
    zero(A_tile); zero(B_tile);   // the ds_reads that would fill these are gone
#endif

    const int m_blocks = g.a.rows() / BLOCK_M;
    const int n_blocks = g.b.rows() / BLOCK_N;

    // L2 swizzle: walk WGM block-rows at a time so that the B panel a group of
    // workgroups reads stays resident. No chiplet term -- NUM_XCDS is 1 here.
    int wgid = blockIdx.x;
    const int WGM = 8;
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
    // the next. A whole K-tile of math sits between the two halves.
    constexpr int stage_a = G::stage_calls<st_a>;
    constexpr int stage_b = G::stage_calls<st_b>;
    float4 buf_a[stage_a];
    float4 buf_b[stage_b];

    // One K-tile of math: read a DOT_SLICE-wide pair of operand tiles out of
    // LDS, multiply, repeat.
    //
    // The obvious next move here is to software-pipeline it -- double-buffer
    // A_tile/B_tile and issue slice k+1's ds_loads before slice k's WMMAs, with
    // lds_wait<SLICE_READS>() instead of lds_wait<0>(). The library has what
    // that needs (load_async and lds_wait).
    //
    // On gfx1100 this was tried and measured *slower*: 63/68 TFLOPs against
    // 66/72. The second set of operand tiles cost 48 VGPRs, which took
    // occupancy from 9 waves/SIMD to 7, and at 9 waves the SIMD was already
    // covering LDS latency by switching waves -- the schedule was buying with
    // registers something it was getting for free.
    //
    // That measurement does not carry over, and this is the one place where the
    // gfx12 fragment change should actually move the answer. The second set
    // costs 24 VGPRs here, not 48, and the occupancy cliff is shallower as a
    // result.  gfx1100 and gfx1201 turn out to have the identical VGPR ->
    // occupancy curve, read off the compiler's resource-usage remarks and
    // bisected around the step that matters: 10 waves/SIMD holds to 144 VGPRs
    // and drops to 9 at 145, then to 8 past 163 and 7 past 180.  So:
    //
    //     gfx1100   164 -> 212 VGPRs   9 -> 7 waves   measured 66/72 -> 63/68
    //     gfx1201   128 -> 152 VGPRs  10 -> 9 waves   unmeasured
    //
    // One step, not two.  Whether one step is cheap enough to pay for the
    // overlap is exactly the question, and it needs a machine.  It is left
    // un-pipelined here only because shipping an unmeasured schedule change
    // would be worse than shipping the known one.
    auto dot_tile = [&](int buf) {
        #pragma unroll
        for (int k = 0; k < NUM_SLICES; k++) {
#if !ABLATE_LDS_READ
            load(A_tile, subtile_inplace<REG_BLOCK_M, DOT_SLICE>(As[buf], {warp_row, k}));
            load(B_tile, subtile_inplace<REG_BLOCK_N, DOT_SLICE>(Bs[buf], {warp_col, k}));
#endif
#if !ABLATE_MMA
            __builtin_amdgcn_s_setprio(1);
            mma_ABt(C_accum, A_tile, B_tile, C_accum);
            __builtin_amdgcn_s_setprio(0);
#else
            keep_live(A_tile); keep_live(B_tile);
#endif
        }
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
    G::load(As[tic], g.a, {0, 0, row, 0});
    G::load(Bs[tic], g.b, {0, 0, col, 0});
#if !ABLATE_GLOBAL
    if (num_tiles > 1) {
        G::load_global_to_register_buffer(buf_a, stage_a, g.a, coord<st_a>{0, 0, row, 1}, As[toc]);
        G::load_global_to_register_buffer(buf_b, stage_b, g.b, coord<st_b>{0, 0, col, 1}, Bs[toc]);
    }
#endif
    __builtin_amdgcn_s_barrier();

    for (int tile = 0; tile < num_tiles - 1; ++tile, tic ^= 1, toc ^= 1) {
#if !ABLATE_GLOBAL
        // Land the in-flight tile, then immediately put the one after it in
        // flight, so the buffer_loads are outstanding across all the math below.
        vmem_load_wait<0>();
        G::store_register_buffer_to_shared(As[toc], buf_a, stage_a);
        G::store_register_buffer_to_shared(Bs[toc], buf_b, stage_b);
        if (tile + 2 < num_tiles) {
            G::load_global_to_register_buffer(buf_a, stage_a, g.a, coord<st_a>{0, 0, row, tile + 2}, As[tic]);
            G::load_global_to_register_buffer(buf_b, stage_b, g.b, coord<st_b>{0, 0, col, tile + 2}, Bs[tic]);
        }
#endif

        dot_tile(tic);

        __builtin_amdgcn_s_barrier();
    }

    // Epilogue: last tile is already resident.
    dot_tile(tic);

    store(g.c, C_accum, {0, 0, row * WARP_ROWS + warp_row, col * WARP_COLS + warp_col});
}

void dispatch_micro(micro_globals g) {
    unsigned long mem_size = g.dynamic_shared_memory();
    hipFuncSetAttribute((void*)micro_tk, hipFuncAttributeMaxDynamicSharedMemorySize, mem_size);
    micro_tk<<<g.grid(), g.block(), mem_size, g.stream>>>(g);
}

PYBIND11_MODULE(tk_kernel, m) {
    m.doc() = "tk_kernel python module";
    py::bind_function<dispatch_micro>(m, "dispatch_micro", &micro_globals::a, &micro_globals::b, &micro_globals::c);
}
