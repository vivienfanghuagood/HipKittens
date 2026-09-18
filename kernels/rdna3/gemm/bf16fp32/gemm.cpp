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

#include "kittens.cuh"
#include "pyutils/pyutils.cuh"
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
};

// The command line sets the tunables as macros, and the config struct below
// wants the same names as members, so capture the values and then get the macros
// out of the way.
namespace cli {
constexpr int block_m = BLOCK_M, block_n = BLOCK_N, k_step = K_STEP,
              dot_slice = DOT_SLICE, num_warps = NUM_WARPS,
              warp_rows = WARP_ROWS, wgm = WGM;
}
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
         int _NUM_WARPS, int _WARP_ROWS, int _WGM>
struct config {
    static constexpr int BLOCK_M = _BLOCK_M, BLOCK_N = _BLOCK_N;
    static constexpr int K_STEP  = _K_STEP,  DOT_SLICE = _DOT_SLICE;
    static constexpr int NUM_WARPS = _NUM_WARPS, WARP_ROWS = _WARP_ROWS;
    static constexpr int WGM = _WGM;

    static constexpr int NUM_THREADS = kittens::WARP_THREADS * NUM_WARPS;
    // Warps tile the block WARP_ROWS (M) by WARP_COLS (N).
    static constexpr int WARP_COLS = NUM_WARPS / WARP_ROWS;
    static_assert(WARP_ROWS * WARP_COLS == NUM_WARPS, "warp grid must cover the block");
    static constexpr int REG_BLOCK_M = BLOCK_M / WARP_ROWS;
    static constexpr int REG_BLOCK_N = BLOCK_N / WARP_COLS;

    using st_a = st_bf<BLOCK_M, K_STEP>;
    using st_b = st_bf<BLOCK_N, K_STEP>;
    using G    = kittens::group<NUM_WARPS>;

    // 2 buffers each of A and B.
    static constexpr size_t SHARED_BYTES = 2 * (sizeof(st_a) + sizeof(st_b));
    static_assert(SHARED_BYTES <= MAX_SHARED_MEMORY, "block does not fit in 64KB of LDS");
};

// What -DBLOCK_M=... on the command line selects. sweep.sh drives this one.
using macro_config = config<cli::block_m, cli::block_n, cli::k_step, cli::dot_slice,
                            cli::num_warps, cli::warp_rows, cli::wgm>;

template<typename C>
__global__ __launch_bounds__(C::NUM_THREADS, MIN_BLOCKS_PER_CU)
void micro_tk(const micro_globals g) {
    constexpr int BLOCK_M = C::BLOCK_M, BLOCK_N = C::BLOCK_N;
    constexpr int K_STEP = C::K_STEP, DOT_SLICE = C::DOT_SLICE;
    constexpr int WARP_ROWS = C::WARP_ROWS, WARP_COLS = C::WARP_COLS;
    constexpr int REG_BLOCK_M = C::REG_BLOCK_M, REG_BLOCK_N = C::REG_BLOCK_N;
    constexpr int WGM = C::WGM;
    using st_a = typename C::st_a;
    using st_b = typename C::st_b;
    using G    = typename C::G;

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
    constexpr int stage_a = G::template stage_calls<st_a>;
    constexpr int stage_b = G::template stage_calls<st_b>;
    float4 buf_a[stage_a];
    float4 buf_b[stage_b];

    // One K-tile of math: read a DOT_SLICE-wide pair of operand tiles out of
    // LDS, multiply, repeat.
    //
    // The obvious next move here is to software-pipeline it -- double-buffer
    // A_tile/B_tile and issue slice k+1's ds_reads before slice k's WMMAs, with
    // lds_wait<SLICE_READS>() instead of lds_wait<0>(). The library has what
    // that needs (load_async and lds_wait). It was tried and it is *slower*:
    // 63/68 TFLOPs against 66/72, because the second set of operand tiles costs
    // 48 VGPRs, which takes occupancy from 9 waves/SIMD to 7. At 9 waves the
    // SIMD already covers LDS latency by switching waves, so the schedule is
    // buying with registers something it was getting for free. Keep this in
    // mind before reaching for ping-pong on a low-occupancy kernel, where the
    // trade goes the other way.
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
        asm volatile("s_waitcnt vmcnt(0)");
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

template<typename C>
static void launch(const micro_globals &g) {
    const unsigned long mem_size = C::SHARED_BYTES;
    hipFuncSetAttribute((void*)micro_tk<C>, hipFuncAttributeMaxDynamicSharedMemorySize, mem_size);
    micro_tk<C><<<dim3((g.a.rows() / C::BLOCK_M) * (g.b.rows() / C::BLOCK_N)),
                  dim3(C::NUM_THREADS), mem_size, g.stream>>>(g);
}

// Two tilings, picked on the number of workgroups the M/N grid produces.
//
//   big    128x128, 8 warps, 32x64 per warp. 164 VGPRs, 9 waves/SIMD, 32 KB of
//          LDS. The most arithmetic intensity that fits without spilling.
//   small  128x64, 8 warps, 32x32 per warp. 103 VGPRs, 12 waves/SIMD, 24 KB.
//          Half the intensity, but twice the workgroups and a third more waves.
//
// The crossover is sharp and it is a tail-quantization effect, not an inner-loop
// one. `big` uses 32 KB of LDS, so four of its workgroups fit per WGP and 48
// WGPs hold 192 at once. At 2048x2048 the grid is 16x16 = 256 workgroups: one
// full pass plus a second pass that is two thirds empty, and the machine idles
// through it. `small` has 512 workgroups of two thirds the cost, which lands
// much closer to a whole number of passes.
//
// Measured, TFLOPs, best of each column in bold in README.md:
//
//   M,N       blocks | big | small
//   1024^2        64 |  36 |  44
//   2048^2       256 |  39 |  55      <- 2048x2048x2048
//   2048^2       256 |  48 |  55      <- 2048x2048x4096
//   2048x4096    512 |  63 |  61
//   4096^2      1024 |  67 |  66
//   4096x8192   2048 |  72 |  69
//   8192^2      4096 |  71 |  59
//
// so the rule is `blocks <= 256`, i.e. up to about 1.3 machine-fulls of `big`.
// A third variant with K_STEP=64 was also measured; it wins nothing anywhere
// that `small` does not win by more, so it is not carried here.
using big_config   = config<128, 128, 32, 16, 8, 4, 8>;
using small_config = config<128,  64, 32, 16, 8, 4, 8>;

#ifndef HK_MULTI_CONFIG
#define HK_MULTI_CONFIG 1
#endif

void dispatch_micro(micro_globals g) {
#if HK_MULTI_CONFIG
    const int blocks = (g.a.rows() / big_config::BLOCK_M) * (g.b.rows() / big_config::BLOCK_N);
    // Fall back to `big` on shapes `small` cannot tile; its BLOCK_N is the
    // smaller of the two, so this only triggers on N not divisible by 128.
    if (blocks <= 256 && g.b.rows() % small_config::BLOCK_N == 0) {
        launch<small_config>(g);
        return;
    }
    launch<big_config>(g);
#else
    launch<macro_config>(g);   // -DBLOCK_M=... from sweep.sh
#endif
}

PYBIND11_MODULE(tk_kernel, m) {
    m.doc() = "tk_kernel python module";
    py::bind_function<dispatch_micro>(m, "dispatch_micro", &micro_globals::a, &micro_globals::b, &micro_globals::c);
}
