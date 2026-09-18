#include "mma.cuh"

#ifdef TEST_WARP_REGISTER_TILE_MMA
// Every test file in this tree names its scaffolding the same way -- test_generator,
// and wrapper structs like vec_load_store. Those are templates with external
// linkage, so across translation units they are one entity with many definitions,
// and the linker keeps whichever it saw first. Internal linkage makes each file's
// version its own.
namespace {


struct test_mma_AB {
    template<typename RT_SHAPE_ACCUM, int K_DIM, int H, int W, int NW, typename K> using valid = std::bool_constant<NW == 1 && (W*H+W*K::value+H*K::value)<=28>; // this is warp-level
    static inline const std::string test_identifier = "reg_mma_AB";
    template<typename RT_SHAPE_ACCUM, int K_DIM, int H, int W, int NW, gl_t GTL_A, gl_t GTL_B, gl_t GTL_C, typename _K> __host__ static void host_func(const std::vector<float> &i_ref, std::vector<float> &o_ref) {
        constexpr int K = _K::value;
        for(int i = 0; i < H*RT_SHAPE_ACCUM::rows; i++) {
            for(int j = 0; j < W*RT_SHAPE_ACCUM::cols; j++) {
                float sum = 0;
                for(int k = 0; k < K*K_DIM; k++) {
                    sum += i_ref[i*K_DIM*K + k]*i_ref[(RT_SHAPE_ACCUM::rows*K_DIM*H*K) + k*RT_SHAPE_ACCUM::cols*W + j];
                }
                o_ref[i*RT_SHAPE_ACCUM::cols*W + j] = sum;
            }
        }
    }
    template<typename RT_SHAPE_ACCUM, int K_DIM, int H, int W, int NW, gl_t GTL_A, gl_t GTL_B, gl_t GTL_C, typename _K> __device__ static void device_func(const GTL_A &a_input, const GTL_B &b_input, const GTL_C &c_output) {
        constexpr int K = _K::value;

        kittens::rt_bf<RT_SHAPE_ACCUM::rows*H, K_DIM*K, kittens::ducks::rt_layout::row> a;
        kittens::rt_bf<K_DIM*K, RT_SHAPE_ACCUM::cols*W, kittens::ducks::rt_layout::col> b;
        kittens::rt_fl<RT_SHAPE_ACCUM::rows*H, RT_SHAPE_ACCUM::cols*W, kittens::ducks::rt_layout::col> c;

        kittens::load(a, a_input, {});
        kittens::load(b, b_input, {});
        __builtin_amdgcn_s_waitcnt(0);
        __builtin_amdgcn_s_barrier();
        kittens::zero(c);
        kittens::mma_AB(c, a, b, c);
        kittens::store(c_output, c, {});
    }
    template<typename RT_SHAPE_ACCUM, int K_DIM, int H, int W, typename K> using make_a_layout = typename kittens::gl<kittens::bf16, 1, 1, RT_SHAPE_ACCUM::rows*H, K_DIM*K::value>;
    template<typename RT_SHAPE_ACCUM, int K_DIM, int H, int W, typename K> using make_b_layout = typename kittens::gl<kittens::bf16, 1, 1, K_DIM*K::value, RT_SHAPE_ACCUM::cols*W>;
    template<typename RT_SHAPE_ACCUM, int K_DIM, int H, int W, typename K> using make_c_layout = typename kittens::gl<kittens::bf16, 1, 1, RT_SHAPE_ACCUM::rows*H, RT_SHAPE_ACCUM::cols*W>;
};
struct test_mma_ABt {
    template<typename RT_SHAPE_ACCUM, int K_DIM, int H, int W, int NW, typename K> using valid = std::bool_constant<NW == 1 && (W*H+W*K::value+H*K::value)<=28>; // this is warp-level
    static inline const std::string test_identifier = "reg_mma_ABt";
    template<typename RT_SHAPE_ACCUM, int K_DIM, int H, int W, int NW, gl_t GTL_A, gl_t GTL_B, gl_t GTL_C, typename _K> __host__ static void host_func(const std::vector<float> &i_ref, std::vector<float> &o_ref) {
        constexpr int K = _K::value;
        for(int i = 0; i < H*RT_SHAPE_ACCUM::rows; i++) {
            for(int j = 0; j < W*RT_SHAPE_ACCUM::cols; j++) {
                float sum = 0;
                for(int k = 0; k < K*K_DIM; k++) {
                    sum += i_ref[i*K*K_DIM+k]*i_ref[RT_SHAPE_ACCUM::rows*K_DIM*K*H + j*K*K_DIM+k];
                }
                o_ref[i*W*RT_SHAPE_ACCUM::cols+j] = sum;
            }
        }
    }
    template<typename RT_SHAPE_ACCUM, int K_DIM, int H, int W, int NW, gl_t GTL_A, gl_t GTL_B, gl_t GTL_C, typename _K> __device__ static void device_func(const GTL_A &a_input, const GTL_B &b_input, const GTL_C &c_output) {
        constexpr int K = _K::value;

        kittens::rt_bf<RT_SHAPE_ACCUM::rows*H, K_DIM*K, kittens::ducks::rt_layout::row> a;
        kittens::rt_bf<RT_SHAPE_ACCUM::cols*W, K_DIM*K, kittens::ducks::rt_layout::row> b;
        kittens::rt_fl<RT_SHAPE_ACCUM::rows*H, RT_SHAPE_ACCUM::cols*W, kittens::ducks::rt_layout::col> c;
        kittens::load(a, a_input, {});
        kittens::load(b, b_input, {});
        __builtin_amdgcn_s_waitcnt(0);
        __builtin_amdgcn_s_barrier();
        kittens::zero(c);
        kittens::mma_ABt(c, a, b, c);
        kittens::store(c_output, c, {});
    }
    template<typename RT_SHAPE_ACCUM, int K_DIM, int H, int W, typename K> using make_a_layout = typename kittens::gl<kittens::bf16, 1, 1, RT_SHAPE_ACCUM::rows*H, K_DIM*K::value>;
    template<typename RT_SHAPE_ACCUM, int K_DIM, int H, int W, typename K> using make_b_layout = typename kittens::gl<kittens::bf16, 1, 1, RT_SHAPE_ACCUM::cols*W, K_DIM*K::value>;
    template<typename RT_SHAPE_ACCUM, int K_DIM, int H, int W, typename K> using make_c_layout = typename kittens::gl<kittens::bf16, 1, 1, RT_SHAPE_ACCUM::rows*H, RT_SHAPE_ACCUM::cols*W>;
};
struct test_mma_AtB {
    template<typename RT_SHAPE_ACCUM, int K_DIM, int H, int W, int NW, typename K> using valid = std::bool_constant<NW == 1 && (W*H+W*K::value+H*K::value)<=28>; // this is warp-level
    static inline const std::string test_identifier = "reg_mma_AtB";
    template<typename RT_SHAPE_ACCUM, int K_DIM, int H, int W, int NW, gl_t GTL_A, gl_t GTL_B, gl_t GTL_C, typename _K> __host__ static void host_func(const std::vector<float> &i_ref, std::vector<float> &o_ref) {
        constexpr int K = _K::value;
        for(int i = 0; i < H*RT_SHAPE_ACCUM::rows; i++) {
            for(int j = 0; j < W*RT_SHAPE_ACCUM::cols; j++) {
                float sum = 0;
                for(int k = 0; k < K*K_DIM; k++) {
                    sum += i_ref[i + k*RT_SHAPE_ACCUM::rows*H]*i_ref[(RT_SHAPE_ACCUM::rows*K_DIM*H*K) + k*RT_SHAPE_ACCUM::cols*W + j];
                }
                o_ref[i*RT_SHAPE_ACCUM::cols*W + j] = sum;
            }
        }
    }
    template<typename RT_SHAPE_ACCUM, int K_DIM, int H, int W, int NW, gl_t GTL_A, gl_t GTL_B, gl_t GTL_C, typename _K> __device__ static void device_func(const GTL_A &a_input, const GTL_B &b_input, const GTL_C &c_output) {
        constexpr int K = _K::value;

        kittens::rt_bf<K_DIM*K, RT_SHAPE_ACCUM::rows*H, kittens::ducks::rt_layout::col> a;
        kittens::rt_bf<K_DIM*K, RT_SHAPE_ACCUM::cols*W, kittens::ducks::rt_layout::col> b;
        kittens::rt_fl<RT_SHAPE_ACCUM::rows*H, RT_SHAPE_ACCUM::cols*W, kittens::ducks::rt_layout::col> c;
        kittens::load(a, a_input, {});
        kittens::load(b, b_input, {});
        __builtin_amdgcn_s_waitcnt(0);
        __builtin_amdgcn_s_barrier();
        kittens::zero(c);
        kittens::mma_AtB(c, a, b, c);
        kittens::store(c_output, c, {});
    }
    template<typename RT_SHAPE_ACCUM, int K_DIM, int H, int W, typename K> using make_a_layout = typename kittens::gl<kittens::bf16, 1, 1, K_DIM*K::value, RT_SHAPE_ACCUM::rows*H>;
    template<typename RT_SHAPE_ACCUM, int K_DIM, int H, int W, typename K> using make_b_layout = typename kittens::gl<kittens::bf16, 1, 1, K_DIM*K::value, RT_SHAPE_ACCUM::cols*W>;
    template<typename RT_SHAPE_ACCUM, int K_DIM, int H, int W, typename K> using make_c_layout = typename kittens::gl<kittens::bf16, 1, 1, RT_SHAPE_ACCUM::rows*H, RT_SHAPE_ACCUM::cols*W>;
};
struct test_mma_AtBt {
    template<typename RT_SHAPE_ACCUM, int K_DIM, int H, int W, int NW, typename K> using valid = std::bool_constant<NW == 1 && (W*H+W*K::value+H*K::value)<=28>; // this is warp-level
    static inline const std::string test_identifier = "reg_mma_AtBt";
    template<typename RT_SHAPE_ACCUM, int K_DIM, int H, int W, int NW, gl_t GTL_A, gl_t GTL_B, gl_t GTL_C, typename _K> __host__ static void host_func(const std::vector<float> &i_ref, std::vector<float> &o_ref) {
        constexpr int K = _K::value;
        for(int i = 0; i < H*RT_SHAPE_ACCUM::rows; i++) {
            for(int j = 0; j < W*RT_SHAPE_ACCUM::cols; j++) {
                float sum = 0;
                for(int k = 0; k < K*K_DIM; k++) {
                    sum += i_ref[i+k*H*RT_SHAPE_ACCUM::rows]*i_ref[RT_SHAPE_ACCUM::rows*K_DIM*K*H + j*K*K_DIM+k];
                }
                o_ref[i*W*RT_SHAPE_ACCUM::cols+j] = sum;
            }
        }
    }
    template<typename RT_SHAPE_ACCUM, int K_DIM, int H, int W, int NW, gl_t GTL_A, gl_t GTL_B, gl_t GTL_C, typename _K> __device__ static void device_func(const GTL_A &a_input, const GTL_B &b_input, const GTL_C &c_output) {
        constexpr int K = _K::value;

        kittens::rt_bf<K_DIM*K, RT_SHAPE_ACCUM::rows*H, kittens::ducks::rt_layout::col> a;
        kittens::rt_bf<RT_SHAPE_ACCUM::cols*W, K_DIM*K, kittens::ducks::rt_layout::row> b;
        kittens::rt_fl<RT_SHAPE_ACCUM::rows*H, RT_SHAPE_ACCUM::cols*W, kittens::ducks::rt_layout::col> c;
        kittens::load(a, a_input, {});
        kittens::load(b, b_input, {});
        __builtin_amdgcn_s_waitcnt(0);
        __builtin_amdgcn_s_barrier();
        kittens::zero(c);
        kittens::mma_AtBt(c, a, b, c);
        kittens::store(c_output, c, {});
    }
    template<typename RT_SHAPE_ACCUM, int K_DIM, int H, int W, typename K> using make_a_layout = typename kittens::gl<kittens::bf16, 1, 1, K_DIM*K::value, RT_SHAPE_ACCUM::rows*H>;
    template<typename RT_SHAPE_ACCUM, int K_DIM, int H, int W, typename K> using make_b_layout = typename kittens::gl<kittens::bf16, 1, 1, RT_SHAPE_ACCUM::cols*W, K_DIM*K::value>;
    template<typename RT_SHAPE_ACCUM, int K_DIM, int H, int W, typename K> using make_c_layout = typename kittens::gl<kittens::bf16, 1, 1, RT_SHAPE_ACCUM::rows*H, RT_SHAPE_ACCUM::cols*W>;
};

// Due to the strange sizes instantiated, we need a custom base wrapper here
template<typename Ker, typename T, typename RT_SHAPE_ACCUM, int K_DIM, int H, int W, int NW, gl_t GTL_A, gl_t GTL_B, gl_t GTL_C, typename... args>
static __global__ void mma_global_wrapper_2d(const GTL_A a_input, const GTL_B b_input, GTL_C c_output) {
    Ker::template device_func<RT_SHAPE_ACCUM, K_DIM, H, W, NW, GTL_A, GTL_B, GTL_C, args...>(a_input, b_input, c_output);
}
template<typename test, typename RT_SHAPE_ACCUM, typename ST_SHAPE, int H, int W, int NUM_WORKERS, typename _K, typename... args>
struct mma_wrapper_2d {
    static void run(test_data& results) {
        using namespace kittens;
        constexpr int K = _K::value;
        constexpr int K_DIM = 16; // v_wmma_f32_16x16x16: one K step is 16.
        constexpr int MN_DIM = RT_SHAPE_ACCUM::rows;
        test_info this_result;

        this_result.label = generate_test_name<RT_SHAPE_ACCUM, H, W, NUM_WORKERS, _K, args...>(test::test_identifier);
        if constexpr (test::template valid<RT_SHAPE_ACCUM, K_DIM, H, W, NUM_WORKERS, _K, args...>::value) {
            // initialize
            kittens::bf16 *d_i, *d_o;
            std::vector<float> i_ref((H+W)*K*MN_DIM*K_DIM);
            std::vector<float> o_ref(H*W*MN_DIM*MN_DIM);
            initialize(&d_i, &d_o, i_ref, o_ref);
            // make descriptors
            using GTL_A = test::template make_a_layout<RT_SHAPE_ACCUM, K_DIM, H, W, _K>;
            using GTL_B = test::template make_b_layout<RT_SHAPE_ACCUM, K_DIM, H, W, _K>;
            using GTL_C = test::template make_c_layout<RT_SHAPE_ACCUM, K_DIM, H, W, _K>;
            GTL_A a_input (d_i,           nullptr, nullptr, nullptr, nullptr);
            GTL_B b_input (d_i + H*K*MN_DIM*K_DIM, nullptr, nullptr, nullptr, nullptr);
            GTL_C c_output(d_o,           nullptr, nullptr, nullptr, nullptr);
            // run kernel
            hipFuncSetAttribute(
                reinterpret_cast<void *>(mma_global_wrapper_2d<test, kittens::bf16, RT_SHAPE_ACCUM, K_DIM, H, W, NUM_WORKERS, GTL_A, GTL_B, GTL_C, _K, args...>),
                hipFuncAttributeMaxDynamicSharedMemorySize,
                kittens::MAX_SHARED_MEMORY
            );
            mma_global_wrapper_2d<test, kittens::bf16, RT_SHAPE_ACCUM, K_DIM, H, W, NUM_WORKERS, GTL_A, GTL_B, GTL_C, _K, args...><<<1, NUM_WORKERS*kittens::WARP_THREADS, kittens::MAX_SHARED_MEMORY>>>(a_input, b_input, c_output);
            // fill in correct results on cpu
            test::template host_func<RT_SHAPE_ACCUM, K_DIM, H, W, NUM_WORKERS, GTL_A, GTL_B, GTL_C, _K, args...>(i_ref, o_ref);
            // check and cleanup
            this_result.result = validate(d_i, d_o, i_ref, o_ref, this_result.label, W*RT_SHAPE_ACCUM::cols, 0.10); // mma's sometimes produce small errors. this appears to be hardware.
        }
        else {
            this_result.result = test_result::INVALID;
        }
        // The cdna4 version of this wrapper sets this_result and then drops it,
        // so an mma failure prints but is not counted in the final tally. Every
        // other wrapper in ../common/testing_commons pushes; this one should too,
        // because on a tree that has never run the tally is the whole report.
        results.push_back(this_result);
    };
};
template<typename test, typename RT_SHAPE_ACCUM, typename ST_SHAPE=kittens::ducks::st_shape::st_16x16, int MAX_H=8, int MAX_W=8, int NUM_WORKERS=1, typename... args> using mma_sweep_size = loop_h<mma_wrapper_2d, test, RT_SHAPE_ACCUM, ST_SHAPE, MAX_H, MAX_W, NUM_WORKERS, MAX_H, args...>;
template<typename test, typename RT_SHAPE_ACCUM, typename ST_SHAPE=kittens::ducks::st_shape::st_16x16, int MAX_H=8, int MAX_W=8, typename... args> using mma_sweep_size_warp = mma_sweep_size<test, RT_SHAPE_ACCUM, ST_SHAPE, MAX_H, MAX_W, 1, args...>;

template<kittens::ducks::rt_shape::all RT_SHAPE, kittens::ducks::st_shape::all ST_SHAPE=kittens::ducks::st_shape::st_16x16>
void test_generator(test_data &results) {
    constexpr int SIZE = INTENSITY_0 ? 1  :
                         INTENSITY_1 ? 2  :
                         INTENSITY_2 ? 4  : 
                         INTENSITY_3 ? 8  :
                         INTENSITY_4 ? 16 : -1;

    mma_sweep_size_warp<test_mma_AB, RT_SHAPE, ST_SHAPE, SIZE, SIZE, std::integral_constant<int, 1>>::run(results);
    mma_sweep_size_warp<test_mma_AB, RT_SHAPE, ST_SHAPE, SIZE, SIZE, std::integral_constant<int, 2>>::run(results);
    mma_sweep_size_warp<test_mma_AB, RT_SHAPE, ST_SHAPE, SIZE, SIZE, std::integral_constant<int, 3>>::run(results);
    mma_sweep_size_warp<test_mma_AB, RT_SHAPE, ST_SHAPE, SIZE, SIZE, std::integral_constant<int, 4>>::run(results);
    mma_sweep_size_warp<test_mma_ABt, RT_SHAPE, ST_SHAPE, SIZE, SIZE, std::integral_constant<int, 1>>::run(results);
    mma_sweep_size_warp<test_mma_ABt, RT_SHAPE, ST_SHAPE, SIZE, SIZE, std::integral_constant<int, 2>>::run(results);
    mma_sweep_size_warp<test_mma_ABt, RT_SHAPE, ST_SHAPE, SIZE, SIZE, std::integral_constant<int, 3>>::run(results);
    mma_sweep_size_warp<test_mma_ABt, RT_SHAPE, ST_SHAPE, SIZE, SIZE, std::integral_constant<int, 4>>::run(results);
    mma_sweep_size_warp<test_mma_AtB, RT_SHAPE, ST_SHAPE, SIZE, SIZE, std::integral_constant<int, 1>>::run(results);
    mma_sweep_size_warp<test_mma_AtB, RT_SHAPE, ST_SHAPE, SIZE, SIZE, std::integral_constant<int, 2>>::run(results);
    mma_sweep_size_warp<test_mma_AtB, RT_SHAPE, ST_SHAPE, SIZE, SIZE, std::integral_constant<int, 3>>::run(results);
    mma_sweep_size_warp<test_mma_AtB, RT_SHAPE, ST_SHAPE, SIZE, SIZE, std::integral_constant<int, 4>>::run(results);
    mma_sweep_size_warp<test_mma_AtBt, RT_SHAPE, ST_SHAPE, SIZE, SIZE, std::integral_constant<int, 1>>::run(results);
    mma_sweep_size_warp<test_mma_AtBt, RT_SHAPE, ST_SHAPE, SIZE, SIZE, std::integral_constant<int, 2>>::run(results);
    mma_sweep_size_warp<test_mma_AtBt, RT_SHAPE, ST_SHAPE, SIZE, SIZE, std::integral_constant<int, 3>>::run(results);
    mma_sweep_size_warp<test_mma_AtBt, RT_SHAPE, ST_SHAPE, SIZE, SIZE, std::integral_constant<int, 4>>::run(results);
}


#ifdef KITTENS_RDNA4
/*
 * fp8 -- gfx12 only, which is why this whole section is behind KITTENS_RDNA4
 * rather than living in its own file. tests/unit/rdna3 and tests/unit/rdna4 are
 * the same sources with a different GPU_TARGET, and keeping that true is worth
 * more than a separate translation unit: if the gfx12 fragment inference is
 * wrong, the diff between the two trees should still not be where you look.
 *
 * gfx12 has four fp8 WMMA opcodes, one per (A, B) encoding pair -- fp8_fp8,
 * fp8_bf8, bf8_fp8, bf8_bf8, where AMD's "fp8" is e4m3 and "bf8" is e5m2. There
 * is no operand-select bit, so the only way to reach all four from the tile API
 * is through the element types of the operand tiles, and the only way to know
 * the dispatch is right is to instantiate all four. Hence the cross product
 * below rather than a single e4m3 smoke test.
 *
 * Hand-rolled instead of going through the sweep harness above, the way
 * tests/unit/cdna4's fp8 tests are: that harness types the input buffer, the
 * output buffer and the reference off one type, and here A, B and C are three
 * different ones (and A and B differ from each other in half the cases).
 *
 * Note the shape: 16x16x16, the same as bf16. CDNA's fp8 MFMA doubles K, so the
 * CDNA trees specialize TILE_*_DIM for it; gfx12 does not, so fp8 here buys
 * registers and LDS bytes, not FLOPs.
 */

/// Fill a device buffer with fp8-rounded randoms, and `ref` with what the fp8
/// values actually are -- so the host reference multiplies exactly the numbers
/// the WMMA sees and the only tolerance needed is for accumulation order.
template<typename T>
static void fp8_fill(T **d, std::vector<float> &ref, int seed) {
    std::mt19937 gen(seed);
    std::uniform_real_distribution<float> dis(-1.0f, 1.0f);
    std::vector<T> h(ref.size());
    for(size_t i = 0; i < ref.size(); i++) {
        h[i]   = T(dis(gen));
        ref[i] = float(h[i]);
    }
    hipMalloc(d, ref.size() * sizeof(T));
    hipMemcpy(*d, h.data(), ref.size() * sizeof(T), hipMemcpyHostToDevice);
    HipCheckError();
}

template<typename T> static inline const char *fp8_name();
template<> inline const char *fp8_name<kittens::fp8e4m3>() { return "e4m3"; }
template<> inline const char *fp8_name<kittens::fp8e5m2>() { return "e5m2"; }

template<typename AT, typename BT, int H, int W, int KK, bool VIA_LDS>
__global__ __launch_bounds__(kittens::WARP_THREADS, 1)
void fp8_mma_kernel(const kittens::gl<AT,    1, 1, 16*H,  16*KK> a_gl,
                    const kittens::gl<BT,    1, 1, 16*KK, 16*W > b_gl,
                    const kittens::gl<float, 1, 1, 16*H,  16*W > c_gl) {
    kittens::rt<AT, 16*H,  16*KK, kittens::ducks::rt_layout::row> a;
    kittens::rt<BT, 16*KK, 16*W,  kittens::ducks::rt_layout::col> b;
    kittens::rt_fl<16*H, 16*W, kittens::ducks::rt_layout::col>    c;

    if constexpr (VIA_LDS) {
        // A the long way round, to exercise the fp8 LDS path. It is the one
        // element width whose vectorized shared<->register transfer is
        // ds_load_b64 / ds_store_b64 rather than b128 -- a lane holds 8
        // elements either way, so the access follows sizeof. Round-tripping
        // through LDS twice must be the identity; if the b64 dispatch or the
        // swizzle disagrees with itself, this is what says so.
        extern __shared__ kittens::alignment_dummy __shm[];
        kittens::shared_allocator<16> al((int*)&__shm[0]);
        using ST_A = kittens::st<AT, 16*H, 16*KK>;
        ST_A &a_sh = al.allocate<ST_A>();

        kittens::load(a_sh, a_gl, {0,0,0,0});
        __builtin_amdgcn_s_barrier();
        kittens::load(a, a_sh);
        kittens::store(a_sh, a);
        __builtin_amdgcn_s_barrier();
        kittens::load(a, a_sh);
    }
    else {
        kittens::load(a, a_gl, {});
    }
    kittens::load(b, b_gl, {});
    __builtin_amdgcn_s_barrier();

    kittens::zero(c);
    kittens::mma_AB(c, a, b, c);
    kittens::store(c_gl, c, {});
}

template<typename AT, typename BT, int H, int W, int KK, bool VIA_LDS>
static void fp8_mma_run(test_data &results) {
    test_info this_result;
    this_result.label = std::string("reg_mma_AB_fp8_A=") + fp8_name<AT>() + "_B=" + fp8_name<BT>()
                      + (VIA_LDS ? "_lds" : "") + "_[" + std::to_string(16*H) + "x"
                      + std::to_string(16*W) + "x" + std::to_string(16*KK) + "]";

    constexpr int M = 16*H, N = 16*W, KD = 16*KK;

    std::vector<float> a_ref(M*KD), b_ref(KD*N), c_ref(M*N), c_out(M*N);
    AT *d_a; BT *d_b; float *d_c;
    fp8_fill(&d_a, a_ref, 42);
    fp8_fill(&d_b, b_ref, 1337);
    hipMalloc(&d_c, M*N*sizeof(float));
    HipCheckError();

    kittens::gl<AT,    1, 1, M,  KD> a_gl(d_a, nullptr, nullptr, nullptr, nullptr);
    kittens::gl<BT,    1, 1, KD, N > b_gl(d_b, nullptr, nullptr, nullptr, nullptr);
    kittens::gl<float, 1, 1, M,  N > c_gl(d_c, nullptr, nullptr, nullptr, nullptr);

    hipFuncSetAttribute(
        reinterpret_cast<const void*>(fp8_mma_kernel<AT, BT, H, W, KK, VIA_LDS>),
        hipFuncAttributeMaxDynamicSharedMemorySize,
        kittens::MAX_SHARED_MEMORY / 2
    );
    fp8_mma_kernel<AT, BT, H, W, KK, VIA_LDS>
        <<<1, kittens::WARP_THREADS, kittens::MAX_SHARED_MEMORY / 2>>>(a_gl, b_gl, c_gl);
    hipDeviceSynchronize();
    HipCheckError();
    hipMemcpy(c_out.data(), d_c, M*N*sizeof(float), hipMemcpyDeviceToHost);
    HipCheckError();

    for(int i = 0; i < M; i++)
        for(int j = 0; j < N; j++) {
            float sum = 0;
            for(int k = 0; k < KD; k++) sum += a_ref[i*KD + k] * b_ref[k*N + j];
            c_ref[i*N + j] = sum;
        }

    // Both sides multiply the identical fp8 values in fp32 and accumulate in
    // fp32, so the tolerance covers accumulation order only -- nothing like the
    // 0.10 the bf16 tests above need for their bf16 output rounding.
    std::cout << "test `" << this_result.label << "` ";
    bool good = true;
    float max_diff = 0; int max_diff_idx = -1;
    for(int i = 0; i < M*N; i++) {
        float diff = std::abs(c_ref[i] - c_out[i]);
        if(diff > 1e-2f + 1e-2f*std::abs(c_ref[i])) good = false;
        if(diff > max_diff) { max_diff = diff; max_diff_idx = i; }
    }
    if(good) std::cout << " -- PASSED" << std::endl;
    else {
        std::cout << " ----- ALERT! FAILED test `" << this_result.label << "` -----" << std::endl;
        std::cout << "Largest mismatch at index " << max_diff_idx << ": ref=" << c_ref[max_diff_idx]
                  << ", out=" << c_out[max_diff_idx] << ", diff=" << max_diff << std::endl;
    }

    hipFree(d_a); hipFree(d_b); hipFree(d_c);
    this_result.result = good ? test_result::PASSED : test_result::FAILED;
    results.push_back(this_result);
}

static void fp8_test_generator(test_data &results) {
    using e4m3 = kittens::fp8e4m3;
    using e5m2 = kittens::fp8e5m2;

    // One base tile first: if the fragment mapping is wrong, this is the
    // smallest thing that says so.
    fp8_mma_run<e4m3, e4m3, 1, 1, 1, false>(results);
    fp8_mma_run<e4m3, e5m2, 1, 1, 1, false>(results);
    fp8_mma_run<e5m2, e4m3, 1, 1, 1, false>(results);
    fp8_mma_run<e5m2, e5m2, 1, 1, 1, false>(results);

    // Then a blocked one, which is where a wrong per-tile stride shows up.
    fp8_mma_run<e4m3, e4m3, 2, 2, 2, false>(results);
    fp8_mma_run<e5m2, e5m2, 2, 2, 2, false>(results);

    // And the LDS round trip, for the ds_*_b64 path.
    fp8_mma_run<e4m3, e4m3, 1, 1, 1, true>(results);
    fp8_mma_run<e4m3, e4m3, 2, 2, 4, true>(results);
}
#endif // KITTENS_RDNA4

} // anonymous namespace
void warp::reg::tile::mma::tests(test_data &results) {
    std::cout << "\n ----- Starting ops/warp/register/tile/mma tests! -----\n" << std::endl;

    test_generator<kittens::ducks::rt_shape::rt_16x16>(results);
#ifdef KITTENS_RDNA4
    fp8_test_generator(results);
#endif
}

#endif