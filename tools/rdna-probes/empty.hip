#include <hip/hip_runtime.h>
#include "kittens.cuh"
using namespace kittens;

__global__ void empty_kernel(const bf16* a, const bf16* b, float* c) {
    rt_base<bf16, ducks::rt_layout::row> A;
    rt_base<bf16, ducks::rt_layout::col> B;
    rt_base<float, ducks::rt_layout::col> C;
    static_assert(decltype(A)::packed_per_thread == 8, "A");
    static_assert(decltype(C)::packed_per_thread == 4, "C");
    #pragma unroll
    for (int i = 0; i < 8; i++) { A.data[i] = bf16_2{a[i*2], a[i*2+1]}; B.data[i] = bf16_2{b[i*2], b[i*2+1]}; }
    #pragma unroll
    for (int i = 0; i < 4; i++) C.data[i] = float2{0.f, 0.f};
    mma_AB_base(C, A, B, C);
    #pragma unroll
    for (int i = 0; i < 4; i++) { c[i*2] = C.data[i].x; c[i*2+1] = C.data[i].y; }
}

__global__ void tile_kernel(const bf16* a, const bf16* b, float* c) {
    rt<bf16, 32, 64, ducks::rt_layout::row> A;
    rt<bf16, 64, 32, ducks::rt_layout::col> B;
    rt<float, 32, 32, ducks::rt_layout::col> C;
    zero(C);
    mma_AB(C, A, B, C);
    c[threadIdx.x] = C.tiles[0][0].data[0].x;
}

int main() {
    rv<float, 64, ducks::rv_layout::align> va;
    rv<bf16,  64, ducks::rv_layout::align> vb;
    rv<float, 64, ducks::rv_layout::ortho> vo;
    static_assert(decltype(va)::inner_dim == 4, "align f32 inner_dim");
    static_assert(decltype(vb)::inner_dim == 8, "align bf16 inner_dim");
    static_assert(decltype(vo)::inner_dim == 1, "ortho inner_dim");
    printf("ok\n");
    return 0;
}
