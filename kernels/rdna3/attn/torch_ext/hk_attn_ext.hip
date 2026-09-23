// A torch custom operator over the RDNA3 attention forward.
//
// This is the layer a framework actually calls. Every Radeon attention backend
// worth replacing -- diffusers, comfyUI, vLLM, SGLang -- reaches
// F.scaled_dot_product_attention, which on ROCm lands in aotriton's attn_fwd.
// The python wrapper next to this file has that function's exact signature and
// forwards anything the kernel cannot take, so swapping it in is one import.
//
//                            --- what it registers ---
//
//   hk_attn::supported(int n, int d, bool causal, int h_q, int h_kv) -> bool
//   hk_attn::why_unsupported(int n, int d, bool causal, int h_q, int h_kv) -> str
//   hk_attn::fwd(Tensor q, Tensor k, Tensor v, float scale, bool causal) -> Tensor
//
// q is (B, H_q, N, D) bf16 and k/v are (B, H_kv, N, D), the layout torch hands
// SDPA after its transpose. H_kv must divide H_q (GQA); equal is MHA.
// `scale` of 0 means 1/sqrt(D), matching torch's default.
//
//                            --- two things to know ---
//
// **Forward only, and one sequence length.** There is no backward: the target is
// diffusion inference. And the kv loop is bounded by q's sequence length, so
// cross attention with n_kv != n_q is rejected rather than silently reading the
// wrong number of keys -- which also settles where a causal diagonal would sit,
// since with n_kv == n_q there is only one convention.
//
// **Contiguity is checked, not fixed.** The globals are plain (B, H, N, D) row
// major with no stride support, so a non-contiguous input would be read wrong.
// The wrapper calls .contiguous() before the op and eats the copy; the op
// itself refuses, so a direct torch.ops caller cannot get silent garbage.
//
// Build: see Makefile in this directory. Load from python with
// torch.ops.load_library, not torch.utils.cpp_extension.load -- the latter runs
// hipify over the source, and this source is already HIP.

#define HK_ATTN_NO_PYBIND 1
#include "../fwd/attn.cpp"

#include <torch/library.h>
#include <torch/types.h>
#include <ATen/ATen.h>
#include <c10/hip/HIPStream.h>
#include <c10/hip/HIPGuard.h>

#include <string>

namespace {

using cfg = macro_config;

// The shape rule itself lives in attn.cpp next to the tiling it comes from --
// these two only add the head-count relation and turn a "no" into a sentence.
// Names are prefixed because attn.cpp already has a `supported` at global scope.
//
// `causal` is a parameter of both and of neither rule: the kernel takes it, so
// it does not make a shape unsupported. It stays in the signature because the
// wrapper asks one question and it used to be the answer.
bool hk_supported(int64_t n, int64_t d, bool causal, int64_t h_q, int64_t h_kv) {
    (void)causal;
    return ::supported<cfg>((int)n, (int)d) && h_kv > 0 && h_q % h_kv == 0;
}

std::string hk_why_unsupported(int64_t n, int64_t d, bool causal, int64_t h_q, int64_t h_kv) {
    if (hk_supported(n, d, causal, h_q, h_kv)) return "";
    if (!::supported<cfg>((int)n, (int)d)) {
        if (d != 64 && d != 128)
            return "head_dim must be 64 or 128, got " + std::to_string(d);
        if (n < cfg::Q_TILE)
            return "seq_len must be at least the Q tile (" + std::to_string(cfg::Q_TILE) +
                   "), got " + std::to_string(n);
        if (n < cfg::KV_BLOCK)
            return "seq_len must be at least the KV block (" + std::to_string(cfg::KV_BLOCK) +
                   "), got " + std::to_string(n);
    }
    return "kv head count (" + std::to_string(h_kv) + ") must divide query head count (" +
           std::to_string(h_q) + ")";
}

at::Tensor hk_fwd(const at::Tensor &q, const at::Tensor &k, const at::Tensor &v,
                  double scale, bool causal) {
    TORCH_CHECK(q.dim() == 4 && k.dim() == 4 && v.dim() == 4,
                "hk_attn::fwd expects (B, H, N, D) tensors");
    TORCH_CHECK(q.scalar_type() == at::kBFloat16 && k.scalar_type() == at::kBFloat16 &&
                v.scalar_type() == at::kBFloat16, "hk_attn::fwd is bf16 only");
    TORCH_CHECK(q.is_contiguous() && k.is_contiguous() && v.is_contiguous(),
                "hk_attn::fwd needs contiguous (B, H, N, D); it has no stride support");
    TORCH_CHECK(k.sizes() == v.sizes(), "hk_attn::fwd needs k and v the same shape");
    TORCH_CHECK(q.is_cuda() && k.is_cuda() && v.is_cuda(), "hk_attn::fwd needs device tensors");
    TORCH_CHECK(k.get_device() == q.get_device() && v.get_device() == q.get_device(),
                "hk_attn::fwd needs q, k and v on one device");

    const int64_t b = q.size(0), h = q.size(1), n = q.size(2), d = q.size(3);
    const int64_t h_kv = k.size(1);
    TORCH_CHECK(k.size(0) == b && k.size(3) == d,
                "hk_attn::fwd needs k and q to agree on batch and head_dim");
    TORCH_CHECK(k.size(2) == n,
                "hk_attn::fwd needs n_kv == n_q; the kv loop is bounded by q's length");
    const std::string why = hk_why_unsupported(n, d, causal, h, h_kv);
    TORCH_CHECK(why.empty(), "hk_attn::fwd: ", why);

    // A DeviceType::CUDA index would be rejected outright by HIPGuardImpl
    // ("initialized with non-HIP DeviceType"), so take the DeviceIndex overload.
    const c10::hip::HIPGuard guard((c10::DeviceIndex)q.get_device());
    at::Tensor o = at::empty_like(q);

    // Aggregate-initialized: gl has no default constructor, so a
    // default-construct-then-assign does not compile.
    micro_globals g{
        .q = _gl_q((bf16 *)q.data_ptr(), (int)b, (int)h, (int)n, (int)d),
        .k = _gl_k((bf16 *)k.data_ptr(), (int)b, (int)h_kv, (int)n, (int)d),
        .v = _gl_v((bf16 *)v.data_ptr(), (int)b, (int)h_kv, (int)n, (int)d),
        .o = _gl_o((bf16 *)o.data_ptr(), (int)b, (int)h, (int)n, (int)d),
        .scale  = (float)scale,
        .causal = causal,
        .stream = c10::hip::getCurrentHIPStream(),
    };
    dispatch_micro(g);
    return o;
}

}  // namespace

TORCH_LIBRARY(hk_attn, m) {
    m.def("supported(int n, int d, bool causal, int h_q, int h_kv) -> bool");
    m.def("why_unsupported(int n, int d, bool causal, int h_q, int h_kv) -> str");
    m.def("fwd(Tensor q, Tensor k, Tensor v, float scale, bool causal) -> Tensor");
}

TORCH_LIBRARY_IMPL(hk_attn, CompositeExplicitAutograd, m) {
    m.impl("supported", TORCH_FN(hk_supported));
    m.impl("why_unsupported", TORCH_FN(hk_why_unsupported));
}

TORCH_LIBRARY_IMPL(hk_attn, CUDA, m) {
    m.impl("fwd", TORCH_FN(hk_fwd));
}
