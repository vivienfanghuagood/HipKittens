// A torch custom operator over the fused GEMM->collective kernels.
//
// This is the layer a framework actually calls. vLLM's and SGLang's tensor
// parallel row-parallel linear is, in both of them, a matmul over a K-slice
// followed by an all-reduce; this replaces both with one dispatch whose
// epilogue pushes each output column block straight into its owner's inbox, so
// the transfer happens while the rest of the GEMM is still running.
//
//                            --- what it registers ---
//
//   hk_dist::init(rank, world, max_tokens, hidden)   build the IPC heaps
//   hk_dist::shutdown()                              tear them down
//   hk_dist::supported(M, N, K_local, which)         would the fused path run?
//   hk_dist::linear_allreduce(x, w) -> (M, N)        x: (M, K_local), w: (N, K_local)
//   hk_dist::linear_reducescatter(x, w) -> (M/w, N)
//
// w is exactly vLLM's RowParallelLinear.weight: (output_size,
// input_size_per_partition), row major, no transpose anywhere.
//
//                            --- two things to know ---
//
// **Host syncs.** On gfx1100 a peer's write into my HBM does not invalidate my
// L2, and no instruction a shader can issue does either. The only thing that
// does is the command processor's system-scope acquire, which the runtime emits
// on the first dispatch after a host sync. So the fused path brackets its
// barrier with two hipStreamSynchronize calls -- a release before, an acquire
// after -- and that means this op blocks the calling thread twice per layer.
// The alternative, an uncached inbox heap (HK_DIST_UNCACHED_INBOX=1), needs
// neither sync but was measured slower end to end (2.57 vs 2.09 ms on prefill
// attn_out). Both are here; the default is the faster one.
//
// **The output is copied.** combine_and_gather has to write into a *symmetric*
// buffer, because every peer pushes its finished column shard into mine, and
// torch's caching allocator cannot hand out fine-grained IPC-exportable memory.
// So the all-reduce form runs into the heap and then copies out, which costs
// about 0.21 ms at M=8192 -- 4% of the fused attn_out, 2% of mlp_down. The
// reduce-scatter form does not pay it: its output is local, so the op writes
// the torch tensor directly. Removing it for all-reduce means teaching torch to
// allocate from the symmetric heap, which is a pluggable-allocator change, not
// a kernel one.
//
// Build: see Makefile in this directory. Load from python with
// torch.ops.load_library, not torch.utils.cpp_extension.load -- the latter runs
// hipify over the source, and this source is already HIP.

#define HK_GEMM_NO_PYBIND 1
#include "../../gemm/bf16fp32/gemm.cpp"
#include "../ipc_heap.cuh"
#include "../fused.cuh"

#include <torch/library.h>
#include <torch/types.h>
#include <ATen/ATen.h>
#include <c10/hip/HIPStream.h>
#include <c10/hip/HIPGuard.h>

#include <memory>
#include <string>

namespace {

using hk_dist::sym_view;
using hk_dist::micro_globals_ar;
using hk_dist::micro_globals_rs;

constexpr int TILE_M = thin_config::BLOCK_M;   // 16: the narrowest row tile

// Process-global, because the heaps are: hipIpcOpenMemHandle is per process,
// the bootstrap rendezvous is per process, and a second set would deadlock
// against the first. One rank, one device, one state.
struct dist_state {
    std::unique_ptr<hk_dist::file_bootstrap> bs;
    std::unique_ptr<hk_dist::ipc_heap>       heap;   // outputs, symmetric
    std::unique_ptr<hk_dist::ipc_heap>       ibox;   // inbox, peer-written
    std::unique_ptr<hk_dist::host_flags>     hf;

    int rank = -1, world = 0, dev = 0;
    int max_m = 0, max_n = 0;
    bool inbox_cached = true;
    long long budget = 0;
    uint32_t epoch = 0;
    bool failed = false;

    bf16 *Cout = nullptr;    // (max_m, max_n) symmetric: the AR gather target
    bf16 *Cown = nullptr;    // (max_m, max_n/world) my column shard
    bf16 *Inbox = nullptr;   // [world][max_m * max_n / world]

    bool ready() const { return rank >= 0; }
};

dist_state st;

void check_ready() {
    TORCH_CHECK(st.ready(), "hk_dist: init() has not been called");
    TORCH_CHECK(!st.failed,
                "hk_dist: a previous cross-rank barrier timed out; this "
                "process group is no longer usable");
}

// The barrier, and the pair of syncs the L2 note above explains.
void sync_ranks(hipStream_t s) {
    hipLaunchKernelGGL(hk_dist::dev_barrier, dim3(1), dim3(hk_dist::MAX_RANKS),
                       0, s, st.hf->device(), ++st.epoch, st.world, st.rank,
                       st.budget, reinterpret_cast<unsigned *>(
                                      st.hf->device() + hk_dist::MAX_RANKS));
}

// Called once per op, after the trailing barrier has been waited on. A timeout
// means a peer is gone or wedged and every number after it is void, so it
// latches rather than being reported once and forgotten.
void check_barrier() {
    const unsigned t = st.hf->host()[hk_dist::MAX_RANKS];
    if (t && !st.failed) {
        st.failed = true;
        TORCH_CHECK(false, "hk_dist: cross-rank device barrier timed out at "
                           "epoch ", t, " -- a peer is gone or wedged");
    }
}

// The shapes the fused epilogues accept, as one function so python can ask
// before committing and the ops can refuse with the same words.
//
// The N-sharded (all-reduce) constraints: a whole warp tile must not straddle
// two owners, and the widest warp tile any config in gemm.cpp emits is 64
// columns, so the shard width has to be a multiple of 64. M is free -- that is
// the entire reason all-reduce shards along N -- which is what lets decode use
// this path at all.
//
// The M-sharded (reduce-scatter) constraints are the ones that reject decode:
// the backed-up last M block needs M >= BLOCK_M, and ownership is decided per
// warp row.
const char *unsupported(int M, int N, int K_local, bool ar) {
    if (!st.ready())                            return "init() has not been called";
    if (N % st.world)                           return "N does not split across ranks";
    if (N % thin_config::BLOCK_N)               return "N does not tile";
    if (K_local % big_config::K_STEP)           return "K_local does not tile";
    if (N > st.max_n)                           return "N exceeds the max_n init() was given";
    if (ar) {
        if ((N / st.world) % 64)                return "the column shard boundary falls inside a warp tile";
        if (((M + TILE_M - 1) / TILE_M) * TILE_M > st.max_m)
                                                return "M exceeds the max_tokens init() was given";
    } else {
        constexpr int BM = big_config::BLOCK_M, RBM = big_config::REG_BLOCK_M;
        if (M % st.world)                       return "M does not split across ranks";
        if (M < BM)                             return "M < BLOCK_M (reduce-scatter cannot shard decode)";
        if (M % RBM)                            return "M is not a multiple of REG_BLOCK_M";
        if ((M / st.world) % RBM)               return "the row shard boundary falls inside a warp tile";
        if (N % big_config::BLOCK_N)            return "N does not tile for the 128-row config";
        if (M > st.max_m)                       return "M exceeds the max_tokens init() was given";
    }
    return nullptr;
}

// x is (M, K_local) bf16 on this rank's device. The GEMM writes whole TILE_M
// row tiles and loads A without a predicate, so below one tile it would read
// and write past the ends of buffers sized at M rows. Pad through a zeroed
// staging tensor; at decode M that is under 150 KB against ~30 MB of weight
// traffic. Above one tile nothing is needed: the last block backs up to
// M - BLOCK_M and recomputes the overlap idempotently.
at::Tensor pad_rows(const at::Tensor &x, int M, int M_pad) {
    if (M == M_pad) return x;
    at::Tensor p = at::zeros({M_pad, x.size(1)}, x.options());
    p.narrow(0, 0, M).copy_(x);
    return p;
}

void check_operands(const at::Tensor &x, const at::Tensor &w) {
    TORCH_CHECK(x.scalar_type() == at::kBFloat16 && w.scalar_type() == at::kBFloat16,
                "hk_dist: both operands must be bfloat16");
    TORCH_CHECK(x.is_contiguous() && w.is_contiguous(),
                "hk_dist: both operands must be contiguous");
    TORCH_CHECK(x.dim() == 2 && w.dim() == 2, "hk_dist: both operands must be 2-D "
                "(flatten the leading dims; the op is shape-agnostic in M)");
    TORCH_CHECK(x.size(1) == w.size(1),
                "hk_dist: x is (M, K_local) and w is (N, K_local); got K ",
                x.size(1), " and ", w.size(1));
    TORCH_CHECK(x.is_cuda() && w.is_cuda(), "hk_dist: both operands must be on device");
    TORCH_CHECK(x.get_device() == st.dev, "hk_dist: operand is on device ",
                x.get_device(), " but init() bound device ", st.dev);
}

// ---------------------------------------------------------------------------
// The ops
// ---------------------------------------------------------------------------

void init(int64_t rank, int64_t world, int64_t max_tokens, int64_t hidden) {
    TORCH_CHECK(!st.ready(), "hk_dist: init() called twice");
    TORCH_CHECK(world >= 1 && world <= hk_dist::MAX_RANKS,
                "hk_dist: world ", world, " outside [1, ", hk_dist::MAX_RANKS, "]");
    TORCH_CHECK(rank >= 0 && rank < world, "hk_dist: rank out of range");
    TORCH_CHECK(hidden % world == 0, "hk_dist: hidden must split across ranks");
    TORCH_CHECK(getenv("HK_DIST_RUN_ID"),
                "hk_dist: HK_DIST_RUN_ID is unset. The python init() broadcasts "
                "one from rank 0 and sets it; call that, not this, directly");

    st.rank = (int)rank;
    st.world = (int)world;
    // Round up to a whole row tile so every buffer can hold what the GEMM
    // writes at the largest M the caller promised.
    st.max_m = (int)(((max_tokens + TILE_M - 1) / TILE_M) * TILE_M);
    st.max_n = (int)hidden;
    st.dev = (int)c10::hip::current_device();
    st.inbox_cached = !(getenv("HK_DIST_UNCACHED_INBOX") &&
                        atoi(getenv("HK_DIST_UNCACHED_INBOX")));

    st.bs = std::make_unique<hk_dist::file_bootstrap>(st.rank, st.world);

    const size_t c_elems = (size_t)st.max_m * st.max_n;
    const size_t own_elems = c_elems / st.world;
    // Both collectives' inboxes are world slots of C/world, so one size serves
    // both: [world][max_m * max_n / world] either way.
    st.heap = std::make_unique<hk_dist::ipc_heap>(
        (c_elems + own_elems) * sizeof(bf16) + (1 << 20), st.dev, *st.bs);
    st.ibox = std::make_unique<hk_dist::ipc_heap>(
        c_elems * sizeof(bf16) + (1 << 20), st.dev, *st.bs, !st.inbox_cached);
    st.Cout = st.heap->allocate<bf16>(c_elems);
    st.Cown = st.heap->allocate<bf16>(own_elems);
    st.Inbox = st.ibox->allocate<bf16>(c_elems);
    // Each rank allocated on its own; translate() is only a subtraction if they
    // all asked for the same thing in the same order, and this is what turns a
    // violation of that into an abort instead of silent corruption.
    st.heap->check_symmetry();
    st.ibox->check_symmetry();

    st.hf = std::make_unique<hk_dist::host_flags>(hk_dist::MAX_RANKS + 1, *st.bs);

    int wall_khz = 0;
    HKD_CHECK(hipDeviceGetAttribute(&wall_khz, hipDeviceAttributeWallClockRate,
                                    st.dev));
    st.budget = (long long)wall_khz * 2000;   // 2 s, so a dead peer terminates

    st.bs->barrier();
}

void shutdown() {
    if (!st.ready()) return;
    st.hf.reset(); st.ibox.reset(); st.heap.reset(); st.bs.reset();
    st = dist_state{};
}

bool supported(int64_t M, int64_t N, int64_t K_local, c10::string_view which) {
    return unsupported((int)M, (int)N, (int)K_local, which != "reduce_scatter")
           == nullptr;
}

std::string why_unsupported(int64_t M, int64_t N, int64_t K_local,
                            c10::string_view which) {
    const char *r = unsupported((int)M, (int)N, (int)K_local,
                                which != "reduce_scatter");
    return r ? std::string(r) : std::string();
}

at::Tensor linear_allreduce(const at::Tensor &x, const at::Tensor &w) {
    check_ready();
    check_operands(x, w);
    const int M = (int)x.size(0), K_local = (int)x.size(1), N = (int)w.size(0);
    if (const char *why = unsupported(M, N, K_local, /*ar=*/true))
        TORCH_CHECK(false, "hk_dist::linear_allreduce: ", why, " (M=", M,
                    " N=", N, " K_local=", K_local, " world=", st.world, ")");

    // The index overload, not the Device one: torch on ROCm still tags its
    // devices DeviceType::CUDA, and HIPGuardImpl rejects that outright
    // ("HIPGuardImpl initialized with non-HIP DeviceType: cuda").
    const c10::hip::HIPGuard guard((c10::DeviceIndex)x.get_device());
    const hipStream_t S = c10::hip::getCurrentHIPStream();
    const int M_pad = ((M + TILE_M - 1) / TILE_M) * TILE_M;
    const int ns = N / st.world;
    const at::Tensor xp = pad_rows(x, M, M_pad);
    at::Tensor out = at::empty({M, N}, x.options());

    micro_globals_ar g{
        .a = _gl_A((bf16 *)xp.data_ptr(), 1, 1, M_pad, K_local),
        .b = _gl_B((bf16 *)w.data_ptr(), 1, 1, N, K_local),
        // Only its dims are read on this path (by the split-K reduction), but
        // they have to be the padded ones: that pass walks c.rows() x c.cols().
        .c = _gl_C(st.Cout, 1, 1, M_pad, N),
        .v = st.ibox->view(),
        .c_own = st.Cown,
        // My *slot* of the inbox. translate()d into an owner's address space it
        // becomes that owner's slot for me, so senders never collide and no
        // rank needs to know which slot it is writing.
        .c_sym = st.Inbox + (size_t)st.rank * (size_t)M_pad * ns,
        .shard_cols = ns,
        .my_rank = st.rank,
        .ws = nullptr,      // launch() owns the split-K scratch
        .stream = S,
    };
    dispatch_any(g);

    if (st.inbox_cached) HKD_CHECK(hipStreamSynchronize(S));
    sync_ranks(S);
    if (st.inbox_cached) HKD_CHECK(hipStreamSynchronize(S));
    // M_pad, not M, and that is not a rounding choice. combine_and_gather
    // derives the inbox slot stride from the M it is given, and the epilogue
    // wrote slot r at r * M_pad * ns -- because the GEMM emits whole 16-row
    // tiles and c_sym was offset by that. Passing M made the two disagree for
    // every M below a tile: rank 1's slot was read 15 rows early and decode
    // came out at rel 0.6-0.8 while every M >= 16 was clean. The extra rows
    // are zero (pad_rows zeroes them, so their products are too) and the copy
    // out below takes only the first M, so the cost is a few gathered rows.
    hipLaunchKernelGGL(hk_dist::combine_and_gather, dim3(1024), dim3(256), 0, S,
                       st.heap->view(), st.Cout, st.Cown, st.Inbox,
                       M_pad, N, ns, st.world, st.rank);
    // Not optional. Without it a peer's next GEMM pushes into my inbox while my
    // combine is still reading it, and its gather overwrites my Cout while I am
    // still copying out of it. The only thing between them is my own stream,
    // which orders nothing on their device. Double-buffering is how to remove
    // it; deleting it is not.
    sync_ranks(S);
    HKD_CHECK(hipStreamSynchronize(S));
    check_barrier();

    HKD_CHECK(hipMemcpyAsync(out.data_ptr(), st.Cout,
                             (size_t)M * N * sizeof(bf16),
                             hipMemcpyDeviceToDevice, S));
    return out;
}

at::Tensor linear_reducescatter(const at::Tensor &x, const at::Tensor &w) {
    check_ready();
    check_operands(x, w);
    const int M = (int)x.size(0), K_local = (int)x.size(1), N = (int)w.size(0);
    if (const char *why = unsupported(M, N, K_local, /*ar=*/false))
        TORCH_CHECK(false, "hk_dist::linear_reducescatter: ", why, " (M=", M,
                    " N=", N, " K_local=", K_local, " world=", st.world, ")");

    // The index overload, not the Device one: torch on ROCm still tags its
    // devices DeviceType::CUDA, and HIPGuardImpl rejects that outright
    // ("HIPGuardImpl initialized with non-HIP DeviceType: cuda").
    const c10::hip::HIPGuard guard((c10::DeviceIndex)x.get_device());
    const hipStream_t S = c10::hip::getCurrentHIPStream();
    const int shard = M / st.world;
    // The output is my row shard and it is local -- peers push into the inbox,
    // never into this -- so unlike the all-reduce form the op can hand the
    // epilogue the torch tensor itself and skip the copy out entirely.
    at::Tensor out = at::empty({shard, N}, x.options());

    micro_globals_rs g{
        .a = _gl_A((bf16 *)x.data_ptr(), 1, 1, M, K_local),
        .b = _gl_B((bf16 *)w.data_ptr(), 1, 1, N, K_local),
        .c = _gl_C((bf16 *)out.data_ptr(), 1, 1, shard, N),
        .v = st.ibox->view(),
        .c_sym = st.Inbox + (size_t)st.rank * (size_t)shard * N,
        .shard_rows = shard,
        .my_rank = st.rank,
        .stream = S,
    };
    dispatch_any(g);

    if (st.inbox_cached) HKD_CHECK(hipStreamSynchronize(S));
    sync_ranks(S);
    if (st.inbox_cached) HKD_CHECK(hipStreamSynchronize(S));
    hipLaunchKernelGGL(hk_dist::combine_n, dim3(1024), dim3(256), 0, S,
                       (bf16 *)out.data_ptr(), st.Inbox,
                       (size_t)shard * N, st.world, st.rank);
    sync_ranks(S);
    HKD_CHECK(hipStreamSynchronize(S));
    check_barrier();
    return out;
}

// The inbox has to start at zero for the *first* iteration of a run: the
// combine adds every peer slot unconditionally, and on iteration one the slots
// a peer has not written yet are whatever hipMalloc left there. The heap
// constructor memsets, so this is only needed if a caller wants to reuse a heap
// across an inbox-corrupting failure; exposed because diagnosing that without
// it is unpleasant.
void reset_inbox() {
    check_ready();
    const hipStream_t S = c10::hip::getCurrentHIPStream();
    HKD_CHECK(hipMemsetAsync(st.Inbox, 0,
                             (size_t)st.max_m * st.max_n * sizeof(bf16), S));
    HKD_CHECK(hipStreamSynchronize(S));
}

}  // namespace

TORCH_LIBRARY(hk_dist, m) {
    m.def("init(int rank, int world, int max_tokens, int hidden) -> ()");
    m.def("shutdown() -> ()");
    m.def("reset_inbox() -> ()");
    m.def("supported(int M, int N, int K_local, str which) -> bool");
    m.def("why_unsupported(int M, int N, int K_local, str which) -> str");
    m.def("linear_allreduce(Tensor x, Tensor w) -> Tensor");
    m.def("linear_reducescatter(Tensor x, Tensor w) -> Tensor");
}

// init/shutdown/supported touch no tensors and are registered for CompositeExplicitAutograd
// so they can be called before any device work; the two linears are HIP-only.
TORCH_LIBRARY_IMPL(hk_dist, CompositeExplicitAutograd, m) {
    m.impl("init", TORCH_FN(init));
    m.impl("shutdown", TORCH_FN(shutdown));
    m.impl("supported", TORCH_FN(supported));
    m.impl("why_unsupported", TORCH_FN(why_unsupported));
    // No tensor argument, so there is nothing for the dispatcher to read a
    // device key off; it has to sit on the catch-all key like init does.
    m.impl("reset_inbox", TORCH_FN(reset_inbox));
}

TORCH_LIBRARY_IMPL(hk_dist, CUDA, m) {
    m.impl("linear_allreduce", TORCH_FN(linear_allreduce));
    m.impl("linear_reducescatter", TORCH_FN(linear_reducescatter));
}
