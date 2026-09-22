// The same symmetric heap as symmem.cuh, in the process model a framework has.
//
// symmem.cuh drives every GPU from one process and reaches peer memory through
// hipDeviceEnablePeerAccess. vLLM and SGLang run one process per rank, so their
// peer memory can only be IPC-mapped. This header exists so the benchmarks can
// run in the model that will actually be deployed, because a number that only
// exists in a benchmark harness is not a reason to integrate anything.
//
// It was written expecting that to cost bandwidth. It does not. iris_p2p.hip
// had read 14.5 GB/s through an IPC mapping against p2p_bw.hip's 27.0 through
// hipDeviceEnablePeerAccess, and the gap survived controlling for heap type,
// store width, buffer size, consumer spin and eager peer-access -- so the
// mapping looked like the only thing left. It was not: this node's peer
// bandwidth is *bimodal per process launch*, about 15.5 GB/s or about 24-27,
// with nothing in between, and both of those numbers are just the two modes of
// the same link. ipc_heap_test reaches 24.0 GB/s through IPC in both uni and
// ring modes. There is no penalty for the deployment process model.
//
// What that costs instead is the right to compare across runs. Two numbers
// measured in two processes have a coin-flip chance of sitting in different
// modes, so any "communication vs computation" claim has to measure both sides
// in the same process and the same run. gemm_rs_mp.hip is built that way.
//
// What changes from symmem.cuh, and what does not:
//
//   * sym_view is reused verbatim. The device side genuinely does not care:
//     translate() is pointer arithmetic against a table of heap bases, and
//     whether a base came from hipMalloc on another device or from
//     hipIpcOpenMemHandle on this one, an offset into it is still the same
//     offset. Nothing in a kernel written against symmem.cuh needs to change.
//   * The host class is per-rank rather than per-node. allocate() hands back
//     one pointer, not a vector of them, and the "every rank allocates the same
//     sizes in the same order" invariant that makes translate() a subtraction
//     is no longer enforceable by construction -- so it is checked instead, see
//     check_symmetry().
//   * Handle exchange and the host barrier are abstracted behind `bootstrap`.
//     The standalone benchmarks use files; the torch extension will hand in a
//     ProcessGroup-backed one. There is deliberately no MPI: the node has none,
//     and both implementations are a dozen lines.
#pragma once

#include <hip/hip_runtime.h>

#include <cerrno>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <thread>
#include <vector>

#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

#include "symmem.cuh"   // sym_view, MAX_RANKS, HKD_CHECK

namespace hk_dist {

// ---------------------------------------------------------------------------
// Bootstrap: an allgather of fixed-size blobs, and a barrier. That is the
// entire set of collectives the heap needs, and neither is on the device.
// ---------------------------------------------------------------------------
struct bootstrap {
    virtual ~bootstrap() = default;
    virtual int rank() const = 0;
    virtual int world() const = 0;
    // `recv` must have world()*bytes of room; rank r's contribution lands at
    // recv + r*bytes, including this rank's own.
    virtual void allgather(const void *send, void *recv, size_t bytes) = 0;
    virtual void barrier() = 0;
};

// File-based bootstrap, for the standalone benchmarks.
//
// One directory per run. A contribution is written to a temporary name and
// renamed into place, because rename is atomic on a local filesystem and a
// reader must never see a half-written IPC handle -- which would not fail
// loudly, it would open some other allocation or corrupt one.
//
// The run id has to be unique per run and shared by all ranks. It is taken from
// the environment rather than generated, because a generated one cannot be
// shared and a fixed one would let a previous run's stale handles be read as
// this run's -- the handles would open successfully and name freed memory.
class file_bootstrap : public bootstrap {
public:
    file_bootstrap(int rank, int world, double timeout_s = 120.0)
        : rank_(rank), world_(world), timeout_s_(timeout_s) {
        const char *id = getenv("HK_DIST_RUN_ID");
        if (!id) {
            fprintf(stderr, "HK_DIST_RUN_ID is not set. It must be the same for "
                            "every rank and different for every run; see run_mp.sh\n");
            abort();
        }
        dir_ = std::string("/tmp/hk_dist_") + id;
        if (mkdir(dir_.c_str(), 0700) != 0 && errno != EEXIST) {
            perror("mkdir"); abort();
        }
    }

    // No cleanup here, deliberately. Rank 0 used to unlink everything it had
    // written, and "once everyone has left" is not something a destructor can
    // know: a peer still inside the *final* barrier, about to read rank 0's
    // file for that generation, finds it gone and waits out the full 120 s
    // timeout before aborting. run_mp.sh removes the directory after every rank
    // has exited, which is the only place that fact is available.
    ~file_bootstrap() override = default;

    int rank() const override { return rank_; }
    int world() const override { return world_; }

    void allgather(const void *send, void *recv, size_t bytes) override {
        const std::string tag = "ag" + std::to_string(gen_++);
        put(tag, send, bytes);
        for (int r = 0; r < world_; r++)
            get(tag, r, (char *)recv + (size_t)r * bytes, bytes);
    }

    void barrier() override {
        const std::string tag = "ba" + std::to_string(gen_++);
        const char one = 1;
        put(tag, &one, 1);
        char sink;
        for (int r = 0; r < world_; r++) get(tag, r, &sink, 1);
    }

private:
    std::string path(const std::string &tag, int r) const {
        return dir_ + "/" + tag + "." + std::to_string(r);
    }

    void put(const std::string &tag, const void *p, size_t bytes) {
        const std::string fin = path(tag, rank_);
        const std::string tmp = fin + ".tmp";
        const int fd = open(tmp.c_str(), O_WRONLY | O_CREAT | O_TRUNC, 0600);
        if (fd < 0) { perror("open"); abort(); }
        if (write(fd, p, bytes) != (ssize_t)bytes) { perror("write"); abort(); }
        if (close(fd) != 0) { perror("close"); abort(); }
        if (rename(tmp.c_str(), fin.c_str()) != 0) { perror("rename"); abort(); }
    }

    // Poll until rank r's contribution appears, with a deadline. A deadline and
    // not an indefinite wait: a rank that died leaves the others spinning on a
    // file that will never arrive, and on a shared node the way out of that is
    // someone else noticing.
    void get(const std::string &tag, int r, void *p, size_t bytes) {
        const std::string fin = path(tag, r);
        const auto deadline =
            std::chrono::steady_clock::now() +
            std::chrono::duration<double>(timeout_s_);
        for (;;) {
            const int fd = open(fin.c_str(), O_RDONLY);
            if (fd >= 0) {
                const ssize_t got = read(fd, p, bytes);
                close(fd);
                if (got == (ssize_t)bytes) return;
                // A rename is atomic, so a short read means the writer wrote
                // fewer bytes than this reader expects -- the two sides disagree
                // about the protocol, which no amount of retrying fixes.
                fprintf(stderr, "bootstrap: %s is %zd bytes, expected %zu\n",
                        fin.c_str(), got, bytes);
                abort();
            }
            if (std::chrono::steady_clock::now() > deadline) {
                fprintf(stderr, "bootstrap: timed out after %.0fs waiting for "
                                "rank %d at %s\n", timeout_s_, r, fin.c_str());
                abort();
            }
            std::this_thread::sleep_for(std::chrono::milliseconds(2));
        }
    }

    int rank_, world_;
    double timeout_s_;
    int gen_ = 0;
    std::string dir_;
};

// ---------------------------------------------------------------------------
// One rank, one device, a heap every other rank can name.
// ---------------------------------------------------------------------------
class ipc_heap {
public:
    // `device` is an index into this process's visible devices. Every rank must
    // see every device -- all the ranks run with the same HIP_VISIBLE_DEVICES
    // and differ only in which one they set -- because hipIpcOpenMemHandle has
    // to be able to reach the exporting device.
    // `uncached` asks for hipDeviceMallocUncached rather than fine-grained.
    // Use it only for buffers a *peer* writes and this rank reads, and expect
    // to pay for it on anything read more than once -- see the note below.
    ipc_heap(size_t bytes_per_rank, int device, bootstrap &bs,
             bool uncached = false)
        : bytes_(bytes_per_rank), dev_(device), bs_(bs), uncached_(uncached) {
        if (bs_.world() > MAX_RANKS) {
            fprintf(stderr, "world %d > MAX_RANKS %d\n", bs_.world(), MAX_RANKS);
            abort();
        }
        HKD_CHECK(hipSetDevice(dev_));

        // Fine-grained for the same reason as symmem.cuh: coarse-grained device
        // memory is only coherent at kernel boundaries, so a peer store landing
        // mid-kernel may never be seen by a consumer spinning in another one.
        //
        // ...but on gfx1100 fine-grained is not enough on its own for a buffer a
        // peer writes and I read from inside a kernel. The peer's write lands in
        // my HBM without invalidating my L2, and nothing a shader can execute
        // invalidates L2: buffer_gl0_inv/buffer_gl1_inv reach L0 and L1 and stop
        // there, and slc/dlc on a load are an L2 *policy* hint (evict-first),
        // not a forced miss. The only thing that invalidates L2 is the command
        // processor's system-scope acquire at dispatch, which the runtime emits
        // after a host sync -- measured: a bare hipStreamSynchronize between the
        // push and the consumer kernel makes the read correct, and a
        // nontemporal load inside the kernel does not, though it does compile to
        // the bypass bits (flat_load_u16 ... slc dlc).
        //
        // hipDeviceMallocUncached keeps the buffer out of L2 in the first place,
        // which removes the question. It is not free: with the *whole* heap
        // uncached the fused GEMM drops from 1.28x to 1.02x, because A and B are
        // read many times each. So this is a per-heap choice, and the inbox --
        // written once by a peer, streamed once by the combine, never reused --
        // is the one buffer that wants it.
        HKD_CHECK(hipExtMallocWithFlags(
            &mine_, bytes_,
            uncached_ ? hipDeviceMallocUncached : hipDeviceMallocFinegrained));
        HKD_CHECK(hipMemset(mine_, 0, bytes_));

        hipIpcMemHandle_t h;
        HKD_CHECK(hipIpcGetMemHandle(&h, mine_));
        std::vector<hipIpcMemHandle_t> all(bs_.world());
        bs_.allgather(&h, all.data(), sizeof h);

        view_.cur_rank_ = bs_.rank();
        view_.world_size_ = bs_.world();
        peers_.assign(bs_.world(), nullptr);
        for (int r = 0; r < bs_.world(); r++) {
            if (r == bs_.rank()) {
                view_.heap_bases_[r] = reinterpret_cast<uintptr_t>(mine_);
                continue;
            }
            // hipIpcMemLazyEnablePeerAccess is the only supported flag; an
            // eager hipDeviceEnablePeerAccess first was tried in iris_p2p.hip
            // and changed neither correctness nor bandwidth.
            void *p = nullptr;
            HKD_CHECK(hipIpcOpenMemHandle(&p, all[r], hipIpcMemLazyEnablePeerAccess));
            peers_[r] = p;
            view_.heap_bases_[r] = reinterpret_cast<uintptr_t>(p);
        }

        HKD_CHECK(hipStreamCreate(&stream_));
        bs_.barrier();   // nobody leaves the constructor before every heap exists
    }

    ~ipc_heap() {
        (void)hipSetDevice(dev_);
        (void)hipStreamDestroy(stream_);
        for (int r = 0; r < (int)peers_.size(); r++)
            if (peers_[r]) (void)hipIpcCloseMemHandle(peers_[r]);
        (void)hipFree(mine_);
    }

    // Bump allocation. Unlike sym_heap::allocate_all this cannot enforce that
    // every rank asks for the same thing in the same order -- each rank is its
    // own process -- so the invariant that makes translate() a subtraction is
    // only a convention here. check_symmetry() is what turns a violation of it
    // into an abort instead of silent corruption.
    template <typename T>
    T *allocate(size_t n) {
        const size_t want = (n * sizeof(T) + 255) & ~size_t(255);
        if (used_ + want > bytes_) {
            fprintf(stderr, "rank %d: symmetric heap exhausted: %zu + %zu > %zu\n",
                    bs_.rank(), used_, want, bytes_);
            abort();
        }
        T *p = reinterpret_cast<T *>(reinterpret_cast<uintptr_t>(mine_) + used_);
        used_ += want;
        // FNV-1a over the sizes, in order. Total size alone would miss two ranks
        // that allocated the same bytes in a different order, which is the
        // failure that produces valid-looking pointers into the wrong object.
        sig_ = (sig_ ^ want) * 1099511628211ull;
        return p;
    }

    // Every rank must have run the same allocation sequence. Collective: call it
    // on all ranks after setup, before any peer traffic.
    void check_symmetry() {
        struct { uint64_t used, sig; } me{used_, sig_}, *all;
        std::vector<char> buf(sizeof me * bs_.world());
        bs_.allgather(&me, buf.data(), sizeof me);
        all = reinterpret_cast<decltype(all)>(buf.data());
        for (int r = 0; r < bs_.world(); r++) {
            if (all[r].used == me.used && all[r].sig == me.sig) continue;
            fprintf(stderr,
                    "rank %d: allocation sequence differs from rank %d "
                    "(%llu bytes/sig %llx vs %llu/%llx). Every rank must "
                    "allocate the same sizes in the same order.\n",
                    bs_.rank(), r, (unsigned long long)me.used,
                    (unsigned long long)me.sig, (unsigned long long)all[r].used,
                    (unsigned long long)all[r].sig);
            abort();
        }
    }

    const sym_view &view() const { return view_; }
    hipStream_t stream() const { return stream_; }
    int rank() const { return bs_.rank(); }
    int world() const { return bs_.world(); }
    int device() const { return dev_; }

    // Finish my own work, then wait for everyone else's. Both halves are needed:
    // the host barrier alone says nothing about kernels still in flight, and the
    // stream sync alone says nothing about the peer.
    void barrier() {
        HKD_CHECK(hipStreamSynchronize(stream_));
        bs_.barrier();
    }

private:
    size_t bytes_, used_ = 0;
    uint64_t sig_ = 1469598103934665603ull;   // FNV-1a offset basis
    int dev_;
    bootstrap &bs_;
    bool uncached_ = false;
    void *mine_ = nullptr;
    std::vector<void *> peers_;
    sym_view view_{};
    hipStream_t stream_{};
};

// ---------------------------------------------------------------------------
// Cross-rank flags, in host memory rather than in the symmetric heap.
//
// The obvious place for a barrier flag is the heap -- it is fine-grained, every
// rank can name it, and symmem.cuh's in-process signal/wait works that way. In
// one process it does work. Across processes over PCIe it does not, and the
// failure is not subtle: bar_probe measured a lockstep round trip through a
// peer-mapped device flag at *seconds*, with every stream topology (one kernel,
// two kernels, waiter on its own stream, event chain, host sync) and with the
// signal as both a release store and an exchange. The arrangements that looked
// fast were fast for a bad reason -- their signals ran ahead of their waits, so
// an early epoch's wait was satisfied by a later epoch's store and no round
// trip ever happened.
//
// A store from GPU A into GPU B's fine-grained memory does eventually arrive,
// but not on any timescale a kernel can spin for: the local L2 is neither
// written through nor bypassed for a peer-mapped line (the ISA for a
// system-scope acquire load is `global_load_b32 glc` + `buffer_gl1_inv` +
// `buffer_gl0_inv` -- L0 and L1, nothing below), so the value sits until
// something evicts it.
//
// Host memory has none of that. It is uncached by the GPU, both ranks reach the
// same physical page over PCIe, and no translate() is needed because there is
// one array, not one per rank. This is what RCCL does for P2P flags on a PCIe
// fabric, and for the same reason.
//
// The page is a POSIX shm segment named after the run id, mmapped by every rank
// and pinned with hipHostRegister. Only flags go here -- it is slow memory for
// bulk traffic, and bulk traffic is what the symmetric heap is for.
// ---------------------------------------------------------------------------
class host_flags {
public:
    host_flags(size_t words, bootstrap &bs) : bs_(bs) {
        const char *id = getenv("HK_DIST_RUN_ID");
        if (!id) { fprintf(stderr, "HK_DIST_RUN_ID is not set\n"); abort(); }
        name_ = std::string("/hk_dist_flags_") + id;
        const long pg = sysconf(_SC_PAGESIZE);
        bytes_ = ((words * sizeof(uint32_t) + pg - 1) / pg) * pg;

        // Rank 0 sizes it before anyone maps it: ftruncate after another rank
        // has mapped the (zero-length) segment leaves that rank with a mapping
        // that faults on first touch.
        if (bs_.rank() == 0) {
            fd_ = shm_open(name_.c_str(), O_CREAT | O_EXCL | O_RDWR, 0600);
            if (fd_ < 0) { perror("shm_open"); abort(); }
            if (ftruncate(fd_, bytes_) != 0) { perror("ftruncate"); abort(); }
        }
        bs_.barrier();
        if (bs_.rank() != 0) {
            fd_ = shm_open(name_.c_str(), O_RDWR, 0600);
            if (fd_ < 0) { perror("shm_open"); abort(); }
        }

        host_ = mmap(nullptr, bytes_, PROT_READ | PROT_WRITE, MAP_SHARED, fd_, 0);
        if (host_ == MAP_FAILED) { perror("mmap"); abort(); }
        HKD_CHECK(hipHostRegister(host_, bytes_,
                                  hipHostRegisterMapped | hipHostRegisterPortable));
        void *d = nullptr;
        HKD_CHECK(hipHostGetDevicePointer(&d, host_, 0));
        dev_ = reinterpret_cast<uint32_t *>(d);

        bs_.barrier();
        // Unlink once everyone holds a mapping: the segment stays alive as long
        // as it is mapped, and a crash then cannot leave a stale one behind for
        // the next run to open and mistake for its own.
        if (bs_.rank() == 0) (void)shm_unlink(name_.c_str());
    }

    ~host_flags() {
        if (host_ && host_ != MAP_FAILED) {
            (void)hipHostUnregister(host_);
            (void)munmap(host_, bytes_);
        }
        if (fd_ >= 0) (void)close(fd_);
    }

    // Device-side pointer, valid on this rank's device. Every rank's pointer
    // names the same physical page, so flags[r] means the same word everywhere.
    uint32_t *device() const { return dev_; }
    uint32_t *host() const { return reinterpret_cast<uint32_t *>(host_); }

private:
    bootstrap &bs_;
    std::string name_;
    size_t bytes_ = 0;
    int fd_ = -1;
    void *host_ = nullptr;
    uint32_t *dev_ = nullptr;
};

}  // namespace hk_dist
