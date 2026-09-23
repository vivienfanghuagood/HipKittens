# Two standalone binaries, no Python. `make` builds both.
#
# CUSTOM_RULES=1 because common.mk's standalone mode builds one target from one
# source, and these are two independent programs that share only a header.
GPU_TARGET ?= RDNA3
BUILD_MODE := standalone
CUSTOM_RULES := 1

include ../../common.mk

# gemm_rs includes gemm.cpp for the tuned inner loop. HK_GEMM_NO_PYBIND (set in
# the source) drops the pybind module *and* the pyutils include, so nothing here
# needs Python headers.
TARGETS := symmem_test gemm_rs ipc_heap_test gemm_rs_mp gemm_ar_mp bar_probe

.PHONY: all clean
all: $(TARGETS)

symmem_test: symmem_test.hip symmem.cuh
	$(HIPCXX) $(HIPFLAGS) $(ICPPFLAGS) $< -o $@

gemm_rs: gemm_rs.hip symmem.cuh ../gemm/bf16fp32/gemm.cpp
	$(HIPCXX) $(HIPFLAGS) $(ICPPFLAGS) $< -o $@

# The multi-process half. Same device-side code, one process per rank, peer
# memory reached through IPC instead of hipDeviceEnablePeerAccess -- which is
# the only process model a framework can offer. Launch with ./run_mp.sh.
ipc_heap_test: ipc_heap_test.hip ipc_heap.cuh symmem.cuh
	$(HIPCXX) $(HIPFLAGS) $(ICPPFLAGS) $< -o $@

# gemm_rs under that process model: the same device code and the same shapes as
# gemm_rs, one process per rank. Launch with ./run_mp.sh.
gemm_rs_mp: gemm_rs_mp.hip ipc_heap.cuh symmem.cuh ../gemm/bf16fp32/gemm.cpp
	$(HIPCXX) $(HIPFLAGS) $(ICPPFLAGS) $< -o $@

# The all-reduce form, sharded along N instead of M. The only one of the two
# decode can use: M=1 cannot be row-sharded. Launch with ./run_mp.sh.
gemm_ar_mp: gemm_ar_mp.hip ipc_heap.cuh symmem.cuh ../gemm/bf16fp32/gemm.cpp
	$(HIPCXX) $(HIPFLAGS) $(ICPPFLAGS) $< -o $@

# Isolates why a cross-rank device barrier costs seconds over IPC. Seconds to
# build and to run; keep it, the answer is not obvious enough to rediscover.
bar_probe: bar_probe.hip ipc_heap.cuh symmem.cuh
	$(HIPCXX) $(HIPFLAGS) $(ICPPFLAGS) $< -o $@

clean:
	rm -f $(TARGETS)
