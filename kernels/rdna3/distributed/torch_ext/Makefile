# A shared object registering torch.ops.hk_dist.*, built with plain hipcc.
#
# Not torch.utils.cpp_extension.load: on ROCm that runs hipify over the source
# first, and this source is already HIP -- hipify would rewrite identifiers that
# are correct, and the diagnostics from the result point at a generated file. A
# .so plus torch.ops.load_library gives the same torch.ops.* surface, CUDA-graph
# capture included, with a build we can read.
#
# The ABI flag is load-bearing. libtorch is built one way or the other and a
# mismatch links cleanly, then crashes on the first std::string crossing the
# boundary. Ask torch rather than assuming.

TORCH_DIR  := $(shell python3 -c "import torch,os;print(os.path.dirname(torch.__file__))")
TORCH_ABI  := $(shell python3 -c "import torch;print(int(torch._C._GLIBCXX_USE_CXX11_ABI))")
ROCM_PATH  ?= /opt/rocm
HIPCXX     ?= $(ROCM_PATH)/bin/hipcc
HK_INCLUDE ?= $(abspath ../../../../include)

CXXFLAGS := -DKITTENS_RDNA3 --offload-arch=gfx1100 -std=c++20 -O3 -w -fPIC \
            -D_GLIBCXX_USE_CXX11_ABI=$(TORCH_ABI) \
            -I$(HK_INCLUDE) -I$(ROCM_PATH)/include -I$(ROCM_PATH)/include/hip \
            -I$(TORCH_DIR)/include -I$(TORCH_DIR)/include/torch/csrc/api/include

LDFLAGS  := -shared -L$(TORCH_DIR)/lib -Wl,-rpath,$(TORCH_DIR)/lib \
            -ltorch -ltorch_hip -lc10 -lc10_hip -lamdhip64

SRC  := hk_dist_ext.hip
DEPS := $(SRC) ../ipc_heap.cuh ../symmem.cuh ../fused.cuh \
        ../../gemm/bf16fp32/gemm.cpp

.PHONY: all clean resources
all: hk_dist_ext.so

hk_dist_ext.so: $(DEPS)
	$(HIPCXX) $(CXXFLAGS) $(LDFLAGS) $(SRC) -o $@

# The check the epilogue notes in gemm.cpp demand after any change to it: this
# GEMM hand-manages s_waitcnt, so a compiler-inserted spill silently breaks the
# waits and corrupts exactly the last accumulator tile. ScratchSize must be 0.
resources:
	$(HIPCXX) $(CXXFLAGS) -Rpass-analysis=kernel-resource-usage -c $(SRC) \
	    -o /dev/null 2>&1 | grep -E 'Function Name|VGPRs|Scratch|Spill'

clean:
	rm -f hk_dist_ext.so
