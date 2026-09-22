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
TARGETS := symmem_test gemm_rs

.PHONY: all clean
all: $(TARGETS)

symmem_test: symmem_test.hip symmem.cuh
	$(HIPCXX) $(HIPFLAGS) $(ICPPFLAGS) $< -o $@

gemm_rs: gemm_rs.hip symmem.cuh ../gemm/bf16fp32/gemm.cpp
	$(HIPCXX) $(HIPFLAGS) $(ICPPFLAGS) $< -o $@

clean:
	rm -f $(TARGETS)
