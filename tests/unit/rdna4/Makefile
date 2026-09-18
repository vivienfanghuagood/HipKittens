# rdna4 (gfx1200/gfx1201, RX 9070) unit tests.
#
# !! These have never been run.  No gfx12 part was available while
# include/rdna4 was written, so the tree is compile-verified only -- see
# README.md.  `make` is the gate; `./unit_tests` is the thing to try first when
# hardware turns up.
#
# Shared rules live in ../common/tests.mk; this file only sets local knobs.

GPU_TARGET     ?= RDNA4
COMP_LEVEL     ?= profile
TEST_INTENSITY ?= 2

# Which tests to run? -DTEST_ALL runs everything; or pick subsections, e.g.
# -DTEST_WARP_MEMORY, or individual tests like -DTEST_WARP_MEMORY_TILE_SHARED_TO_REGISTER.
TEST_DEFINES   ?= -DTEST_ALL

include ../common/tests.mk
