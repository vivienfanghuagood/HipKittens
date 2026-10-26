# rdna4 (gfx1200/gfx1201, RX 9070) unit tests.
#
# Green on a gfx1201 (Radeon AI PRO R9700), section by section -- see README.md.
#
# Shared rules live in ../common/tests.mk; this file only sets local knobs.

GPU_TARGET     ?= RDNA4
COMP_LEVEL     ?= profile
TEST_INTENSITY ?= 2

# Which tests to run? -DTEST_ALL runs everything; or pick subsections, e.g.
# -DTEST_WARP_MEMORY, or individual tests like -DTEST_WARP_MEMORY_TILE_SHARED_TO_REGISTER.
TEST_DEFINES   ?= -DTEST_ALL

include ../common/tests.mk
