# See LICENSE.txt for license details.
PCC = g++-5.4
CXX_FLAGS += -std=c++11 -O3 -Wall -g
PAR_FLAG = -fopenmp

ifneq (,$(findstring icpc,$(CXX)))
	PAR_FLAG = -openmp
endif

ifneq (,$(findstring sunCC,$(CXX)))
	CXX_FLAGS = -std=c++11 -xO3 -m64 -xtarget=native
	PAR_FLAG = -xopenmp
endif

ifeq ($(DEBUG), 1)
	CXX_FLAGS = -std=c++11 -O0 -Wall -g
endif

ifneq ($(SERIAL), 1)
	CXX_FLAGS += $(PAR_FLAG)
endif 

ifeq ($(CILK), 1)
    CXX_FLAGS -= $(PAR_FLAG)
	CXX_FLAGS += -fcilkplus -lcilkrts -O2 -DCILK  -DBYTERLE
endif

KERNELS = bc bfs cc cc_sv pr sssp tc
SUITE = $(KERNELS) converter

.PHONY: all
all: $(SUITE)

% : src/%.cc src/*.h
	$(PCC) $(CXX_FLAGS) $< -o $@

# Testing
include test/test.mk

# Benchmark Automation
include benchmark/bench.mk


.PHONY: clean
clean:
	rm -f $(SUITE) test/out/*
