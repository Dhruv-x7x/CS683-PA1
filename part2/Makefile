CXX = g++
CXXFLAGS = -O0 -mavx2 -msse -mavx512f -Wall -g
# -mavx2 -msse -mavx512f use these commands for simd
TARGET = emb
SRCS = emb.cpp
OBJS = $(SRCS:.cpp=.o)

$(TARGET): $(OBJS)
	$(CXX) $(CXXFLAGS) -o $(TARGET) $(OBJS)

%.o: %.cpp
	$(CXX) $(CXXFLAGS) -c $< -o $@

perf-naive: $(TARGET)
	perf stat -e cache-misses,L1-dcache-load-misses,LLC-load-misses,cycles,instructions,l2_rqsts.demand_data_rd_miss,sw_prefetch_access.any,sw_prefetch_access.t0,sw_prefetch_access.t1_t2,sw_prefetch_access.nta,L1-dcache-load ./$(TARGET) naive

perf-prefetch: $(TARGET)
	perf stat -e cache-misses,L1-dcache-load-misses,LLC-load-misses,cycles,instructions,l2_rqsts.demand_data_rd_miss,sw_prefetch_access.any,sw_prefetch_access.t0,sw_prefetch_access.t1_t2,sw_prefetch_access.nta,L1-dcache-load ./$(TARGET) prefetch

perf-simd: $(TARGET)
	perf stat -e L1-dcache-load-misses,LLC-load-misses,cycles,instructions,l2_rqsts.demand_data_rd_miss,sw_prefetch_access.any,L1-dcache-load,fp_arith_inst_retired.scalar_single,fp_arith_inst_retired.128b_packed_single,fp_arith_inst_retired.256b_packed_single,fp_arith_inst_retired.512b_packed_single ./$(TARGET) simd

perf-prefetch-simd: $(TARGET)
	perf stat -e L1-dcache-load-misses,LLC-load-misses,cycles,instructions,l2_rqsts.demand_data_rd_miss,sw_prefetch_access.any,L1-dcache-load ./$(TARGET) prefetch_simd

clean:
	rm -f $(OBJS) $(TARGET)
