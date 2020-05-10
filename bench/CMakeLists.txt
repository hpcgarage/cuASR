set(BENCHMARK_ENABLE_TESTING OFF)
add_subdirectory(benchmark)

# CPU benchmarks
add_executable(cpu_bench cpu_bench.cpp)
target_include_directories(cpu_bench PRIVATE benchmark/include)
target_link_libraries(cpu_bench
  benchmark
  benchmark_main
  ${FWGPU_LIB_NAME}
  ${CMAKE_THREAD_LIBS_INIT}
)

# GPU Benchmarks
cuda_add_executable(gpu_bench gpu_bench.cu)
target_include_directories(gpu_bench PRIVATE benchmark/include)
target_link_libraries(gpu_bench
  benchmark
  benchmark_main
  ${FWGPU_LIB_NAME}
  ${CMAKE_THREAD_LIBS_INIT}
)
