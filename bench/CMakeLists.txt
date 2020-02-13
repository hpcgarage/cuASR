set(BENCHMARK_ENABLE_TESTING OFF)

add_subdirectory(benchmark)
include_directories(${PROJECT_SOURCE_DIR} benchmark/include)

# CPU benchmarks
set(cpu_bench_src
  cpu_bench.cpp
)
add_executable(cpu_bench ${cpu_bench_src})
target_link_libraries(cpu_bench
  benchmark
  benchmark_main
  ${FWGPU_LIB_NAME}
  ${CMAKE_THREAD_LIBS_INIT}
)

# GPU Benchmarks
set(gpu_bench_src
  gpu_bench.cpp
)
add_executable(gpu_bench ${gpu_bench_src})
target_link_libraries(gpu_bench
  benchmark
  benchmark_main
  ${FWGPU_LIB_NAME}
  ${CMAKE_THREAD_LIBS_INIT}
)
