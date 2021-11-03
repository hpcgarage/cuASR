# SM50 defualt configurations
add_executable(cuasr_bench_srgemm_device_sm50_defaults
  sm50_defaults.cu
)
target_include_directories(
  cuasr_bench_srgemm_device_sm50_defaults
  PRIVATE
  ${PROJECT_SOURCE_DIR}/include/
  ${PROJECT_SOURCE_DIR}/tools/include/
  ${PROJECT_SOURCE_DIR}/cutlass/include/
  ${PROJECT_SOURCE_DIR}/cutlass/tools/util/include/
)
target_link_libraries(cuasr_bench_srgemm_device_sm50_defaults
  benchmark
  benchmark_main
  ${cuASR_LIB_NAME}
)

# SM80 defualt configurations
add_executable(cuasr_bench_srgemm_device_sm80_defaults
  sm80_defaults.cu
)
target_include_directories(
  cuasr_bench_srgemm_device_sm80_defaults
  PRIVATE
  ${PROJECT_SOURCE_DIR}/include/
  ${PROJECT_SOURCE_DIR}/tools/include/
  ${PROJECT_SOURCE_DIR}/cutlass/include/
  ${PROJECT_SOURCE_DIR}/cutlass/tools/util/include/
)
target_link_libraries(cuasr_bench_srgemm_device_sm80_defaults
  benchmark
  benchmark_main
  ${cuASR_LIB_NAME}
)

# All shmoo benchmarks
file(GLOB SIMT_BENCH_SRCS CONFIGURE_DEPENDS sm50_simt_*.cu)
add_executable(cuasr_bench_srgemm_device_shmoo
  ${SIMT_BENCH_SRCS}
)
target_include_directories(
  cuasr_bench_srgemm_device_shmoo
  PRIVATE
  ${PROJECT_SOURCE_DIR}/include/
  ${PROJECT_SOURCE_DIR}/tools/include/
  ${PROJECT_SOURCE_DIR}/cutlass/include/
  ${PROJECT_SOURCE_DIR}/cutlass/tools/util/include/
)
target_link_libraries(cuasr_bench_srgemm_device_shmoo
  benchmark
  benchmark_main
  ${cuASR_LIB_NAME}
)
if(NOT DEFINED CUASR_BENCH_LEVEL)
  set(CUASR_BENCH_LEVEL 0)
endif()
target_compile_definitions(cuasr_bench_srgemm_device_shmoo
  PRIVATE CUASR_BENCH_LEVEL=${CUASR_BENCH_LEVEL}
)
