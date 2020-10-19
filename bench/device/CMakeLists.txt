file(GLOB SIMT_BENCH_SRCS CONFIGURE_DEPENDS *.cu)
add_executable(cuasr_bench_srgemm_device
  ${SIMT_BENCH_SRCS}
)
target_include_directories(
  cuasr_bench_srgemm_device
  PRIVATE
  ${PROJECT_SOURCE_DIR}/include/
  ${PROJECT_SOURCE_DIR}/tools/include/
  ${PROJECT_SOURCE_DIR}/cutlass/include/
  ${PROJECT_SOURCE_DIR}/cutlass/tools/util/include/
)
target_link_libraries(cuasr_bench_srgemm_device
  benchmark
  benchmark_main
  ${cuASR_LIB_NAME}
)
if(NOT DEFINED CUASR_BENCH_LEVEL)
  set(CUASR_BENCH_LEVEL 0)
endif()
target_compile_definitions(cuasr_bench_srgemm_device
  PRIVATE CUASR_BENCH_LEVEL=${CUASR_BENCH_LEVEL}
)
