# SM50 default tests
add_executable(cuasr_test_srgemm_device_sm50_defaults
  ${PROJECT_SOURCE_DIR}/test/harness.cpp
  sm50_defaults.cu
)
target_include_directories(
  cuasr_test_srgemm_device_sm50_defaults
  PRIVATE
  ${PROJECT_SOURCE_DIR}/include/
  ${PROJECT_SOURCE_DIR}/tools/include/
  ${PROJECT_SOURCE_DIR}/cutlass/include/
  ${PROJECT_SOURCE_DIR}/cutlass/tools/util/include/
)
target_link_libraries(cuasr_test_srgemm_device_sm50_defaults
  gtest
  ${cuASR_LIB_NAME}
)
add_test(
  NAME cuasr_test_srgemm_device_sm50_defaults
  COMMAND cuasr_test_srgemm_device_sm50_defaults
)

# SM50 all shmoo tests
file(GLOB SM50_SIMT_TEST_SRCS CONFIGURE_DEPENDS sm50_simt_*.cu)
add_executable(cuasr_test_srgemm_device_sm50_shmoo
  ${PROJECT_SOURCE_DIR}/test/harness.cpp
  ${SM50_SIMT_TEST_SRCS}
)
target_include_directories(
  cuasr_test_srgemm_device_sm50_shmoo
  PRIVATE
  ${PROJECT_SOURCE_DIR}/include/
  ${PROJECT_SOURCE_DIR}/tools/include/
  ${PROJECT_SOURCE_DIR}/cutlass/include/
  ${PROJECT_SOURCE_DIR}/cutlass/tools/util/include/
)
target_link_libraries(cuasr_test_srgemm_device_sm50_shmoo
  gtest
  ${cuASR_LIB_NAME}
)
add_test(
  NAME cuasr_test_srgemm_device_sm50_shmoo
  COMMAND cuasr_test_srgemm_device_sm50_shmoo
)
if(NOT DEFINED CUASR_TEST_LEVEL)
  set(CUASR_TEST_LEVEL 0)
endif()
target_compile_definitions(cuasr_test_srgemm_device_sm50_shmoo
  PRIVATE CUASR_TEST_LEVEL=${CUASR_TEST_LEVEL}
)

# SM80 default tests
add_executable(cuasr_test_srgemm_device_sm80_defaults
  ${PROJECT_SOURCE_DIR}/test/harness.cpp
  sm80_defaults.cu
)
target_include_directories(
  cuasr_test_srgemm_device_sm80_defaults
  PRIVATE
  ${PROJECT_SOURCE_DIR}/include/
  ${PROJECT_SOURCE_DIR}/tools/include/
  ${PROJECT_SOURCE_DIR}/cutlass/include/
  ${PROJECT_SOURCE_DIR}/cutlass/tools/util/include/
)
target_link_libraries(cuasr_test_srgemm_device_sm80_defaults
  gtest
  ${cuASR_LIB_NAME}
)
add_test(
  NAME cuasr_test_srgemm_device_sm80_defaults
  COMMAND cuasr_test_srgemm_device_sm80_defaults
)
