file(GLOB SIMT_TEST_SRCS CONFIGURE_DEPENDS *.cu)
add_executable(cuasr_test_srgemm_device
  ${PROJECT_SOURCE_DIR}/test/harness.cpp
  ${SIMT_TEST_SRCS}
)
target_include_directories(
  cuasr_test_srgemm_device
  PRIVATE
  ${PROJECT_SOURCE_DIR}/include/
  ${PROJECT_SOURCE_DIR}/tools/include/
  ${PROJECT_SOURCE_DIR}/cutlass/include/
  ${PROJECT_SOURCE_DIR}/cutlass/tools/util/include/
)
target_link_libraries(cuasr_test_srgemm_device
  gtest
  ${cuASR_LIB_NAME}
)
add_test(
  NAME cuasr_test_srgemm_device
  COMMAND cuasr_test_srgemm_device
)
if(NOT DEFINED CUASR_TEST_LEVEL)
  set(CUASR_TEST_LEVEL 0)
endif()
target_compile_definitions(cuasr_test_srgemm_device
  PRIVATE CUASR_TEST_LEVEL=${CUASR_TEST_LEVEL}
)
