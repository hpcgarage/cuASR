# first make sure we have gtest checked-out and its up-to-date
if(GIT_FOUND AND EXISTS "${PROJECT_SOURCE_DIR}/.git")
  message(STATUS "Checking submodule version for google/googletest")
  execute_process(
    COMMAND ${GIT_EXECUTABLE} submodule update --init ${PROJECT_SOURCE_DIR}/test/gtest
    WORKING_DIRECTORY ${PROJECT_SOURCE_DIR}
    OUTPUT_VARIABLE GIT_SUBMOD_STDOUT OUTPUT_STRIP_TRAILING_WHITESPACE
    ERROR_VARIABLE GIT_SUBMOD_STDERR ERROR_STRIP_TRAILING_WHITESPACE
    RESULT_VARIABLE GIT_SUBMOD_RESULT
  )
  if(NOT GIT_SUBMOD_RESULT EQUAL "0")
    message(FATAL_ERROR "git submodule update --init failed with ${GIT_SUBMOD_RESULT}, please checkout gtest manually. Git stdout was ${GIT_SUBMOD_STDOUT}. Git stderr was ${GIT_SUBMOD_STDERR}.")
  elseif(NOT ${GIT_SUBMOD_STDOUT} STREQUAL "")
    message(STATUS ${GIT_SUBMOD_STDOUT})
  endif()
endif()

if(NOT EXISTS "${PROJECT_SOURCE_DIR}/test/gtest/googletest/include")
  message(FATAL_ERROR "GTest submodule is not present and automatic checkout failed, please checkout gtest manually.")
endif()

add_subdirectory(gtest)

### Matrix tests
cuda_add_executable(Matrix_tests harness.cpp Matrix_test.cpp)
target_include_directories(Matrix_tests PRIVATE gtest/googletest/include)
target_link_libraries(Matrix_tests gtest ${cuASR_LIB_NAME})
add_test(
  NAME Matrix_tests
  COMMAND Matrix_tests
)

### GEMM tests
cuda_add_executable(Gemm_tests harness.cpp Gemm_test.cu)
target_include_directories(Gemm_tests PRIVATE gtest/googletest/include)
target_link_libraries(Gemm_tests gtest ${cuASR_LIB_NAME})
add_test(
  NAME Gemm_tests
  COMMAND Gemm_tests
)

### SemiRing GEMM tests
cuda_add_executable(Srgemm_tests harness.cpp Srgemm_test.cu)
target_include_directories(Srgemm_tests PRIVATE gtest/googletest/include)
target_link_libraries(Srgemm_tests gtest ${cuASR_LIB_NAME})
add_test(
  NAME Srgemm_tests
  COMMAND Srgemm_tests
)
