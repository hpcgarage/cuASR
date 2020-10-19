# first make sure we have benchmark checked-out and its up-to-date
if(GIT_FOUND AND EXISTS "${PROJECT_SOURCE_DIR}/.git")
  message(STATUS "Checking submodule version for google/benchmark")
  execute_process(
    COMMAND ${GIT_EXECUTABLE} submodule update --init ${PROJECT_SOURCE_DIR}/bench/benchmark
    WORKING_DIRECTORY ${PROJECT_SOURCE_DIR}
    OUTPUT_VARIABLE GIT_SUBMOD_STDOUT OUTPUT_STRIP_TRAILING_WHITESPACE
    ERROR_VARIABLE GIT_SUBMOD_STDERR ERROR_STRIP_TRAILING_WHITESPACE
    RESULT_VARIABLE GIT_SUBMOD_RESULT
  )
  if(NOT GIT_SUBMOD_RESULT EQUAL "0")
    message(FATAL_ERROR "git submodule update --init failed with ${GIT_SUBMOD_RESULT}, please checkout benchmark manually. Git stdout was ${GIT_SUBMOD_STDOUT}. Git stderr was ${GIT_SUBMOD_STDERR}.")
  elseif(NOT ${GIT_SUBMOD_STDOUT} STREQUAL "")
    message(STATUS ${GIT_SUBMOD_STDOUT})
  endif()
endif()

if(NOT EXISTS "${PROJECT_SOURCE_DIR}/bench/benchmark/include")
  message(FATAL_ERROR "GTest submodule is not present and automatic checkout failed, please checkout benchmark manually.")
endif()

set(BENCHMARK_ENABLE_TESTING OFF)
add_subdirectory(benchmark)
add_subdirectory(device)
