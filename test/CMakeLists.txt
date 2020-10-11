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
add_subdirectory(regress)
add_subdirectory(device)
