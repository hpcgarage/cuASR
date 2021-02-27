add_executable(splitk_srgemm
  splitk_srgemm.cu
)
target_include_directories(splitk_srgemm
  PRIVATE
  ${CUDA_INCLUDE_DIRS}
  ${PROJECT_SOURCE_DIR}/include
  ${PROJECT_SOURCE_DIR}/tools/include/
  ${PROJECT_SOURCE_DIR}/cutlass/include
  ${PROJECT_SOURCE_DIR}/cutlass/tools/util/include/
)
target_compile_options(splitk_srgemm
  PUBLIC
  # C++ compiler flags
  $<$<AND:$<COMPILE_LANGUAGE:CXX>,$<CONFIG:${uppercase_CMAKE_BUILD_TYPE}>>:
    ${cuASR_CXX_FLAGS_${uppercase_CMAKE_BUILD_TYPE}}>

  # CUDA compiler flags
  $<$<AND:$<COMPILE_LANGUAGE:CUDA>,$<CONFIG:${uppercase_CMAKE_BUILD_TYPE}>>:
    ${cuASR_CUDA_FLAGS_${uppercase_CMAKE_BUILD_TYPE}}>
)
