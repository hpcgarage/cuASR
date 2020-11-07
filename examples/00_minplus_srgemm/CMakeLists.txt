add_executable(minplus_srgemm
  minplus_srgemm.cu
)
target_include_directories(minplus_srgemm
  PRIVATE
  ${PROJECT_SOURCE_DIR}/include
  ${PROJECT_SOURCE_DIR}/cutlass/include
  ${CUDA_INCLUDE_DIRS}
)
target_compile_options(minplus_srgemm
  PUBLIC
  # C++ compiler flags
  $<$<AND:$<COMPILE_LANGUAGE:CXX>,$<CONFIG:${uppercase_CMAKE_BUILD_TYPE}>>:
    ${cuASR_CXX_FLAGS_${uppercase_CMAKE_BUILD_TYPE}}>

  # CUDA compiler flags
  $<$<AND:$<COMPILE_LANGUAGE:CUDA>,$<CONFIG:${uppercase_CMAKE_BUILD_TYPE}>>:
    ${cuASR_CUDA_FLAGS_${uppercase_CMAKE_BUILD_TYPE}}>
)
