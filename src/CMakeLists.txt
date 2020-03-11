set(fwgpu_src
  cublas_sgemm.cu
  cutlass_sgemm.cu
)

cuda_add_library(${FWGPU_LIB_NAME} ${FWGPU_LIB_TYPE} ${fwgpu_src})

target_link_libraries(
  ${FWGPU_LIB_NAME}
  ${CUDA_CUBLAS_LIBRARIES}
  ${CMAKE_THREAD_LIBS_INIT}
)

include_directories(
  ${PROJECT_SOURCE_DIR}
  "${PROJECT_SOURCE_DIR}/cutlass/include"
  ${CUDA_INCLUDE_DIRS})

install(TARGETS ${FWGPU_LIB_NAME} LIBRARY DESTINATION lib ARCHIVE DESTINATION lib)
install(DIRECTORY ${FWGPU_LIB_NAME}/include DESTINATION include)
