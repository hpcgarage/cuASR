#include <cublas_v2.h>

#include "include/Matrix.hpp"

namespace fwgpu {

void cublas_sgemm(
    const float *A, const float *B, float *C, const int m, const int k, const int n) {
  const int lda         = m;
  const int ldb         = k;
  const int ldc         = m;
  const float tmp_alpha = 1;
  const float tmp_beta  = 0;
  const float *alpha    = &tmp_alpha;
  const float *beta     = &tmp_beta;

  // context handle create
  cublasHandle_t handle;
  cublasCreate(&handle);

  cublasSgemm(
      handle, CUBLAS_OP_N, CUBLAS_OP_N, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc);

  // tear down context before exiting method
  cublasDestroy(handle);
}

} // namespace fwgpu
