#include <cublas_v2.h>

#include "include/Matrix.hpp"
#include "include/gpu_gemm.cuh"
#include "include/gpu_srgemm.cuh"
#include "include/utils.hpp"

namespace fwgpu {
namespace testing {
  auto gpu_sgemm_naive_entry(const Matrix<float> &A, const Matrix<float> &B)
      -> Matrix<float> {
    const auto m         = A.num_rows();
    const auto k         = A.num_cols(); // B.num_rows();
    const auto n         = B.num_cols();
    auto output_bytesize = m * n * sizeof(float);
    float *c_bytes       = new float[m * n];

    // allocate for inputs and outputs on device
    float *d_A, *d_B, *d_C;
    fwgpu::malloc_device((void**)(&d_A), A.bytesize());
    fwgpu::malloc_device((void**)(&d_B), B.bytesize());
    fwgpu::malloc_device((void**)(&d_C), output_bytesize);

    // copy inputs to device
    fwgpu::memcpy_h2d(d_A, A.get_buf(), A.bytesize());
    fwgpu::memcpy_h2d(d_B, B.get_buf(), B.bytesize());

    dim3 threads(16, 16);
    dim3 blocks((m - 1) / 16 + 1, (n - 1) / 16 + 1);
    fwgpu::gpu_gemm_naive<float><<<blocks, threads>>>(m, n, k, d_A, d_B, d_C);

    // copy output to host
    fwgpu::memcpy_d2h(c_bytes, d_C, output_bytesize);

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    cudaDeviceSynchronize();
    return Matrix<float>(m, n, c_bytes);
  }

  auto gpu_srgemm_naive_entry(const Matrix<float> &A, const Matrix<float> &B)
      -> Matrix<float> {
    const auto m         = A.num_rows();
    const auto k         = A.num_cols(); // B.num_rows();
    const auto n         = B.num_cols();
    auto output_bytesize = m * n * sizeof(float);
    float *c_bytes       = new float[m * n];

    // allocate for inputs and outputs on device
    float *d_A, *d_B, *d_C;
    fwgpu::malloc_device((void**)(&d_A), A.bytesize());
    fwgpu::malloc_device((void**)(&d_B), B.bytesize());
    fwgpu::malloc_device((void**)(&d_C), output_bytesize);

    // copy inputs to device
    fwgpu::memcpy_h2d(d_A, A.get_buf(), A.bytesize());
    fwgpu::memcpy_h2d(d_B, B.get_buf(), B.bytesize());

    dim3 threads(16, 16);
    dim3 blocks((m - 1) / 16 + 1, (n - 1) / 16 + 1);
    fwgpu::gpu_srgemm_naive<float><<<blocks, threads>>>(m, n, k, d_A, d_B, d_C);

    // copy output to host
    fwgpu::memcpy_d2h(c_bytes, d_C, output_bytesize);

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    cudaDeviceSynchronize();
    return Matrix<float>(m, n, c_bytes);
  }

  auto cublas_sgemm_entry(const Matrix<float> &A, const Matrix<float> &B)
      -> Matrix<float> {
    const auto m         = A.num_rows();
    const auto k         = A.num_cols(); // B.num_rows();
    const auto n         = B.num_cols();
    auto output_bytesize = m * n * sizeof(float);
    float *c_bytes       = new float[m * n];

    // allocate for inputs and outputs on device
    float *d_A, *d_B, *d_C;
    fwgpu::malloc_device((void**)(&d_A), A.bytesize());
    fwgpu::malloc_device((void**)(&d_B), B.bytesize());
    fwgpu::malloc_device((void**)(&d_C), output_bytesize);

    // copy inputs to device
    fwgpu::memcpy_h2d(d_A, A.get_buf(), A.bytesize());
    fwgpu::memcpy_h2d(d_B, B.get_buf(), B.bytesize());

    fwgpu::cublas_sgemm(d_A, d_B, d_C, m, k, n);

    // copy output to host
    fwgpu::memcpy_d2h(c_bytes, d_C, output_bytesize);

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    return Matrix<float>(m, n, c_bytes);
  }

  auto cutlass_sgemm_entry(const Matrix<float> &A, const Matrix<float> &B)
      -> Matrix<float> {
    const auto m = A.num_rows();
    const auto k = A.num_cols(); // B.num_rows();
    const auto n = B.num_cols();

    int lda              = m;
    int ldb              = k;
    int ldc              = m;
    float alpha          = 1.0;
    float beta           = 0.0;
    auto output_bytesize = m * n * sizeof(float);
    float *c_bytes       = new float[m * n];

    // allocate for inputs and outputs on device
    float *d_A, *d_B, *d_C;
    fwgpu::malloc_device((void**)(&d_A), A.bytesize());
    fwgpu::malloc_device((void**)(&d_B), B.bytesize());
    fwgpu::malloc_device((void**)(&d_C), output_bytesize);

    // copy inputs to device
    fwgpu::memcpy_h2d(d_A, A.get_buf(), A.bytesize());
    fwgpu::memcpy_h2d(d_B, B.get_buf(), B.bytesize());

    fwgpu::cutlass_sgemm_nn(m, n, k, alpha, d_A, lda, d_B, ldb, beta, d_C, ldc);

    // copy output to host
    fwgpu::memcpy_d2h(c_bytes, d_C, output_bytesize);

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    return Matrix<float>(m, n, c_bytes);
  }

  auto cutlass_srsgemm_entry(const Matrix<float> &A, const Matrix<float> &B)
      -> Matrix<float> {
    const auto m = A.num_rows();
    const auto k = A.num_cols(); // B.num_rows();
    const auto n = B.num_cols();

    int lda              = m;
    int ldb              = k;
    int ldc              = m;
    auto output_bytesize = m * n * sizeof(float);
    float *c_bytes       = new float[m * n];

    // allocate for inputs and outputs on device
    float *d_A, *d_B, *d_C;
    fwgpu::malloc_device((void**)(&d_A), A.bytesize());
    fwgpu::malloc_device((void**)(&d_B), B.bytesize());
    fwgpu::malloc_device((void**)(&d_C), output_bytesize);

    // copy inputs to device
    fwgpu::memcpy_h2d(d_A, A.get_buf(), A.bytesize());
    fwgpu::memcpy_h2d(d_B, B.get_buf(), B.bytesize());

    fwgpu::cutlass_srsgemm_nn(m, n, k, d_A, lda, d_B, ldb, d_C, ldc);

    // copy output to host
    fwgpu::memcpy_d2h(c_bytes, d_C, output_bytesize);

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    return Matrix<float>(m, n, c_bytes);
  }
} // namespace testing
} // namespace fwgpu
