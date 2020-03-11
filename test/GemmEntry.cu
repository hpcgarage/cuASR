#include <cublas_v2.h>

#include "include/Matrix.hpp"
#include "include/gpu_gemm.cuh"
#include "include/gpu_srgemm.cuh"

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
    cudaMalloc(&d_A, A.bytesize());
    cudaMalloc(&d_B, B.bytesize());
    cudaMalloc(&d_C, output_bytesize);

    // copy inputs to device
    cudaMemcpy(d_A, A.get_buf(), A.bytesize(), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, B.get_buf(), B.bytesize(), cudaMemcpyHostToDevice);

    dim3 threads(16, 16);
    dim3 blocks((m - 1) / 16 + 1, (n - 1) / 16 + 1);
    fwgpu::gpu_gemm_naive<float><<<blocks, threads>>>(m, n, k, d_A, d_B, d_C);

    // copy output to host
    cudaMemcpy(c_bytes, d_C, output_bytesize, cudaMemcpyDeviceToHost);
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
    cudaMalloc(&d_A, A.bytesize());
    cudaMalloc(&d_B, B.bytesize());
    cudaMalloc(&d_C, output_bytesize);

    // copy inputs to device
    cudaMemcpy(d_A, A.get_buf(), A.bytesize(), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, B.get_buf(), B.bytesize(), cudaMemcpyHostToDevice);

    dim3 threads(16, 16);
    dim3 blocks((m - 1) / 16 + 1, (n - 1) / 16 + 1);
    fwgpu::gpu_srgemm_naive<float><<<blocks, threads>>>(m, n, k, d_A, d_B, d_C);

    // copy output to host
    cudaMemcpy(c_bytes, d_C, output_bytesize, cudaMemcpyDeviceToHost);
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
    cudaMalloc(&d_A, A.bytesize());
    cudaMalloc(&d_B, B.bytesize());
    cudaMalloc(&d_C, output_bytesize);

    // copy inputs to device
    cudaMemcpy(d_A, A.get_buf(), A.bytesize(), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, B.get_buf(), B.bytesize(), cudaMemcpyHostToDevice);

    fwgpu::cublas_sgemm(d_A, d_B, d_C, m, k, n);

    // copy output to host
    cudaMemcpy(c_bytes, d_C, output_bytesize, cudaMemcpyDeviceToHost);

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    return Matrix<float>(m, n, c_bytes);
  }

  auto cutlass_sgemm_entry(const Matrix<float> &A, const Matrix<float> &B)
      -> Matrix<float> {
    const auto m         = A.num_rows();
    const auto k         = A.num_cols(); // B.num_rows();
    const auto n         = B.num_cols();

    int lda = m;
    int ldb = k;
    int ldc = m;
    float alpha = 1.0;
    float beta  = 0.0;
    auto output_bytesize = m * n * sizeof(float);
    float *c_bytes       = new float[m * n];

    // allocate for inputs and outputs on device
    float *d_A, *d_B, *d_C;
    cudaMalloc(&d_A, A.bytesize());
    cudaMalloc(&d_B, B.bytesize());
    cudaMalloc(&d_C, output_bytesize);

    // copy inputs to device
    cudaMemcpy(d_A, A.get_buf(), A.bytesize(), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, B.get_buf(), B.bytesize(), cudaMemcpyHostToDevice);

    fwgpu::cutlass_sgemm_nn(
      m, n, k, alpha, d_A, lda, d_B, ldb, beta, d_C, ldc);

    // copy output to host
    cudaMemcpy(c_bytes, d_C, output_bytesize, cudaMemcpyDeviceToHost);

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    return Matrix<float>(m, n, c_bytes);
  }
} // namespace testing
} // namespace fwgpu
