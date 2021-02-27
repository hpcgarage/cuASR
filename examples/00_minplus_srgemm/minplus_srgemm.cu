#include <chrono>
#include <iostream>
#include <random>

#include "cuasr/gemm/device/default_srgemm_configuration.h"
#include "cuasr/gemm/device/srgemm.h"
#include "cuasr/functional.h"

auto cuasr_minplus_srsgemm_nt_n(
    int M,
    int N,
    int K,
    float const *A,
    int lda,
    float const *B,
    int ldb,
    float *C,
    int ldc,
    float *D,
    bool do_epilogue_min,
    cudaStream_t stream = nullptr) -> int {
  // compile time configuration of this srgemm kernel using OperatorClass
  using OperatorClass    = cutlass::arch::OpClassSimt;
  using SmArch           = cutlass::arch::Sm50;
  using AdditionOp       = cuasr::minimum<float>;
  using MultiplicationOp = cuasr::plus<float>;

  using TropicalConfig = typename cuasr::gemm::device::DefaultSemiRingConfiguration<
      float, float, float, float, OperatorClass, //
      AdditionOp, MultiplicationOp, SmArch>;

  using ColumnMajor = cutlass::layout::ColumnMajor;
  using RowMajor    = cutlass::layout::RowMajor;

  using cuASR_MinPlus_SGEMM = cuasr::gemm::device::Srgemm<
      AdditionOp,       // Thread level SemiRing operator
      MultiplicationOp, // Thread level SemiRing operator
      float,            // element type of A
      ColumnMajor,      // layout of A
      float,            // element type of B
      RowMajor,         // layout of B
      float,            // element t  ype of C
      ColumnMajor,         // layout of C
      float             // element type of D
      >;

  float alpha = MultiplicationOp::Identity;
  float beta
      = do_epilogue_min ? MultiplicationOp::Identity : MultiplicationOp::Annihilator;

  // construct kernel arguments struct
  cuASR_MinPlus_SGEMM::Arguments args(
      { M, N, K },    // Problem dimensions
      { A, lda },     // Tensor-ref for source matrix A
      { B, ldb },     // Tensor-ref for source matrix B
      { C, ldc },     // Tensor-ref for source matrix C
      { D, ldc },     // Tensor-ref for destination matrix D
      { alpha, beta } //
  );

  // launch SRGEMM kernel
  cuASR_MinPlus_SGEMM minplus_gemm;
  cutlass::Status status = minplus_gemm(args, nullptr, stream);
  return static_cast<int>(status);
}

auto cuasr_minplus_srsgemm_nt_n(
    int M,
    int N,
    int K,
    float const *A,
    int lda,
    float const *B,
    int ldb,
    float *C,
    int ldc,
    bool do_epilogue_min,
    cudaStream_t stream) -> int {
  return cuasr_minplus_srsgemm_nt_n(
      M, N, K, A, lda, B, ldb, C, ldc, C, do_epilogue_min, stream);
}

auto rng_init_matrix(float *buf, int len, int seed, float min = 0.5, float max = 1.5)
    -> void {
  auto rng  = std::mt19937_64(seed);
  auto dist = std::uniform_real_distribution<float>(min, max);
  for (auto i = 0; i < len; ++i) {
    buf[i] = dist(rng);
  }
}

int main() {
  using namespace std::chrono;
  // problem size
  constexpr int M       = 4096;
  constexpr int N       = 4096;
  constexpr int K       = 4096;
  constexpr int repeats = 1;

  std::cout << "Running tropical SRGEMM on A = " << M << 'x' << K << " and B = " << K
            << 'x' << N << '\n';

  std::cout << "Allocating and initializing host/device buffers\n";
  float *A = new float[M * K];
  float *B = new float[K * N];
  float *C = new float[M * N];

  rng_init_matrix(A, M * K, 3090 + 0);
  rng_init_matrix(B, K * N, 3090 + 1);
  rng_init_matrix(C, M * N, 3090 + 2);

  float *d_A, *d_B, *d_C;
  cudaMalloc((void **)&d_A, sizeof(float) * M * K);
  cudaMalloc((void **)&d_B, sizeof(float) * K * N);
  cudaMalloc((void **)&d_C, sizeof(float) * M * N);

  cudaMemcpy(d_A, A, sizeof(float) * M * K, cudaMemcpyHostToDevice);
  cudaMemcpy(d_B, B, sizeof(float) * K * N, cudaMemcpyHostToDevice);
  cudaMemcpy(d_C, C, sizeof(float) * M * N, cudaMemcpyHostToDevice);

  auto retval = 0;
  auto start  = high_resolution_clock::now();
  for (int i = 0; i < repeats; ++i) {
    retval |= cuasr_minplus_srsgemm_nt_n(M, N, K, d_A, M, d_B, K, d_C, M, true, nullptr);
    cudaDeviceSynchronize();
  }
  auto end               = high_resolution_clock::now();
  auto delta = duration_cast<nanoseconds>(end - start).count();

  if (retval) {
    std::cout << "Error code " << retval << '\n';
    return retval;
  }

  std::cout << "Min-Plus SRGEMM FLOP/s = " << (repeats * 2.0 * M * N * K) / (delta / 1'000'000'000.0)
            << '\n';
  return 0;
}
