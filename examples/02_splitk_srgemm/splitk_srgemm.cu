/***************************************************************************************************
 * Copyright (c) 2020, Vijay Thakkar (thakkarv@gatech.edu).  All rights reserved.
 **************************************************************************************************/

#include <chrono>
#include <iostream>
#include <random>

#include "cuasr/functional.h"
#include "cuasr/gemm/device/default_srgemm_configuration.h"
#include "cuasr/gemm/device/srgemm.h"
#include "cuasr/gemm/device/srgemm_splitk_parallel.h"

#include "cutlass/gemm/device/gemm_splitk_parallel.h"
#include "cutlass/util/device_memory.h"

auto cuasr_splitk_minplus_srsgemm_tn_t(
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
    int split_k_slices,
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

  using cuASR_SplitK_SRGEMM = cuasr::gemm::device::SrgemmSplitKParallel<
      AdditionOp,       // Thread level SemiRing operator
      MultiplicationOp, // Thread level SemiRing operator
      float,            // element type of A
      RowMajor,         // layout of A
      float,            // element type of B
      ColumnMajor,      // layout of B
      float,            // element t  ype of C
      RowMajor,         // layout of C
      float             // element type of D
      >;

  // setup runtime configuration
  float alpha = MultiplicationOp::Identity;
  float beta
      = do_epilogue_min ? MultiplicationOp::Identity : MultiplicationOp::Annihilator;

  // construct kernel arguments struct
  cuASR_SplitK_SRGEMM::Arguments args(
      { M, N, K },     // Problem dimensions
      { A, lda },      // Tensor-ref for source matrix A
      { B, ldb },      // Tensor-ref for source matrix B
      { C, ldc },      // Tensor-ref for source matrix C
      { D, ldc },      // Tensor-ref for destination matrix D
      { alpha, beta }, // epilogue scalars
      split_k_slices   // number of K dimension slices
  );

  // using the arguments, query for extra workspace required parallel reduction
  size_t workspace_size = cuASR_SplitK_SRGEMM::get_workspace_size(args);
  cutlass::device_memory::allocation<uint8_t> workspace(workspace_size);

  // construct cuasr kernel depending on templates
  cuASR_SplitK_SRGEMM splitk_minplus_srgemm_op;

  // Initialize cuasr kernel with arguments and workspace ptr
  cutlass::Status status = splitk_minplus_srgemm_op.initialize(args, workspace.get());

  // launch split-K parallel SRGEMM kernel
  status = splitk_minplus_srgemm_op();

  return static_cast<int>(status);
}

auto cuasr_splitk_minplus_srsgemm_tn_t(
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
    int split_k_slices,
    cudaStream_t stream) -> int {
  return cuasr_splitk_minplus_srsgemm_tn_t(
      M, N, K, A, lda, B, ldb, C, ldc, C, do_epilogue_min, split_k_slices, stream);
}

auto cuasr_minplus_srsgemm_tn_t(
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
      RowMajor,         // layout of A
      float,            // element type of B
      ColumnMajor,      // layout of B
      float,            // element t  ype of C
      RowMajor,         // layout of C
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

auto cuasr_minplus_srsgemm_tn_t(
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
  return cuasr_minplus_srsgemm_tn_t(
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

int main(int argc, const char *argv[]) {
  using namespace std::chrono;

  // problem size
  constexpr int M       = 128;
  constexpr int N       = 128;
  constexpr int K       = 128 * 32;
  constexpr int lda     = N; // num cols if row major, num rows if col major
  constexpr int ldb     = K; // num cols if row major, num rows if col major
  constexpr int ldc     = N; // num cols if row major, num rows if col major
  constexpr int repeats = 10;
  int split_k_slices    = 8;
  if (argc > 1) {
    split_k_slices = std::atoi(argv[1]);
  }

  std::cout << "Running tropical SRGEMM on A = " << M << 'x' << K << " and B = " << K
            << 'x' << N << " with " << split_k_slices << " split-K slices." << '\n';

  std::cout << "Allocating and initializing host/device buffers\n";
  float *A        = new float[M * K];
  float *B        = new float[K * N];
  float *C        = new float[M * N];
  float *C_splitk = new float[M * N];

  rng_init_matrix(A, M * K, 3090 + 0);
  rng_init_matrix(B, K * N, 3090 + 1);
  rng_init_matrix(C, M * N, 3090 + 2);

  auto retval = 0;

  float *d_A, *d_B, *d_C_regular, *d_C_splitk;
  retval |= cudaMalloc((void **)&d_A, sizeof(float) * M * K);
  retval |= cudaMalloc((void **)&d_B, sizeof(float) * K * N);
  retval |= cudaMalloc((void **)&d_C_regular, sizeof(float) * M * N);
  retval |= cudaMalloc((void **)&d_C_splitk, sizeof(float) * M * N);

  retval |= cudaMemcpy(d_A, A, sizeof(float) * M * K, cudaMemcpyHostToDevice);
  retval |= cudaMemcpy(d_B, B, sizeof(float) * K * N, cudaMemcpyHostToDevice);
  retval |= cudaMemcpy(d_C_regular, C, sizeof(float) * M * N, cudaMemcpyHostToDevice);
  retval |= cudaMemcpy(d_C_splitk, C, sizeof(float) * M * N, cudaMemcpyHostToDevice);

  if (retval > 0) {
    std::cout << "Could not allocate or copy to device.\n";
    return retval;
  }

  // run the tests
  auto start = high_resolution_clock::now();
  for (int i = 0; i < repeats; ++i) {
    retval |= cuasr_minplus_srsgemm_tn_t(
        M, N, K, d_A, M, d_B, K, d_C_regular, M, true, nullptr);
    retval |= cudaDeviceSynchronize();
  }
  auto end           = high_resolution_clock::now();
  auto delta_regular = duration_cast<nanoseconds>(end - start).count();

  retval = 0;
  start  = high_resolution_clock::now();
  for (int i = 0; i < repeats; ++i) {
    retval |= cuasr_splitk_minplus_srsgemm_tn_t(
        M, N, K, d_A, lda, d_B, ldb, d_C_splitk, ldc, true, split_k_slices, nullptr);
    retval |= cudaDeviceSynchronize();
  }
  end               = high_resolution_clock::now();
  auto delta_splitk = duration_cast<nanoseconds>(end - start).count();

  if (retval) {
    std::cout << "Error code " << retval << '\n';
    return retval;
  }

  // print perf numbers
  std::cout << "Min-Plus SRGEMM FLOP/s = "
            << (repeats * 2.0 * M * N * K) / (delta_regular / 1'000'000'000.0) << '\n';

  std::cout << "Min-Plus Split-K SRGEMM FLOP/s = "
            << (repeats * 2.0 * M * N * K) / (delta_splitk / 1'000'000'000.0) << '\n';

  std::cout << "Split-K speedup over regular = "
            << static_cast<double>(delta_regular) / delta_splitk << '\n';

  // verify correct
  cudaMemcpy(C, d_C_regular, sizeof(float) * M * N, cudaMemcpyDeviceToHost);
  cudaMemcpy(C_splitk, d_C_splitk, sizeof(float) * M * N, cudaMemcpyDeviceToHost);
  auto is_correct = true;
  for (int n = 0; n < N; ++n) {
    for (int m = 0; m < M; ++m) {
      is_correct &= (C[(ldc * n) + m] == C_splitk[(ldc * n) + m]);
    }
  }

  if (is_correct) {
    std::cout << "Split-K matches regular SRGEMM\n";
    return 0;
  }
  else {
    std::cout << "Split-K does NOT match regular SRGEMM\n";
    return 1;
  }
}
