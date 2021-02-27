#include <chrono>
#include <iostream>
#include <random>

#include "cuasr/gemm/device/default_srgemm_configuration.h"
#include "cuasr/gemm/device/srgemm.h"
#include "cuasr/functional.h"

#include "cuasr/reference/srgemm/host_srgemm.h"

/* cuASR for Galois Field Semiring GEMM : A Demo of cuasr extension
 *
 * In this example, we show how to define a custom semiring GEMM operator
 * that is not supported by the provided default SRGEMM configurations in cuASR.
 *
 * Galois Field SRGEMM explained here is an implementation of GEMM over GF(2) field
 * arithmetic. cuasr/functional.h already contians an implementation of binary_and<T>
 * operation, so we must define a binary_xor<T> here in order to define our own out of
 * library ring.
 *
 * GF(2) GEMM:
 *   Addition operator = binary XOR
 *   Multiplication Operator = binary AND
 *   Zero = Addition Identity = false
 *   Multiplicative Annihilator = false
 *
 * The primary thing that needs to be done for this is contained in the anonymous
 * namespace below. All cuasr ring operators are defined as default constructible structs
 * that contain many overloads of operator() with which the cuasr SRGEMM core kernel can
 * invoke them. Although verbose, the different scalar and cutlass::Array<T,N> overloads
 * of each operator allow for optimizations to be done, primarily for unrolling. These
 * structs need minimal knowledge of CUDA and are still quite short to implement at around
 * 50 lines.
 *
 * This operator struct must also contain a constexpr definition of the Identity and
 * Annihilator elements for the user defined operator, as these are used within the core
 * cuasr SRGEMM kernel to initialize the accumulators and during the epilogue to see if a
 * load from the C matrix is needed. In our case of xor operation, this is as simple as
 * including `static T constexpr Identity = static_cast<T>(false);` in the struct
 * definition.
 *
 * After the operator struct is defined, the rest is some simple boilerplate for
 * instantiating the cuasr::gemm::device::Srgemm template such as input matrix data types,
 * leading dimensions, alignments as well as the tile shapes for threadblock, warp and
 * instruction level SRGEMM. In the case of SIMT SRGEMM, only valid `InstructionShape` is
 * <1, 1, 1> since each lane processes a single element at a time. ThreadblockShape and
 * WarpShape are the two main points of optimization as they affect the amount of shared
 * memory and register usage and unrolling. Since SRGEMM only supports SIMT instructions,
 * OperatorClass must be set to OpClassSimt. SmArch can be set to Sm50 for SRGEMM on
 * Maxwell or later which only supports 2 stage SRGEMM. Support for Sm80 (Ampere)
 * multi-stage pipelined SRGEMM is planned for the future.
 */

// clang-format off
namespace {
template <typename T, int N = 1>
struct binary_xor {
  static T constexpr Identity = static_cast<T>(false);

  // expose base scalar operator
  __host__ __device__
  T operator()(T lhs, T const &rhs) const {
    lhs ^= rhs;
    return lhs;
  }

  __host__ __device__
  cutlass::Array<T, N>
  operator()(cutlass::Array<T, N> const &lhs, cutlass::Array<T, N> const &rhs) const {
    cutlass::Array<T, N> result;
    #pragma unroll
    for (int i = 0; i < N; ++i) {
      result[i] = this->operator()(lhs[i], rhs[i]);
    }
    return result;
  }

  __host__ __device__
  cutlass::Array<T, N>
  operator()(cutlass::Array<T, N> const &lhs, T const &scalar) const {
    cutlass::Array<T, N> result;
    #pragma unroll
    for (int i = 0; i < N; ++i) {
      result[i] = this->operator()(lhs[i], scalar);
    }
    return result;
  }

  __host__ __device__
  cutlass::Array<T, N>
  operator()(T const &scalar, cutlass::Array<T, N> const &rhs) const {
    cutlass::Array<T, N> result;
    #pragma unroll
    for (int i = 0; i < N; ++i) {
      result[i] = this->operator()(scalar, rhs[i]);
    }
    return result;
  }
};
} // namespace
// clang-format on

// GF(2) xor-and SRGEMM
auto cuasr_gf_srgemm_nnn(
    int M,
    int N,
    int K,
    int const *A,
    int lda,
    int const *B,
    int ldb,
    int *C,
    int ldc,
    int *D,
    bool do_epilogue_and,
    cudaStream_t stream = nullptr) -> int {
  // compile time configuration of this srgemm kernel
  using OperatorClass = cutlass::arch::OpClassSimt;
  using SmArch        = cutlass::arch::Sm50;

  using AdditionOp       = binary_xor<int>;
  using MultiplicationOp = cuasr::binary_and<int>;
  using EpilogueOutputOp = cuasr::epilogue::thread::SemiringLinearCombination<
      AdditionOp, MultiplicationOp, int, 1>;

  static int constexpr AlignmentA = 1;
  static int constexpr AlignmentB = 1;
  using ThreadblockShape          = cutlass::gemm::GemmShape<64, 128, 8>;
  using WarpShape                 = cutlass::gemm::GemmShape<16, 64, 8>;
  using InstructionShape          = cutlass::gemm::GemmShape<1, 1, 1>;
  using ThreadblockSwizzle =
      typename cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>;
  static int constexpr Stages = 2;

  using RowMajor = cutlass::layout::RowMajor;

  using cuASRGaloisFieldSrgemm = cuasr::gemm::device::Srgemm<
      AdditionOp,         // Thread level SemiRing operator
      MultiplicationOp,   // Thread level SemiRing operator
      int,                // element type of A
      RowMajor,           // layout of A
      int,                // element type of B
      RowMajor,           // layout of B
      int,                // element t  ype of C
      RowMajor,           // layout of C
      int,                // element type of D
      OperatorClass,      // Logical operator class (SIMT/Tensor)
      SmArch,             // CUDA architecture
      ThreadblockShape,   // GEMM shape at CTA level
      WarpShape,          // GEMM shape at Warp level
      InstructionShape,   // GEMM shape at thread level
      EpilogueOutputOp,   // Epilogue operator at thread level
      ThreadblockSwizzle, // GEMM threadblock swizzler
      Stages,             // Pipeline stages for shmem
      AlignmentA,         // Alignment of A elements
      AlignmentB,         // Alignment of B elements
      false               // SplitKSerial
      >;

  int alpha = MultiplicationOp::Identity;
  int beta = do_epilogue_and ? MultiplicationOp::Identity : MultiplicationOp::Annihilator;

  // construct kernel arguments struct
  cuASRGaloisFieldSrgemm::Arguments args(
      { M, N, K },    // Problem dimensions
      { A, lda },     // Tensor-ref for source matrix A
      { B, ldb },     // Tensor-ref for source matrix B
      { C, ldc },     // Tensor-ref for source matrix C
      { D, ldc },     // Tensor-ref for destination matrix D
      { alpha, beta } //
  );

  // launch SRGEMM kernel
  cuASRGaloisFieldSrgemm gf_srgemm;
  cutlass::Status status = gf_srgemm(args, nullptr, stream);
  return static_cast<int>(status);
}

auto cuasr_gf_srgemm_nnn(
    int M,
    int N,
    int K,
    int const *A,
    int lda,
    int const *B,
    int ldb,
    int *C,
    int ldc,
    bool do_epilogue_and,
    cudaStream_t stream) -> int {
  return cuasr_gf_srgemm_nnn(M, N, K, A, lda, B, ldb, C, ldc, C, do_epilogue_and, stream);
}

auto rng_init_matrix(int *buf, int len, int seed) -> void {
  auto rng  = std::mt19937_64(seed);
  auto dist = std::bernoulli_distribution(0.025);
  for (auto i = 0; i < len; ++i) {
    buf[i] = static_cast<int>(dist(rng));
  }
}

// compares result of SRGEMM to a CPU kernel as reference
auto compare_host_reference(
    int M,
    int N,
    int K,
    int alpha,
    int *A,
    int lda,
    int *B,
    int ldb,
    int beta,
    int *C,
    int ldc,
    int *reference_D,
    int *device_D) -> bool {
  using AdditionOp       = binary_xor<int>;
  using MultiplicationOp = cuasr::binary_and<int>;
  using EpilogueOutputOp = cuasr::epilogue::thread::SemiringLinearCombination<
      AdditionOp, MultiplicationOp, int, 1>;
  using RowMajor = cutlass::layout::RowMajor;

  cuasr::reference::host::Srgemm<
      AdditionOp,                                    //
      MultiplicationOp,                              //
      int, RowMajor,                                 //
      int, RowMajor,                                 //
      int, RowMajor,                                 //
      typename EpilogueOutputOp::ElementCompute,     //
      typename EpilogueOutputOp::ElementAccumulator, //
      EpilogueOutputOp>
      reference_srgemm;

  reference_srgemm(
      { M, N, K },                            //
      alpha, { A, lda }, { B, ldb },          //
      beta, { C, ldc }, { reference_D, ldc }, //
      AdditionOp::Identity);

  auto is_correct = true;
  for (int n = 0; n < N; ++n) {
    for (int m = 0; m < M; ++m) {
      is_correct &= (reference_D[(ldc * n) + m] == device_D[(ldc * n) + m]);
    }
  }
  return is_correct;
}


int main() {
  using namespace std::chrono;
  // problem size
  constexpr int M                = 512; // 4096
  constexpr int N                = 512;
  constexpr int K                = 512;
  constexpr bool do_epilogue_and = true;

  std::cout << "Running Xor-And Galois Field SRGEMM on A = " << M << 'x' << K
            << " and B = " << K << 'x' << N << '\n';

  // input matrices
  std::cout << "Allocating and initializing host/device buffers\n";
  int *A = new int[M * K];
  int *B = new int[K * N];
  int *C = new int[M * N];

  // output
  int *reference_D = new int[M * N];
  int *device_D    = new int[M * N];

  rng_init_matrix(A, M * K, 3090 + 0);
  rng_init_matrix(B, K * N, 3090 + 1);
  rng_init_matrix(C, M * N, 3090 + 2);

  int *d_A, *d_B, *d_C;
  cudaMalloc((void **)&d_A, sizeof(int) * M * K);
  cudaMalloc((void **)&d_B, sizeof(int) * K * N);
  cudaMalloc((void **)&d_C, sizeof(int) * M * N);

  cudaMemcpy(d_A, A, sizeof(int) * M * K, cudaMemcpyHostToDevice);
  cudaMemcpy(d_B, B, sizeof(int) * K * N, cudaMemcpyHostToDevice);
  cudaMemcpy(d_C, C, sizeof(int) * M * N, cudaMemcpyHostToDevice);

  auto start = high_resolution_clock::now();

  auto retval
      = cuasr_gf_srgemm_nnn(M, N, K, d_A, M, d_B, K, d_C, M, do_epilogue_and, nullptr);
  retval |= cudaDeviceSynchronize();
  auto end               = high_resolution_clock::now();
  duration<double> delta = (end - start);

  if (retval) {
    std::cout << "Error code " << retval << '\n';
    return retval;
  }

  std::cout << "Xor-And Galois Field SRGEMM FLOP/s = "
            << (2.0 * M * N * K) / delta.count() << '\n';

  cudaMemcpy(device_D, d_C, sizeof(int) * M * N, cudaMemcpyDeviceToHost);

  // compare against host
  std::cout << "Comparing against reference host-side SRGEMM : ";
  int alpha = cuasr::binary_and<int>::Identity;
  int beta  = do_epilogue_and ? cuasr::binary_and<int>::Identity
                             : cuasr::binary_and<int>::Annihilator;
  auto is_correct = compare_host_reference(
      M, N, K, alpha, A, M, B, N, beta, C, M, reference_D, device_D);

  if (is_correct) {
    std::cout << "PASSED!\n";
  }
  else {
    std::cout << "FAILED!\n";
  }
  return !is_correct;
}
