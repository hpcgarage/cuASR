# cuASR: CUDA Algebra for SemiRings

`cuASR` (pronounced quasar) is a template library for semi-ring linear algebra on CUDA GPUs. It is based on [NVIDIA Cutlass](https://github.com/NVIDIA/cutlass) open source project and extends the matrix multiplication to all algebraic rings. This library's key design philosophy is to offer users with the following key features:

- Dense linear algebra library with semiring operators as first class citizens.
- Header only template library.
- Out of the box tuned configurations for commonly used semirings found in the GraphBLAS specification.
- Minimal knowledge of CUDA needed to extend
- Ability to tune each semiring's tiling strategy and other optimizations independently from each other.

This library is intended to be used as a building block for other libraries that offer BLAS like dense linear algebra routines that need to operate on semirings, such as a backend for an implementation of GraphBLAS. For now, IEEE 32-bit and 64-bit precision floating point datatypes (`float` and `double`) are supported, but future support for half precision and integer datatypes is also possible. Integral datatype based SRGEMMs work, as can be seen in the examples below with Xor-And Galois Field SRGEMM, but are not fully supported across all possible operators yet.

# 1. What Is A Semiring and Why Should I Care?

In abstract algebra, a [ring](https://en.wikipedia.org/wiki/Ring_(mathematics)) is a set of two binary operations that generalize the notion of addition and multiplication on a set of elements. A ring's addition operator is commutative and has an additive identity that acts as an abstract `0` element. Multiplication operator is associative, distributive and also has an identity element that acts as an abstract `1` element. A [semiring](https://en.wikipedia.org/wiki/Semiring) is similar to a ring in every aspect, but without the requirement that there exist an additive inverse for every possible element in the ring.

Normally, BLAS libraries are defined as operations over real numbers,`+` and `x` only, however, as shown by GraphBLAS, there are many algorithms in graph processing that can be expressed as linear algebra operations over other semirings. For example, shortest path problems can be formulated as GEMV (in the case of single source shortest path) or GEMM (in the case of all pairs shortest path) over the tropical semiring (min-plus algebra). More practical applications and deeper justifications can be found in the [GraphBLAS 1.0 spec](http://www.mit.edu/~kepner/GraphBLAS/GraphBLAS-Math-release.pdf) and [Dr. Jeremy Kepner's prior work](http://persagen.com/files/misc/kepner2015graphs.pdf). Traditionally, GraphBLAS primarily concerns sparse BLAS operations such as SpMV and SpMM, but we have strong reasons to believe that dense linear algebra operations with user customizable semiring operators can play a key role in large scale graph analytics.

# 2. Building Tests and Benchmarks:

cuASR is a template library and therefore header only, but includes an exhaustive list of tests and benchmarks. The build system is based on `CMake`. Basic checkout and build instructions are as follows:

```sh
$ git clone --recurse-submodules https://github.com/hpcgarage/semiring-gemm /path/to/repo
$ cd /path/to/repo
$ mkdir build && cd build
$ cmake .. -G Ninja -DCUASR_CUDA_ARCHS="70 75"
$ ninja
```

This will build the tests and benchmarks in configurations that are most likely to be used in real world workloads. More extensive benchmarks and tests can also be built; see Test and B

Notable build flags:

| Build Flag | Usage Description |
|-|-|
| `CUASR_CUDA_ARCHS` | lists the CUDA SM architectures the fat binaries should be built to target. `CUASR_CUDA_ARCHS="60 61 70 72 75"` (all Pascal and Volta GPUs) will be used if no value is specified, but this can really hurt compile times for tests and benchmarks; Limit CUDA architectures to the smallest subset you forsee running the tests and benchmarks on.
| `CUASR_TEST` | Set to `ON` by default and controls whether tests will be built or not. Set to `OFF` to disable building all tests. |
| `CUASR_BENCH` | Set to `ON` by default and controls whether benchmarks will be built or not. Set to `OFF` to disable building all benchmarks. |
| `CUASR_EXAMPLES` | Set to `ON` by default and controls whether examples will be built or not. Set to `OFF` to disable building all examples. |
| `CMAKE_BUILD_TYPE` | Set to `RELEASE` by default but can bet set to `DEBUG` which enables both host side and device side kernel debugging support. Debugging library internals should not be required for users of `cuASR`, however this flags allows for easy integration into other CMake projects. |

## Test Infrastructure:
This project uses the [`google-test`](https://github.com/google/googletest) project for test infrastructure. [`test`](test) directory contains auto-generated testes for device level semiring GEMM across a wide range of threadblock and warp tile sizes and for `float` as well as `double` datatypes. All tests are registered as CTest tests.

This results in a massive amount of total tests cases, and therefore, tests are split into three increasing levels of coverage. L0 tests contain the most common threadblock and warp shapes, which become more and more exotic (unlikely to be seen in real world usage and yield good performance) at L1 and L2. Top level CMake configuration exposes two test related options. `CUASR_TEST_LEVEL={0|1|2}` is set to `0` by default, the lowest test coverage level.

Test directory also contains some legacy regression tests for cuASR 0.1 that are retained in tree. These tests depend on deprecated builds of the previous versions, and should not used for any other purpose.

Running test cases is easy. To run all tests, just run `ninja test`. Device level semiring GEMM tests can be run directly by executing `build/test/device/cuasr_test_srgemm_device`.

## Benchmark Infrastructure:
This project uses the [`google-benchmark`](https://github.com/google/benchmark) micro-benchmark library for reliably measuring the flop rate of kernel. A custom `FLOP/s` counter is added to each benchmark.

Similar to the tests above, benchmarks are split into the identical three level hierarchy with increasing level of benchmark configuration space coverage. The configurations for L0, L1 and L2 are identical to that of L0, L1 and L2 tests. We benchmark only what we test.

Similar to test case build flags, benchmark suite level can be controlled with the build flag `CUASR_BENCH_LEVEL={0|1|2}`, which is set to level `0` by default. Device level semiring GEMM benchmarks can be run by executing `build/bench/device/cuasr_bench_srgemm_device --benchmark_counters_tabular=true`.

# 3. Using cuASR:

## Using Tuned GraphBLAS Semirings

cuASR provides eight semi-rings that have configurations pre-tuned for threadblock and warp tile sizes. These semirings are

- Plus-Multiply (regular GEMM)
- Min-Plus (tropical semiring)
- Max-Plus
- Min-Multiply
- Max-Multiply
- Min-Max
- Max-Min
- Or-And (binary arithmetic on floating point numbers)

The following shows how to use and wrap min-plus tropical semiring GEMM in a BLAS-3 like API wrapper, where input matrices are already allocated and moved to device memory. Code block below is an excerpt taken from [`examples/00_minplus_srgemm`](examples/00_minplus_srgemm/minplus_srgemm.cu) which shows how easy it is to instance a template SRGEMM for any of these pre-tuned semirings.

```cpp
#include "cuasr/functional.h"
#include "cuasr/device/default_srgemm_configuration.h"
#include "cuasr/device/srgemm.h"

auto cuasr_minplus_srsgemm_nt(
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
  // compile time configuration of this srgemm kernel
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
      AdditionOp,       // Thread level semiring add operator
      MultiplicationOp, // Thread level semiRing multiply operator
      float,            // element type of A
      ColumnMajor,      // layout of A
      float,            // element type of B
      RowMajor,         // layout of B
      float,            // element type of C
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
      { alpha, beta } // True if we perform a final min with source matrix C
  );

  // launch SRGEMM kernel
  cuASR_MinPlus_SGEMM minplus_gemm;
  cutlass::Status status = minplus_gemm(args, nullptr, stream);
  return static_cast<int>(status);
}
```

## Defining Custom Semirings:

cuASR is designed with the intention of allowing users of the library to write arbitrary custom semiring operators and create a SRGEMM on top of them with minimal effort and lines of code, minimal knowledge of CUDA while still promising for close to the peak performance of the hardware. This is further explained in the [design of core operator structs](#Core-Operator-Structs) section below.

In this example, we show how to define a custom semiring GEMM operator that is not supported by the provided default SRGEMM configurations in cuASR.

Galois Field SRGEMM explained here is an implementation of GEMM over GF(2) field arithmetic. [`cuasr/functional.h`](include/cuasr/functional.h) already contains an implementation of `binary_and<T>` operation, so we must define a `binary_xor<T>` here in order to define our custom semiring.

GF(2) SRGEMM has the following properties:
- Addition operator = binary XOR
- Multiplication Operator = binary AND
- Zero = Additive Identity = `false`
- Multiplicative Annihilator = `false`

All cuASR ring operators are defined as default constructible structs that contain four overloads of `operator()` with which the cuASR SRGEMM core kernel can invoke them. Different `cutlass::Array<T,N>` overloads of the operator allow for unrolling to be performed. At around 50 lines of code, these structs require minimal effort or knowledge of CUDA to implement.

These operator structs must also contain a `constexpr` definition of the `Identity` and/or `Annihilator` elements for the user defined operator, as these are used within the core cuASR SRGEMM kernel to initialize the accumulators and during the epilogue to see if a load from the source matrix is needed. Refer to the design sections about [operators](#Core-Operator-Structs) and [semiring epilogue](#Generic-Semiring-Linear-Combination-Epilogue) below for details. This this example for xor operation, this is as simple as including `static T constexpr Identity = static_cast<T>(false);` in the struct definition.

After the operator struct is defined, the rest is some simple boilerplate for instantiating the `cuasr::gemm::device::Srgemm` template such as input matrix data types, leading dimensions, alignments as well as the tile shapes for threadblock, warp and instruction level SRGEMM. In the case of SIMT SRGEMM, only valid `InstructionShape` is
`<1, 1, 1>` since each lane processes a single element at a time. `ThreadblockShape` and `WarpShape` are the two main points of optimization as they affect tile sizes of shared memory, register blocking and unrolling amounts. Since SRGEMM only supports SIMT instructions, `OperatorClass` must be set to `OpClassSimt`. `SmArch` can be set to `Sm50` for SRGEMM on Pascal or later, which only supports 2 stage SRGEMM. Support for `Sm80` (Ampere) multi-stage pipelined SRGEMM is planned for the future.

The code excerpt below is taken from [`examples/01_userdefined_semiring`](examples/01_userdefined_semiring/userdefined_semiring.cu).

```cpp
template <typename T, int N = 1>
struct binary_xor {
  static T constexpr Identity = static_cast<T>(false);
  // scalar operator
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

  // ... other overloads for cutlass::Array<T, N> here ...
};

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
```
# 4. Design of cuASR:

## Core Operator Structs:

At the core design of are default constructible structs that we refer to as operators. These structs contain four `operator()` implementations (one scalar, three array based) that perform the semiring addition or multiplication computation.

These structs always have the following signature:

```cpp
template<
  typename Element, // datatype on which to perform the operation
  typename N = 1    // size of the cutlass::Array on which to perform the operation
>
struct semiring_op;
```

An example can be seen above in the form of `binary_xor`. Since the array has automatic storage whose size is known ahead of time, these are unrolled operations on registers in the GPU. The primary point of optimization that allows for pluggable operators while having peak performance is the fact that these structs are passed in as templates. At the core `cuasr::arch::Srmma` level, these structs are default constructed, and then used with their `operator()`. This allows for the CUDA compiler to inline the invocation of the operator. The fact that same struct has both the scalar and array operators allows for the same operators to be used for both the thread level SRMMA operation as well as during the SRGEMM epilogue.

Each operator struct must also contain a `constexpr` definition for `Identity` value in case for use as addition operator or `Annihilator` value in the case for use as a multiplication operator. A struct can also contain both if the operator is intended to be used as either. This makes the definition of the SRGEMM with these template structs algebraically precise. This allows us to use `Identity` value is for the addition operator as the initial value of all accumulators as the `0` under addition. Multiplication operator's `Annihilator` value is used during the SRGEMM epilogue, as described in the section below about the [semiring linear combination epilogue](#Generic-Semiring-Linear-Combination-Epilogue).

[`cuasr/functional.h`](include/cuasr/functional.h) contains many operator structs that have been predefined, and custom operators are easy to implement as described in [the custom SRGEMM example](#Defining-Custom-Semirings). These structs, once defined, are passed to the device level cuASR SRGEMM template, `cuasr::gemm::device::Srgemm`, as the first two parameters.

## Thread Level Semiring MMA:

When a device level SRGEMM template, `cuasr::gemm::device::Srgemm`, is instantiated with addition and multiplication operator structs, the operator structs are eventually used at thread level semiring multiply add operation after passing through the threadblock and wrap level SRMMA. This is where the operator structs are default constructed, and the `operator()` called with the thread level inputs, as seen below.

```cpp
namespace cuasr::arch {
template <
  // ... datatype and GEMM shape template params
  typename AdditionOp,
  typename MultiplicationOp
>
struct Srmma {
  using Shape = cutlass::gemm::GemmShape<1, 1, 1>;

  // operators must be default contructible and contain a binary operator()
  AdditionOp add;
  MultiplicationOp mult;

  __host__ __device__
  void operator()(
    cutlass::Array<ElementC, 1> &d,
    cutlass::Array<ElementA, 1> const &a,
    cutlass::Array<ElementB, 1> const &b,
    cutlass::Array<ElementC, 1> const &c
  ) {
    d[0] = add(c[0], mult(a[0], b[0]));
  }
};
}
```

Since these structs are passed into the SRGEMM template hierarchy at compile time, both the add and multiply `operator()` can be inlined by the compiler. This allows the compiler to perform optimal register allocation, instruction selection and instruction schedule for any arbitrary pairings of add and multiply operators. To illustrate this, we can take cuASR GEMM for regular arithmetic as an example. By using `cuasr::plus<T>` and `cuasr::multiplies<T>` as the operators for add and multiply, we can easily implement regular GEMM with cuASR. In this case, even though we do not specialize `cuasr::gemm::thread::Srmma` to use fused multiply-add, the compiler is able to optimize the two operators and replace them with a single `FMA` instruction. We do not sacrifice any performance for traditional BLAS-3 GEMM on regular arithmetic, while gaining the flexibility to implement arbitrary semirings with minimal code duplication.

## Generic Semiring Linear Combination Epilogue:

The epilogue of a GEMM performs the final linear scaling and addition with the source matrix to obtain the output result. With regular arithmetic, this results in performing `Output = (alpha*C) + (beta*AB)` operation, where `AB` is the result of multiplying matrix `A` with `B`.

Just like cuASR SRGEMM replaces the core compute arithmetic with semiring operators, the BLAS scalar epilogue is abstracted to its semiring linear scaling analog. Under the semiring linear combination epilogue, the output matrix is calculated as follows:

```cpp
AdditionOp add_op;        // semiring add operator struct
MultiplicationOp mult_op; // semiring multiply operator struct

// element wise operations for all matrices
D[:, :] = add_op(
  mult_op(alpha, AB[:, :])
  mult_op(beta, C[:, :]),
);
```

In the case of GEMM with regular arithmetic, default values of alpha and beta are generally set to `alpha = 1` and `beta = 0` so that the output matrix `D = AB`. In the case of SRGEMM, we also abstract these default values to be `alpha = MultiplicationOp::Identity` and `beta = MultiplicationOp::Annihilator` such that output matrix `D = AB` where `AB` is now the result of semiring matrix multiply. Because of the [cuASR operator struct design](#Core-Operator-Structs) described earlier, semiring linear combination is short and easy to implement at around 50 lines of code, and can be found under ['cuasr/epilogue/thread/semiring_linear_combination.h'](include/cuasr/epilogue/thread/semiring_linear_combination.h).

This design allows for a lot of flexibility in the desired output of any arbitrary semiring GEMM. Let us take All Pairs Shortest Path (APSP) as an example. APSP is analogous to performing a GEMM over min-plus or tropical semiring on the graph adjacency matrix.

For a single GPU run of Floyd-Warshall, it is often the case that we have the same input matrix for both `A` and `B` and the source matrix `C` is empty; There is no need to calculate `beta * C`. In this case, `beta` can bet set to a `cuasr::plus<T>::Annihilator` value such that multiplying any element of semiring with it results in additive operator's identity (`minimum<T>::Identity`, i.e. `inf`). This eliminates the load of the source matrix `C` from global memory, and the product of SRGEMM, `AB`, is written directly to the output `D`. In the general case, load of source matrix C can be eliminated for an SRGEMM if `mult_op(beta, MultiplicationOp::Identity) == AdditionOp::Identity`.

In a distributed APSP computation, however, an epilogue min needs to be performed with the source matrix `C`. In that case, `beta` can be initialized to `plus<T>::Identity` which would result in the output being set to an element wise `D = min(C, AB)`.

## Pre-tuned configurations:

[`cuasr/device/default_srgemm_configuration.h`](include/cuasr/device/default_srgemm_configuration.h)  pre-tuned configurations for 8 semirings together with type alias for their epilogues. These reduce the boilerplate required to implement commonly used SRGEMMs, such as the [min-plus tropical GEMM example](#Using-Tuned-GraphBLAS-Semirings).

## Acknowledgements:

This would would not have been possible without the open source [NVIDIA Cutlass](https://github.com/NVIDIA/cutlass) library. We would like to thank the authors of Cutlass for making it open source, and providing us valuable feedback during the design process of cuASR.
