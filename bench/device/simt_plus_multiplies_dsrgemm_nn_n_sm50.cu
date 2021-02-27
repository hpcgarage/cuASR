/***************************************************************************************************
* Copyright (c) 2020, Vijay Thakkar (thakkarv@gatech.edu).
**************************************************************************************************/
//////////////////////////////////////////////////////////////////////
//  THIS BENCHMARK FILE IS GENERATED AUTOMATICALLY : DO NOT MODIFY  //
//////////////////////////////////////////////////////////////////////

#include "benchmark/benchmark.h"

#include "cuasr/gemm/device/default_srgemm_configuration.h"
#include "cuasr/gemm/device/srgemm.h"
#include "cuasr/functional.h"

#include "harness.h"

////////////////////////////////////////////////////////////////////////////////
// Elements / Thread:   2 x   4
//    Threads / Warp:   4 x   8
//     Warps / Block:   1 x   1
//       Threadblock:   8 x  32 x   8
#if defined(CUASR_BENCH_LEVEL) and (CUASR_BENCH_LEVEL >= 1)
static void BM_SM50_device_plus_multiplies_dsrgemm_nn_n_8x32x8_8x32x1_2x4_4x8_1x1(benchmark::State &state) {
  const auto N = static_cast<int>(state.range(0));
  using precision = double;
  using OpClass   = cutlass::arch::OpClassSimt;
  using SmArch    = cutlass::arch::Sm50;

  using ThreadblockShape = cutlass::gemm::GemmShape<8, 32, 8>;
  using WarpShape        = cutlass::gemm::GemmShape<8, 32, 8>;
  using InstructionShape = cutlass::gemm::GemmShape<1, 1, 1>;

  using Config = typename cuasr::gemm::device::DefaultSemiRingConfiguration< //
      precision, precision, precision, precision, OpClass,                   //
      cuasr::plus<precision>, cuasr::multiplies<precision>, SmArch>;

  using AddOp            = Config::AdditionOp;
  using MultOp           = Config::MultiplicationOp;
  using EpilogueOutputOp = Config::EpilogueOutputOp;

  using Srgemm = cuasr::gemm::device::Srgemm<                           //
      AddOp, MultOp,                                                    //
      precision, cutlass::layout::ColumnMajor,                             //
      precision, cutlass::layout::ColumnMajor,                             //
      precision, cutlass::layout::ColumnMajor,                             //
      precision, OpClass, SmArch,                                       //
      ThreadblockShape, WarpShape, InstructionShape, EpilogueOutputOp,  //
      cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>, 2>;

  // setup bench harness
  cuasr::bench::device::BenchHarness<Srgemm> bench({ N, N, N });

  // benchmark loop
  for (auto _ : state) {
    benchmark::DoNotOptimize(bench.run());
    cudaDeviceSynchronize();
  }

  double flops_per_itr = 2.0 * N * N * N;
  state.counters["Flop/s"]
      = benchmark::Counter(flops_per_itr, benchmark::Counter::kIsIterationInvariantRate);
}
BENCHMARK(BM_SM50_device_plus_multiplies_dsrgemm_nn_n_8x32x8_8x32x1_2x4_4x8_1x1)
    ->RangeMultiplier(2)->Range(256, 4096);
#endif

////////////////////////////////////////////////////////////////////////////////
// Elements / Thread:   4 x   4
//    Threads / Warp:   4 x   8
//     Warps / Block:   1 x   1
//       Threadblock:  16 x  32 x   8
#if defined(CUASR_BENCH_LEVEL) and (CUASR_BENCH_LEVEL >= 1)
static void BM_SM50_device_plus_multiplies_dsrgemm_nn_n_16x32x8_16x32x1_4x4_4x8_1x1(benchmark::State &state) {
  const auto N = static_cast<int>(state.range(0));
  using precision = double;
  using OpClass   = cutlass::arch::OpClassSimt;
  using SmArch    = cutlass::arch::Sm50;

  using ThreadblockShape = cutlass::gemm::GemmShape<16, 32, 8>;
  using WarpShape        = cutlass::gemm::GemmShape<16, 32, 8>;
  using InstructionShape = cutlass::gemm::GemmShape<1, 1, 1>;

  using Config = typename cuasr::gemm::device::DefaultSemiRingConfiguration< //
      precision, precision, precision, precision, OpClass,                   //
      cuasr::plus<precision>, cuasr::multiplies<precision>, SmArch>;

  using AddOp            = Config::AdditionOp;
  using MultOp           = Config::MultiplicationOp;
  using EpilogueOutputOp = Config::EpilogueOutputOp;

  using Srgemm = cuasr::gemm::device::Srgemm<                           //
      AddOp, MultOp,                                                    //
      precision, cutlass::layout::ColumnMajor,                             //
      precision, cutlass::layout::ColumnMajor,                             //
      precision, cutlass::layout::ColumnMajor,                             //
      precision, OpClass, SmArch,                                       //
      ThreadblockShape, WarpShape, InstructionShape, EpilogueOutputOp,  //
      cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>, 2>;

  // setup bench harness
  cuasr::bench::device::BenchHarness<Srgemm> bench({ N, N, N });

  // benchmark loop
  for (auto _ : state) {
    benchmark::DoNotOptimize(bench.run());
    cudaDeviceSynchronize();
  }

  double flops_per_itr = 2.0 * N * N * N;
  state.counters["Flop/s"]
      = benchmark::Counter(flops_per_itr, benchmark::Counter::kIsIterationInvariantRate);
}
BENCHMARK(BM_SM50_device_plus_multiplies_dsrgemm_nn_n_16x32x8_16x32x1_4x4_4x8_1x1)
    ->RangeMultiplier(2)->Range(256, 4096);
#endif

////////////////////////////////////////////////////////////////////////////////
// Elements / Thread:   4 x   8
//    Threads / Warp:   4 x   8
//     Warps / Block:   1 x   1
//       Threadblock:  16 x  64 x   8
#if defined(CUASR_BENCH_LEVEL) and (CUASR_BENCH_LEVEL >= 1)
static void BM_SM50_device_plus_multiplies_dsrgemm_nn_n_16x64x8_16x64x1_4x8_4x8_1x1(benchmark::State &state) {
  const auto N = static_cast<int>(state.range(0));
  using precision = double;
  using OpClass   = cutlass::arch::OpClassSimt;
  using SmArch    = cutlass::arch::Sm50;

  using ThreadblockShape = cutlass::gemm::GemmShape<16, 64, 8>;
  using WarpShape        = cutlass::gemm::GemmShape<16, 64, 8>;
  using InstructionShape = cutlass::gemm::GemmShape<1, 1, 1>;

  using Config = typename cuasr::gemm::device::DefaultSemiRingConfiguration< //
      precision, precision, precision, precision, OpClass,                   //
      cuasr::plus<precision>, cuasr::multiplies<precision>, SmArch>;

  using AddOp            = Config::AdditionOp;
  using MultOp           = Config::MultiplicationOp;
  using EpilogueOutputOp = Config::EpilogueOutputOp;

  using Srgemm = cuasr::gemm::device::Srgemm<                           //
      AddOp, MultOp,                                                    //
      precision, cutlass::layout::ColumnMajor,                             //
      precision, cutlass::layout::ColumnMajor,                             //
      precision, cutlass::layout::ColumnMajor,                             //
      precision, OpClass, SmArch,                                       //
      ThreadblockShape, WarpShape, InstructionShape, EpilogueOutputOp,  //
      cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>, 2>;

  // setup bench harness
  cuasr::bench::device::BenchHarness<Srgemm> bench({ N, N, N });

  // benchmark loop
  for (auto _ : state) {
    benchmark::DoNotOptimize(bench.run());
    cudaDeviceSynchronize();
  }

  double flops_per_itr = 2.0 * N * N * N;
  state.counters["Flop/s"]
      = benchmark::Counter(flops_per_itr, benchmark::Counter::kIsIterationInvariantRate);
}
BENCHMARK(BM_SM50_device_plus_multiplies_dsrgemm_nn_n_16x64x8_16x64x1_4x8_4x8_1x1)
    ->RangeMultiplier(2)->Range(256, 4096);
#endif

////////////////////////////////////////////////////////////////////////////////
// Elements / Thread:   8 x   4
//    Threads / Warp:   4 x   8
//     Warps / Block:   1 x   1
//       Threadblock:  32 x  32 x   8
#if defined(CUASR_BENCH_LEVEL) and (CUASR_BENCH_LEVEL >= 0)
static void BM_SM50_device_plus_multiplies_dsrgemm_nn_n_32x32x8_32x32x1_8x4_4x8_1x1(benchmark::State &state) {
  const auto N = static_cast<int>(state.range(0));
  using precision = double;
  using OpClass   = cutlass::arch::OpClassSimt;
  using SmArch    = cutlass::arch::Sm50;

  using ThreadblockShape = cutlass::gemm::GemmShape<32, 32, 8>;
  using WarpShape        = cutlass::gemm::GemmShape<32, 32, 8>;
  using InstructionShape = cutlass::gemm::GemmShape<1, 1, 1>;

  using Config = typename cuasr::gemm::device::DefaultSemiRingConfiguration< //
      precision, precision, precision, precision, OpClass,                   //
      cuasr::plus<precision>, cuasr::multiplies<precision>, SmArch>;

  using AddOp            = Config::AdditionOp;
  using MultOp           = Config::MultiplicationOp;
  using EpilogueOutputOp = Config::EpilogueOutputOp;

  using Srgemm = cuasr::gemm::device::Srgemm<                           //
      AddOp, MultOp,                                                    //
      precision, cutlass::layout::ColumnMajor,                             //
      precision, cutlass::layout::ColumnMajor,                             //
      precision, cutlass::layout::ColumnMajor,                             //
      precision, OpClass, SmArch,                                       //
      ThreadblockShape, WarpShape, InstructionShape, EpilogueOutputOp,  //
      cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>, 2>;

  // setup bench harness
  cuasr::bench::device::BenchHarness<Srgemm> bench({ N, N, N });

  // benchmark loop
  for (auto _ : state) {
    benchmark::DoNotOptimize(bench.run());
    cudaDeviceSynchronize();
  }

  double flops_per_itr = 2.0 * N * N * N;
  state.counters["Flop/s"]
      = benchmark::Counter(flops_per_itr, benchmark::Counter::kIsIterationInvariantRate);
}
BENCHMARK(BM_SM50_device_plus_multiplies_dsrgemm_nn_n_32x32x8_32x32x1_8x4_4x8_1x1)
    ->RangeMultiplier(2)->Range(256, 4096);
#endif

////////////////////////////////////////////////////////////////////////////////
// Elements / Thread:   2 x   2
//    Threads / Warp:   4 x   8
//     Warps / Block:   1 x   2
//       Threadblock:   8 x  32 x   8
#if defined(CUASR_BENCH_LEVEL) and (CUASR_BENCH_LEVEL >= 2)
static void BM_SM50_device_plus_multiplies_dsrgemm_nn_n_8x32x8_8x16x1_2x2_4x8_1x2(benchmark::State &state) {
  const auto N = static_cast<int>(state.range(0));
  using precision = double;
  using OpClass   = cutlass::arch::OpClassSimt;
  using SmArch    = cutlass::arch::Sm50;

  using ThreadblockShape = cutlass::gemm::GemmShape<8, 32, 8>;
  using WarpShape        = cutlass::gemm::GemmShape<8, 16, 8>;
  using InstructionShape = cutlass::gemm::GemmShape<1, 1, 1>;

  using Config = typename cuasr::gemm::device::DefaultSemiRingConfiguration< //
      precision, precision, precision, precision, OpClass,                   //
      cuasr::plus<precision>, cuasr::multiplies<precision>, SmArch>;

  using AddOp            = Config::AdditionOp;
  using MultOp           = Config::MultiplicationOp;
  using EpilogueOutputOp = Config::EpilogueOutputOp;

  using Srgemm = cuasr::gemm::device::Srgemm<                           //
      AddOp, MultOp,                                                    //
      precision, cutlass::layout::ColumnMajor,                             //
      precision, cutlass::layout::ColumnMajor,                             //
      precision, cutlass::layout::ColumnMajor,                             //
      precision, OpClass, SmArch,                                       //
      ThreadblockShape, WarpShape, InstructionShape, EpilogueOutputOp,  //
      cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>, 2>;

  // setup bench harness
  cuasr::bench::device::BenchHarness<Srgemm> bench({ N, N, N });

  // benchmark loop
  for (auto _ : state) {
    benchmark::DoNotOptimize(bench.run());
    cudaDeviceSynchronize();
  }

  double flops_per_itr = 2.0 * N * N * N;
  state.counters["Flop/s"]
      = benchmark::Counter(flops_per_itr, benchmark::Counter::kIsIterationInvariantRate);
}
BENCHMARK(BM_SM50_device_plus_multiplies_dsrgemm_nn_n_8x32x8_8x16x1_2x2_4x8_1x2)
    ->RangeMultiplier(2)->Range(256, 4096);
#endif

////////////////////////////////////////////////////////////////////////////////
// Elements / Thread:   2 x   4
//    Threads / Warp:   4 x   8
//     Warps / Block:   1 x   2
//       Threadblock:   8 x  64 x   8
#if defined(CUASR_BENCH_LEVEL) and (CUASR_BENCH_LEVEL >= 1)
static void BM_SM50_device_plus_multiplies_dsrgemm_nn_n_8x64x8_8x32x1_2x4_4x8_1x2(benchmark::State &state) {
  const auto N = static_cast<int>(state.range(0));
  using precision = double;
  using OpClass   = cutlass::arch::OpClassSimt;
  using SmArch    = cutlass::arch::Sm50;

  using ThreadblockShape = cutlass::gemm::GemmShape<8, 64, 8>;
  using WarpShape        = cutlass::gemm::GemmShape<8, 32, 8>;
  using InstructionShape = cutlass::gemm::GemmShape<1, 1, 1>;

  using Config = typename cuasr::gemm::device::DefaultSemiRingConfiguration< //
      precision, precision, precision, precision, OpClass,                   //
      cuasr::plus<precision>, cuasr::multiplies<precision>, SmArch>;

  using AddOp            = Config::AdditionOp;
  using MultOp           = Config::MultiplicationOp;
  using EpilogueOutputOp = Config::EpilogueOutputOp;

  using Srgemm = cuasr::gemm::device::Srgemm<                           //
      AddOp, MultOp,                                                    //
      precision, cutlass::layout::ColumnMajor,                             //
      precision, cutlass::layout::ColumnMajor,                             //
      precision, cutlass::layout::ColumnMajor,                             //
      precision, OpClass, SmArch,                                       //
      ThreadblockShape, WarpShape, InstructionShape, EpilogueOutputOp,  //
      cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>, 2>;

  // setup bench harness
  cuasr::bench::device::BenchHarness<Srgemm> bench({ N, N, N });

  // benchmark loop
  for (auto _ : state) {
    benchmark::DoNotOptimize(bench.run());
    cudaDeviceSynchronize();
  }

  double flops_per_itr = 2.0 * N * N * N;
  state.counters["Flop/s"]
      = benchmark::Counter(flops_per_itr, benchmark::Counter::kIsIterationInvariantRate);
}
BENCHMARK(BM_SM50_device_plus_multiplies_dsrgemm_nn_n_8x64x8_8x32x1_2x4_4x8_1x2)
    ->RangeMultiplier(2)->Range(256, 4096);
#endif

////////////////////////////////////////////////////////////////////////////////
// Elements / Thread:   4 x   2
//    Threads / Warp:   4 x   8
//     Warps / Block:   1 x   2
//       Threadblock:  16 x  32 x   8
#if defined(CUASR_BENCH_LEVEL) and (CUASR_BENCH_LEVEL >= 2)
static void BM_SM50_device_plus_multiplies_dsrgemm_nn_n_16x32x8_16x16x1_4x2_4x8_1x2(benchmark::State &state) {
  const auto N = static_cast<int>(state.range(0));
  using precision = double;
  using OpClass   = cutlass::arch::OpClassSimt;
  using SmArch    = cutlass::arch::Sm50;

  using ThreadblockShape = cutlass::gemm::GemmShape<16, 32, 8>;
  using WarpShape        = cutlass::gemm::GemmShape<16, 16, 8>;
  using InstructionShape = cutlass::gemm::GemmShape<1, 1, 1>;

  using Config = typename cuasr::gemm::device::DefaultSemiRingConfiguration< //
      precision, precision, precision, precision, OpClass,                   //
      cuasr::plus<precision>, cuasr::multiplies<precision>, SmArch>;

  using AddOp            = Config::AdditionOp;
  using MultOp           = Config::MultiplicationOp;
  using EpilogueOutputOp = Config::EpilogueOutputOp;

  using Srgemm = cuasr::gemm::device::Srgemm<                           //
      AddOp, MultOp,                                                    //
      precision, cutlass::layout::ColumnMajor,                             //
      precision, cutlass::layout::ColumnMajor,                             //
      precision, cutlass::layout::ColumnMajor,                             //
      precision, OpClass, SmArch,                                       //
      ThreadblockShape, WarpShape, InstructionShape, EpilogueOutputOp,  //
      cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>, 2>;

  // setup bench harness
  cuasr::bench::device::BenchHarness<Srgemm> bench({ N, N, N });

  // benchmark loop
  for (auto _ : state) {
    benchmark::DoNotOptimize(bench.run());
    cudaDeviceSynchronize();
  }

  double flops_per_itr = 2.0 * N * N * N;
  state.counters["Flop/s"]
      = benchmark::Counter(flops_per_itr, benchmark::Counter::kIsIterationInvariantRate);
}
BENCHMARK(BM_SM50_device_plus_multiplies_dsrgemm_nn_n_16x32x8_16x16x1_4x2_4x8_1x2)
    ->RangeMultiplier(2)->Range(256, 4096);
#endif

////////////////////////////////////////////////////////////////////////////////
// Elements / Thread:   4 x   4
//    Threads / Warp:   4 x   8
//     Warps / Block:   1 x   2
//       Threadblock:  16 x  64 x   8
#if defined(CUASR_BENCH_LEVEL) and (CUASR_BENCH_LEVEL >= 2)
static void BM_SM50_device_plus_multiplies_dsrgemm_nn_n_16x64x8_16x32x1_4x4_4x8_1x2(benchmark::State &state) {
  const auto N = static_cast<int>(state.range(0));
  using precision = double;
  using OpClass   = cutlass::arch::OpClassSimt;
  using SmArch    = cutlass::arch::Sm50;

  using ThreadblockShape = cutlass::gemm::GemmShape<16, 64, 8>;
  using WarpShape        = cutlass::gemm::GemmShape<16, 32, 8>;
  using InstructionShape = cutlass::gemm::GemmShape<1, 1, 1>;

  using Config = typename cuasr::gemm::device::DefaultSemiRingConfiguration< //
      precision, precision, precision, precision, OpClass,                   //
      cuasr::plus<precision>, cuasr::multiplies<precision>, SmArch>;

  using AddOp            = Config::AdditionOp;
  using MultOp           = Config::MultiplicationOp;
  using EpilogueOutputOp = Config::EpilogueOutputOp;

  using Srgemm = cuasr::gemm::device::Srgemm<                           //
      AddOp, MultOp,                                                    //
      precision, cutlass::layout::ColumnMajor,                             //
      precision, cutlass::layout::ColumnMajor,                             //
      precision, cutlass::layout::ColumnMajor,                             //
      precision, OpClass, SmArch,                                       //
      ThreadblockShape, WarpShape, InstructionShape, EpilogueOutputOp,  //
      cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>, 2>;

  // setup bench harness
  cuasr::bench::device::BenchHarness<Srgemm> bench({ N, N, N });

  // benchmark loop
  for (auto _ : state) {
    benchmark::DoNotOptimize(bench.run());
    cudaDeviceSynchronize();
  }

  double flops_per_itr = 2.0 * N * N * N;
  state.counters["Flop/s"]
      = benchmark::Counter(flops_per_itr, benchmark::Counter::kIsIterationInvariantRate);
}
BENCHMARK(BM_SM50_device_plus_multiplies_dsrgemm_nn_n_16x64x8_16x32x1_4x4_4x8_1x2)
    ->RangeMultiplier(2)->Range(256, 4096);
#endif

////////////////////////////////////////////////////////////////////////////////
// Elements / Thread:   4 x   8
//    Threads / Warp:   4 x   8
//     Warps / Block:   1 x   2
//       Threadblock:  16 x 128 x   8
#if defined(CUASR_BENCH_LEVEL) and (CUASR_BENCH_LEVEL >= 1)
static void BM_SM50_device_plus_multiplies_dsrgemm_nn_n_16x128x8_16x64x1_4x8_4x8_1x2(benchmark::State &state) {
  const auto N = static_cast<int>(state.range(0));
  using precision = double;
  using OpClass   = cutlass::arch::OpClassSimt;
  using SmArch    = cutlass::arch::Sm50;

  using ThreadblockShape = cutlass::gemm::GemmShape<16, 128, 8>;
  using WarpShape        = cutlass::gemm::GemmShape<16, 64, 8>;
  using InstructionShape = cutlass::gemm::GemmShape<1, 1, 1>;

  using Config = typename cuasr::gemm::device::DefaultSemiRingConfiguration< //
      precision, precision, precision, precision, OpClass,                   //
      cuasr::plus<precision>, cuasr::multiplies<precision>, SmArch>;

  using AddOp            = Config::AdditionOp;
  using MultOp           = Config::MultiplicationOp;
  using EpilogueOutputOp = Config::EpilogueOutputOp;

  using Srgemm = cuasr::gemm::device::Srgemm<                           //
      AddOp, MultOp,                                                    //
      precision, cutlass::layout::ColumnMajor,                             //
      precision, cutlass::layout::ColumnMajor,                             //
      precision, cutlass::layout::ColumnMajor,                             //
      precision, OpClass, SmArch,                                       //
      ThreadblockShape, WarpShape, InstructionShape, EpilogueOutputOp,  //
      cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>, 2>;

  // setup bench harness
  cuasr::bench::device::BenchHarness<Srgemm> bench({ N, N, N });

  // benchmark loop
  for (auto _ : state) {
    benchmark::DoNotOptimize(bench.run());
    cudaDeviceSynchronize();
  }

  double flops_per_itr = 2.0 * N * N * N;
  state.counters["Flop/s"]
      = benchmark::Counter(flops_per_itr, benchmark::Counter::kIsIterationInvariantRate);
}
BENCHMARK(BM_SM50_device_plus_multiplies_dsrgemm_nn_n_16x128x8_16x64x1_4x8_4x8_1x2)
    ->RangeMultiplier(2)->Range(256, 4096);
#endif

////////////////////////////////////////////////////////////////////////////////
// Elements / Thread:   4 x   4
//    Threads / Warp:   8 x   4
//     Warps / Block:   1 x   2
//       Threadblock:  32 x  32 x   8
#if defined(CUASR_BENCH_LEVEL) and (CUASR_BENCH_LEVEL >= 1)
static void BM_SM50_device_plus_multiplies_dsrgemm_nn_n_32x32x8_32x16x1_4x4_8x4_1x2(benchmark::State &state) {
  const auto N = static_cast<int>(state.range(0));
  using precision = double;
  using OpClass   = cutlass::arch::OpClassSimt;
  using SmArch    = cutlass::arch::Sm50;

  using ThreadblockShape = cutlass::gemm::GemmShape<32, 32, 8>;
  using WarpShape        = cutlass::gemm::GemmShape<32, 16, 8>;
  using InstructionShape = cutlass::gemm::GemmShape<1, 1, 1>;

  using Config = typename cuasr::gemm::device::DefaultSemiRingConfiguration< //
      precision, precision, precision, precision, OpClass,                   //
      cuasr::plus<precision>, cuasr::multiplies<precision>, SmArch>;

  using AddOp            = Config::AdditionOp;
  using MultOp           = Config::MultiplicationOp;
  using EpilogueOutputOp = Config::EpilogueOutputOp;

  using Srgemm = cuasr::gemm::device::Srgemm<                           //
      AddOp, MultOp,                                                    //
      precision, cutlass::layout::ColumnMajor,                             //
      precision, cutlass::layout::ColumnMajor,                             //
      precision, cutlass::layout::ColumnMajor,                             //
      precision, OpClass, SmArch,                                       //
      ThreadblockShape, WarpShape, InstructionShape, EpilogueOutputOp,  //
      cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>, 2>;

  // setup bench harness
  cuasr::bench::device::BenchHarness<Srgemm> bench({ N, N, N });

  // benchmark loop
  for (auto _ : state) {
    benchmark::DoNotOptimize(bench.run());
    cudaDeviceSynchronize();
  }

  double flops_per_itr = 2.0 * N * N * N;
  state.counters["Flop/s"]
      = benchmark::Counter(flops_per_itr, benchmark::Counter::kIsIterationInvariantRate);
}
BENCHMARK(BM_SM50_device_plus_multiplies_dsrgemm_nn_n_32x32x8_32x16x1_4x4_8x4_1x2)
    ->RangeMultiplier(2)->Range(256, 4096);
#endif

////////////////////////////////////////////////////////////////////////////////
// Elements / Thread:   8 x   4
//    Threads / Warp:   4 x   8
//     Warps / Block:   1 x   2
//       Threadblock:  32 x  64 x   8
#if defined(CUASR_BENCH_LEVEL) and (CUASR_BENCH_LEVEL >= 1)
static void BM_SM50_device_plus_multiplies_dsrgemm_nn_n_32x64x8_32x32x1_8x4_4x8_1x2(benchmark::State &state) {
  const auto N = static_cast<int>(state.range(0));
  using precision = double;
  using OpClass   = cutlass::arch::OpClassSimt;
  using SmArch    = cutlass::arch::Sm50;

  using ThreadblockShape = cutlass::gemm::GemmShape<32, 64, 8>;
  using WarpShape        = cutlass::gemm::GemmShape<32, 32, 8>;
  using InstructionShape = cutlass::gemm::GemmShape<1, 1, 1>;

  using Config = typename cuasr::gemm::device::DefaultSemiRingConfiguration< //
      precision, precision, precision, precision, OpClass,                   //
      cuasr::plus<precision>, cuasr::multiplies<precision>, SmArch>;

  using AddOp            = Config::AdditionOp;
  using MultOp           = Config::MultiplicationOp;
  using EpilogueOutputOp = Config::EpilogueOutputOp;

  using Srgemm = cuasr::gemm::device::Srgemm<                           //
      AddOp, MultOp,                                                    //
      precision, cutlass::layout::ColumnMajor,                             //
      precision, cutlass::layout::ColumnMajor,                             //
      precision, cutlass::layout::ColumnMajor,                             //
      precision, OpClass, SmArch,                                       //
      ThreadblockShape, WarpShape, InstructionShape, EpilogueOutputOp,  //
      cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>, 2>;

  // setup bench harness
  cuasr::bench::device::BenchHarness<Srgemm> bench({ N, N, N });

  // benchmark loop
  for (auto _ : state) {
    benchmark::DoNotOptimize(bench.run());
    cudaDeviceSynchronize();
  }

  double flops_per_itr = 2.0 * N * N * N;
  state.counters["Flop/s"]
      = benchmark::Counter(flops_per_itr, benchmark::Counter::kIsIterationInvariantRate);
}
BENCHMARK(BM_SM50_device_plus_multiplies_dsrgemm_nn_n_32x64x8_32x32x1_8x4_4x8_1x2)
    ->RangeMultiplier(2)->Range(256, 4096);
#endif

////////////////////////////////////////////////////////////////////////////////
// Elements / Thread:   4 x   4
//    Threads / Warp:   4 x   8
//     Warps / Block:   2 x   1
//       Threadblock:  32 x  32 x   8
#if defined(CUASR_BENCH_LEVEL) and (CUASR_BENCH_LEVEL >= 2)
static void BM_SM50_device_plus_multiplies_dsrgemm_nn_n_32x32x8_16x32x1_4x4_4x8_2x1(benchmark::State &state) {
  const auto N = static_cast<int>(state.range(0));
  using precision = double;
  using OpClass   = cutlass::arch::OpClassSimt;
  using SmArch    = cutlass::arch::Sm50;

  using ThreadblockShape = cutlass::gemm::GemmShape<32, 32, 8>;
  using WarpShape        = cutlass::gemm::GemmShape<16, 32, 8>;
  using InstructionShape = cutlass::gemm::GemmShape<1, 1, 1>;

  using Config = typename cuasr::gemm::device::DefaultSemiRingConfiguration< //
      precision, precision, precision, precision, OpClass,                   //
      cuasr::plus<precision>, cuasr::multiplies<precision>, SmArch>;

  using AddOp            = Config::AdditionOp;
  using MultOp           = Config::MultiplicationOp;
  using EpilogueOutputOp = Config::EpilogueOutputOp;

  using Srgemm = cuasr::gemm::device::Srgemm<                           //
      AddOp, MultOp,                                                    //
      precision, cutlass::layout::ColumnMajor,                             //
      precision, cutlass::layout::ColumnMajor,                             //
      precision, cutlass::layout::ColumnMajor,                             //
      precision, OpClass, SmArch,                                       //
      ThreadblockShape, WarpShape, InstructionShape, EpilogueOutputOp,  //
      cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>, 2>;

  // setup bench harness
  cuasr::bench::device::BenchHarness<Srgemm> bench({ N, N, N });

  // benchmark loop
  for (auto _ : state) {
    benchmark::DoNotOptimize(bench.run());
    cudaDeviceSynchronize();
  }

  double flops_per_itr = 2.0 * N * N * N;
  state.counters["Flop/s"]
      = benchmark::Counter(flops_per_itr, benchmark::Counter::kIsIterationInvariantRate);
}
BENCHMARK(BM_SM50_device_plus_multiplies_dsrgemm_nn_n_32x32x8_16x32x1_4x4_4x8_2x1)
    ->RangeMultiplier(2)->Range(256, 4096);
#endif

////////////////////////////////////////////////////////////////////////////////
// Elements / Thread:   8 x   4
//    Threads / Warp:   4 x   8
//     Warps / Block:   2 x   1
//       Threadblock:  64 x  32 x   8
#if defined(CUASR_BENCH_LEVEL) and (CUASR_BENCH_LEVEL >= 1)
static void BM_SM50_device_plus_multiplies_dsrgemm_nn_n_64x32x8_32x32x1_8x4_4x8_2x1(benchmark::State &state) {
  const auto N = static_cast<int>(state.range(0));
  using precision = double;
  using OpClass   = cutlass::arch::OpClassSimt;
  using SmArch    = cutlass::arch::Sm50;

  using ThreadblockShape = cutlass::gemm::GemmShape<64, 32, 8>;
  using WarpShape        = cutlass::gemm::GemmShape<32, 32, 8>;
  using InstructionShape = cutlass::gemm::GemmShape<1, 1, 1>;

  using Config = typename cuasr::gemm::device::DefaultSemiRingConfiguration< //
      precision, precision, precision, precision, OpClass,                   //
      cuasr::plus<precision>, cuasr::multiplies<precision>, SmArch>;

  using AddOp            = Config::AdditionOp;
  using MultOp           = Config::MultiplicationOp;
  using EpilogueOutputOp = Config::EpilogueOutputOp;

  using Srgemm = cuasr::gemm::device::Srgemm<                           //
      AddOp, MultOp,                                                    //
      precision, cutlass::layout::ColumnMajor,                             //
      precision, cutlass::layout::ColumnMajor,                             //
      precision, cutlass::layout::ColumnMajor,                             //
      precision, OpClass, SmArch,                                       //
      ThreadblockShape, WarpShape, InstructionShape, EpilogueOutputOp,  //
      cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>, 2>;

  // setup bench harness
  cuasr::bench::device::BenchHarness<Srgemm> bench({ N, N, N });

  // benchmark loop
  for (auto _ : state) {
    benchmark::DoNotOptimize(bench.run());
    cudaDeviceSynchronize();
  }

  double flops_per_itr = 2.0 * N * N * N;
  state.counters["Flop/s"]
      = benchmark::Counter(flops_per_itr, benchmark::Counter::kIsIterationInvariantRate);
}
BENCHMARK(BM_SM50_device_plus_multiplies_dsrgemm_nn_n_64x32x8_32x32x1_8x4_4x8_2x1)
    ->RangeMultiplier(2)->Range(256, 4096);
#endif

////////////////////////////////////////////////////////////////////////////////
// Elements / Thread:   2 x   2
//    Threads / Warp:   4 x   8
//     Warps / Block:   2 x   2
//       Threadblock:  16 x  32 x   8
#if defined(CUASR_BENCH_LEVEL) and (CUASR_BENCH_LEVEL >= 2)
static void BM_SM50_device_plus_multiplies_dsrgemm_nn_n_16x32x8_8x16x1_2x2_4x8_2x2(benchmark::State &state) {
  const auto N = static_cast<int>(state.range(0));
  using precision = double;
  using OpClass   = cutlass::arch::OpClassSimt;
  using SmArch    = cutlass::arch::Sm50;

  using ThreadblockShape = cutlass::gemm::GemmShape<16, 32, 8>;
  using WarpShape        = cutlass::gemm::GemmShape<8, 16, 8>;
  using InstructionShape = cutlass::gemm::GemmShape<1, 1, 1>;

  using Config = typename cuasr::gemm::device::DefaultSemiRingConfiguration< //
      precision, precision, precision, precision, OpClass,                   //
      cuasr::plus<precision>, cuasr::multiplies<precision>, SmArch>;

  using AddOp            = Config::AdditionOp;
  using MultOp           = Config::MultiplicationOp;
  using EpilogueOutputOp = Config::EpilogueOutputOp;

  using Srgemm = cuasr::gemm::device::Srgemm<                           //
      AddOp, MultOp,                                                    //
      precision, cutlass::layout::ColumnMajor,                             //
      precision, cutlass::layout::ColumnMajor,                             //
      precision, cutlass::layout::ColumnMajor,                             //
      precision, OpClass, SmArch,                                       //
      ThreadblockShape, WarpShape, InstructionShape, EpilogueOutputOp,  //
      cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>, 2>;

  // setup bench harness
  cuasr::bench::device::BenchHarness<Srgemm> bench({ N, N, N });

  // benchmark loop
  for (auto _ : state) {
    benchmark::DoNotOptimize(bench.run());
    cudaDeviceSynchronize();
  }

  double flops_per_itr = 2.0 * N * N * N;
  state.counters["Flop/s"]
      = benchmark::Counter(flops_per_itr, benchmark::Counter::kIsIterationInvariantRate);
}
BENCHMARK(BM_SM50_device_plus_multiplies_dsrgemm_nn_n_16x32x8_8x16x1_2x2_4x8_2x2)
    ->RangeMultiplier(2)->Range(256, 4096);
#endif

////////////////////////////////////////////////////////////////////////////////
// Elements / Thread:   2 x   4
//    Threads / Warp:   4 x   8
//     Warps / Block:   2 x   2
//       Threadblock:  16 x  64 x   8
#if defined(CUASR_BENCH_LEVEL) and (CUASR_BENCH_LEVEL >= 2)
static void BM_SM50_device_plus_multiplies_dsrgemm_nn_n_16x64x8_8x32x1_2x4_4x8_2x2(benchmark::State &state) {
  const auto N = static_cast<int>(state.range(0));
  using precision = double;
  using OpClass   = cutlass::arch::OpClassSimt;
  using SmArch    = cutlass::arch::Sm50;

  using ThreadblockShape = cutlass::gemm::GemmShape<16, 64, 8>;
  using WarpShape        = cutlass::gemm::GemmShape<8, 32, 8>;
  using InstructionShape = cutlass::gemm::GemmShape<1, 1, 1>;

  using Config = typename cuasr::gemm::device::DefaultSemiRingConfiguration< //
      precision, precision, precision, precision, OpClass,                   //
      cuasr::plus<precision>, cuasr::multiplies<precision>, SmArch>;

  using AddOp            = Config::AdditionOp;
  using MultOp           = Config::MultiplicationOp;
  using EpilogueOutputOp = Config::EpilogueOutputOp;

  using Srgemm = cuasr::gemm::device::Srgemm<                           //
      AddOp, MultOp,                                                    //
      precision, cutlass::layout::ColumnMajor,                             //
      precision, cutlass::layout::ColumnMajor,                             //
      precision, cutlass::layout::ColumnMajor,                             //
      precision, OpClass, SmArch,                                       //
      ThreadblockShape, WarpShape, InstructionShape, EpilogueOutputOp,  //
      cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>, 2>;

  // setup bench harness
  cuasr::bench::device::BenchHarness<Srgemm> bench({ N, N, N });

  // benchmark loop
  for (auto _ : state) {
    benchmark::DoNotOptimize(bench.run());
    cudaDeviceSynchronize();
  }

  double flops_per_itr = 2.0 * N * N * N;
  state.counters["Flop/s"]
      = benchmark::Counter(flops_per_itr, benchmark::Counter::kIsIterationInvariantRate);
}
BENCHMARK(BM_SM50_device_plus_multiplies_dsrgemm_nn_n_16x64x8_8x32x1_2x4_4x8_2x2)
    ->RangeMultiplier(2)->Range(256, 4096);
#endif

////////////////////////////////////////////////////////////////////////////////
// Elements / Thread:   4 x   2
//    Threads / Warp:   4 x   8
//     Warps / Block:   2 x   2
//       Threadblock:  32 x  32 x   8
#if defined(CUASR_BENCH_LEVEL) and (CUASR_BENCH_LEVEL >= 2)
static void BM_SM50_device_plus_multiplies_dsrgemm_nn_n_32x32x8_16x16x1_4x2_4x8_2x2(benchmark::State &state) {
  const auto N = static_cast<int>(state.range(0));
  using precision = double;
  using OpClass   = cutlass::arch::OpClassSimt;
  using SmArch    = cutlass::arch::Sm50;

  using ThreadblockShape = cutlass::gemm::GemmShape<32, 32, 8>;
  using WarpShape        = cutlass::gemm::GemmShape<16, 16, 8>;
  using InstructionShape = cutlass::gemm::GemmShape<1, 1, 1>;

  using Config = typename cuasr::gemm::device::DefaultSemiRingConfiguration< //
      precision, precision, precision, precision, OpClass,                   //
      cuasr::plus<precision>, cuasr::multiplies<precision>, SmArch>;

  using AddOp            = Config::AdditionOp;
  using MultOp           = Config::MultiplicationOp;
  using EpilogueOutputOp = Config::EpilogueOutputOp;

  using Srgemm = cuasr::gemm::device::Srgemm<                           //
      AddOp, MultOp,                                                    //
      precision, cutlass::layout::ColumnMajor,                             //
      precision, cutlass::layout::ColumnMajor,                             //
      precision, cutlass::layout::ColumnMajor,                             //
      precision, OpClass, SmArch,                                       //
      ThreadblockShape, WarpShape, InstructionShape, EpilogueOutputOp,  //
      cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>, 2>;

  // setup bench harness
  cuasr::bench::device::BenchHarness<Srgemm> bench({ N, N, N });

  // benchmark loop
  for (auto _ : state) {
    benchmark::DoNotOptimize(bench.run());
    cudaDeviceSynchronize();
  }

  double flops_per_itr = 2.0 * N * N * N;
  state.counters["Flop/s"]
      = benchmark::Counter(flops_per_itr, benchmark::Counter::kIsIterationInvariantRate);
}
BENCHMARK(BM_SM50_device_plus_multiplies_dsrgemm_nn_n_32x32x8_16x16x1_4x2_4x8_2x2)
    ->RangeMultiplier(2)->Range(256, 4096);
#endif

////////////////////////////////////////////////////////////////////////////////
// Elements / Thread:   4 x   4
//    Threads / Warp:   4 x   8
//     Warps / Block:   2 x   2
//       Threadblock:  32 x  64 x   8
#if defined(CUASR_BENCH_LEVEL) and (CUASR_BENCH_LEVEL >= 2)
static void BM_SM50_device_plus_multiplies_dsrgemm_nn_n_32x64x8_16x32x1_4x4_4x8_2x2(benchmark::State &state) {
  const auto N = static_cast<int>(state.range(0));
  using precision = double;
  using OpClass   = cutlass::arch::OpClassSimt;
  using SmArch    = cutlass::arch::Sm50;

  using ThreadblockShape = cutlass::gemm::GemmShape<32, 64, 8>;
  using WarpShape        = cutlass::gemm::GemmShape<16, 32, 8>;
  using InstructionShape = cutlass::gemm::GemmShape<1, 1, 1>;

  using Config = typename cuasr::gemm::device::DefaultSemiRingConfiguration< //
      precision, precision, precision, precision, OpClass,                   //
      cuasr::plus<precision>, cuasr::multiplies<precision>, SmArch>;

  using AddOp            = Config::AdditionOp;
  using MultOp           = Config::MultiplicationOp;
  using EpilogueOutputOp = Config::EpilogueOutputOp;

  using Srgemm = cuasr::gemm::device::Srgemm<                           //
      AddOp, MultOp,                                                    //
      precision, cutlass::layout::ColumnMajor,                             //
      precision, cutlass::layout::ColumnMajor,                             //
      precision, cutlass::layout::ColumnMajor,                             //
      precision, OpClass, SmArch,                                       //
      ThreadblockShape, WarpShape, InstructionShape, EpilogueOutputOp,  //
      cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>, 2>;

  // setup bench harness
  cuasr::bench::device::BenchHarness<Srgemm> bench({ N, N, N });

  // benchmark loop
  for (auto _ : state) {
    benchmark::DoNotOptimize(bench.run());
    cudaDeviceSynchronize();
  }

  double flops_per_itr = 2.0 * N * N * N;
  state.counters["Flop/s"]
      = benchmark::Counter(flops_per_itr, benchmark::Counter::kIsIterationInvariantRate);
}
BENCHMARK(BM_SM50_device_plus_multiplies_dsrgemm_nn_n_32x64x8_16x32x1_4x4_4x8_2x2)
    ->RangeMultiplier(2)->Range(256, 4096);
#endif

////////////////////////////////////////////////////////////////////////////////
// Elements / Thread:   4 x   8
//    Threads / Warp:   4 x   8
//     Warps / Block:   2 x   2
//       Threadblock:  32 x 128 x   8
#if defined(CUASR_BENCH_LEVEL) and (CUASR_BENCH_LEVEL >= 1)
static void BM_SM50_device_plus_multiplies_dsrgemm_nn_n_32x128x8_16x64x1_4x8_4x8_2x2(benchmark::State &state) {
  const auto N = static_cast<int>(state.range(0));
  using precision = double;
  using OpClass   = cutlass::arch::OpClassSimt;
  using SmArch    = cutlass::arch::Sm50;

  using ThreadblockShape = cutlass::gemm::GemmShape<32, 128, 8>;
  using WarpShape        = cutlass::gemm::GemmShape<16, 64, 8>;
  using InstructionShape = cutlass::gemm::GemmShape<1, 1, 1>;

  using Config = typename cuasr::gemm::device::DefaultSemiRingConfiguration< //
      precision, precision, precision, precision, OpClass,                   //
      cuasr::plus<precision>, cuasr::multiplies<precision>, SmArch>;

  using AddOp            = Config::AdditionOp;
  using MultOp           = Config::MultiplicationOp;
  using EpilogueOutputOp = Config::EpilogueOutputOp;

  using Srgemm = cuasr::gemm::device::Srgemm<                           //
      AddOp, MultOp,                                                    //
      precision, cutlass::layout::ColumnMajor,                             //
      precision, cutlass::layout::ColumnMajor,                             //
      precision, cutlass::layout::ColumnMajor,                             //
      precision, OpClass, SmArch,                                       //
      ThreadblockShape, WarpShape, InstructionShape, EpilogueOutputOp,  //
      cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>, 2>;

  // setup bench harness
  cuasr::bench::device::BenchHarness<Srgemm> bench({ N, N, N });

  // benchmark loop
  for (auto _ : state) {
    benchmark::DoNotOptimize(bench.run());
    cudaDeviceSynchronize();
  }

  double flops_per_itr = 2.0 * N * N * N;
  state.counters["Flop/s"]
      = benchmark::Counter(flops_per_itr, benchmark::Counter::kIsIterationInvariantRate);
}
BENCHMARK(BM_SM50_device_plus_multiplies_dsrgemm_nn_n_32x128x8_16x64x1_4x8_4x8_2x2)
    ->RangeMultiplier(2)->Range(256, 4096);
#endif

////////////////////////////////////////////////////////////////////////////////
// Elements / Thread:   4 x   4
//    Threads / Warp:   8 x   4
//     Warps / Block:   2 x   2
//       Threadblock:  64 x  32 x   8
#if defined(CUASR_BENCH_LEVEL) and (CUASR_BENCH_LEVEL >= 2)
static void BM_SM50_device_plus_multiplies_dsrgemm_nn_n_64x32x8_32x16x1_4x4_8x4_2x2(benchmark::State &state) {
  const auto N = static_cast<int>(state.range(0));
  using precision = double;
  using OpClass   = cutlass::arch::OpClassSimt;
  using SmArch    = cutlass::arch::Sm50;

  using ThreadblockShape = cutlass::gemm::GemmShape<64, 32, 8>;
  using WarpShape        = cutlass::gemm::GemmShape<32, 16, 8>;
  using InstructionShape = cutlass::gemm::GemmShape<1, 1, 1>;

  using Config = typename cuasr::gemm::device::DefaultSemiRingConfiguration< //
      precision, precision, precision, precision, OpClass,                   //
      cuasr::plus<precision>, cuasr::multiplies<precision>, SmArch>;

  using AddOp            = Config::AdditionOp;
  using MultOp           = Config::MultiplicationOp;
  using EpilogueOutputOp = Config::EpilogueOutputOp;

  using Srgemm = cuasr::gemm::device::Srgemm<                           //
      AddOp, MultOp,                                                    //
      precision, cutlass::layout::ColumnMajor,                             //
      precision, cutlass::layout::ColumnMajor,                             //
      precision, cutlass::layout::ColumnMajor,                             //
      precision, OpClass, SmArch,                                       //
      ThreadblockShape, WarpShape, InstructionShape, EpilogueOutputOp,  //
      cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>, 2>;

  // setup bench harness
  cuasr::bench::device::BenchHarness<Srgemm> bench({ N, N, N });

  // benchmark loop
  for (auto _ : state) {
    benchmark::DoNotOptimize(bench.run());
    cudaDeviceSynchronize();
  }

  double flops_per_itr = 2.0 * N * N * N;
  state.counters["Flop/s"]
      = benchmark::Counter(flops_per_itr, benchmark::Counter::kIsIterationInvariantRate);
}
BENCHMARK(BM_SM50_device_plus_multiplies_dsrgemm_nn_n_64x32x8_32x16x1_4x4_8x4_2x2)
    ->RangeMultiplier(2)->Range(256, 4096);
#endif

////////////////////////////////////////////////////////////////////////////////
// Elements / Thread:   8 x   4
//    Threads / Warp:   4 x   8
//     Warps / Block:   2 x   2
//       Threadblock:  64 x  64 x   8
#if defined(CUASR_BENCH_LEVEL) and (CUASR_BENCH_LEVEL >= 0)
static void BM_SM50_device_plus_multiplies_dsrgemm_nn_n_64x64x8_32x32x1_8x4_4x8_2x2(benchmark::State &state) {
  const auto N = static_cast<int>(state.range(0));
  using precision = double;
  using OpClass   = cutlass::arch::OpClassSimt;
  using SmArch    = cutlass::arch::Sm50;

  using ThreadblockShape = cutlass::gemm::GemmShape<64, 64, 8>;
  using WarpShape        = cutlass::gemm::GemmShape<32, 32, 8>;
  using InstructionShape = cutlass::gemm::GemmShape<1, 1, 1>;

  using Config = typename cuasr::gemm::device::DefaultSemiRingConfiguration< //
      precision, precision, precision, precision, OpClass,                   //
      cuasr::plus<precision>, cuasr::multiplies<precision>, SmArch>;

  using AddOp            = Config::AdditionOp;
  using MultOp           = Config::MultiplicationOp;
  using EpilogueOutputOp = Config::EpilogueOutputOp;

  using Srgemm = cuasr::gemm::device::Srgemm<                           //
      AddOp, MultOp,                                                    //
      precision, cutlass::layout::ColumnMajor,                             //
      precision, cutlass::layout::ColumnMajor,                             //
      precision, cutlass::layout::ColumnMajor,                             //
      precision, OpClass, SmArch,                                       //
      ThreadblockShape, WarpShape, InstructionShape, EpilogueOutputOp,  //
      cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>, 2>;

  // setup bench harness
  cuasr::bench::device::BenchHarness<Srgemm> bench({ N, N, N });

  // benchmark loop
  for (auto _ : state) {
    benchmark::DoNotOptimize(bench.run());
    cudaDeviceSynchronize();
  }

  double flops_per_itr = 2.0 * N * N * N;
  state.counters["Flop/s"]
      = benchmark::Counter(flops_per_itr, benchmark::Counter::kIsIterationInvariantRate);
}
BENCHMARK(BM_SM50_device_plus_multiplies_dsrgemm_nn_n_64x64x8_32x32x1_8x4_4x8_2x2)
    ->RangeMultiplier(2)->Range(256, 4096);
#endif

////////////////////////////////////////////////////////////////////////////////
// Elements / Thread:   8 x   4
//    Threads / Warp:   8 x   4
//     Warps / Block:   2 x   2
//       Threadblock: 128 x  32 x   8
#if defined(CUASR_BENCH_LEVEL) and (CUASR_BENCH_LEVEL >= 1)
static void BM_SM50_device_plus_multiplies_dsrgemm_nn_n_128x32x8_64x16x1_8x4_8x4_2x2(benchmark::State &state) {
  const auto N = static_cast<int>(state.range(0));
  using precision = double;
  using OpClass   = cutlass::arch::OpClassSimt;
  using SmArch    = cutlass::arch::Sm50;

  using ThreadblockShape = cutlass::gemm::GemmShape<128, 32, 8>;
  using WarpShape        = cutlass::gemm::GemmShape<64, 16, 8>;
  using InstructionShape = cutlass::gemm::GemmShape<1, 1, 1>;

  using Config = typename cuasr::gemm::device::DefaultSemiRingConfiguration< //
      precision, precision, precision, precision, OpClass,                   //
      cuasr::plus<precision>, cuasr::multiplies<precision>, SmArch>;

  using AddOp            = Config::AdditionOp;
  using MultOp           = Config::MultiplicationOp;
  using EpilogueOutputOp = Config::EpilogueOutputOp;

  using Srgemm = cuasr::gemm::device::Srgemm<                           //
      AddOp, MultOp,                                                    //
      precision, cutlass::layout::ColumnMajor,                             //
      precision, cutlass::layout::ColumnMajor,                             //
      precision, cutlass::layout::ColumnMajor,                             //
      precision, OpClass, SmArch,                                       //
      ThreadblockShape, WarpShape, InstructionShape, EpilogueOutputOp,  //
      cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>, 2>;

  // setup bench harness
  cuasr::bench::device::BenchHarness<Srgemm> bench({ N, N, N });

  // benchmark loop
  for (auto _ : state) {
    benchmark::DoNotOptimize(bench.run());
    cudaDeviceSynchronize();
  }

  double flops_per_itr = 2.0 * N * N * N;
  state.counters["Flop/s"]
      = benchmark::Counter(flops_per_itr, benchmark::Counter::kIsIterationInvariantRate);
}
BENCHMARK(BM_SM50_device_plus_multiplies_dsrgemm_nn_n_128x32x8_64x16x1_8x4_8x4_2x2)
    ->RangeMultiplier(2)->Range(256, 4096);
#endif

////////////////////////////////////////////////////////////////////////////////
// Elements / Thread:   2 x   2
//    Threads / Warp:   4 x   8
//     Warps / Block:   2 x   4
//       Threadblock:  16 x  64 x  16
#if defined(CUASR_BENCH_LEVEL) and (CUASR_BENCH_LEVEL >= 2)
static void BM_SM50_device_plus_multiplies_dsrgemm_nn_n_16x64x16_8x16x1_2x2_4x8_2x4(benchmark::State &state) {
  const auto N = static_cast<int>(state.range(0));
  using precision = double;
  using OpClass   = cutlass::arch::OpClassSimt;
  using SmArch    = cutlass::arch::Sm50;

  using ThreadblockShape = cutlass::gemm::GemmShape<16, 64, 16>;
  using WarpShape        = cutlass::gemm::GemmShape<8, 16, 16>;
  using InstructionShape = cutlass::gemm::GemmShape<1, 1, 1>;

  using Config = typename cuasr::gemm::device::DefaultSemiRingConfiguration< //
      precision, precision, precision, precision, OpClass,                   //
      cuasr::plus<precision>, cuasr::multiplies<precision>, SmArch>;

  using AddOp            = Config::AdditionOp;
  using MultOp           = Config::MultiplicationOp;
  using EpilogueOutputOp = Config::EpilogueOutputOp;

  using Srgemm = cuasr::gemm::device::Srgemm<                           //
      AddOp, MultOp,                                                    //
      precision, cutlass::layout::ColumnMajor,                             //
      precision, cutlass::layout::ColumnMajor,                             //
      precision, cutlass::layout::ColumnMajor,                             //
      precision, OpClass, SmArch,                                       //
      ThreadblockShape, WarpShape, InstructionShape, EpilogueOutputOp,  //
      cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>, 2>;

  // setup bench harness
  cuasr::bench::device::BenchHarness<Srgemm> bench({ N, N, N });

  // benchmark loop
  for (auto _ : state) {
    benchmark::DoNotOptimize(bench.run());
    cudaDeviceSynchronize();
  }

  double flops_per_itr = 2.0 * N * N * N;
  state.counters["Flop/s"]
      = benchmark::Counter(flops_per_itr, benchmark::Counter::kIsIterationInvariantRate);
}
BENCHMARK(BM_SM50_device_plus_multiplies_dsrgemm_nn_n_16x64x16_8x16x1_2x2_4x8_2x4)
    ->RangeMultiplier(2)->Range(256, 4096);
#endif

////////////////////////////////////////////////////////////////////////////////
// Elements / Thread:   2 x   4
//    Threads / Warp:   4 x   8
//     Warps / Block:   2 x   4
//       Threadblock:  16 x 128 x  16
#if defined(CUASR_BENCH_LEVEL) and (CUASR_BENCH_LEVEL >= 2)
static void BM_SM50_device_plus_multiplies_dsrgemm_nn_n_16x128x16_8x32x1_2x4_4x8_2x4(benchmark::State &state) {
  const auto N = static_cast<int>(state.range(0));
  using precision = double;
  using OpClass   = cutlass::arch::OpClassSimt;
  using SmArch    = cutlass::arch::Sm50;

  using ThreadblockShape = cutlass::gemm::GemmShape<16, 128, 16>;
  using WarpShape        = cutlass::gemm::GemmShape<8, 32, 16>;
  using InstructionShape = cutlass::gemm::GemmShape<1, 1, 1>;

  using Config = typename cuasr::gemm::device::DefaultSemiRingConfiguration< //
      precision, precision, precision, precision, OpClass,                   //
      cuasr::plus<precision>, cuasr::multiplies<precision>, SmArch>;

  using AddOp            = Config::AdditionOp;
  using MultOp           = Config::MultiplicationOp;
  using EpilogueOutputOp = Config::EpilogueOutputOp;

  using Srgemm = cuasr::gemm::device::Srgemm<                           //
      AddOp, MultOp,                                                    //
      precision, cutlass::layout::ColumnMajor,                             //
      precision, cutlass::layout::ColumnMajor,                             //
      precision, cutlass::layout::ColumnMajor,                             //
      precision, OpClass, SmArch,                                       //
      ThreadblockShape, WarpShape, InstructionShape, EpilogueOutputOp,  //
      cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>, 2>;

  // setup bench harness
  cuasr::bench::device::BenchHarness<Srgemm> bench({ N, N, N });

  // benchmark loop
  for (auto _ : state) {
    benchmark::DoNotOptimize(bench.run());
    cudaDeviceSynchronize();
  }

  double flops_per_itr = 2.0 * N * N * N;
  state.counters["Flop/s"]
      = benchmark::Counter(flops_per_itr, benchmark::Counter::kIsIterationInvariantRate);
}
BENCHMARK(BM_SM50_device_plus_multiplies_dsrgemm_nn_n_16x128x16_8x32x1_2x4_4x8_2x4)
    ->RangeMultiplier(2)->Range(256, 4096);
#endif

////////////////////////////////////////////////////////////////////////////////
// Elements / Thread:   2 x   2
//    Threads / Warp:   8 x   4
//     Warps / Block:   2 x   4
//       Threadblock:  32 x  32 x   8
#if defined(CUASR_BENCH_LEVEL) and (CUASR_BENCH_LEVEL >= 2)
static void BM_SM50_device_plus_multiplies_dsrgemm_nn_n_32x32x8_16x8x1_2x2_8x4_2x4(benchmark::State &state) {
  const auto N = static_cast<int>(state.range(0));
  using precision = double;
  using OpClass   = cutlass::arch::OpClassSimt;
  using SmArch    = cutlass::arch::Sm50;

  using ThreadblockShape = cutlass::gemm::GemmShape<32, 32, 8>;
  using WarpShape        = cutlass::gemm::GemmShape<16, 8, 8>;
  using InstructionShape = cutlass::gemm::GemmShape<1, 1, 1>;

  using Config = typename cuasr::gemm::device::DefaultSemiRingConfiguration< //
      precision, precision, precision, precision, OpClass,                   //
      cuasr::plus<precision>, cuasr::multiplies<precision>, SmArch>;

  using AddOp            = Config::AdditionOp;
  using MultOp           = Config::MultiplicationOp;
  using EpilogueOutputOp = Config::EpilogueOutputOp;

  using Srgemm = cuasr::gemm::device::Srgemm<                           //
      AddOp, MultOp,                                                    //
      precision, cutlass::layout::ColumnMajor,                             //
      precision, cutlass::layout::ColumnMajor,                             //
      precision, cutlass::layout::ColumnMajor,                             //
      precision, OpClass, SmArch,                                       //
      ThreadblockShape, WarpShape, InstructionShape, EpilogueOutputOp,  //
      cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>, 2>;

  // setup bench harness
  cuasr::bench::device::BenchHarness<Srgemm> bench({ N, N, N });

  // benchmark loop
  for (auto _ : state) {
    benchmark::DoNotOptimize(bench.run());
    cudaDeviceSynchronize();
  }

  double flops_per_itr = 2.0 * N * N * N;
  state.counters["Flop/s"]
      = benchmark::Counter(flops_per_itr, benchmark::Counter::kIsIterationInvariantRate);
}
BENCHMARK(BM_SM50_device_plus_multiplies_dsrgemm_nn_n_32x32x8_16x8x1_2x2_8x4_2x4)
    ->RangeMultiplier(2)->Range(256, 4096);
#endif

////////////////////////////////////////////////////////////////////////////////
// Elements / Thread:   4 x   2
//    Threads / Warp:   4 x   8
//     Warps / Block:   2 x   4
//       Threadblock:  32 x  64 x   8
#if defined(CUASR_BENCH_LEVEL) and (CUASR_BENCH_LEVEL >= 2)
static void BM_SM50_device_plus_multiplies_dsrgemm_nn_n_32x64x8_16x16x1_4x2_4x8_2x4(benchmark::State &state) {
  const auto N = static_cast<int>(state.range(0));
  using precision = double;
  using OpClass   = cutlass::arch::OpClassSimt;
  using SmArch    = cutlass::arch::Sm50;

  using ThreadblockShape = cutlass::gemm::GemmShape<32, 64, 8>;
  using WarpShape        = cutlass::gemm::GemmShape<16, 16, 8>;
  using InstructionShape = cutlass::gemm::GemmShape<1, 1, 1>;

  using Config = typename cuasr::gemm::device::DefaultSemiRingConfiguration< //
      precision, precision, precision, precision, OpClass,                   //
      cuasr::plus<precision>, cuasr::multiplies<precision>, SmArch>;

  using AddOp            = Config::AdditionOp;
  using MultOp           = Config::MultiplicationOp;
  using EpilogueOutputOp = Config::EpilogueOutputOp;

  using Srgemm = cuasr::gemm::device::Srgemm<                           //
      AddOp, MultOp,                                                    //
      precision, cutlass::layout::ColumnMajor,                             //
      precision, cutlass::layout::ColumnMajor,                             //
      precision, cutlass::layout::ColumnMajor,                             //
      precision, OpClass, SmArch,                                       //
      ThreadblockShape, WarpShape, InstructionShape, EpilogueOutputOp,  //
      cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>, 2>;

  // setup bench harness
  cuasr::bench::device::BenchHarness<Srgemm> bench({ N, N, N });

  // benchmark loop
  for (auto _ : state) {
    benchmark::DoNotOptimize(bench.run());
    cudaDeviceSynchronize();
  }

  double flops_per_itr = 2.0 * N * N * N;
  state.counters["Flop/s"]
      = benchmark::Counter(flops_per_itr, benchmark::Counter::kIsIterationInvariantRate);
}
BENCHMARK(BM_SM50_device_plus_multiplies_dsrgemm_nn_n_32x64x8_16x16x1_4x2_4x8_2x4)
    ->RangeMultiplier(2)->Range(256, 4096);
#endif

////////////////////////////////////////////////////////////////////////////////
// Elements / Thread:   4 x   4
//    Threads / Warp:   4 x   8
//     Warps / Block:   2 x   4
//       Threadblock:  32 x 128 x   8
#if defined(CUASR_BENCH_LEVEL) and (CUASR_BENCH_LEVEL >= 2)
static void BM_SM50_device_plus_multiplies_dsrgemm_nn_n_32x128x8_16x32x1_4x4_4x8_2x4(benchmark::State &state) {
  const auto N = static_cast<int>(state.range(0));
  using precision = double;
  using OpClass   = cutlass::arch::OpClassSimt;
  using SmArch    = cutlass::arch::Sm50;

  using ThreadblockShape = cutlass::gemm::GemmShape<32, 128, 8>;
  using WarpShape        = cutlass::gemm::GemmShape<16, 32, 8>;
  using InstructionShape = cutlass::gemm::GemmShape<1, 1, 1>;

  using Config = typename cuasr::gemm::device::DefaultSemiRingConfiguration< //
      precision, precision, precision, precision, OpClass,                   //
      cuasr::plus<precision>, cuasr::multiplies<precision>, SmArch>;

  using AddOp            = Config::AdditionOp;
  using MultOp           = Config::MultiplicationOp;
  using EpilogueOutputOp = Config::EpilogueOutputOp;

  using Srgemm = cuasr::gemm::device::Srgemm<                           //
      AddOp, MultOp,                                                    //
      precision, cutlass::layout::ColumnMajor,                             //
      precision, cutlass::layout::ColumnMajor,                             //
      precision, cutlass::layout::ColumnMajor,                             //
      precision, OpClass, SmArch,                                       //
      ThreadblockShape, WarpShape, InstructionShape, EpilogueOutputOp,  //
      cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>, 2>;

  // setup bench harness
  cuasr::bench::device::BenchHarness<Srgemm> bench({ N, N, N });

  // benchmark loop
  for (auto _ : state) {
    benchmark::DoNotOptimize(bench.run());
    cudaDeviceSynchronize();
  }

  double flops_per_itr = 2.0 * N * N * N;
  state.counters["Flop/s"]
      = benchmark::Counter(flops_per_itr, benchmark::Counter::kIsIterationInvariantRate);
}
BENCHMARK(BM_SM50_device_plus_multiplies_dsrgemm_nn_n_32x128x8_16x32x1_4x4_4x8_2x4)
    ->RangeMultiplier(2)->Range(256, 4096);
#endif

////////////////////////////////////////////////////////////////////////////////
// Elements / Thread:   4 x   4
//    Threads / Warp:   8 x   4
//     Warps / Block:   2 x   4
//       Threadblock:  64 x  64 x   8
#if defined(CUASR_BENCH_LEVEL) and (CUASR_BENCH_LEVEL >= 1)
static void BM_SM50_device_plus_multiplies_dsrgemm_nn_n_64x64x8_32x16x1_4x4_8x4_2x4(benchmark::State &state) {
  const auto N = static_cast<int>(state.range(0));
  using precision = double;
  using OpClass   = cutlass::arch::OpClassSimt;
  using SmArch    = cutlass::arch::Sm50;

  using ThreadblockShape = cutlass::gemm::GemmShape<64, 64, 8>;
  using WarpShape        = cutlass::gemm::GemmShape<32, 16, 8>;
  using InstructionShape = cutlass::gemm::GemmShape<1, 1, 1>;

  using Config = typename cuasr::gemm::device::DefaultSemiRingConfiguration< //
      precision, precision, precision, precision, OpClass,                   //
      cuasr::plus<precision>, cuasr::multiplies<precision>, SmArch>;

  using AddOp            = Config::AdditionOp;
  using MultOp           = Config::MultiplicationOp;
  using EpilogueOutputOp = Config::EpilogueOutputOp;

  using Srgemm = cuasr::gemm::device::Srgemm<                           //
      AddOp, MultOp,                                                    //
      precision, cutlass::layout::ColumnMajor,                             //
      precision, cutlass::layout::ColumnMajor,                             //
      precision, cutlass::layout::ColumnMajor,                             //
      precision, OpClass, SmArch,                                       //
      ThreadblockShape, WarpShape, InstructionShape, EpilogueOutputOp,  //
      cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>, 2>;

  // setup bench harness
  cuasr::bench::device::BenchHarness<Srgemm> bench({ N, N, N });

  // benchmark loop
  for (auto _ : state) {
    benchmark::DoNotOptimize(bench.run());
    cudaDeviceSynchronize();
  }

  double flops_per_itr = 2.0 * N * N * N;
  state.counters["Flop/s"]
      = benchmark::Counter(flops_per_itr, benchmark::Counter::kIsIterationInvariantRate);
}
BENCHMARK(BM_SM50_device_plus_multiplies_dsrgemm_nn_n_64x64x8_32x16x1_4x4_8x4_2x4)
    ->RangeMultiplier(2)->Range(256, 4096);
#endif

////////////////////////////////////////////////////////////////////////////////
// Elements / Thread:   2 x   2
//    Threads / Warp:   4 x   8
//     Warps / Block:   4 x   2
//       Threadblock:  32 x  32 x   8
#if defined(CUASR_BENCH_LEVEL) and (CUASR_BENCH_LEVEL >= 2)
static void BM_SM50_device_plus_multiplies_dsrgemm_nn_n_32x32x8_8x16x1_2x2_4x8_4x2(benchmark::State &state) {
  const auto N = static_cast<int>(state.range(0));
  using precision = double;
  using OpClass   = cutlass::arch::OpClassSimt;
  using SmArch    = cutlass::arch::Sm50;

  using ThreadblockShape = cutlass::gemm::GemmShape<32, 32, 8>;
  using WarpShape        = cutlass::gemm::GemmShape<8, 16, 8>;
  using InstructionShape = cutlass::gemm::GemmShape<1, 1, 1>;

  using Config = typename cuasr::gemm::device::DefaultSemiRingConfiguration< //
      precision, precision, precision, precision, OpClass,                   //
      cuasr::plus<precision>, cuasr::multiplies<precision>, SmArch>;

  using AddOp            = Config::AdditionOp;
  using MultOp           = Config::MultiplicationOp;
  using EpilogueOutputOp = Config::EpilogueOutputOp;

  using Srgemm = cuasr::gemm::device::Srgemm<                           //
      AddOp, MultOp,                                                    //
      precision, cutlass::layout::ColumnMajor,                             //
      precision, cutlass::layout::ColumnMajor,                             //
      precision, cutlass::layout::ColumnMajor,                             //
      precision, OpClass, SmArch,                                       //
      ThreadblockShape, WarpShape, InstructionShape, EpilogueOutputOp,  //
      cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>, 2>;

  // setup bench harness
  cuasr::bench::device::BenchHarness<Srgemm> bench({ N, N, N });

  // benchmark loop
  for (auto _ : state) {
    benchmark::DoNotOptimize(bench.run());
    cudaDeviceSynchronize();
  }

  double flops_per_itr = 2.0 * N * N * N;
  state.counters["Flop/s"]
      = benchmark::Counter(flops_per_itr, benchmark::Counter::kIsIterationInvariantRate);
}
BENCHMARK(BM_SM50_device_plus_multiplies_dsrgemm_nn_n_32x32x8_8x16x1_2x2_4x8_4x2)
    ->RangeMultiplier(2)->Range(256, 4096);
#endif

////////////////////////////////////////////////////////////////////////////////
// Elements / Thread:   4 x   2
//    Threads / Warp:   4 x   8
//     Warps / Block:   4 x   2
//       Threadblock:  64 x  32 x   8
#if defined(CUASR_BENCH_LEVEL) and (CUASR_BENCH_LEVEL >= 2)
static void BM_SM50_device_plus_multiplies_dsrgemm_nn_n_64x32x8_16x16x1_4x2_4x8_4x2(benchmark::State &state) {
  const auto N = static_cast<int>(state.range(0));
  using precision = double;
  using OpClass   = cutlass::arch::OpClassSimt;
  using SmArch    = cutlass::arch::Sm50;

  using ThreadblockShape = cutlass::gemm::GemmShape<64, 32, 8>;
  using WarpShape        = cutlass::gemm::GemmShape<16, 16, 8>;
  using InstructionShape = cutlass::gemm::GemmShape<1, 1, 1>;

  using Config = typename cuasr::gemm::device::DefaultSemiRingConfiguration< //
      precision, precision, precision, precision, OpClass,                   //
      cuasr::plus<precision>, cuasr::multiplies<precision>, SmArch>;

  using AddOp            = Config::AdditionOp;
  using MultOp           = Config::MultiplicationOp;
  using EpilogueOutputOp = Config::EpilogueOutputOp;

  using Srgemm = cuasr::gemm::device::Srgemm<                           //
      AddOp, MultOp,                                                    //
      precision, cutlass::layout::ColumnMajor,                             //
      precision, cutlass::layout::ColumnMajor,                             //
      precision, cutlass::layout::ColumnMajor,                             //
      precision, OpClass, SmArch,                                       //
      ThreadblockShape, WarpShape, InstructionShape, EpilogueOutputOp,  //
      cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>, 2>;

  // setup bench harness
  cuasr::bench::device::BenchHarness<Srgemm> bench({ N, N, N });

  // benchmark loop
  for (auto _ : state) {
    benchmark::DoNotOptimize(bench.run());
    cudaDeviceSynchronize();
  }

  double flops_per_itr = 2.0 * N * N * N;
  state.counters["Flop/s"]
      = benchmark::Counter(flops_per_itr, benchmark::Counter::kIsIterationInvariantRate);
}
BENCHMARK(BM_SM50_device_plus_multiplies_dsrgemm_nn_n_64x32x8_16x16x1_4x2_4x8_4x2)
    ->RangeMultiplier(2)->Range(256, 4096);
#endif

////////////////////////////////////////////////////////////////////////////////
// Elements / Thread:   4 x   4
//    Threads / Warp:   4 x   8
//     Warps / Block:   4 x   2
//       Threadblock:  64 x  64 x   8
#if defined(CUASR_BENCH_LEVEL) and (CUASR_BENCH_LEVEL >= 2)
static void BM_SM50_device_plus_multiplies_dsrgemm_nn_n_64x64x8_16x32x1_4x4_4x8_4x2(benchmark::State &state) {
  const auto N = static_cast<int>(state.range(0));
  using precision = double;
  using OpClass   = cutlass::arch::OpClassSimt;
  using SmArch    = cutlass::arch::Sm50;

  using ThreadblockShape = cutlass::gemm::GemmShape<64, 64, 8>;
  using WarpShape        = cutlass::gemm::GemmShape<16, 32, 8>;
  using InstructionShape = cutlass::gemm::GemmShape<1, 1, 1>;

  using Config = typename cuasr::gemm::device::DefaultSemiRingConfiguration< //
      precision, precision, precision, precision, OpClass,                   //
      cuasr::plus<precision>, cuasr::multiplies<precision>, SmArch>;

  using AddOp            = Config::AdditionOp;
  using MultOp           = Config::MultiplicationOp;
  using EpilogueOutputOp = Config::EpilogueOutputOp;

  using Srgemm = cuasr::gemm::device::Srgemm<                           //
      AddOp, MultOp,                                                    //
      precision, cutlass::layout::ColumnMajor,                             //
      precision, cutlass::layout::ColumnMajor,                             //
      precision, cutlass::layout::ColumnMajor,                             //
      precision, OpClass, SmArch,                                       //
      ThreadblockShape, WarpShape, InstructionShape, EpilogueOutputOp,  //
      cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>, 2>;

  // setup bench harness
  cuasr::bench::device::BenchHarness<Srgemm> bench({ N, N, N });

  // benchmark loop
  for (auto _ : state) {
    benchmark::DoNotOptimize(bench.run());
    cudaDeviceSynchronize();
  }

  double flops_per_itr = 2.0 * N * N * N;
  state.counters["Flop/s"]
      = benchmark::Counter(flops_per_itr, benchmark::Counter::kIsIterationInvariantRate);
}
BENCHMARK(BM_SM50_device_plus_multiplies_dsrgemm_nn_n_64x64x8_16x32x1_4x4_4x8_4x2)
    ->RangeMultiplier(2)->Range(256, 4096);
#endif

////////////////////////////////////////////////////////////////////////////////
// Elements / Thread:   4 x   4
//    Threads / Warp:   8 x   4
//     Warps / Block:   4 x   2
//       Threadblock: 128 x  32 x   8
#if defined(CUASR_BENCH_LEVEL) and (CUASR_BENCH_LEVEL >= 2)
static void BM_SM50_device_plus_multiplies_dsrgemm_nn_n_128x32x8_32x16x1_4x4_8x4_4x2(benchmark::State &state) {
  const auto N = static_cast<int>(state.range(0));
  using precision = double;
  using OpClass   = cutlass::arch::OpClassSimt;
  using SmArch    = cutlass::arch::Sm50;

  using ThreadblockShape = cutlass::gemm::GemmShape<128, 32, 8>;
  using WarpShape        = cutlass::gemm::GemmShape<32, 16, 8>;
  using InstructionShape = cutlass::gemm::GemmShape<1, 1, 1>;

  using Config = typename cuasr::gemm::device::DefaultSemiRingConfiguration< //
      precision, precision, precision, precision, OpClass,                   //
      cuasr::plus<precision>, cuasr::multiplies<precision>, SmArch>;

  using AddOp            = Config::AdditionOp;
  using MultOp           = Config::MultiplicationOp;
  using EpilogueOutputOp = Config::EpilogueOutputOp;

  using Srgemm = cuasr::gemm::device::Srgemm<                           //
      AddOp, MultOp,                                                    //
      precision, cutlass::layout::ColumnMajor,                             //
      precision, cutlass::layout::ColumnMajor,                             //
      precision, cutlass::layout::ColumnMajor,                             //
      precision, OpClass, SmArch,                                       //
      ThreadblockShape, WarpShape, InstructionShape, EpilogueOutputOp,  //
      cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>, 2>;

  // setup bench harness
  cuasr::bench::device::BenchHarness<Srgemm> bench({ N, N, N });

  // benchmark loop
  for (auto _ : state) {
    benchmark::DoNotOptimize(bench.run());
    cudaDeviceSynchronize();
  }

  double flops_per_itr = 2.0 * N * N * N;
  state.counters["Flop/s"]
      = benchmark::Counter(flops_per_itr, benchmark::Counter::kIsIterationInvariantRate);
}
BENCHMARK(BM_SM50_device_plus_multiplies_dsrgemm_nn_n_128x32x8_32x16x1_4x4_8x4_4x2)
    ->RangeMultiplier(2)->Range(256, 4096);
#endif

////////////////////////////////////////////////////////////////////////////////
// Elements / Thread:   2 x   2
//    Threads / Warp:   4 x   8
//     Warps / Block:   4 x   4
//       Threadblock:  32 x  64 x  16
#if defined(CUASR_BENCH_LEVEL) and (CUASR_BENCH_LEVEL >= 2)
static void BM_SM50_device_plus_multiplies_dsrgemm_nn_n_32x64x16_8x16x1_2x2_4x8_4x4(benchmark::State &state) {
  const auto N = static_cast<int>(state.range(0));
  using precision = double;
  using OpClass   = cutlass::arch::OpClassSimt;
  using SmArch    = cutlass::arch::Sm50;

  using ThreadblockShape = cutlass::gemm::GemmShape<32, 64, 16>;
  using WarpShape        = cutlass::gemm::GemmShape<8, 16, 16>;
  using InstructionShape = cutlass::gemm::GemmShape<1, 1, 1>;

  using Config = typename cuasr::gemm::device::DefaultSemiRingConfiguration< //
      precision, precision, precision, precision, OpClass,                   //
      cuasr::plus<precision>, cuasr::multiplies<precision>, SmArch>;

  using AddOp            = Config::AdditionOp;
  using MultOp           = Config::MultiplicationOp;
  using EpilogueOutputOp = Config::EpilogueOutputOp;

  using Srgemm = cuasr::gemm::device::Srgemm<                           //
      AddOp, MultOp,                                                    //
      precision, cutlass::layout::ColumnMajor,                             //
      precision, cutlass::layout::ColumnMajor,                             //
      precision, cutlass::layout::ColumnMajor,                             //
      precision, OpClass, SmArch,                                       //
      ThreadblockShape, WarpShape, InstructionShape, EpilogueOutputOp,  //
      cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>, 2>;

  // setup bench harness
  cuasr::bench::device::BenchHarness<Srgemm> bench({ N, N, N });

  // benchmark loop
  for (auto _ : state) {
    benchmark::DoNotOptimize(bench.run());
    cudaDeviceSynchronize();
  }

  double flops_per_itr = 2.0 * N * N * N;
  state.counters["Flop/s"]
      = benchmark::Counter(flops_per_itr, benchmark::Counter::kIsIterationInvariantRate);
}
BENCHMARK(BM_SM50_device_plus_multiplies_dsrgemm_nn_n_32x64x16_8x16x1_2x2_4x8_4x4)
    ->RangeMultiplier(2)->Range(256, 4096);
#endif

////////////////////////////////////////////////////////////////////////////////
// Elements / Thread:   2 x   4
//    Threads / Warp:   4 x   8
//     Warps / Block:   4 x   4
//       Threadblock:  32 x 128 x  16
#if defined(CUASR_BENCH_LEVEL) and (CUASR_BENCH_LEVEL >= 2)
static void BM_SM50_device_plus_multiplies_dsrgemm_nn_n_32x128x16_8x32x1_2x4_4x8_4x4(benchmark::State &state) {
  const auto N = static_cast<int>(state.range(0));
  using precision = double;
  using OpClass   = cutlass::arch::OpClassSimt;
  using SmArch    = cutlass::arch::Sm50;

  using ThreadblockShape = cutlass::gemm::GemmShape<32, 128, 16>;
  using WarpShape        = cutlass::gemm::GemmShape<8, 32, 16>;
  using InstructionShape = cutlass::gemm::GemmShape<1, 1, 1>;

  using Config = typename cuasr::gemm::device::DefaultSemiRingConfiguration< //
      precision, precision, precision, precision, OpClass,                   //
      cuasr::plus<precision>, cuasr::multiplies<precision>, SmArch>;

  using AddOp            = Config::AdditionOp;
  using MultOp           = Config::MultiplicationOp;
  using EpilogueOutputOp = Config::EpilogueOutputOp;

  using Srgemm = cuasr::gemm::device::Srgemm<                           //
      AddOp, MultOp,                                                    //
      precision, cutlass::layout::ColumnMajor,                             //
      precision, cutlass::layout::ColumnMajor,                             //
      precision, cutlass::layout::ColumnMajor,                             //
      precision, OpClass, SmArch,                                       //
      ThreadblockShape, WarpShape, InstructionShape, EpilogueOutputOp,  //
      cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>, 2>;

  // setup bench harness
  cuasr::bench::device::BenchHarness<Srgemm> bench({ N, N, N });

  // benchmark loop
  for (auto _ : state) {
    benchmark::DoNotOptimize(bench.run());
    cudaDeviceSynchronize();
  }

  double flops_per_itr = 2.0 * N * N * N;
  state.counters["Flop/s"]
      = benchmark::Counter(flops_per_itr, benchmark::Counter::kIsIterationInvariantRate);
}
BENCHMARK(BM_SM50_device_plus_multiplies_dsrgemm_nn_n_32x128x16_8x32x1_2x4_4x8_4x4)
    ->RangeMultiplier(2)->Range(256, 4096);
#endif

////////////////////////////////////////////////////////////////////////////////
// Elements / Thread:   2 x   2
//    Threads / Warp:   8 x   4
//     Warps / Block:   4 x   4
//       Threadblock:  64 x  32 x  16
#if defined(CUASR_BENCH_LEVEL) and (CUASR_BENCH_LEVEL >= 2)
static void BM_SM50_device_plus_multiplies_dsrgemm_nn_n_64x32x16_16x8x1_2x2_8x4_4x4(benchmark::State &state) {
  const auto N = static_cast<int>(state.range(0));
  using precision = double;
  using OpClass   = cutlass::arch::OpClassSimt;
  using SmArch    = cutlass::arch::Sm50;

  using ThreadblockShape = cutlass::gemm::GemmShape<64, 32, 16>;
  using WarpShape        = cutlass::gemm::GemmShape<16, 8, 16>;
  using InstructionShape = cutlass::gemm::GemmShape<1, 1, 1>;

  using Config = typename cuasr::gemm::device::DefaultSemiRingConfiguration< //
      precision, precision, precision, precision, OpClass,                   //
      cuasr::plus<precision>, cuasr::multiplies<precision>, SmArch>;

  using AddOp            = Config::AdditionOp;
  using MultOp           = Config::MultiplicationOp;
  using EpilogueOutputOp = Config::EpilogueOutputOp;

  using Srgemm = cuasr::gemm::device::Srgemm<                           //
      AddOp, MultOp,                                                    //
      precision, cutlass::layout::ColumnMajor,                             //
      precision, cutlass::layout::ColumnMajor,                             //
      precision, cutlass::layout::ColumnMajor,                             //
      precision, OpClass, SmArch,                                       //
      ThreadblockShape, WarpShape, InstructionShape, EpilogueOutputOp,  //
      cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>, 2>;

  // setup bench harness
  cuasr::bench::device::BenchHarness<Srgemm> bench({ N, N, N });

  // benchmark loop
  for (auto _ : state) {
    benchmark::DoNotOptimize(bench.run());
    cudaDeviceSynchronize();
  }

  double flops_per_itr = 2.0 * N * N * N;
  state.counters["Flop/s"]
      = benchmark::Counter(flops_per_itr, benchmark::Counter::kIsIterationInvariantRate);
}
BENCHMARK(BM_SM50_device_plus_multiplies_dsrgemm_nn_n_64x32x16_16x8x1_2x2_8x4_4x4)
    ->RangeMultiplier(2)->Range(256, 4096);
#endif

////////////////////////////////////////////////////////////////////////////////
// Elements / Thread:   4 x   2
//    Threads / Warp:   4 x   8
//     Warps / Block:   4 x   4
//       Threadblock:  64 x  64 x   8
#if defined(CUASR_BENCH_LEVEL) and (CUASR_BENCH_LEVEL >= 2)
static void BM_SM50_device_plus_multiplies_dsrgemm_nn_n_64x64x8_16x16x1_4x2_4x8_4x4(benchmark::State &state) {
  const auto N = static_cast<int>(state.range(0));
  using precision = double;
  using OpClass   = cutlass::arch::OpClassSimt;
  using SmArch    = cutlass::arch::Sm50;

  using ThreadblockShape = cutlass::gemm::GemmShape<64, 64, 8>;
  using WarpShape        = cutlass::gemm::GemmShape<16, 16, 8>;
  using InstructionShape = cutlass::gemm::GemmShape<1, 1, 1>;

  using Config = typename cuasr::gemm::device::DefaultSemiRingConfiguration< //
      precision, precision, precision, precision, OpClass,                   //
      cuasr::plus<precision>, cuasr::multiplies<precision>, SmArch>;

  using AddOp            = Config::AdditionOp;
  using MultOp           = Config::MultiplicationOp;
  using EpilogueOutputOp = Config::EpilogueOutputOp;

  using Srgemm = cuasr::gemm::device::Srgemm<                           //
      AddOp, MultOp,                                                    //
      precision, cutlass::layout::ColumnMajor,                             //
      precision, cutlass::layout::ColumnMajor,                             //
      precision, cutlass::layout::ColumnMajor,                             //
      precision, OpClass, SmArch,                                       //
      ThreadblockShape, WarpShape, InstructionShape, EpilogueOutputOp,  //
      cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>, 2>;

  // setup bench harness
  cuasr::bench::device::BenchHarness<Srgemm> bench({ N, N, N });

  // benchmark loop
  for (auto _ : state) {
    benchmark::DoNotOptimize(bench.run());
    cudaDeviceSynchronize();
  }

  double flops_per_itr = 2.0 * N * N * N;
  state.counters["Flop/s"]
      = benchmark::Counter(flops_per_itr, benchmark::Counter::kIsIterationInvariantRate);
}
BENCHMARK(BM_SM50_device_plus_multiplies_dsrgemm_nn_n_64x64x8_16x16x1_4x2_4x8_4x4)
    ->RangeMultiplier(2)->Range(256, 4096);
#endif

////////////////////////////////////////////////////////////////////////////////
// Elements / Thread:   4 x   2
//    Threads / Warp:   8 x   4
//     Warps / Block:   4 x   4
//       Threadblock: 128 x  32 x  16
#if defined(CUASR_BENCH_LEVEL) and (CUASR_BENCH_LEVEL >= 2)
static void BM_SM50_device_plus_multiplies_dsrgemm_nn_n_128x32x16_32x8x1_4x2_8x4_4x4(benchmark::State &state) {
  const auto N = static_cast<int>(state.range(0));
  using precision = double;
  using OpClass   = cutlass::arch::OpClassSimt;
  using SmArch    = cutlass::arch::Sm50;

  using ThreadblockShape = cutlass::gemm::GemmShape<128, 32, 16>;
  using WarpShape        = cutlass::gemm::GemmShape<32, 8, 16>;
  using InstructionShape = cutlass::gemm::GemmShape<1, 1, 1>;

  using Config = typename cuasr::gemm::device::DefaultSemiRingConfiguration< //
      precision, precision, precision, precision, OpClass,                   //
      cuasr::plus<precision>, cuasr::multiplies<precision>, SmArch>;

  using AddOp            = Config::AdditionOp;
  using MultOp           = Config::MultiplicationOp;
  using EpilogueOutputOp = Config::EpilogueOutputOp;

  using Srgemm = cuasr::gemm::device::Srgemm<                           //
      AddOp, MultOp,                                                    //
      precision, cutlass::layout::ColumnMajor,                             //
      precision, cutlass::layout::ColumnMajor,                             //
      precision, cutlass::layout::ColumnMajor,                             //
      precision, OpClass, SmArch,                                       //
      ThreadblockShape, WarpShape, InstructionShape, EpilogueOutputOp,  //
      cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>, 2>;

  // setup bench harness
  cuasr::bench::device::BenchHarness<Srgemm> bench({ N, N, N });

  // benchmark loop
  for (auto _ : state) {
    benchmark::DoNotOptimize(bench.run());
    cudaDeviceSynchronize();
  }

  double flops_per_itr = 2.0 * N * N * N;
  state.counters["Flop/s"]
      = benchmark::Counter(flops_per_itr, benchmark::Counter::kIsIterationInvariantRate);
}
BENCHMARK(BM_SM50_device_plus_multiplies_dsrgemm_nn_n_128x32x16_32x8x1_4x2_8x4_4x4)
    ->RangeMultiplier(2)->Range(256, 4096);
#endif

