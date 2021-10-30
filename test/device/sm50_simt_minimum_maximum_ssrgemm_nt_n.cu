/***************************************************************************************************
* Copyright (c) 2020, Vijay Thakkar (thakkarv@gatech.edu).
**************************************************************************************************/
/////////////////////////////////////////////////////////////////
//  THIS TEST FILE IS GENERATED AUTOMATICALLY : DO NOT MODIFY  //
/////////////////////////////////////////////////////////////////

#include "gtest/gtest.h"

/// from upstream cutlass
#include "cutlass/cutlass.h"
#include "cutlass/gemm/gemm.h"
#include "cutlass/gemm/threadblock/threadblock_swizzle.h"

/// from cuasr lib
#include "cuasr/gemm/device/default_srgemm_configuration.h"
#include "cuasr/gemm/device/srgemm.h"
#include "cuasr/functional.h"

/// from cuasr tools
#include "cuasr/reference/srgemm/host_srgemm.h"

/// from local test dir
#include "testbed.h"

////////////////////////////////////////////////////////////////////////////////
// Elements / Thread:   2 x   4
//    Threads / Warp:   4 x   8
//     Warps / Block:   1 x   1
//       Threadblock:   8 x  32 x   8
#if defined(CUASR_TEST_LEVEL) and (CUASR_TEST_LEVEL >= 1)
TEST(SM50_device_minimum_maximum_ssrgemm_nt_n, 8x32x8_8x32x1_2x4_4x8_1x1) {
  using precision = float;
  using OpClass   = cutlass::arch::OpClassSimt;
  using SmArch    = cutlass::arch::Sm50;

  using ThreadblockShape = cutlass::gemm::GemmShape<8, 32, 8>;
  using WarpShape        = cutlass::gemm::GemmShape<8, 32, 8>;
  using InstructionShape = cutlass::gemm::GemmShape<1, 1, 1>;

  using Config = typename cuasr::gemm::device::DefaultSemiRingConfiguration< //
      precision, precision, precision, precision, OpClass,                   //
      cuasr::minimum<precision>, cuasr::maximum<precision>, SmArch>;

  using AddOp            = Config::AdditionOp;
  using MultOp           = Config::MultiplicationOp;
  using EpilogueOutputOp = Config::EpilogueOutputOp;

  using Srgemm = cuasr::gemm::device::Srgemm<                           //
      AddOp, MultOp,                                                    //
      precision, cutlass::layout::ColumnMajor,                             //
      precision, cutlass::layout::RowMajor,                             //
      precision, cutlass::layout::ColumnMajor,                             //
      precision, OpClass, SmArch,                                       //
      ThreadblockShape, WarpShape, InstructionShape, EpilogueOutputOp,  //
      cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>, 2>;

  EXPECT_TRUE(cuasr::test::gemm::device::TestAllGemm<Srgemm>());
}
#endif

////////////////////////////////////////////////////////////////////////////////
// Elements / Thread:   4 x   4
//    Threads / Warp:   4 x   8
//     Warps / Block:   1 x   1
//       Threadblock:  16 x  32 x   8
#if defined(CUASR_TEST_LEVEL) and (CUASR_TEST_LEVEL >= 1)
TEST(SM50_device_minimum_maximum_ssrgemm_nt_n, 16x32x8_16x32x1_4x4_4x8_1x1) {
  using precision = float;
  using OpClass   = cutlass::arch::OpClassSimt;
  using SmArch    = cutlass::arch::Sm50;

  using ThreadblockShape = cutlass::gemm::GemmShape<16, 32, 8>;
  using WarpShape        = cutlass::gemm::GemmShape<16, 32, 8>;
  using InstructionShape = cutlass::gemm::GemmShape<1, 1, 1>;

  using Config = typename cuasr::gemm::device::DefaultSemiRingConfiguration< //
      precision, precision, precision, precision, OpClass,                   //
      cuasr::minimum<precision>, cuasr::maximum<precision>, SmArch>;

  using AddOp            = Config::AdditionOp;
  using MultOp           = Config::MultiplicationOp;
  using EpilogueOutputOp = Config::EpilogueOutputOp;

  using Srgemm = cuasr::gemm::device::Srgemm<                           //
      AddOp, MultOp,                                                    //
      precision, cutlass::layout::ColumnMajor,                             //
      precision, cutlass::layout::RowMajor,                             //
      precision, cutlass::layout::ColumnMajor,                             //
      precision, OpClass, SmArch,                                       //
      ThreadblockShape, WarpShape, InstructionShape, EpilogueOutputOp,  //
      cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>, 2>;

  EXPECT_TRUE(cuasr::test::gemm::device::TestAllGemm<Srgemm>());
}
#endif

////////////////////////////////////////////////////////////////////////////////
// Elements / Thread:   4 x   8
//    Threads / Warp:   4 x   8
//     Warps / Block:   1 x   1
//       Threadblock:  16 x  64 x   8
#if defined(CUASR_TEST_LEVEL) and (CUASR_TEST_LEVEL >= 1)
TEST(SM50_device_minimum_maximum_ssrgemm_nt_n, 16x64x8_16x64x1_4x8_4x8_1x1) {
  using precision = float;
  using OpClass   = cutlass::arch::OpClassSimt;
  using SmArch    = cutlass::arch::Sm50;

  using ThreadblockShape = cutlass::gemm::GemmShape<16, 64, 8>;
  using WarpShape        = cutlass::gemm::GemmShape<16, 64, 8>;
  using InstructionShape = cutlass::gemm::GemmShape<1, 1, 1>;

  using Config = typename cuasr::gemm::device::DefaultSemiRingConfiguration< //
      precision, precision, precision, precision, OpClass,                   //
      cuasr::minimum<precision>, cuasr::maximum<precision>, SmArch>;

  using AddOp            = Config::AdditionOp;
  using MultOp           = Config::MultiplicationOp;
  using EpilogueOutputOp = Config::EpilogueOutputOp;

  using Srgemm = cuasr::gemm::device::Srgemm<                           //
      AddOp, MultOp,                                                    //
      precision, cutlass::layout::ColumnMajor,                             //
      precision, cutlass::layout::RowMajor,                             //
      precision, cutlass::layout::ColumnMajor,                             //
      precision, OpClass, SmArch,                                       //
      ThreadblockShape, WarpShape, InstructionShape, EpilogueOutputOp,  //
      cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>, 2>;

  EXPECT_TRUE(cuasr::test::gemm::device::TestAllGemm<Srgemm>());
}
#endif

////////////////////////////////////////////////////////////////////////////////
// Elements / Thread:   8 x   4
//    Threads / Warp:   4 x   8
//     Warps / Block:   1 x   1
//       Threadblock:  32 x  32 x   8
#if defined(CUASR_TEST_LEVEL) and (CUASR_TEST_LEVEL >= 1)
TEST(SM50_device_minimum_maximum_ssrgemm_nt_n, 32x32x8_32x32x1_8x4_4x8_1x1) {
  using precision = float;
  using OpClass   = cutlass::arch::OpClassSimt;
  using SmArch    = cutlass::arch::Sm50;

  using ThreadblockShape = cutlass::gemm::GemmShape<32, 32, 8>;
  using WarpShape        = cutlass::gemm::GemmShape<32, 32, 8>;
  using InstructionShape = cutlass::gemm::GemmShape<1, 1, 1>;

  using Config = typename cuasr::gemm::device::DefaultSemiRingConfiguration< //
      precision, precision, precision, precision, OpClass,                   //
      cuasr::minimum<precision>, cuasr::maximum<precision>, SmArch>;

  using AddOp            = Config::AdditionOp;
  using MultOp           = Config::MultiplicationOp;
  using EpilogueOutputOp = Config::EpilogueOutputOp;

  using Srgemm = cuasr::gemm::device::Srgemm<                           //
      AddOp, MultOp,                                                    //
      precision, cutlass::layout::ColumnMajor,                             //
      precision, cutlass::layout::RowMajor,                             //
      precision, cutlass::layout::ColumnMajor,                             //
      precision, OpClass, SmArch,                                       //
      ThreadblockShape, WarpShape, InstructionShape, EpilogueOutputOp,  //
      cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>, 2>;

  EXPECT_TRUE(cuasr::test::gemm::device::TestAllGemm<Srgemm>());
}
#endif

////////////////////////////////////////////////////////////////////////////////
// Elements / Thread:   8 x   8
//    Threads / Warp:   4 x   8
//     Warps / Block:   1 x   1
//       Threadblock:  32 x  64 x   8
#if defined(CUASR_TEST_LEVEL) and (CUASR_TEST_LEVEL >= 1)
TEST(SM50_device_minimum_maximum_ssrgemm_nt_n, 32x64x8_32x64x1_8x8_4x8_1x1) {
  using precision = float;
  using OpClass   = cutlass::arch::OpClassSimt;
  using SmArch    = cutlass::arch::Sm50;

  using ThreadblockShape = cutlass::gemm::GemmShape<32, 64, 8>;
  using WarpShape        = cutlass::gemm::GemmShape<32, 64, 8>;
  using InstructionShape = cutlass::gemm::GemmShape<1, 1, 1>;

  using Config = typename cuasr::gemm::device::DefaultSemiRingConfiguration< //
      precision, precision, precision, precision, OpClass,                   //
      cuasr::minimum<precision>, cuasr::maximum<precision>, SmArch>;

  using AddOp            = Config::AdditionOp;
  using MultOp           = Config::MultiplicationOp;
  using EpilogueOutputOp = Config::EpilogueOutputOp;

  using Srgemm = cuasr::gemm::device::Srgemm<                           //
      AddOp, MultOp,                                                    //
      precision, cutlass::layout::ColumnMajor,                             //
      precision, cutlass::layout::RowMajor,                             //
      precision, cutlass::layout::ColumnMajor,                             //
      precision, OpClass, SmArch,                                       //
      ThreadblockShape, WarpShape, InstructionShape, EpilogueOutputOp,  //
      cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>, 2>;

  EXPECT_TRUE(cuasr::test::gemm::device::TestAllGemm<Srgemm>());
}
#endif

////////////////////////////////////////////////////////////////////////////////
// Elements / Thread:   8 x   8
//    Threads / Warp:   8 x   4
//     Warps / Block:   1 x   1
//       Threadblock:  64 x  32 x   8
#if defined(CUASR_TEST_LEVEL) and (CUASR_TEST_LEVEL >= 1)
TEST(SM50_device_minimum_maximum_ssrgemm_nt_n, 64x32x8_64x32x1_8x8_8x4_1x1) {
  using precision = float;
  using OpClass   = cutlass::arch::OpClassSimt;
  using SmArch    = cutlass::arch::Sm50;

  using ThreadblockShape = cutlass::gemm::GemmShape<64, 32, 8>;
  using WarpShape        = cutlass::gemm::GemmShape<64, 32, 8>;
  using InstructionShape = cutlass::gemm::GemmShape<1, 1, 1>;

  using Config = typename cuasr::gemm::device::DefaultSemiRingConfiguration< //
      precision, precision, precision, precision, OpClass,                   //
      cuasr::minimum<precision>, cuasr::maximum<precision>, SmArch>;

  using AddOp            = Config::AdditionOp;
  using MultOp           = Config::MultiplicationOp;
  using EpilogueOutputOp = Config::EpilogueOutputOp;

  using Srgemm = cuasr::gemm::device::Srgemm<                           //
      AddOp, MultOp,                                                    //
      precision, cutlass::layout::ColumnMajor,                             //
      precision, cutlass::layout::RowMajor,                             //
      precision, cutlass::layout::ColumnMajor,                             //
      precision, OpClass, SmArch,                                       //
      ThreadblockShape, WarpShape, InstructionShape, EpilogueOutputOp,  //
      cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>, 2>;

  EXPECT_TRUE(cuasr::test::gemm::device::TestAllGemm<Srgemm>());
}
#endif

////////////////////////////////////////////////////////////////////////////////
// Elements / Thread:   2 x   2
//    Threads / Warp:   4 x   8
//     Warps / Block:   1 x   2
//       Threadblock:   8 x  32 x   8
#if defined(CUASR_TEST_LEVEL) and (CUASR_TEST_LEVEL >= 2)
TEST(SM50_device_minimum_maximum_ssrgemm_nt_n, 8x32x8_8x16x1_2x2_4x8_1x2) {
  using precision = float;
  using OpClass   = cutlass::arch::OpClassSimt;
  using SmArch    = cutlass::arch::Sm50;

  using ThreadblockShape = cutlass::gemm::GemmShape<8, 32, 8>;
  using WarpShape        = cutlass::gemm::GemmShape<8, 16, 8>;
  using InstructionShape = cutlass::gemm::GemmShape<1, 1, 1>;

  using Config = typename cuasr::gemm::device::DefaultSemiRingConfiguration< //
      precision, precision, precision, precision, OpClass,                   //
      cuasr::minimum<precision>, cuasr::maximum<precision>, SmArch>;

  using AddOp            = Config::AdditionOp;
  using MultOp           = Config::MultiplicationOp;
  using EpilogueOutputOp = Config::EpilogueOutputOp;

  using Srgemm = cuasr::gemm::device::Srgemm<                           //
      AddOp, MultOp,                                                    //
      precision, cutlass::layout::ColumnMajor,                             //
      precision, cutlass::layout::RowMajor,                             //
      precision, cutlass::layout::ColumnMajor,                             //
      precision, OpClass, SmArch,                                       //
      ThreadblockShape, WarpShape, InstructionShape, EpilogueOutputOp,  //
      cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>, 2>;

  EXPECT_TRUE(cuasr::test::gemm::device::TestAllGemm<Srgemm>());
}
#endif

////////////////////////////////////////////////////////////////////////////////
// Elements / Thread:   2 x   4
//    Threads / Warp:   4 x   8
//     Warps / Block:   1 x   2
//       Threadblock:   8 x  64 x   8
#if defined(CUASR_TEST_LEVEL) and (CUASR_TEST_LEVEL >= 1)
TEST(SM50_device_minimum_maximum_ssrgemm_nt_n, 8x64x8_8x32x1_2x4_4x8_1x2) {
  using precision = float;
  using OpClass   = cutlass::arch::OpClassSimt;
  using SmArch    = cutlass::arch::Sm50;

  using ThreadblockShape = cutlass::gemm::GemmShape<8, 64, 8>;
  using WarpShape        = cutlass::gemm::GemmShape<8, 32, 8>;
  using InstructionShape = cutlass::gemm::GemmShape<1, 1, 1>;

  using Config = typename cuasr::gemm::device::DefaultSemiRingConfiguration< //
      precision, precision, precision, precision, OpClass,                   //
      cuasr::minimum<precision>, cuasr::maximum<precision>, SmArch>;

  using AddOp            = Config::AdditionOp;
  using MultOp           = Config::MultiplicationOp;
  using EpilogueOutputOp = Config::EpilogueOutputOp;

  using Srgemm = cuasr::gemm::device::Srgemm<                           //
      AddOp, MultOp,                                                    //
      precision, cutlass::layout::ColumnMajor,                             //
      precision, cutlass::layout::RowMajor,                             //
      precision, cutlass::layout::ColumnMajor,                             //
      precision, OpClass, SmArch,                                       //
      ThreadblockShape, WarpShape, InstructionShape, EpilogueOutputOp,  //
      cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>, 2>;

  EXPECT_TRUE(cuasr::test::gemm::device::TestAllGemm<Srgemm>());
}
#endif

////////////////////////////////////////////////////////////////////////////////
// Elements / Thread:   4 x   2
//    Threads / Warp:   4 x   8
//     Warps / Block:   1 x   2
//       Threadblock:  16 x  32 x   8
#if defined(CUASR_TEST_LEVEL) and (CUASR_TEST_LEVEL >= 2)
TEST(SM50_device_minimum_maximum_ssrgemm_nt_n, 16x32x8_16x16x1_4x2_4x8_1x2) {
  using precision = float;
  using OpClass   = cutlass::arch::OpClassSimt;
  using SmArch    = cutlass::arch::Sm50;

  using ThreadblockShape = cutlass::gemm::GemmShape<16, 32, 8>;
  using WarpShape        = cutlass::gemm::GemmShape<16, 16, 8>;
  using InstructionShape = cutlass::gemm::GemmShape<1, 1, 1>;

  using Config = typename cuasr::gemm::device::DefaultSemiRingConfiguration< //
      precision, precision, precision, precision, OpClass,                   //
      cuasr::minimum<precision>, cuasr::maximum<precision>, SmArch>;

  using AddOp            = Config::AdditionOp;
  using MultOp           = Config::MultiplicationOp;
  using EpilogueOutputOp = Config::EpilogueOutputOp;

  using Srgemm = cuasr::gemm::device::Srgemm<                           //
      AddOp, MultOp,                                                    //
      precision, cutlass::layout::ColumnMajor,                             //
      precision, cutlass::layout::RowMajor,                             //
      precision, cutlass::layout::ColumnMajor,                             //
      precision, OpClass, SmArch,                                       //
      ThreadblockShape, WarpShape, InstructionShape, EpilogueOutputOp,  //
      cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>, 2>;

  EXPECT_TRUE(cuasr::test::gemm::device::TestAllGemm<Srgemm>());
}
#endif

////////////////////////////////////////////////////////////////////////////////
// Elements / Thread:   4 x   4
//    Threads / Warp:   4 x   8
//     Warps / Block:   1 x   2
//       Threadblock:  16 x  64 x   8
#if defined(CUASR_TEST_LEVEL) and (CUASR_TEST_LEVEL >= 2)
TEST(SM50_device_minimum_maximum_ssrgemm_nt_n, 16x64x8_16x32x1_4x4_4x8_1x2) {
  using precision = float;
  using OpClass   = cutlass::arch::OpClassSimt;
  using SmArch    = cutlass::arch::Sm50;

  using ThreadblockShape = cutlass::gemm::GemmShape<16, 64, 8>;
  using WarpShape        = cutlass::gemm::GemmShape<16, 32, 8>;
  using InstructionShape = cutlass::gemm::GemmShape<1, 1, 1>;

  using Config = typename cuasr::gemm::device::DefaultSemiRingConfiguration< //
      precision, precision, precision, precision, OpClass,                   //
      cuasr::minimum<precision>, cuasr::maximum<precision>, SmArch>;

  using AddOp            = Config::AdditionOp;
  using MultOp           = Config::MultiplicationOp;
  using EpilogueOutputOp = Config::EpilogueOutputOp;

  using Srgemm = cuasr::gemm::device::Srgemm<                           //
      AddOp, MultOp,                                                    //
      precision, cutlass::layout::ColumnMajor,                             //
      precision, cutlass::layout::RowMajor,                             //
      precision, cutlass::layout::ColumnMajor,                             //
      precision, OpClass, SmArch,                                       //
      ThreadblockShape, WarpShape, InstructionShape, EpilogueOutputOp,  //
      cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>, 2>;

  EXPECT_TRUE(cuasr::test::gemm::device::TestAllGemm<Srgemm>());
}
#endif

////////////////////////////////////////////////////////////////////////////////
// Elements / Thread:   4 x   8
//    Threads / Warp:   4 x   8
//     Warps / Block:   1 x   2
//       Threadblock:  16 x 128 x   8
#if defined(CUASR_TEST_LEVEL) and (CUASR_TEST_LEVEL >= 1)
TEST(SM50_device_minimum_maximum_ssrgemm_nt_n, 16x128x8_16x64x1_4x8_4x8_1x2) {
  using precision = float;
  using OpClass   = cutlass::arch::OpClassSimt;
  using SmArch    = cutlass::arch::Sm50;

  using ThreadblockShape = cutlass::gemm::GemmShape<16, 128, 8>;
  using WarpShape        = cutlass::gemm::GemmShape<16, 64, 8>;
  using InstructionShape = cutlass::gemm::GemmShape<1, 1, 1>;

  using Config = typename cuasr::gemm::device::DefaultSemiRingConfiguration< //
      precision, precision, precision, precision, OpClass,                   //
      cuasr::minimum<precision>, cuasr::maximum<precision>, SmArch>;

  using AddOp            = Config::AdditionOp;
  using MultOp           = Config::MultiplicationOp;
  using EpilogueOutputOp = Config::EpilogueOutputOp;

  using Srgemm = cuasr::gemm::device::Srgemm<                           //
      AddOp, MultOp,                                                    //
      precision, cutlass::layout::ColumnMajor,                             //
      precision, cutlass::layout::RowMajor,                             //
      precision, cutlass::layout::ColumnMajor,                             //
      precision, OpClass, SmArch,                                       //
      ThreadblockShape, WarpShape, InstructionShape, EpilogueOutputOp,  //
      cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>, 2>;

  EXPECT_TRUE(cuasr::test::gemm::device::TestAllGemm<Srgemm>());
}
#endif

////////////////////////////////////////////////////////////////////////////////
// Elements / Thread:   4 x   4
//    Threads / Warp:   8 x   4
//     Warps / Block:   1 x   2
//       Threadblock:  32 x  32 x   8
#if defined(CUASR_TEST_LEVEL) and (CUASR_TEST_LEVEL >= 2)
TEST(SM50_device_minimum_maximum_ssrgemm_nt_n, 32x32x8_32x16x1_4x4_8x4_1x2) {
  using precision = float;
  using OpClass   = cutlass::arch::OpClassSimt;
  using SmArch    = cutlass::arch::Sm50;

  using ThreadblockShape = cutlass::gemm::GemmShape<32, 32, 8>;
  using WarpShape        = cutlass::gemm::GemmShape<32, 16, 8>;
  using InstructionShape = cutlass::gemm::GemmShape<1, 1, 1>;

  using Config = typename cuasr::gemm::device::DefaultSemiRingConfiguration< //
      precision, precision, precision, precision, OpClass,                   //
      cuasr::minimum<precision>, cuasr::maximum<precision>, SmArch>;

  using AddOp            = Config::AdditionOp;
  using MultOp           = Config::MultiplicationOp;
  using EpilogueOutputOp = Config::EpilogueOutputOp;

  using Srgemm = cuasr::gemm::device::Srgemm<                           //
      AddOp, MultOp,                                                    //
      precision, cutlass::layout::ColumnMajor,                             //
      precision, cutlass::layout::RowMajor,                             //
      precision, cutlass::layout::ColumnMajor,                             //
      precision, OpClass, SmArch,                                       //
      ThreadblockShape, WarpShape, InstructionShape, EpilogueOutputOp,  //
      cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>, 2>;

  EXPECT_TRUE(cuasr::test::gemm::device::TestAllGemm<Srgemm>());
}
#endif

////////////////////////////////////////////////////////////////////////////////
// Elements / Thread:   8 x   4
//    Threads / Warp:   4 x   8
//     Warps / Block:   1 x   2
//       Threadblock:  32 x  64 x   8
#if defined(CUASR_TEST_LEVEL) and (CUASR_TEST_LEVEL >= 2)
TEST(SM50_device_minimum_maximum_ssrgemm_nt_n, 32x64x8_32x32x1_8x4_4x8_1x2) {
  using precision = float;
  using OpClass   = cutlass::arch::OpClassSimt;
  using SmArch    = cutlass::arch::Sm50;

  using ThreadblockShape = cutlass::gemm::GemmShape<32, 64, 8>;
  using WarpShape        = cutlass::gemm::GemmShape<32, 32, 8>;
  using InstructionShape = cutlass::gemm::GemmShape<1, 1, 1>;

  using Config = typename cuasr::gemm::device::DefaultSemiRingConfiguration< //
      precision, precision, precision, precision, OpClass,                   //
      cuasr::minimum<precision>, cuasr::maximum<precision>, SmArch>;

  using AddOp            = Config::AdditionOp;
  using MultOp           = Config::MultiplicationOp;
  using EpilogueOutputOp = Config::EpilogueOutputOp;

  using Srgemm = cuasr::gemm::device::Srgemm<                           //
      AddOp, MultOp,                                                    //
      precision, cutlass::layout::ColumnMajor,                             //
      precision, cutlass::layout::RowMajor,                             //
      precision, cutlass::layout::ColumnMajor,                             //
      precision, OpClass, SmArch,                                       //
      ThreadblockShape, WarpShape, InstructionShape, EpilogueOutputOp,  //
      cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>, 2>;

  EXPECT_TRUE(cuasr::test::gemm::device::TestAllGemm<Srgemm>());
}
#endif

////////////////////////////////////////////////////////////////////////////////
// Elements / Thread:   8 x   8
//    Threads / Warp:   4 x   8
//     Warps / Block:   1 x   2
//       Threadblock:  32 x 128 x   8
#if defined(CUASR_TEST_LEVEL) and (CUASR_TEST_LEVEL >= 1)
TEST(SM50_device_minimum_maximum_ssrgemm_nt_n, 32x128x8_32x64x1_8x8_4x8_1x2) {
  using precision = float;
  using OpClass   = cutlass::arch::OpClassSimt;
  using SmArch    = cutlass::arch::Sm50;

  using ThreadblockShape = cutlass::gemm::GemmShape<32, 128, 8>;
  using WarpShape        = cutlass::gemm::GemmShape<32, 64, 8>;
  using InstructionShape = cutlass::gemm::GemmShape<1, 1, 1>;

  using Config = typename cuasr::gemm::device::DefaultSemiRingConfiguration< //
      precision, precision, precision, precision, OpClass,                   //
      cuasr::minimum<precision>, cuasr::maximum<precision>, SmArch>;

  using AddOp            = Config::AdditionOp;
  using MultOp           = Config::MultiplicationOp;
  using EpilogueOutputOp = Config::EpilogueOutputOp;

  using Srgemm = cuasr::gemm::device::Srgemm<                           //
      AddOp, MultOp,                                                    //
      precision, cutlass::layout::ColumnMajor,                             //
      precision, cutlass::layout::RowMajor,                             //
      precision, cutlass::layout::ColumnMajor,                             //
      precision, OpClass, SmArch,                                       //
      ThreadblockShape, WarpShape, InstructionShape, EpilogueOutputOp,  //
      cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>, 2>;

  EXPECT_TRUE(cuasr::test::gemm::device::TestAllGemm<Srgemm>());
}
#endif

////////////////////////////////////////////////////////////////////////////////
// Elements / Thread:   8 x   8
//    Threads / Warp:   8 x   4
//     Warps / Block:   1 x   2
//       Threadblock:  64 x  64 x   8
#if defined(CUASR_TEST_LEVEL) and (CUASR_TEST_LEVEL >= 0)
TEST(SM50_device_minimum_maximum_ssrgemm_nt_n, 64x64x8_64x32x1_8x8_8x4_1x2) {
  using precision = float;
  using OpClass   = cutlass::arch::OpClassSimt;
  using SmArch    = cutlass::arch::Sm50;

  using ThreadblockShape = cutlass::gemm::GemmShape<64, 64, 8>;
  using WarpShape        = cutlass::gemm::GemmShape<64, 32, 8>;
  using InstructionShape = cutlass::gemm::GemmShape<1, 1, 1>;

  using Config = typename cuasr::gemm::device::DefaultSemiRingConfiguration< //
      precision, precision, precision, precision, OpClass,                   //
      cuasr::minimum<precision>, cuasr::maximum<precision>, SmArch>;

  using AddOp            = Config::AdditionOp;
  using MultOp           = Config::MultiplicationOp;
  using EpilogueOutputOp = Config::EpilogueOutputOp;

  using Srgemm = cuasr::gemm::device::Srgemm<                           //
      AddOp, MultOp,                                                    //
      precision, cutlass::layout::ColumnMajor,                             //
      precision, cutlass::layout::RowMajor,                             //
      precision, cutlass::layout::ColumnMajor,                             //
      precision, OpClass, SmArch,                                       //
      ThreadblockShape, WarpShape, InstructionShape, EpilogueOutputOp,  //
      cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>, 2>;

  EXPECT_TRUE(cuasr::test::gemm::device::TestAllGemm<Srgemm>());
}
#endif

////////////////////////////////////////////////////////////////////////////////
// Elements / Thread:   4 x   4
//    Threads / Warp:   4 x   8
//     Warps / Block:   2 x   1
//       Threadblock:  32 x  32 x   8
#if defined(CUASR_TEST_LEVEL) and (CUASR_TEST_LEVEL >= 2)
TEST(SM50_device_minimum_maximum_ssrgemm_nt_n, 32x32x8_16x32x1_4x4_4x8_2x1) {
  using precision = float;
  using OpClass   = cutlass::arch::OpClassSimt;
  using SmArch    = cutlass::arch::Sm50;

  using ThreadblockShape = cutlass::gemm::GemmShape<32, 32, 8>;
  using WarpShape        = cutlass::gemm::GemmShape<16, 32, 8>;
  using InstructionShape = cutlass::gemm::GemmShape<1, 1, 1>;

  using Config = typename cuasr::gemm::device::DefaultSemiRingConfiguration< //
      precision, precision, precision, precision, OpClass,                   //
      cuasr::minimum<precision>, cuasr::maximum<precision>, SmArch>;

  using AddOp            = Config::AdditionOp;
  using MultOp           = Config::MultiplicationOp;
  using EpilogueOutputOp = Config::EpilogueOutputOp;

  using Srgemm = cuasr::gemm::device::Srgemm<                           //
      AddOp, MultOp,                                                    //
      precision, cutlass::layout::ColumnMajor,                             //
      precision, cutlass::layout::RowMajor,                             //
      precision, cutlass::layout::ColumnMajor,                             //
      precision, OpClass, SmArch,                                       //
      ThreadblockShape, WarpShape, InstructionShape, EpilogueOutputOp,  //
      cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>, 2>;

  EXPECT_TRUE(cuasr::test::gemm::device::TestAllGemm<Srgemm>());
}
#endif

////////////////////////////////////////////////////////////////////////////////
// Elements / Thread:   8 x   4
//    Threads / Warp:   4 x   8
//     Warps / Block:   2 x   1
//       Threadblock:  64 x  32 x   8
#if defined(CUASR_TEST_LEVEL) and (CUASR_TEST_LEVEL >= 2)
TEST(SM50_device_minimum_maximum_ssrgemm_nt_n, 64x32x8_32x32x1_8x4_4x8_2x1) {
  using precision = float;
  using OpClass   = cutlass::arch::OpClassSimt;
  using SmArch    = cutlass::arch::Sm50;

  using ThreadblockShape = cutlass::gemm::GemmShape<64, 32, 8>;
  using WarpShape        = cutlass::gemm::GemmShape<32, 32, 8>;
  using InstructionShape = cutlass::gemm::GemmShape<1, 1, 1>;

  using Config = typename cuasr::gemm::device::DefaultSemiRingConfiguration< //
      precision, precision, precision, precision, OpClass,                   //
      cuasr::minimum<precision>, cuasr::maximum<precision>, SmArch>;

  using AddOp            = Config::AdditionOp;
  using MultOp           = Config::MultiplicationOp;
  using EpilogueOutputOp = Config::EpilogueOutputOp;

  using Srgemm = cuasr::gemm::device::Srgemm<                           //
      AddOp, MultOp,                                                    //
      precision, cutlass::layout::ColumnMajor,                             //
      precision, cutlass::layout::RowMajor,                             //
      precision, cutlass::layout::ColumnMajor,                             //
      precision, OpClass, SmArch,                                       //
      ThreadblockShape, WarpShape, InstructionShape, EpilogueOutputOp,  //
      cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>, 2>;

  EXPECT_TRUE(cuasr::test::gemm::device::TestAllGemm<Srgemm>());
}
#endif

////////////////////////////////////////////////////////////////////////////////
// Elements / Thread:   8 x   8
//    Threads / Warp:   4 x   8
//     Warps / Block:   2 x   1
//       Threadblock:  64 x  64 x   8
#if defined(CUASR_TEST_LEVEL) and (CUASR_TEST_LEVEL >= 1)
TEST(SM50_device_minimum_maximum_ssrgemm_nt_n, 64x64x8_32x64x1_8x8_4x8_2x1) {
  using precision = float;
  using OpClass   = cutlass::arch::OpClassSimt;
  using SmArch    = cutlass::arch::Sm50;

  using ThreadblockShape = cutlass::gemm::GemmShape<64, 64, 8>;
  using WarpShape        = cutlass::gemm::GemmShape<32, 64, 8>;
  using InstructionShape = cutlass::gemm::GemmShape<1, 1, 1>;

  using Config = typename cuasr::gemm::device::DefaultSemiRingConfiguration< //
      precision, precision, precision, precision, OpClass,                   //
      cuasr::minimum<precision>, cuasr::maximum<precision>, SmArch>;

  using AddOp            = Config::AdditionOp;
  using MultOp           = Config::MultiplicationOp;
  using EpilogueOutputOp = Config::EpilogueOutputOp;

  using Srgemm = cuasr::gemm::device::Srgemm<                           //
      AddOp, MultOp,                                                    //
      precision, cutlass::layout::ColumnMajor,                             //
      precision, cutlass::layout::RowMajor,                             //
      precision, cutlass::layout::ColumnMajor,                             //
      precision, OpClass, SmArch,                                       //
      ThreadblockShape, WarpShape, InstructionShape, EpilogueOutputOp,  //
      cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>, 2>;

  EXPECT_TRUE(cuasr::test::gemm::device::TestAllGemm<Srgemm>());
}
#endif

////////////////////////////////////////////////////////////////////////////////
// Elements / Thread:   8 x   8
//    Threads / Warp:   8 x   4
//     Warps / Block:   2 x   1
//       Threadblock: 128 x  32 x   8
#if defined(CUASR_TEST_LEVEL) and (CUASR_TEST_LEVEL >= 1)
TEST(SM50_device_minimum_maximum_ssrgemm_nt_n, 128x32x8_64x32x1_8x8_8x4_2x1) {
  using precision = float;
  using OpClass   = cutlass::arch::OpClassSimt;
  using SmArch    = cutlass::arch::Sm50;

  using ThreadblockShape = cutlass::gemm::GemmShape<128, 32, 8>;
  using WarpShape        = cutlass::gemm::GemmShape<64, 32, 8>;
  using InstructionShape = cutlass::gemm::GemmShape<1, 1, 1>;

  using Config = typename cuasr::gemm::device::DefaultSemiRingConfiguration< //
      precision, precision, precision, precision, OpClass,                   //
      cuasr::minimum<precision>, cuasr::maximum<precision>, SmArch>;

  using AddOp            = Config::AdditionOp;
  using MultOp           = Config::MultiplicationOp;
  using EpilogueOutputOp = Config::EpilogueOutputOp;

  using Srgemm = cuasr::gemm::device::Srgemm<                           //
      AddOp, MultOp,                                                    //
      precision, cutlass::layout::ColumnMajor,                             //
      precision, cutlass::layout::RowMajor,                             //
      precision, cutlass::layout::ColumnMajor,                             //
      precision, OpClass, SmArch,                                       //
      ThreadblockShape, WarpShape, InstructionShape, EpilogueOutputOp,  //
      cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>, 2>;

  EXPECT_TRUE(cuasr::test::gemm::device::TestAllGemm<Srgemm>());
}
#endif

////////////////////////////////////////////////////////////////////////////////
// Elements / Thread:   2 x   2
//    Threads / Warp:   4 x   8
//     Warps / Block:   2 x   2
//       Threadblock:  16 x  32 x   8
#if defined(CUASR_TEST_LEVEL) and (CUASR_TEST_LEVEL >= 2)
TEST(SM50_device_minimum_maximum_ssrgemm_nt_n, 16x32x8_8x16x1_2x2_4x8_2x2) {
  using precision = float;
  using OpClass   = cutlass::arch::OpClassSimt;
  using SmArch    = cutlass::arch::Sm50;

  using ThreadblockShape = cutlass::gemm::GemmShape<16, 32, 8>;
  using WarpShape        = cutlass::gemm::GemmShape<8, 16, 8>;
  using InstructionShape = cutlass::gemm::GemmShape<1, 1, 1>;

  using Config = typename cuasr::gemm::device::DefaultSemiRingConfiguration< //
      precision, precision, precision, precision, OpClass,                   //
      cuasr::minimum<precision>, cuasr::maximum<precision>, SmArch>;

  using AddOp            = Config::AdditionOp;
  using MultOp           = Config::MultiplicationOp;
  using EpilogueOutputOp = Config::EpilogueOutputOp;

  using Srgemm = cuasr::gemm::device::Srgemm<                           //
      AddOp, MultOp,                                                    //
      precision, cutlass::layout::ColumnMajor,                             //
      precision, cutlass::layout::RowMajor,                             //
      precision, cutlass::layout::ColumnMajor,                             //
      precision, OpClass, SmArch,                                       //
      ThreadblockShape, WarpShape, InstructionShape, EpilogueOutputOp,  //
      cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>, 2>;

  EXPECT_TRUE(cuasr::test::gemm::device::TestAllGemm<Srgemm>());
}
#endif

////////////////////////////////////////////////////////////////////////////////
// Elements / Thread:   2 x   4
//    Threads / Warp:   4 x   8
//     Warps / Block:   2 x   2
//       Threadblock:  16 x  64 x   8
#if defined(CUASR_TEST_LEVEL) and (CUASR_TEST_LEVEL >= 2)
TEST(SM50_device_minimum_maximum_ssrgemm_nt_n, 16x64x8_8x32x1_2x4_4x8_2x2) {
  using precision = float;
  using OpClass   = cutlass::arch::OpClassSimt;
  using SmArch    = cutlass::arch::Sm50;

  using ThreadblockShape = cutlass::gemm::GemmShape<16, 64, 8>;
  using WarpShape        = cutlass::gemm::GemmShape<8, 32, 8>;
  using InstructionShape = cutlass::gemm::GemmShape<1, 1, 1>;

  using Config = typename cuasr::gemm::device::DefaultSemiRingConfiguration< //
      precision, precision, precision, precision, OpClass,                   //
      cuasr::minimum<precision>, cuasr::maximum<precision>, SmArch>;

  using AddOp            = Config::AdditionOp;
  using MultOp           = Config::MultiplicationOp;
  using EpilogueOutputOp = Config::EpilogueOutputOp;

  using Srgemm = cuasr::gemm::device::Srgemm<                           //
      AddOp, MultOp,                                                    //
      precision, cutlass::layout::ColumnMajor,                             //
      precision, cutlass::layout::RowMajor,                             //
      precision, cutlass::layout::ColumnMajor,                             //
      precision, OpClass, SmArch,                                       //
      ThreadblockShape, WarpShape, InstructionShape, EpilogueOutputOp,  //
      cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>, 2>;

  EXPECT_TRUE(cuasr::test::gemm::device::TestAllGemm<Srgemm>());
}
#endif

////////////////////////////////////////////////////////////////////////////////
// Elements / Thread:   4 x   2
//    Threads / Warp:   4 x   8
//     Warps / Block:   2 x   2
//       Threadblock:  32 x  32 x   8
#if defined(CUASR_TEST_LEVEL) and (CUASR_TEST_LEVEL >= 2)
TEST(SM50_device_minimum_maximum_ssrgemm_nt_n, 32x32x8_16x16x1_4x2_4x8_2x2) {
  using precision = float;
  using OpClass   = cutlass::arch::OpClassSimt;
  using SmArch    = cutlass::arch::Sm50;

  using ThreadblockShape = cutlass::gemm::GemmShape<32, 32, 8>;
  using WarpShape        = cutlass::gemm::GemmShape<16, 16, 8>;
  using InstructionShape = cutlass::gemm::GemmShape<1, 1, 1>;

  using Config = typename cuasr::gemm::device::DefaultSemiRingConfiguration< //
      precision, precision, precision, precision, OpClass,                   //
      cuasr::minimum<precision>, cuasr::maximum<precision>, SmArch>;

  using AddOp            = Config::AdditionOp;
  using MultOp           = Config::MultiplicationOp;
  using EpilogueOutputOp = Config::EpilogueOutputOp;

  using Srgemm = cuasr::gemm::device::Srgemm<                           //
      AddOp, MultOp,                                                    //
      precision, cutlass::layout::ColumnMajor,                             //
      precision, cutlass::layout::RowMajor,                             //
      precision, cutlass::layout::ColumnMajor,                             //
      precision, OpClass, SmArch,                                       //
      ThreadblockShape, WarpShape, InstructionShape, EpilogueOutputOp,  //
      cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>, 2>;

  EXPECT_TRUE(cuasr::test::gemm::device::TestAllGemm<Srgemm>());
}
#endif

////////////////////////////////////////////////////////////////////////////////
// Elements / Thread:   4 x   4
//    Threads / Warp:   4 x   8
//     Warps / Block:   2 x   2
//       Threadblock:  32 x  64 x   8
#if defined(CUASR_TEST_LEVEL) and (CUASR_TEST_LEVEL >= 2)
TEST(SM50_device_minimum_maximum_ssrgemm_nt_n, 32x64x8_16x32x1_4x4_4x8_2x2) {
  using precision = float;
  using OpClass   = cutlass::arch::OpClassSimt;
  using SmArch    = cutlass::arch::Sm50;

  using ThreadblockShape = cutlass::gemm::GemmShape<32, 64, 8>;
  using WarpShape        = cutlass::gemm::GemmShape<16, 32, 8>;
  using InstructionShape = cutlass::gemm::GemmShape<1, 1, 1>;

  using Config = typename cuasr::gemm::device::DefaultSemiRingConfiguration< //
      precision, precision, precision, precision, OpClass,                   //
      cuasr::minimum<precision>, cuasr::maximum<precision>, SmArch>;

  using AddOp            = Config::AdditionOp;
  using MultOp           = Config::MultiplicationOp;
  using EpilogueOutputOp = Config::EpilogueOutputOp;

  using Srgemm = cuasr::gemm::device::Srgemm<                           //
      AddOp, MultOp,                                                    //
      precision, cutlass::layout::ColumnMajor,                             //
      precision, cutlass::layout::RowMajor,                             //
      precision, cutlass::layout::ColumnMajor,                             //
      precision, OpClass, SmArch,                                       //
      ThreadblockShape, WarpShape, InstructionShape, EpilogueOutputOp,  //
      cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>, 2>;

  EXPECT_TRUE(cuasr::test::gemm::device::TestAllGemm<Srgemm>());
}
#endif

////////////////////////////////////////////////////////////////////////////////
// Elements / Thread:   4 x   8
//    Threads / Warp:   4 x   8
//     Warps / Block:   2 x   2
//       Threadblock:  32 x 128 x   8
#if defined(CUASR_TEST_LEVEL) and (CUASR_TEST_LEVEL >= 2)
TEST(SM50_device_minimum_maximum_ssrgemm_nt_n, 32x128x8_16x64x1_4x8_4x8_2x2) {
  using precision = float;
  using OpClass   = cutlass::arch::OpClassSimt;
  using SmArch    = cutlass::arch::Sm50;

  using ThreadblockShape = cutlass::gemm::GemmShape<32, 128, 8>;
  using WarpShape        = cutlass::gemm::GemmShape<16, 64, 8>;
  using InstructionShape = cutlass::gemm::GemmShape<1, 1, 1>;

  using Config = typename cuasr::gemm::device::DefaultSemiRingConfiguration< //
      precision, precision, precision, precision, OpClass,                   //
      cuasr::minimum<precision>, cuasr::maximum<precision>, SmArch>;

  using AddOp            = Config::AdditionOp;
  using MultOp           = Config::MultiplicationOp;
  using EpilogueOutputOp = Config::EpilogueOutputOp;

  using Srgemm = cuasr::gemm::device::Srgemm<                           //
      AddOp, MultOp,                                                    //
      precision, cutlass::layout::ColumnMajor,                             //
      precision, cutlass::layout::RowMajor,                             //
      precision, cutlass::layout::ColumnMajor,                             //
      precision, OpClass, SmArch,                                       //
      ThreadblockShape, WarpShape, InstructionShape, EpilogueOutputOp,  //
      cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>, 2>;

  EXPECT_TRUE(cuasr::test::gemm::device::TestAllGemm<Srgemm>());
}
#endif

////////////////////////////////////////////////////////////////////////////////
// Elements / Thread:   4 x   4
//    Threads / Warp:   8 x   4
//     Warps / Block:   2 x   2
//       Threadblock:  64 x  32 x   8
#if defined(CUASR_TEST_LEVEL) and (CUASR_TEST_LEVEL >= 2)
TEST(SM50_device_minimum_maximum_ssrgemm_nt_n, 64x32x8_32x16x1_4x4_8x4_2x2) {
  using precision = float;
  using OpClass   = cutlass::arch::OpClassSimt;
  using SmArch    = cutlass::arch::Sm50;

  using ThreadblockShape = cutlass::gemm::GemmShape<64, 32, 8>;
  using WarpShape        = cutlass::gemm::GemmShape<32, 16, 8>;
  using InstructionShape = cutlass::gemm::GemmShape<1, 1, 1>;

  using Config = typename cuasr::gemm::device::DefaultSemiRingConfiguration< //
      precision, precision, precision, precision, OpClass,                   //
      cuasr::minimum<precision>, cuasr::maximum<precision>, SmArch>;

  using AddOp            = Config::AdditionOp;
  using MultOp           = Config::MultiplicationOp;
  using EpilogueOutputOp = Config::EpilogueOutputOp;

  using Srgemm = cuasr::gemm::device::Srgemm<                           //
      AddOp, MultOp,                                                    //
      precision, cutlass::layout::ColumnMajor,                             //
      precision, cutlass::layout::RowMajor,                             //
      precision, cutlass::layout::ColumnMajor,                             //
      precision, OpClass, SmArch,                                       //
      ThreadblockShape, WarpShape, InstructionShape, EpilogueOutputOp,  //
      cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>, 2>;

  EXPECT_TRUE(cuasr::test::gemm::device::TestAllGemm<Srgemm>());
}
#endif

////////////////////////////////////////////////////////////////////////////////
// Elements / Thread:   8 x   4
//    Threads / Warp:   4 x   8
//     Warps / Block:   2 x   2
//       Threadblock:  64 x  64 x   8
#if defined(CUASR_TEST_LEVEL) and (CUASR_TEST_LEVEL >= 2)
TEST(SM50_device_minimum_maximum_ssrgemm_nt_n, 64x64x8_32x32x1_8x4_4x8_2x2) {
  using precision = float;
  using OpClass   = cutlass::arch::OpClassSimt;
  using SmArch    = cutlass::arch::Sm50;

  using ThreadblockShape = cutlass::gemm::GemmShape<64, 64, 8>;
  using WarpShape        = cutlass::gemm::GemmShape<32, 32, 8>;
  using InstructionShape = cutlass::gemm::GemmShape<1, 1, 1>;

  using Config = typename cuasr::gemm::device::DefaultSemiRingConfiguration< //
      precision, precision, precision, precision, OpClass,                   //
      cuasr::minimum<precision>, cuasr::maximum<precision>, SmArch>;

  using AddOp            = Config::AdditionOp;
  using MultOp           = Config::MultiplicationOp;
  using EpilogueOutputOp = Config::EpilogueOutputOp;

  using Srgemm = cuasr::gemm::device::Srgemm<                           //
      AddOp, MultOp,                                                    //
      precision, cutlass::layout::ColumnMajor,                             //
      precision, cutlass::layout::RowMajor,                             //
      precision, cutlass::layout::ColumnMajor,                             //
      precision, OpClass, SmArch,                                       //
      ThreadblockShape, WarpShape, InstructionShape, EpilogueOutputOp,  //
      cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>, 2>;

  EXPECT_TRUE(cuasr::test::gemm::device::TestAllGemm<Srgemm>());
}
#endif

////////////////////////////////////////////////////////////////////////////////
// Elements / Thread:   8 x   8
//    Threads / Warp:   4 x   8
//     Warps / Block:   2 x   2
//       Threadblock:  64 x 128 x   8
#if defined(CUASR_TEST_LEVEL) and (CUASR_TEST_LEVEL >= 1)
TEST(SM50_device_minimum_maximum_ssrgemm_nt_n, 64x128x8_32x64x1_8x8_4x8_2x2) {
  using precision = float;
  using OpClass   = cutlass::arch::OpClassSimt;
  using SmArch    = cutlass::arch::Sm50;

  using ThreadblockShape = cutlass::gemm::GemmShape<64, 128, 8>;
  using WarpShape        = cutlass::gemm::GemmShape<32, 64, 8>;
  using InstructionShape = cutlass::gemm::GemmShape<1, 1, 1>;

  using Config = typename cuasr::gemm::device::DefaultSemiRingConfiguration< //
      precision, precision, precision, precision, OpClass,                   //
      cuasr::minimum<precision>, cuasr::maximum<precision>, SmArch>;

  using AddOp            = Config::AdditionOp;
  using MultOp           = Config::MultiplicationOp;
  using EpilogueOutputOp = Config::EpilogueOutputOp;

  using Srgemm = cuasr::gemm::device::Srgemm<                           //
      AddOp, MultOp,                                                    //
      precision, cutlass::layout::ColumnMajor,                             //
      precision, cutlass::layout::RowMajor,                             //
      precision, cutlass::layout::ColumnMajor,                             //
      precision, OpClass, SmArch,                                       //
      ThreadblockShape, WarpShape, InstructionShape, EpilogueOutputOp,  //
      cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>, 2>;

  EXPECT_TRUE(cuasr::test::gemm::device::TestAllGemm<Srgemm>());
}
#endif

////////////////////////////////////////////////////////////////////////////////
// Elements / Thread:   8 x   4
//    Threads / Warp:   8 x   4
//     Warps / Block:   2 x   2
//       Threadblock: 128 x  32 x   8
#if defined(CUASR_TEST_LEVEL) and (CUASR_TEST_LEVEL >= 2)
TEST(SM50_device_minimum_maximum_ssrgemm_nt_n, 128x32x8_64x16x1_8x4_8x4_2x2) {
  using precision = float;
  using OpClass   = cutlass::arch::OpClassSimt;
  using SmArch    = cutlass::arch::Sm50;

  using ThreadblockShape = cutlass::gemm::GemmShape<128, 32, 8>;
  using WarpShape        = cutlass::gemm::GemmShape<64, 16, 8>;
  using InstructionShape = cutlass::gemm::GemmShape<1, 1, 1>;

  using Config = typename cuasr::gemm::device::DefaultSemiRingConfiguration< //
      precision, precision, precision, precision, OpClass,                   //
      cuasr::minimum<precision>, cuasr::maximum<precision>, SmArch>;

  using AddOp            = Config::AdditionOp;
  using MultOp           = Config::MultiplicationOp;
  using EpilogueOutputOp = Config::EpilogueOutputOp;

  using Srgemm = cuasr::gemm::device::Srgemm<                           //
      AddOp, MultOp,                                                    //
      precision, cutlass::layout::ColumnMajor,                             //
      precision, cutlass::layout::RowMajor,                             //
      precision, cutlass::layout::ColumnMajor,                             //
      precision, OpClass, SmArch,                                       //
      ThreadblockShape, WarpShape, InstructionShape, EpilogueOutputOp,  //
      cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>, 2>;

  EXPECT_TRUE(cuasr::test::gemm::device::TestAllGemm<Srgemm>());
}
#endif

////////////////////////////////////////////////////////////////////////////////
// Elements / Thread:   8 x   8
//    Threads / Warp:   8 x   4
//     Warps / Block:   2 x   2
//       Threadblock: 128 x  64 x   8
#if defined(CUASR_TEST_LEVEL) and (CUASR_TEST_LEVEL >= 1)
TEST(SM50_device_minimum_maximum_ssrgemm_nt_n, 128x64x8_64x32x1_8x8_8x4_2x2) {
  using precision = float;
  using OpClass   = cutlass::arch::OpClassSimt;
  using SmArch    = cutlass::arch::Sm50;

  using ThreadblockShape = cutlass::gemm::GemmShape<128, 64, 8>;
  using WarpShape        = cutlass::gemm::GemmShape<64, 32, 8>;
  using InstructionShape = cutlass::gemm::GemmShape<1, 1, 1>;

  using Config = typename cuasr::gemm::device::DefaultSemiRingConfiguration< //
      precision, precision, precision, precision, OpClass,                   //
      cuasr::minimum<precision>, cuasr::maximum<precision>, SmArch>;

  using AddOp            = Config::AdditionOp;
  using MultOp           = Config::MultiplicationOp;
  using EpilogueOutputOp = Config::EpilogueOutputOp;

  using Srgemm = cuasr::gemm::device::Srgemm<                           //
      AddOp, MultOp,                                                    //
      precision, cutlass::layout::ColumnMajor,                             //
      precision, cutlass::layout::RowMajor,                             //
      precision, cutlass::layout::ColumnMajor,                             //
      precision, OpClass, SmArch,                                       //
      ThreadblockShape, WarpShape, InstructionShape, EpilogueOutputOp,  //
      cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>, 2>;

  EXPECT_TRUE(cuasr::test::gemm::device::TestAllGemm<Srgemm>());
}
#endif

////////////////////////////////////////////////////////////////////////////////
// Elements / Thread:   2 x   2
//    Threads / Warp:   4 x   8
//     Warps / Block:   2 x   4
//       Threadblock:  16 x  64 x  16
#if defined(CUASR_TEST_LEVEL) and (CUASR_TEST_LEVEL >= 2)
TEST(SM50_device_minimum_maximum_ssrgemm_nt_n, 16x64x16_8x16x1_2x2_4x8_2x4) {
  using precision = float;
  using OpClass   = cutlass::arch::OpClassSimt;
  using SmArch    = cutlass::arch::Sm50;

  using ThreadblockShape = cutlass::gemm::GemmShape<16, 64, 16>;
  using WarpShape        = cutlass::gemm::GemmShape<8, 16, 16>;
  using InstructionShape = cutlass::gemm::GemmShape<1, 1, 1>;

  using Config = typename cuasr::gemm::device::DefaultSemiRingConfiguration< //
      precision, precision, precision, precision, OpClass,                   //
      cuasr::minimum<precision>, cuasr::maximum<precision>, SmArch>;

  using AddOp            = Config::AdditionOp;
  using MultOp           = Config::MultiplicationOp;
  using EpilogueOutputOp = Config::EpilogueOutputOp;

  using Srgemm = cuasr::gemm::device::Srgemm<                           //
      AddOp, MultOp,                                                    //
      precision, cutlass::layout::ColumnMajor,                             //
      precision, cutlass::layout::RowMajor,                             //
      precision, cutlass::layout::ColumnMajor,                             //
      precision, OpClass, SmArch,                                       //
      ThreadblockShape, WarpShape, InstructionShape, EpilogueOutputOp,  //
      cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>, 2>;

  EXPECT_TRUE(cuasr::test::gemm::device::TestAllGemm<Srgemm>());
}
#endif

////////////////////////////////////////////////////////////////////////////////
// Elements / Thread:   2 x   4
//    Threads / Warp:   4 x   8
//     Warps / Block:   2 x   4
//       Threadblock:  16 x 128 x  16
#if defined(CUASR_TEST_LEVEL) and (CUASR_TEST_LEVEL >= 2)
TEST(SM50_device_minimum_maximum_ssrgemm_nt_n, 16x128x16_8x32x1_2x4_4x8_2x4) {
  using precision = float;
  using OpClass   = cutlass::arch::OpClassSimt;
  using SmArch    = cutlass::arch::Sm50;

  using ThreadblockShape = cutlass::gemm::GemmShape<16, 128, 16>;
  using WarpShape        = cutlass::gemm::GemmShape<8, 32, 16>;
  using InstructionShape = cutlass::gemm::GemmShape<1, 1, 1>;

  using Config = typename cuasr::gemm::device::DefaultSemiRingConfiguration< //
      precision, precision, precision, precision, OpClass,                   //
      cuasr::minimum<precision>, cuasr::maximum<precision>, SmArch>;

  using AddOp            = Config::AdditionOp;
  using MultOp           = Config::MultiplicationOp;
  using EpilogueOutputOp = Config::EpilogueOutputOp;

  using Srgemm = cuasr::gemm::device::Srgemm<                           //
      AddOp, MultOp,                                                    //
      precision, cutlass::layout::ColumnMajor,                             //
      precision, cutlass::layout::RowMajor,                             //
      precision, cutlass::layout::ColumnMajor,                             //
      precision, OpClass, SmArch,                                       //
      ThreadblockShape, WarpShape, InstructionShape, EpilogueOutputOp,  //
      cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>, 2>;

  EXPECT_TRUE(cuasr::test::gemm::device::TestAllGemm<Srgemm>());
}
#endif

////////////////////////////////////////////////////////////////////////////////
// Elements / Thread:   2 x   2
//    Threads / Warp:   8 x   4
//     Warps / Block:   2 x   4
//       Threadblock:  32 x  32 x   8
#if defined(CUASR_TEST_LEVEL) and (CUASR_TEST_LEVEL >= 2)
TEST(SM50_device_minimum_maximum_ssrgemm_nt_n, 32x32x8_16x8x1_2x2_8x4_2x4) {
  using precision = float;
  using OpClass   = cutlass::arch::OpClassSimt;
  using SmArch    = cutlass::arch::Sm50;

  using ThreadblockShape = cutlass::gemm::GemmShape<32, 32, 8>;
  using WarpShape        = cutlass::gemm::GemmShape<16, 8, 8>;
  using InstructionShape = cutlass::gemm::GemmShape<1, 1, 1>;

  using Config = typename cuasr::gemm::device::DefaultSemiRingConfiguration< //
      precision, precision, precision, precision, OpClass,                   //
      cuasr::minimum<precision>, cuasr::maximum<precision>, SmArch>;

  using AddOp            = Config::AdditionOp;
  using MultOp           = Config::MultiplicationOp;
  using EpilogueOutputOp = Config::EpilogueOutputOp;

  using Srgemm = cuasr::gemm::device::Srgemm<                           //
      AddOp, MultOp,                                                    //
      precision, cutlass::layout::ColumnMajor,                             //
      precision, cutlass::layout::RowMajor,                             //
      precision, cutlass::layout::ColumnMajor,                             //
      precision, OpClass, SmArch,                                       //
      ThreadblockShape, WarpShape, InstructionShape, EpilogueOutputOp,  //
      cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>, 2>;

  EXPECT_TRUE(cuasr::test::gemm::device::TestAllGemm<Srgemm>());
}
#endif

////////////////////////////////////////////////////////////////////////////////
// Elements / Thread:   4 x   2
//    Threads / Warp:   4 x   8
//     Warps / Block:   2 x   4
//       Threadblock:  32 x  64 x   8
#if defined(CUASR_TEST_LEVEL) and (CUASR_TEST_LEVEL >= 2)
TEST(SM50_device_minimum_maximum_ssrgemm_nt_n, 32x64x8_16x16x1_4x2_4x8_2x4) {
  using precision = float;
  using OpClass   = cutlass::arch::OpClassSimt;
  using SmArch    = cutlass::arch::Sm50;

  using ThreadblockShape = cutlass::gemm::GemmShape<32, 64, 8>;
  using WarpShape        = cutlass::gemm::GemmShape<16, 16, 8>;
  using InstructionShape = cutlass::gemm::GemmShape<1, 1, 1>;

  using Config = typename cuasr::gemm::device::DefaultSemiRingConfiguration< //
      precision, precision, precision, precision, OpClass,                   //
      cuasr::minimum<precision>, cuasr::maximum<precision>, SmArch>;

  using AddOp            = Config::AdditionOp;
  using MultOp           = Config::MultiplicationOp;
  using EpilogueOutputOp = Config::EpilogueOutputOp;

  using Srgemm = cuasr::gemm::device::Srgemm<                           //
      AddOp, MultOp,                                                    //
      precision, cutlass::layout::ColumnMajor,                             //
      precision, cutlass::layout::RowMajor,                             //
      precision, cutlass::layout::ColumnMajor,                             //
      precision, OpClass, SmArch,                                       //
      ThreadblockShape, WarpShape, InstructionShape, EpilogueOutputOp,  //
      cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>, 2>;

  EXPECT_TRUE(cuasr::test::gemm::device::TestAllGemm<Srgemm>());
}
#endif

////////////////////////////////////////////////////////////////////////////////
// Elements / Thread:   4 x   4
//    Threads / Warp:   4 x   8
//     Warps / Block:   2 x   4
//       Threadblock:  32 x 128 x   8
#if defined(CUASR_TEST_LEVEL) and (CUASR_TEST_LEVEL >= 2)
TEST(SM50_device_minimum_maximum_ssrgemm_nt_n, 32x128x8_16x32x1_4x4_4x8_2x4) {
  using precision = float;
  using OpClass   = cutlass::arch::OpClassSimt;
  using SmArch    = cutlass::arch::Sm50;

  using ThreadblockShape = cutlass::gemm::GemmShape<32, 128, 8>;
  using WarpShape        = cutlass::gemm::GemmShape<16, 32, 8>;
  using InstructionShape = cutlass::gemm::GemmShape<1, 1, 1>;

  using Config = typename cuasr::gemm::device::DefaultSemiRingConfiguration< //
      precision, precision, precision, precision, OpClass,                   //
      cuasr::minimum<precision>, cuasr::maximum<precision>, SmArch>;

  using AddOp            = Config::AdditionOp;
  using MultOp           = Config::MultiplicationOp;
  using EpilogueOutputOp = Config::EpilogueOutputOp;

  using Srgemm = cuasr::gemm::device::Srgemm<                           //
      AddOp, MultOp,                                                    //
      precision, cutlass::layout::ColumnMajor,                             //
      precision, cutlass::layout::RowMajor,                             //
      precision, cutlass::layout::ColumnMajor,                             //
      precision, OpClass, SmArch,                                       //
      ThreadblockShape, WarpShape, InstructionShape, EpilogueOutputOp,  //
      cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>, 2>;

  EXPECT_TRUE(cuasr::test::gemm::device::TestAllGemm<Srgemm>());
}
#endif

////////////////////////////////////////////////////////////////////////////////
// Elements / Thread:   4 x   8
//    Threads / Warp:   4 x   8
//     Warps / Block:   2 x   4
//       Threadblock:  32 x 256 x   8
#if defined(CUASR_TEST_LEVEL) and (CUASR_TEST_LEVEL >= 1)
TEST(SM50_device_minimum_maximum_ssrgemm_nt_n, 32x256x8_16x64x1_4x8_4x8_2x4) {
  using precision = float;
  using OpClass   = cutlass::arch::OpClassSimt;
  using SmArch    = cutlass::arch::Sm50;

  using ThreadblockShape = cutlass::gemm::GemmShape<32, 256, 8>;
  using WarpShape        = cutlass::gemm::GemmShape<16, 64, 8>;
  using InstructionShape = cutlass::gemm::GemmShape<1, 1, 1>;

  using Config = typename cuasr::gemm::device::DefaultSemiRingConfiguration< //
      precision, precision, precision, precision, OpClass,                   //
      cuasr::minimum<precision>, cuasr::maximum<precision>, SmArch>;

  using AddOp            = Config::AdditionOp;
  using MultOp           = Config::MultiplicationOp;
  using EpilogueOutputOp = Config::EpilogueOutputOp;

  using Srgemm = cuasr::gemm::device::Srgemm<                           //
      AddOp, MultOp,                                                    //
      precision, cutlass::layout::ColumnMajor,                             //
      precision, cutlass::layout::RowMajor,                             //
      precision, cutlass::layout::ColumnMajor,                             //
      precision, OpClass, SmArch,                                       //
      ThreadblockShape, WarpShape, InstructionShape, EpilogueOutputOp,  //
      cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>, 2>;

  EXPECT_TRUE(cuasr::test::gemm::device::TestAllGemm<Srgemm>());
}
#endif

////////////////////////////////////////////////////////////////////////////////
// Elements / Thread:   4 x   4
//    Threads / Warp:   8 x   4
//     Warps / Block:   2 x   4
//       Threadblock:  64 x  64 x   8
#if defined(CUASR_TEST_LEVEL) and (CUASR_TEST_LEVEL >= 2)
TEST(SM50_device_minimum_maximum_ssrgemm_nt_n, 64x64x8_32x16x1_4x4_8x4_2x4) {
  using precision = float;
  using OpClass   = cutlass::arch::OpClassSimt;
  using SmArch    = cutlass::arch::Sm50;

  using ThreadblockShape = cutlass::gemm::GemmShape<64, 64, 8>;
  using WarpShape        = cutlass::gemm::GemmShape<32, 16, 8>;
  using InstructionShape = cutlass::gemm::GemmShape<1, 1, 1>;

  using Config = typename cuasr::gemm::device::DefaultSemiRingConfiguration< //
      precision, precision, precision, precision, OpClass,                   //
      cuasr::minimum<precision>, cuasr::maximum<precision>, SmArch>;

  using AddOp            = Config::AdditionOp;
  using MultOp           = Config::MultiplicationOp;
  using EpilogueOutputOp = Config::EpilogueOutputOp;

  using Srgemm = cuasr::gemm::device::Srgemm<                           //
      AddOp, MultOp,                                                    //
      precision, cutlass::layout::ColumnMajor,                             //
      precision, cutlass::layout::RowMajor,                             //
      precision, cutlass::layout::ColumnMajor,                             //
      precision, OpClass, SmArch,                                       //
      ThreadblockShape, WarpShape, InstructionShape, EpilogueOutputOp,  //
      cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>, 2>;

  EXPECT_TRUE(cuasr::test::gemm::device::TestAllGemm<Srgemm>());
}
#endif

////////////////////////////////////////////////////////////////////////////////
// Elements / Thread:   8 x   4
//    Threads / Warp:   4 x   8
//     Warps / Block:   2 x   4
//       Threadblock:  64 x 128 x   8
#if defined(CUASR_TEST_LEVEL) and (CUASR_TEST_LEVEL >= 2)
TEST(SM50_device_minimum_maximum_ssrgemm_nt_n, 64x128x8_32x32x1_8x4_4x8_2x4) {
  using precision = float;
  using OpClass   = cutlass::arch::OpClassSimt;
  using SmArch    = cutlass::arch::Sm50;

  using ThreadblockShape = cutlass::gemm::GemmShape<64, 128, 8>;
  using WarpShape        = cutlass::gemm::GemmShape<32, 32, 8>;
  using InstructionShape = cutlass::gemm::GemmShape<1, 1, 1>;

  using Config = typename cuasr::gemm::device::DefaultSemiRingConfiguration< //
      precision, precision, precision, precision, OpClass,                   //
      cuasr::minimum<precision>, cuasr::maximum<precision>, SmArch>;

  using AddOp            = Config::AdditionOp;
  using MultOp           = Config::MultiplicationOp;
  using EpilogueOutputOp = Config::EpilogueOutputOp;

  using Srgemm = cuasr::gemm::device::Srgemm<                           //
      AddOp, MultOp,                                                    //
      precision, cutlass::layout::ColumnMajor,                             //
      precision, cutlass::layout::RowMajor,                             //
      precision, cutlass::layout::ColumnMajor,                             //
      precision, OpClass, SmArch,                                       //
      ThreadblockShape, WarpShape, InstructionShape, EpilogueOutputOp,  //
      cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>, 2>;

  EXPECT_TRUE(cuasr::test::gemm::device::TestAllGemm<Srgemm>());
}
#endif

////////////////////////////////////////////////////////////////////////////////
// Elements / Thread:   8 x   8
//    Threads / Warp:   4 x   8
//     Warps / Block:   2 x   4
//       Threadblock:  64 x 256 x   8
#if defined(CUASR_TEST_LEVEL) and (CUASR_TEST_LEVEL >= 1)
TEST(SM50_device_minimum_maximum_ssrgemm_nt_n, 64x256x8_32x64x1_8x8_4x8_2x4) {
  using precision = float;
  using OpClass   = cutlass::arch::OpClassSimt;
  using SmArch    = cutlass::arch::Sm50;

  using ThreadblockShape = cutlass::gemm::GemmShape<64, 256, 8>;
  using WarpShape        = cutlass::gemm::GemmShape<32, 64, 8>;
  using InstructionShape = cutlass::gemm::GemmShape<1, 1, 1>;

  using Config = typename cuasr::gemm::device::DefaultSemiRingConfiguration< //
      precision, precision, precision, precision, OpClass,                   //
      cuasr::minimum<precision>, cuasr::maximum<precision>, SmArch>;

  using AddOp            = Config::AdditionOp;
  using MultOp           = Config::MultiplicationOp;
  using EpilogueOutputOp = Config::EpilogueOutputOp;

  using Srgemm = cuasr::gemm::device::Srgemm<                           //
      AddOp, MultOp,                                                    //
      precision, cutlass::layout::ColumnMajor,                             //
      precision, cutlass::layout::RowMajor,                             //
      precision, cutlass::layout::ColumnMajor,                             //
      precision, OpClass, SmArch,                                       //
      ThreadblockShape, WarpShape, InstructionShape, EpilogueOutputOp,  //
      cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>, 2>;

  EXPECT_TRUE(cuasr::test::gemm::device::TestAllGemm<Srgemm>());
}
#endif

////////////////////////////////////////////////////////////////////////////////
// Elements / Thread:   8 x   8
//    Threads / Warp:   8 x   4
//     Warps / Block:   2 x   4
//       Threadblock: 128 x 128 x   8
#if defined(CUASR_TEST_LEVEL) and (CUASR_TEST_LEVEL >= 0)
TEST(SM50_device_minimum_maximum_ssrgemm_nt_n, 128x128x8_64x32x1_8x8_8x4_2x4) {
  using precision = float;
  using OpClass   = cutlass::arch::OpClassSimt;
  using SmArch    = cutlass::arch::Sm50;

  using ThreadblockShape = cutlass::gemm::GemmShape<128, 128, 8>;
  using WarpShape        = cutlass::gemm::GemmShape<64, 32, 8>;
  using InstructionShape = cutlass::gemm::GemmShape<1, 1, 1>;

  using Config = typename cuasr::gemm::device::DefaultSemiRingConfiguration< //
      precision, precision, precision, precision, OpClass,                   //
      cuasr::minimum<precision>, cuasr::maximum<precision>, SmArch>;

  using AddOp            = Config::AdditionOp;
  using MultOp           = Config::MultiplicationOp;
  using EpilogueOutputOp = Config::EpilogueOutputOp;

  using Srgemm = cuasr::gemm::device::Srgemm<                           //
      AddOp, MultOp,                                                    //
      precision, cutlass::layout::ColumnMajor,                             //
      precision, cutlass::layout::RowMajor,                             //
      precision, cutlass::layout::ColumnMajor,                             //
      precision, OpClass, SmArch,                                       //
      ThreadblockShape, WarpShape, InstructionShape, EpilogueOutputOp,  //
      cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>, 2>;

  EXPECT_TRUE(cuasr::test::gemm::device::TestAllGemm<Srgemm>());
}
#endif

////////////////////////////////////////////////////////////////////////////////
// Elements / Thread:   2 x   2
//    Threads / Warp:   4 x   8
//     Warps / Block:   4 x   2
//       Threadblock:  32 x  32 x   8
#if defined(CUASR_TEST_LEVEL) and (CUASR_TEST_LEVEL >= 2)
TEST(SM50_device_minimum_maximum_ssrgemm_nt_n, 32x32x8_8x16x1_2x2_4x8_4x2) {
  using precision = float;
  using OpClass   = cutlass::arch::OpClassSimt;
  using SmArch    = cutlass::arch::Sm50;

  using ThreadblockShape = cutlass::gemm::GemmShape<32, 32, 8>;
  using WarpShape        = cutlass::gemm::GemmShape<8, 16, 8>;
  using InstructionShape = cutlass::gemm::GemmShape<1, 1, 1>;

  using Config = typename cuasr::gemm::device::DefaultSemiRingConfiguration< //
      precision, precision, precision, precision, OpClass,                   //
      cuasr::minimum<precision>, cuasr::maximum<precision>, SmArch>;

  using AddOp            = Config::AdditionOp;
  using MultOp           = Config::MultiplicationOp;
  using EpilogueOutputOp = Config::EpilogueOutputOp;

  using Srgemm = cuasr::gemm::device::Srgemm<                           //
      AddOp, MultOp,                                                    //
      precision, cutlass::layout::ColumnMajor,                             //
      precision, cutlass::layout::RowMajor,                             //
      precision, cutlass::layout::ColumnMajor,                             //
      precision, OpClass, SmArch,                                       //
      ThreadblockShape, WarpShape, InstructionShape, EpilogueOutputOp,  //
      cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>, 2>;

  EXPECT_TRUE(cuasr::test::gemm::device::TestAllGemm<Srgemm>());
}
#endif

////////////////////////////////////////////////////////////////////////////////
// Elements / Thread:   4 x   2
//    Threads / Warp:   4 x   8
//     Warps / Block:   4 x   2
//       Threadblock:  64 x  32 x   8
#if defined(CUASR_TEST_LEVEL) and (CUASR_TEST_LEVEL >= 2)
TEST(SM50_device_minimum_maximum_ssrgemm_nt_n, 64x32x8_16x16x1_4x2_4x8_4x2) {
  using precision = float;
  using OpClass   = cutlass::arch::OpClassSimt;
  using SmArch    = cutlass::arch::Sm50;

  using ThreadblockShape = cutlass::gemm::GemmShape<64, 32, 8>;
  using WarpShape        = cutlass::gemm::GemmShape<16, 16, 8>;
  using InstructionShape = cutlass::gemm::GemmShape<1, 1, 1>;

  using Config = typename cuasr::gemm::device::DefaultSemiRingConfiguration< //
      precision, precision, precision, precision, OpClass,                   //
      cuasr::minimum<precision>, cuasr::maximum<precision>, SmArch>;

  using AddOp            = Config::AdditionOp;
  using MultOp           = Config::MultiplicationOp;
  using EpilogueOutputOp = Config::EpilogueOutputOp;

  using Srgemm = cuasr::gemm::device::Srgemm<                           //
      AddOp, MultOp,                                                    //
      precision, cutlass::layout::ColumnMajor,                             //
      precision, cutlass::layout::RowMajor,                             //
      precision, cutlass::layout::ColumnMajor,                             //
      precision, OpClass, SmArch,                                       //
      ThreadblockShape, WarpShape, InstructionShape, EpilogueOutputOp,  //
      cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>, 2>;

  EXPECT_TRUE(cuasr::test::gemm::device::TestAllGemm<Srgemm>());
}
#endif

////////////////////////////////////////////////////////////////////////////////
// Elements / Thread:   4 x   4
//    Threads / Warp:   4 x   8
//     Warps / Block:   4 x   2
//       Threadblock:  64 x  64 x   8
#if defined(CUASR_TEST_LEVEL) and (CUASR_TEST_LEVEL >= 2)
TEST(SM50_device_minimum_maximum_ssrgemm_nt_n, 64x64x8_16x32x1_4x4_4x8_4x2) {
  using precision = float;
  using OpClass   = cutlass::arch::OpClassSimt;
  using SmArch    = cutlass::arch::Sm50;

  using ThreadblockShape = cutlass::gemm::GemmShape<64, 64, 8>;
  using WarpShape        = cutlass::gemm::GemmShape<16, 32, 8>;
  using InstructionShape = cutlass::gemm::GemmShape<1, 1, 1>;

  using Config = typename cuasr::gemm::device::DefaultSemiRingConfiguration< //
      precision, precision, precision, precision, OpClass,                   //
      cuasr::minimum<precision>, cuasr::maximum<precision>, SmArch>;

  using AddOp            = Config::AdditionOp;
  using MultOp           = Config::MultiplicationOp;
  using EpilogueOutputOp = Config::EpilogueOutputOp;

  using Srgemm = cuasr::gemm::device::Srgemm<                           //
      AddOp, MultOp,                                                    //
      precision, cutlass::layout::ColumnMajor,                             //
      precision, cutlass::layout::RowMajor,                             //
      precision, cutlass::layout::ColumnMajor,                             //
      precision, OpClass, SmArch,                                       //
      ThreadblockShape, WarpShape, InstructionShape, EpilogueOutputOp,  //
      cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>, 2>;

  EXPECT_TRUE(cuasr::test::gemm::device::TestAllGemm<Srgemm>());
}
#endif

////////////////////////////////////////////////////////////////////////////////
// Elements / Thread:   4 x   4
//    Threads / Warp:   8 x   4
//     Warps / Block:   4 x   2
//       Threadblock: 128 x  32 x   8
#if defined(CUASR_TEST_LEVEL) and (CUASR_TEST_LEVEL >= 2)
TEST(SM50_device_minimum_maximum_ssrgemm_nt_n, 128x32x8_32x16x1_4x4_8x4_4x2) {
  using precision = float;
  using OpClass   = cutlass::arch::OpClassSimt;
  using SmArch    = cutlass::arch::Sm50;

  using ThreadblockShape = cutlass::gemm::GemmShape<128, 32, 8>;
  using WarpShape        = cutlass::gemm::GemmShape<32, 16, 8>;
  using InstructionShape = cutlass::gemm::GemmShape<1, 1, 1>;

  using Config = typename cuasr::gemm::device::DefaultSemiRingConfiguration< //
      precision, precision, precision, precision, OpClass,                   //
      cuasr::minimum<precision>, cuasr::maximum<precision>, SmArch>;

  using AddOp            = Config::AdditionOp;
  using MultOp           = Config::MultiplicationOp;
  using EpilogueOutputOp = Config::EpilogueOutputOp;

  using Srgemm = cuasr::gemm::device::Srgemm<                           //
      AddOp, MultOp,                                                    //
      precision, cutlass::layout::ColumnMajor,                             //
      precision, cutlass::layout::RowMajor,                             //
      precision, cutlass::layout::ColumnMajor,                             //
      precision, OpClass, SmArch,                                       //
      ThreadblockShape, WarpShape, InstructionShape, EpilogueOutputOp,  //
      cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>, 2>;

  EXPECT_TRUE(cuasr::test::gemm::device::TestAllGemm<Srgemm>());
}
#endif

////////////////////////////////////////////////////////////////////////////////
// Elements / Thread:   8 x   4
//    Threads / Warp:   4 x   8
//     Warps / Block:   4 x   2
//       Threadblock: 128 x  64 x   8
#if defined(CUASR_TEST_LEVEL) and (CUASR_TEST_LEVEL >= 2)
TEST(SM50_device_minimum_maximum_ssrgemm_nt_n, 128x64x8_32x32x1_8x4_4x8_4x2) {
  using precision = float;
  using OpClass   = cutlass::arch::OpClassSimt;
  using SmArch    = cutlass::arch::Sm50;

  using ThreadblockShape = cutlass::gemm::GemmShape<128, 64, 8>;
  using WarpShape        = cutlass::gemm::GemmShape<32, 32, 8>;
  using InstructionShape = cutlass::gemm::GemmShape<1, 1, 1>;

  using Config = typename cuasr::gemm::device::DefaultSemiRingConfiguration< //
      precision, precision, precision, precision, OpClass,                   //
      cuasr::minimum<precision>, cuasr::maximum<precision>, SmArch>;

  using AddOp            = Config::AdditionOp;
  using MultOp           = Config::MultiplicationOp;
  using EpilogueOutputOp = Config::EpilogueOutputOp;

  using Srgemm = cuasr::gemm::device::Srgemm<                           //
      AddOp, MultOp,                                                    //
      precision, cutlass::layout::ColumnMajor,                             //
      precision, cutlass::layout::RowMajor,                             //
      precision, cutlass::layout::ColumnMajor,                             //
      precision, OpClass, SmArch,                                       //
      ThreadblockShape, WarpShape, InstructionShape, EpilogueOutputOp,  //
      cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>, 2>;

  EXPECT_TRUE(cuasr::test::gemm::device::TestAllGemm<Srgemm>());
}
#endif

////////////////////////////////////////////////////////////////////////////////
// Elements / Thread:   8 x   8
//    Threads / Warp:   4 x   8
//     Warps / Block:   4 x   2
//       Threadblock: 128 x 128 x   8
#if defined(CUASR_TEST_LEVEL) and (CUASR_TEST_LEVEL >= 1)
TEST(SM50_device_minimum_maximum_ssrgemm_nt_n, 128x128x8_32x64x1_8x8_4x8_4x2) {
  using precision = float;
  using OpClass   = cutlass::arch::OpClassSimt;
  using SmArch    = cutlass::arch::Sm50;

  using ThreadblockShape = cutlass::gemm::GemmShape<128, 128, 8>;
  using WarpShape        = cutlass::gemm::GemmShape<32, 64, 8>;
  using InstructionShape = cutlass::gemm::GemmShape<1, 1, 1>;

  using Config = typename cuasr::gemm::device::DefaultSemiRingConfiguration< //
      precision, precision, precision, precision, OpClass,                   //
      cuasr::minimum<precision>, cuasr::maximum<precision>, SmArch>;

  using AddOp            = Config::AdditionOp;
  using MultOp           = Config::MultiplicationOp;
  using EpilogueOutputOp = Config::EpilogueOutputOp;

  using Srgemm = cuasr::gemm::device::Srgemm<                           //
      AddOp, MultOp,                                                    //
      precision, cutlass::layout::ColumnMajor,                             //
      precision, cutlass::layout::RowMajor,                             //
      precision, cutlass::layout::ColumnMajor,                             //
      precision, OpClass, SmArch,                                       //
      ThreadblockShape, WarpShape, InstructionShape, EpilogueOutputOp,  //
      cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>, 2>;

  EXPECT_TRUE(cuasr::test::gemm::device::TestAllGemm<Srgemm>());
}
#endif

////////////////////////////////////////////////////////////////////////////////
// Elements / Thread:   8 x   4
//    Threads / Warp:   8 x   4
//     Warps / Block:   4 x   2
//       Threadblock: 256 x  32 x   8
#if defined(CUASR_TEST_LEVEL) and (CUASR_TEST_LEVEL >= 1)
TEST(SM50_device_minimum_maximum_ssrgemm_nt_n, 256x32x8_64x16x1_8x4_8x4_4x2) {
  using precision = float;
  using OpClass   = cutlass::arch::OpClassSimt;
  using SmArch    = cutlass::arch::Sm50;

  using ThreadblockShape = cutlass::gemm::GemmShape<256, 32, 8>;
  using WarpShape        = cutlass::gemm::GemmShape<64, 16, 8>;
  using InstructionShape = cutlass::gemm::GemmShape<1, 1, 1>;

  using Config = typename cuasr::gemm::device::DefaultSemiRingConfiguration< //
      precision, precision, precision, precision, OpClass,                   //
      cuasr::minimum<precision>, cuasr::maximum<precision>, SmArch>;

  using AddOp            = Config::AdditionOp;
  using MultOp           = Config::MultiplicationOp;
  using EpilogueOutputOp = Config::EpilogueOutputOp;

  using Srgemm = cuasr::gemm::device::Srgemm<                           //
      AddOp, MultOp,                                                    //
      precision, cutlass::layout::ColumnMajor,                             //
      precision, cutlass::layout::RowMajor,                             //
      precision, cutlass::layout::ColumnMajor,                             //
      precision, OpClass, SmArch,                                       //
      ThreadblockShape, WarpShape, InstructionShape, EpilogueOutputOp,  //
      cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>, 2>;

  EXPECT_TRUE(cuasr::test::gemm::device::TestAllGemm<Srgemm>());
}
#endif

////////////////////////////////////////////////////////////////////////////////
// Elements / Thread:   8 x   8
//    Threads / Warp:   8 x   4
//     Warps / Block:   4 x   2
//       Threadblock: 256 x  64 x   8
#if defined(CUASR_TEST_LEVEL) and (CUASR_TEST_LEVEL >= 1)
TEST(SM50_device_minimum_maximum_ssrgemm_nt_n, 256x64x8_64x32x1_8x8_8x4_4x2) {
  using precision = float;
  using OpClass   = cutlass::arch::OpClassSimt;
  using SmArch    = cutlass::arch::Sm50;

  using ThreadblockShape = cutlass::gemm::GemmShape<256, 64, 8>;
  using WarpShape        = cutlass::gemm::GemmShape<64, 32, 8>;
  using InstructionShape = cutlass::gemm::GemmShape<1, 1, 1>;

  using Config = typename cuasr::gemm::device::DefaultSemiRingConfiguration< //
      precision, precision, precision, precision, OpClass,                   //
      cuasr::minimum<precision>, cuasr::maximum<precision>, SmArch>;

  using AddOp            = Config::AdditionOp;
  using MultOp           = Config::MultiplicationOp;
  using EpilogueOutputOp = Config::EpilogueOutputOp;

  using Srgemm = cuasr::gemm::device::Srgemm<                           //
      AddOp, MultOp,                                                    //
      precision, cutlass::layout::ColumnMajor,                             //
      precision, cutlass::layout::RowMajor,                             //
      precision, cutlass::layout::ColumnMajor,                             //
      precision, OpClass, SmArch,                                       //
      ThreadblockShape, WarpShape, InstructionShape, EpilogueOutputOp,  //
      cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>, 2>;

  EXPECT_TRUE(cuasr::test::gemm::device::TestAllGemm<Srgemm>());
}
#endif

////////////////////////////////////////////////////////////////////////////////
// Elements / Thread:   2 x   2
//    Threads / Warp:   4 x   8
//     Warps / Block:   4 x   4
//       Threadblock:  32 x  64 x  16
#if defined(CUASR_TEST_LEVEL) and (CUASR_TEST_LEVEL >= 2)
TEST(SM50_device_minimum_maximum_ssrgemm_nt_n, 32x64x16_8x16x1_2x2_4x8_4x4) {
  using precision = float;
  using OpClass   = cutlass::arch::OpClassSimt;
  using SmArch    = cutlass::arch::Sm50;

  using ThreadblockShape = cutlass::gemm::GemmShape<32, 64, 16>;
  using WarpShape        = cutlass::gemm::GemmShape<8, 16, 16>;
  using InstructionShape = cutlass::gemm::GemmShape<1, 1, 1>;

  using Config = typename cuasr::gemm::device::DefaultSemiRingConfiguration< //
      precision, precision, precision, precision, OpClass,                   //
      cuasr::minimum<precision>, cuasr::maximum<precision>, SmArch>;

  using AddOp            = Config::AdditionOp;
  using MultOp           = Config::MultiplicationOp;
  using EpilogueOutputOp = Config::EpilogueOutputOp;

  using Srgemm = cuasr::gemm::device::Srgemm<                           //
      AddOp, MultOp,                                                    //
      precision, cutlass::layout::ColumnMajor,                             //
      precision, cutlass::layout::RowMajor,                             //
      precision, cutlass::layout::ColumnMajor,                             //
      precision, OpClass, SmArch,                                       //
      ThreadblockShape, WarpShape, InstructionShape, EpilogueOutputOp,  //
      cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>, 2>;

  EXPECT_TRUE(cuasr::test::gemm::device::TestAllGemm<Srgemm>());
}
#endif

////////////////////////////////////////////////////////////////////////////////
// Elements / Thread:   2 x   4
//    Threads / Warp:   4 x   8
//     Warps / Block:   4 x   4
//       Threadblock:  32 x 128 x  16
#if defined(CUASR_TEST_LEVEL) and (CUASR_TEST_LEVEL >= 2)
TEST(SM50_device_minimum_maximum_ssrgemm_nt_n, 32x128x16_8x32x1_2x4_4x8_4x4) {
  using precision = float;
  using OpClass   = cutlass::arch::OpClassSimt;
  using SmArch    = cutlass::arch::Sm50;

  using ThreadblockShape = cutlass::gemm::GemmShape<32, 128, 16>;
  using WarpShape        = cutlass::gemm::GemmShape<8, 32, 16>;
  using InstructionShape = cutlass::gemm::GemmShape<1, 1, 1>;

  using Config = typename cuasr::gemm::device::DefaultSemiRingConfiguration< //
      precision, precision, precision, precision, OpClass,                   //
      cuasr::minimum<precision>, cuasr::maximum<precision>, SmArch>;

  using AddOp            = Config::AdditionOp;
  using MultOp           = Config::MultiplicationOp;
  using EpilogueOutputOp = Config::EpilogueOutputOp;

  using Srgemm = cuasr::gemm::device::Srgemm<                           //
      AddOp, MultOp,                                                    //
      precision, cutlass::layout::ColumnMajor,                             //
      precision, cutlass::layout::RowMajor,                             //
      precision, cutlass::layout::ColumnMajor,                             //
      precision, OpClass, SmArch,                                       //
      ThreadblockShape, WarpShape, InstructionShape, EpilogueOutputOp,  //
      cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>, 2>;

  EXPECT_TRUE(cuasr::test::gemm::device::TestAllGemm<Srgemm>());
}
#endif

////////////////////////////////////////////////////////////////////////////////
// Elements / Thread:   2 x   2
//    Threads / Warp:   8 x   4
//     Warps / Block:   4 x   4
//       Threadblock:  64 x  32 x  16
#if defined(CUASR_TEST_LEVEL) and (CUASR_TEST_LEVEL >= 2)
TEST(SM50_device_minimum_maximum_ssrgemm_nt_n, 64x32x16_16x8x1_2x2_8x4_4x4) {
  using precision = float;
  using OpClass   = cutlass::arch::OpClassSimt;
  using SmArch    = cutlass::arch::Sm50;

  using ThreadblockShape = cutlass::gemm::GemmShape<64, 32, 16>;
  using WarpShape        = cutlass::gemm::GemmShape<16, 8, 16>;
  using InstructionShape = cutlass::gemm::GemmShape<1, 1, 1>;

  using Config = typename cuasr::gemm::device::DefaultSemiRingConfiguration< //
      precision, precision, precision, precision, OpClass,                   //
      cuasr::minimum<precision>, cuasr::maximum<precision>, SmArch>;

  using AddOp            = Config::AdditionOp;
  using MultOp           = Config::MultiplicationOp;
  using EpilogueOutputOp = Config::EpilogueOutputOp;

  using Srgemm = cuasr::gemm::device::Srgemm<                           //
      AddOp, MultOp,                                                    //
      precision, cutlass::layout::ColumnMajor,                             //
      precision, cutlass::layout::RowMajor,                             //
      precision, cutlass::layout::ColumnMajor,                             //
      precision, OpClass, SmArch,                                       //
      ThreadblockShape, WarpShape, InstructionShape, EpilogueOutputOp,  //
      cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>, 2>;

  EXPECT_TRUE(cuasr::test::gemm::device::TestAllGemm<Srgemm>());
}
#endif

////////////////////////////////////////////////////////////////////////////////
// Elements / Thread:   4 x   2
//    Threads / Warp:   4 x   8
//     Warps / Block:   4 x   4
//       Threadblock:  64 x  64 x   8
#if defined(CUASR_TEST_LEVEL) and (CUASR_TEST_LEVEL >= 2)
TEST(SM50_device_minimum_maximum_ssrgemm_nt_n, 64x64x8_16x16x1_4x2_4x8_4x4) {
  using precision = float;
  using OpClass   = cutlass::arch::OpClassSimt;
  using SmArch    = cutlass::arch::Sm50;

  using ThreadblockShape = cutlass::gemm::GemmShape<64, 64, 8>;
  using WarpShape        = cutlass::gemm::GemmShape<16, 16, 8>;
  using InstructionShape = cutlass::gemm::GemmShape<1, 1, 1>;

  using Config = typename cuasr::gemm::device::DefaultSemiRingConfiguration< //
      precision, precision, precision, precision, OpClass,                   //
      cuasr::minimum<precision>, cuasr::maximum<precision>, SmArch>;

  using AddOp            = Config::AdditionOp;
  using MultOp           = Config::MultiplicationOp;
  using EpilogueOutputOp = Config::EpilogueOutputOp;

  using Srgemm = cuasr::gemm::device::Srgemm<                           //
      AddOp, MultOp,                                                    //
      precision, cutlass::layout::ColumnMajor,                             //
      precision, cutlass::layout::RowMajor,                             //
      precision, cutlass::layout::ColumnMajor,                             //
      precision, OpClass, SmArch,                                       //
      ThreadblockShape, WarpShape, InstructionShape, EpilogueOutputOp,  //
      cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>, 2>;

  EXPECT_TRUE(cuasr::test::gemm::device::TestAllGemm<Srgemm>());
}
#endif

////////////////////////////////////////////////////////////////////////////////
// Elements / Thread:   4 x   4
//    Threads / Warp:   4 x   8
//     Warps / Block:   4 x   4
//       Threadblock:  64 x 128 x   8
#if defined(CUASR_TEST_LEVEL) and (CUASR_TEST_LEVEL >= 2)
TEST(SM50_device_minimum_maximum_ssrgemm_nt_n, 64x128x8_16x32x1_4x4_4x8_4x4) {
  using precision = float;
  using OpClass   = cutlass::arch::OpClassSimt;
  using SmArch    = cutlass::arch::Sm50;

  using ThreadblockShape = cutlass::gemm::GemmShape<64, 128, 8>;
  using WarpShape        = cutlass::gemm::GemmShape<16, 32, 8>;
  using InstructionShape = cutlass::gemm::GemmShape<1, 1, 1>;

  using Config = typename cuasr::gemm::device::DefaultSemiRingConfiguration< //
      precision, precision, precision, precision, OpClass,                   //
      cuasr::minimum<precision>, cuasr::maximum<precision>, SmArch>;

  using AddOp            = Config::AdditionOp;
  using MultOp           = Config::MultiplicationOp;
  using EpilogueOutputOp = Config::EpilogueOutputOp;

  using Srgemm = cuasr::gemm::device::Srgemm<                           //
      AddOp, MultOp,                                                    //
      precision, cutlass::layout::ColumnMajor,                             //
      precision, cutlass::layout::RowMajor,                             //
      precision, cutlass::layout::ColumnMajor,                             //
      precision, OpClass, SmArch,                                       //
      ThreadblockShape, WarpShape, InstructionShape, EpilogueOutputOp,  //
      cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>, 2>;

  EXPECT_TRUE(cuasr::test::gemm::device::TestAllGemm<Srgemm>());
}
#endif

////////////////////////////////////////////////////////////////////////////////
// Elements / Thread:   4 x   8
//    Threads / Warp:   4 x   8
//     Warps / Block:   4 x   4
//       Threadblock:  64 x 256 x   8
#if defined(CUASR_TEST_LEVEL) and (CUASR_TEST_LEVEL >= 2)
TEST(SM50_device_minimum_maximum_ssrgemm_nt_n, 64x256x8_16x64x1_4x8_4x8_4x4) {
  using precision = float;
  using OpClass   = cutlass::arch::OpClassSimt;
  using SmArch    = cutlass::arch::Sm50;

  using ThreadblockShape = cutlass::gemm::GemmShape<64, 256, 8>;
  using WarpShape        = cutlass::gemm::GemmShape<16, 64, 8>;
  using InstructionShape = cutlass::gemm::GemmShape<1, 1, 1>;

  using Config = typename cuasr::gemm::device::DefaultSemiRingConfiguration< //
      precision, precision, precision, precision, OpClass,                   //
      cuasr::minimum<precision>, cuasr::maximum<precision>, SmArch>;

  using AddOp            = Config::AdditionOp;
  using MultOp           = Config::MultiplicationOp;
  using EpilogueOutputOp = Config::EpilogueOutputOp;

  using Srgemm = cuasr::gemm::device::Srgemm<                           //
      AddOp, MultOp,                                                    //
      precision, cutlass::layout::ColumnMajor,                             //
      precision, cutlass::layout::RowMajor,                             //
      precision, cutlass::layout::ColumnMajor,                             //
      precision, OpClass, SmArch,                                       //
      ThreadblockShape, WarpShape, InstructionShape, EpilogueOutputOp,  //
      cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>, 2>;

  EXPECT_TRUE(cuasr::test::gemm::device::TestAllGemm<Srgemm>());
}
#endif

////////////////////////////////////////////////////////////////////////////////
// Elements / Thread:   4 x   2
//    Threads / Warp:   8 x   4
//     Warps / Block:   4 x   4
//       Threadblock: 128 x  32 x  16
#if defined(CUASR_TEST_LEVEL) and (CUASR_TEST_LEVEL >= 2)
TEST(SM50_device_minimum_maximum_ssrgemm_nt_n, 128x32x16_32x8x1_4x2_8x4_4x4) {
  using precision = float;
  using OpClass   = cutlass::arch::OpClassSimt;
  using SmArch    = cutlass::arch::Sm50;

  using ThreadblockShape = cutlass::gemm::GemmShape<128, 32, 16>;
  using WarpShape        = cutlass::gemm::GemmShape<32, 8, 16>;
  using InstructionShape = cutlass::gemm::GemmShape<1, 1, 1>;

  using Config = typename cuasr::gemm::device::DefaultSemiRingConfiguration< //
      precision, precision, precision, precision, OpClass,                   //
      cuasr::minimum<precision>, cuasr::maximum<precision>, SmArch>;

  using AddOp            = Config::AdditionOp;
  using MultOp           = Config::MultiplicationOp;
  using EpilogueOutputOp = Config::EpilogueOutputOp;

  using Srgemm = cuasr::gemm::device::Srgemm<                           //
      AddOp, MultOp,                                                    //
      precision, cutlass::layout::ColumnMajor,                             //
      precision, cutlass::layout::RowMajor,                             //
      precision, cutlass::layout::ColumnMajor,                             //
      precision, OpClass, SmArch,                                       //
      ThreadblockShape, WarpShape, InstructionShape, EpilogueOutputOp,  //
      cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>, 2>;

  EXPECT_TRUE(cuasr::test::gemm::device::TestAllGemm<Srgemm>());
}
#endif

////////////////////////////////////////////////////////////////////////////////
// Elements / Thread:   4 x   4
//    Threads / Warp:   8 x   4
//     Warps / Block:   4 x   4
//       Threadblock: 128 x  64 x   8
#if defined(CUASR_TEST_LEVEL) and (CUASR_TEST_LEVEL >= 2)
TEST(SM50_device_minimum_maximum_ssrgemm_nt_n, 128x64x8_32x16x1_4x4_8x4_4x4) {
  using precision = float;
  using OpClass   = cutlass::arch::OpClassSimt;
  using SmArch    = cutlass::arch::Sm50;

  using ThreadblockShape = cutlass::gemm::GemmShape<128, 64, 8>;
  using WarpShape        = cutlass::gemm::GemmShape<32, 16, 8>;
  using InstructionShape = cutlass::gemm::GemmShape<1, 1, 1>;

  using Config = typename cuasr::gemm::device::DefaultSemiRingConfiguration< //
      precision, precision, precision, precision, OpClass,                   //
      cuasr::minimum<precision>, cuasr::maximum<precision>, SmArch>;

  using AddOp            = Config::AdditionOp;
  using MultOp           = Config::MultiplicationOp;
  using EpilogueOutputOp = Config::EpilogueOutputOp;

  using Srgemm = cuasr::gemm::device::Srgemm<                           //
      AddOp, MultOp,                                                    //
      precision, cutlass::layout::ColumnMajor,                             //
      precision, cutlass::layout::RowMajor,                             //
      precision, cutlass::layout::ColumnMajor,                             //
      precision, OpClass, SmArch,                                       //
      ThreadblockShape, WarpShape, InstructionShape, EpilogueOutputOp,  //
      cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>, 2>;

  EXPECT_TRUE(cuasr::test::gemm::device::TestAllGemm<Srgemm>());
}
#endif

////////////////////////////////////////////////////////////////////////////////
// Elements / Thread:   8 x   4
//    Threads / Warp:   4 x   8
//     Warps / Block:   4 x   4
//       Threadblock: 128 x 128 x   8
#if defined(CUASR_TEST_LEVEL) and (CUASR_TEST_LEVEL >= 2)
TEST(SM50_device_minimum_maximum_ssrgemm_nt_n, 128x128x8_32x32x1_8x4_4x8_4x4) {
  using precision = float;
  using OpClass   = cutlass::arch::OpClassSimt;
  using SmArch    = cutlass::arch::Sm50;

  using ThreadblockShape = cutlass::gemm::GemmShape<128, 128, 8>;
  using WarpShape        = cutlass::gemm::GemmShape<32, 32, 8>;
  using InstructionShape = cutlass::gemm::GemmShape<1, 1, 1>;

  using Config = typename cuasr::gemm::device::DefaultSemiRingConfiguration< //
      precision, precision, precision, precision, OpClass,                   //
      cuasr::minimum<precision>, cuasr::maximum<precision>, SmArch>;

  using AddOp            = Config::AdditionOp;
  using MultOp           = Config::MultiplicationOp;
  using EpilogueOutputOp = Config::EpilogueOutputOp;

  using Srgemm = cuasr::gemm::device::Srgemm<                           //
      AddOp, MultOp,                                                    //
      precision, cutlass::layout::ColumnMajor,                             //
      precision, cutlass::layout::RowMajor,                             //
      precision, cutlass::layout::ColumnMajor,                             //
      precision, OpClass, SmArch,                                       //
      ThreadblockShape, WarpShape, InstructionShape, EpilogueOutputOp,  //
      cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>, 2>;

  EXPECT_TRUE(cuasr::test::gemm::device::TestAllGemm<Srgemm>());
}
#endif

////////////////////////////////////////////////////////////////////////////////
// Elements / Thread:   8 x   4
//    Threads / Warp:   8 x   4
//     Warps / Block:   4 x   4
//       Threadblock: 256 x  64 x   8
#if defined(CUASR_TEST_LEVEL) and (CUASR_TEST_LEVEL >= 2)
TEST(SM50_device_minimum_maximum_ssrgemm_nt_n, 256x64x8_64x16x1_8x4_8x4_4x4) {
  using precision = float;
  using OpClass   = cutlass::arch::OpClassSimt;
  using SmArch    = cutlass::arch::Sm50;

  using ThreadblockShape = cutlass::gemm::GemmShape<256, 64, 8>;
  using WarpShape        = cutlass::gemm::GemmShape<64, 16, 8>;
  using InstructionShape = cutlass::gemm::GemmShape<1, 1, 1>;

  using Config = typename cuasr::gemm::device::DefaultSemiRingConfiguration< //
      precision, precision, precision, precision, OpClass,                   //
      cuasr::minimum<precision>, cuasr::maximum<precision>, SmArch>;

  using AddOp            = Config::AdditionOp;
  using MultOp           = Config::MultiplicationOp;
  using EpilogueOutputOp = Config::EpilogueOutputOp;

  using Srgemm = cuasr::gemm::device::Srgemm<                           //
      AddOp, MultOp,                                                    //
      precision, cutlass::layout::ColumnMajor,                             //
      precision, cutlass::layout::RowMajor,                             //
      precision, cutlass::layout::ColumnMajor,                             //
      precision, OpClass, SmArch,                                       //
      ThreadblockShape, WarpShape, InstructionShape, EpilogueOutputOp,  //
      cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>, 2>;

  EXPECT_TRUE(cuasr::test::gemm::device::TestAllGemm<Srgemm>());
}
#endif

