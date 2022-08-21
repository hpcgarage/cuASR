/***************************************************************************************************
* Copyright (c) 2022, Vijay Thakkar (thakkarv@gatech.edu).
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

///////////////////////////////////////////////////////////////////////////////

TEST(SM80_default_plus_mult_srgemm, f64_tt_n) {
  using precision = double;
  using OpClass   = cutlass::arch::OpClassSimt;
  using SmArch    = cutlass::arch::Sm80;
  using RingOp = cuasr::plus_mult<precision>;

  using Srgemm = cuasr::gemm::device::Srgemm<                           //
      RingOp,                                                           //
      precision, cutlass::layout::RowMajor,                   //
      precision, cutlass::layout::RowMajor,                   //
      precision, cutlass::layout::ColumnMajor,
      precision, OpClass, SmArch>;

  EXPECT_TRUE(cuasr::test::gemm::device::TestAllGemm<Srgemm>());
}

///////////////////////////////////////////////////////////////////////////////

TEST(SM80_default_plus_mult_srgemm, f64_tt_t) {
  using precision = double;
  using OpClass   = cutlass::arch::OpClassSimt;
  using SmArch    = cutlass::arch::Sm80;
  using RingOp = cuasr::plus_mult<precision>;

  using Srgemm = cuasr::gemm::device::Srgemm<                           //
      RingOp,                                                           //
      precision, cutlass::layout::RowMajor,                   //
      precision, cutlass::layout::RowMajor,                   //
      precision, cutlass::layout::RowMajor,
      precision, OpClass, SmArch>;

  EXPECT_TRUE(cuasr::test::gemm::device::TestAllGemm<Srgemm>());
}

///////////////////////////////////////////////////////////////////////////////

TEST(SM80_default_plus_mult_srgemm, f64_tn_n) {
  using precision = double;
  using OpClass   = cutlass::arch::OpClassSimt;
  using SmArch    = cutlass::arch::Sm80;
  using RingOp = cuasr::plus_mult<precision>;

  using Srgemm = cuasr::gemm::device::Srgemm<                           //
      RingOp,                                                           //
      precision, cutlass::layout::RowMajor,                   //
      precision, cutlass::layout::ColumnMajor,                   //
      precision, cutlass::layout::ColumnMajor,
      precision, OpClass, SmArch>;

  EXPECT_TRUE(cuasr::test::gemm::device::TestAllGemm<Srgemm>());
}

///////////////////////////////////////////////////////////////////////////////

TEST(SM80_default_plus_mult_srgemm, f64_tn_t) {
  using precision = double;
  using OpClass   = cutlass::arch::OpClassSimt;
  using SmArch    = cutlass::arch::Sm80;
  using RingOp = cuasr::plus_mult<precision>;

  using Srgemm = cuasr::gemm::device::Srgemm<                           //
      RingOp,                                                           //
      precision, cutlass::layout::RowMajor,                   //
      precision, cutlass::layout::ColumnMajor,                   //
      precision, cutlass::layout::RowMajor,
      precision, OpClass, SmArch>;

  EXPECT_TRUE(cuasr::test::gemm::device::TestAllGemm<Srgemm>());
}

///////////////////////////////////////////////////////////////////////////////

TEST(SM80_default_plus_mult_srgemm, f64_nt_n) {
  using precision = double;
  using OpClass   = cutlass::arch::OpClassSimt;
  using SmArch    = cutlass::arch::Sm80;
  using RingOp = cuasr::plus_mult<precision>;

  using Srgemm = cuasr::gemm::device::Srgemm<                           //
      RingOp,                                                           //
      precision, cutlass::layout::ColumnMajor,                   //
      precision, cutlass::layout::RowMajor,                   //
      precision, cutlass::layout::ColumnMajor,
      precision, OpClass, SmArch>;

  EXPECT_TRUE(cuasr::test::gemm::device::TestAllGemm<Srgemm>());
}

///////////////////////////////////////////////////////////////////////////////

TEST(SM80_default_plus_mult_srgemm, f64_nt_t) {
  using precision = double;
  using OpClass   = cutlass::arch::OpClassSimt;
  using SmArch    = cutlass::arch::Sm80;
  using RingOp = cuasr::plus_mult<precision>;

  using Srgemm = cuasr::gemm::device::Srgemm<                           //
      RingOp,                                                           //
      precision, cutlass::layout::ColumnMajor,                   //
      precision, cutlass::layout::RowMajor,                   //
      precision, cutlass::layout::RowMajor,
      precision, OpClass, SmArch>;

  EXPECT_TRUE(cuasr::test::gemm::device::TestAllGemm<Srgemm>());
}

///////////////////////////////////////////////////////////////////////////////

TEST(SM80_default_plus_mult_srgemm, f64_nn_n) {
  using precision = double;
  using OpClass   = cutlass::arch::OpClassSimt;
  using SmArch    = cutlass::arch::Sm80;
  using RingOp = cuasr::plus_mult<precision>;

  using Srgemm = cuasr::gemm::device::Srgemm<                           //
      RingOp,                                                           //
      precision, cutlass::layout::ColumnMajor,                   //
      precision, cutlass::layout::ColumnMajor,                   //
      precision, cutlass::layout::ColumnMajor,
      precision, OpClass, SmArch>;

  EXPECT_TRUE(cuasr::test::gemm::device::TestAllGemm<Srgemm>());
}

///////////////////////////////////////////////////////////////////////////////

TEST(SM80_default_plus_mult_srgemm, f64_nn_t) {
  using precision = double;
  using OpClass   = cutlass::arch::OpClassSimt;
  using SmArch    = cutlass::arch::Sm80;
  using RingOp = cuasr::plus_mult<precision>;

  using Srgemm = cuasr::gemm::device::Srgemm<                           //
      RingOp,                                                           //
      precision, cutlass::layout::ColumnMajor,                   //
      precision, cutlass::layout::ColumnMajor,                   //
      precision, cutlass::layout::RowMajor,
      precision, OpClass, SmArch>;

  EXPECT_TRUE(cuasr::test::gemm::device::TestAllGemm<Srgemm>());
}

///////////////////////////////////////////////////////////////////////////////

TEST(SM80_default_plus_mult_srgemm, f32_tt_n) {
  using precision = float;
  using OpClass   = cutlass::arch::OpClassSimt;
  using SmArch    = cutlass::arch::Sm80;
  using RingOp = cuasr::plus_mult<precision>;

  using Srgemm = cuasr::gemm::device::Srgemm<                           //
      RingOp,                                                           //
      precision, cutlass::layout::RowMajor,                   //
      precision, cutlass::layout::RowMajor,                   //
      precision, cutlass::layout::ColumnMajor,
      precision, OpClass, SmArch>;

  EXPECT_TRUE(cuasr::test::gemm::device::TestAllGemm<Srgemm>());
}

///////////////////////////////////////////////////////////////////////////////

TEST(SM80_default_plus_mult_srgemm, f32_tt_t) {
  using precision = float;
  using OpClass   = cutlass::arch::OpClassSimt;
  using SmArch    = cutlass::arch::Sm80;
  using RingOp = cuasr::plus_mult<precision>;

  using Srgemm = cuasr::gemm::device::Srgemm<                           //
      RingOp,                                                           //
      precision, cutlass::layout::RowMajor,                   //
      precision, cutlass::layout::RowMajor,                   //
      precision, cutlass::layout::RowMajor,
      precision, OpClass, SmArch>;

  EXPECT_TRUE(cuasr::test::gemm::device::TestAllGemm<Srgemm>());
}

///////////////////////////////////////////////////////////////////////////////

TEST(SM80_default_plus_mult_srgemm, f32_tn_n) {
  using precision = float;
  using OpClass   = cutlass::arch::OpClassSimt;
  using SmArch    = cutlass::arch::Sm80;
  using RingOp = cuasr::plus_mult<precision>;

  using Srgemm = cuasr::gemm::device::Srgemm<                           //
      RingOp,                                                           //
      precision, cutlass::layout::RowMajor,                   //
      precision, cutlass::layout::ColumnMajor,                   //
      precision, cutlass::layout::ColumnMajor,
      precision, OpClass, SmArch>;

  EXPECT_TRUE(cuasr::test::gemm::device::TestAllGemm<Srgemm>());
}

///////////////////////////////////////////////////////////////////////////////

TEST(SM80_default_plus_mult_srgemm, f32_tn_t) {
  using precision = float;
  using OpClass   = cutlass::arch::OpClassSimt;
  using SmArch    = cutlass::arch::Sm80;
  using RingOp = cuasr::plus_mult<precision>;

  using Srgemm = cuasr::gemm::device::Srgemm<                           //
      RingOp,                                                           //
      precision, cutlass::layout::RowMajor,                   //
      precision, cutlass::layout::ColumnMajor,                   //
      precision, cutlass::layout::RowMajor,
      precision, OpClass, SmArch>;

  EXPECT_TRUE(cuasr::test::gemm::device::TestAllGemm<Srgemm>());
}

///////////////////////////////////////////////////////////////////////////////

TEST(SM80_default_plus_mult_srgemm, f32_nt_n) {
  using precision = float;
  using OpClass   = cutlass::arch::OpClassSimt;
  using SmArch    = cutlass::arch::Sm80;
  using RingOp = cuasr::plus_mult<precision>;

  using Srgemm = cuasr::gemm::device::Srgemm<                           //
      RingOp,                                                           //
      precision, cutlass::layout::ColumnMajor,                   //
      precision, cutlass::layout::RowMajor,                   //
      precision, cutlass::layout::ColumnMajor,
      precision, OpClass, SmArch>;

  EXPECT_TRUE(cuasr::test::gemm::device::TestAllGemm<Srgemm>());
}

///////////////////////////////////////////////////////////////////////////////

TEST(SM80_default_plus_mult_srgemm, f32_nt_t) {
  using precision = float;
  using OpClass   = cutlass::arch::OpClassSimt;
  using SmArch    = cutlass::arch::Sm80;
  using RingOp = cuasr::plus_mult<precision>;

  using Srgemm = cuasr::gemm::device::Srgemm<                           //
      RingOp,                                                           //
      precision, cutlass::layout::ColumnMajor,                   //
      precision, cutlass::layout::RowMajor,                   //
      precision, cutlass::layout::RowMajor,
      precision, OpClass, SmArch>;

  EXPECT_TRUE(cuasr::test::gemm::device::TestAllGemm<Srgemm>());
}

///////////////////////////////////////////////////////////////////////////////

TEST(SM80_default_plus_mult_srgemm, f32_nn_n) {
  using precision = float;
  using OpClass   = cutlass::arch::OpClassSimt;
  using SmArch    = cutlass::arch::Sm80;
  using RingOp = cuasr::plus_mult<precision>;

  using Srgemm = cuasr::gemm::device::Srgemm<                           //
      RingOp,                                                           //
      precision, cutlass::layout::ColumnMajor,                   //
      precision, cutlass::layout::ColumnMajor,                   //
      precision, cutlass::layout::ColumnMajor,
      precision, OpClass, SmArch>;

  EXPECT_TRUE(cuasr::test::gemm::device::TestAllGemm<Srgemm>());
}

///////////////////////////////////////////////////////////////////////////////

TEST(SM80_default_plus_mult_srgemm, f32_nn_t) {
  using precision = float;
  using OpClass   = cutlass::arch::OpClassSimt;
  using SmArch    = cutlass::arch::Sm80;
  using RingOp = cuasr::plus_mult<precision>;

  using Srgemm = cuasr::gemm::device::Srgemm<                           //
      RingOp,                                                           //
      precision, cutlass::layout::ColumnMajor,                   //
      precision, cutlass::layout::ColumnMajor,                   //
      precision, cutlass::layout::RowMajor,
      precision, OpClass, SmArch>;

  EXPECT_TRUE(cuasr::test::gemm::device::TestAllGemm<Srgemm>());
}

///////////////////////////////////////////////////////////////////////////////

TEST(SM80_default_min_plus_srgemm, f64_tt_n) {
  using precision = double;
  using OpClass   = cutlass::arch::OpClassSimt;
  using SmArch    = cutlass::arch::Sm80;
  using RingOp = cuasr::min_plus<precision>;

  using Srgemm = cuasr::gemm::device::Srgemm<                           //
      RingOp,                                                           //
      precision, cutlass::layout::RowMajor,                   //
      precision, cutlass::layout::RowMajor,                   //
      precision, cutlass::layout::ColumnMajor,
      precision, OpClass, SmArch>;

  EXPECT_TRUE(cuasr::test::gemm::device::TestAllGemm<Srgemm>());
}

///////////////////////////////////////////////////////////////////////////////

TEST(SM80_default_min_plus_srgemm, f64_tt_t) {
  using precision = double;
  using OpClass   = cutlass::arch::OpClassSimt;
  using SmArch    = cutlass::arch::Sm80;
  using RingOp = cuasr::min_plus<precision>;

  using Srgemm = cuasr::gemm::device::Srgemm<                           //
      RingOp,                                                           //
      precision, cutlass::layout::RowMajor,                   //
      precision, cutlass::layout::RowMajor,                   //
      precision, cutlass::layout::RowMajor,
      precision, OpClass, SmArch>;

  EXPECT_TRUE(cuasr::test::gemm::device::TestAllGemm<Srgemm>());
}

///////////////////////////////////////////////////////////////////////////////

TEST(SM80_default_min_plus_srgemm, f64_tn_n) {
  using precision = double;
  using OpClass   = cutlass::arch::OpClassSimt;
  using SmArch    = cutlass::arch::Sm80;
  using RingOp = cuasr::min_plus<precision>;

  using Srgemm = cuasr::gemm::device::Srgemm<                           //
      RingOp,                                                           //
      precision, cutlass::layout::RowMajor,                   //
      precision, cutlass::layout::ColumnMajor,                   //
      precision, cutlass::layout::ColumnMajor,
      precision, OpClass, SmArch>;

  EXPECT_TRUE(cuasr::test::gemm::device::TestAllGemm<Srgemm>());
}

///////////////////////////////////////////////////////////////////////////////

TEST(SM80_default_min_plus_srgemm, f64_tn_t) {
  using precision = double;
  using OpClass   = cutlass::arch::OpClassSimt;
  using SmArch    = cutlass::arch::Sm80;
  using RingOp = cuasr::min_plus<precision>;

  using Srgemm = cuasr::gemm::device::Srgemm<                           //
      RingOp,                                                           //
      precision, cutlass::layout::RowMajor,                   //
      precision, cutlass::layout::ColumnMajor,                   //
      precision, cutlass::layout::RowMajor,
      precision, OpClass, SmArch>;

  EXPECT_TRUE(cuasr::test::gemm::device::TestAllGemm<Srgemm>());
}

///////////////////////////////////////////////////////////////////////////////

TEST(SM80_default_min_plus_srgemm, f64_nt_n) {
  using precision = double;
  using OpClass   = cutlass::arch::OpClassSimt;
  using SmArch    = cutlass::arch::Sm80;
  using RingOp = cuasr::min_plus<precision>;

  using Srgemm = cuasr::gemm::device::Srgemm<                           //
      RingOp,                                                           //
      precision, cutlass::layout::ColumnMajor,                   //
      precision, cutlass::layout::RowMajor,                   //
      precision, cutlass::layout::ColumnMajor,
      precision, OpClass, SmArch>;

  EXPECT_TRUE(cuasr::test::gemm::device::TestAllGemm<Srgemm>());
}

///////////////////////////////////////////////////////////////////////////////

TEST(SM80_default_min_plus_srgemm, f64_nt_t) {
  using precision = double;
  using OpClass   = cutlass::arch::OpClassSimt;
  using SmArch    = cutlass::arch::Sm80;
  using RingOp = cuasr::min_plus<precision>;

  using Srgemm = cuasr::gemm::device::Srgemm<                           //
      RingOp,                                                           //
      precision, cutlass::layout::ColumnMajor,                   //
      precision, cutlass::layout::RowMajor,                   //
      precision, cutlass::layout::RowMajor,
      precision, OpClass, SmArch>;

  EXPECT_TRUE(cuasr::test::gemm::device::TestAllGemm<Srgemm>());
}

///////////////////////////////////////////////////////////////////////////////

TEST(SM80_default_min_plus_srgemm, f64_nn_n) {
  using precision = double;
  using OpClass   = cutlass::arch::OpClassSimt;
  using SmArch    = cutlass::arch::Sm80;
  using RingOp = cuasr::min_plus<precision>;

  using Srgemm = cuasr::gemm::device::Srgemm<                           //
      RingOp,                                                           //
      precision, cutlass::layout::ColumnMajor,                   //
      precision, cutlass::layout::ColumnMajor,                   //
      precision, cutlass::layout::ColumnMajor,
      precision, OpClass, SmArch>;

  EXPECT_TRUE(cuasr::test::gemm::device::TestAllGemm<Srgemm>());
}

///////////////////////////////////////////////////////////////////////////////

TEST(SM80_default_min_plus_srgemm, f64_nn_t) {
  using precision = double;
  using OpClass   = cutlass::arch::OpClassSimt;
  using SmArch    = cutlass::arch::Sm80;
  using RingOp = cuasr::min_plus<precision>;

  using Srgemm = cuasr::gemm::device::Srgemm<                           //
      RingOp,                                                           //
      precision, cutlass::layout::ColumnMajor,                   //
      precision, cutlass::layout::ColumnMajor,                   //
      precision, cutlass::layout::RowMajor,
      precision, OpClass, SmArch>;

  EXPECT_TRUE(cuasr::test::gemm::device::TestAllGemm<Srgemm>());
}

///////////////////////////////////////////////////////////////////////////////

TEST(SM80_default_min_plus_srgemm, f32_tt_n) {
  using precision = float;
  using OpClass   = cutlass::arch::OpClassSimt;
  using SmArch    = cutlass::arch::Sm80;
  using RingOp = cuasr::min_plus<precision>;

  using Srgemm = cuasr::gemm::device::Srgemm<                           //
      RingOp,                                                           //
      precision, cutlass::layout::RowMajor,                   //
      precision, cutlass::layout::RowMajor,                   //
      precision, cutlass::layout::ColumnMajor,
      precision, OpClass, SmArch>;

  EXPECT_TRUE(cuasr::test::gemm::device::TestAllGemm<Srgemm>());
}

///////////////////////////////////////////////////////////////////////////////

TEST(SM80_default_min_plus_srgemm, f32_tt_t) {
  using precision = float;
  using OpClass   = cutlass::arch::OpClassSimt;
  using SmArch    = cutlass::arch::Sm80;
  using RingOp = cuasr::min_plus<precision>;

  using Srgemm = cuasr::gemm::device::Srgemm<                           //
      RingOp,                                                           //
      precision, cutlass::layout::RowMajor,                   //
      precision, cutlass::layout::RowMajor,                   //
      precision, cutlass::layout::RowMajor,
      precision, OpClass, SmArch>;

  EXPECT_TRUE(cuasr::test::gemm::device::TestAllGemm<Srgemm>());
}

///////////////////////////////////////////////////////////////////////////////

TEST(SM80_default_min_plus_srgemm, f32_tn_n) {
  using precision = float;
  using OpClass   = cutlass::arch::OpClassSimt;
  using SmArch    = cutlass::arch::Sm80;
  using RingOp = cuasr::min_plus<precision>;

  using Srgemm = cuasr::gemm::device::Srgemm<                           //
      RingOp,                                                           //
      precision, cutlass::layout::RowMajor,                   //
      precision, cutlass::layout::ColumnMajor,                   //
      precision, cutlass::layout::ColumnMajor,
      precision, OpClass, SmArch>;

  EXPECT_TRUE(cuasr::test::gemm::device::TestAllGemm<Srgemm>());
}

///////////////////////////////////////////////////////////////////////////////

TEST(SM80_default_min_plus_srgemm, f32_tn_t) {
  using precision = float;
  using OpClass   = cutlass::arch::OpClassSimt;
  using SmArch    = cutlass::arch::Sm80;
  using RingOp = cuasr::min_plus<precision>;

  using Srgemm = cuasr::gemm::device::Srgemm<                           //
      RingOp,                                                           //
      precision, cutlass::layout::RowMajor,                   //
      precision, cutlass::layout::ColumnMajor,                   //
      precision, cutlass::layout::RowMajor,
      precision, OpClass, SmArch>;

  EXPECT_TRUE(cuasr::test::gemm::device::TestAllGemm<Srgemm>());
}

///////////////////////////////////////////////////////////////////////////////

TEST(SM80_default_min_plus_srgemm, f32_nt_n) {
  using precision = float;
  using OpClass   = cutlass::arch::OpClassSimt;
  using SmArch    = cutlass::arch::Sm80;
  using RingOp = cuasr::min_plus<precision>;

  using Srgemm = cuasr::gemm::device::Srgemm<                           //
      RingOp,                                                           //
      precision, cutlass::layout::ColumnMajor,                   //
      precision, cutlass::layout::RowMajor,                   //
      precision, cutlass::layout::ColumnMajor,
      precision, OpClass, SmArch>;

  EXPECT_TRUE(cuasr::test::gemm::device::TestAllGemm<Srgemm>());
}

///////////////////////////////////////////////////////////////////////////////

TEST(SM80_default_min_plus_srgemm, f32_nt_t) {
  using precision = float;
  using OpClass   = cutlass::arch::OpClassSimt;
  using SmArch    = cutlass::arch::Sm80;
  using RingOp = cuasr::min_plus<precision>;

  using Srgemm = cuasr::gemm::device::Srgemm<                           //
      RingOp,                                                           //
      precision, cutlass::layout::ColumnMajor,                   //
      precision, cutlass::layout::RowMajor,                   //
      precision, cutlass::layout::RowMajor,
      precision, OpClass, SmArch>;

  EXPECT_TRUE(cuasr::test::gemm::device::TestAllGemm<Srgemm>());
}

///////////////////////////////////////////////////////////////////////////////

TEST(SM80_default_min_plus_srgemm, f32_nn_n) {
  using precision = float;
  using OpClass   = cutlass::arch::OpClassSimt;
  using SmArch    = cutlass::arch::Sm80;
  using RingOp = cuasr::min_plus<precision>;

  using Srgemm = cuasr::gemm::device::Srgemm<                           //
      RingOp,                                                           //
      precision, cutlass::layout::ColumnMajor,                   //
      precision, cutlass::layout::ColumnMajor,                   //
      precision, cutlass::layout::ColumnMajor,
      precision, OpClass, SmArch>;

  EXPECT_TRUE(cuasr::test::gemm::device::TestAllGemm<Srgemm>());
}

///////////////////////////////////////////////////////////////////////////////

TEST(SM80_default_min_plus_srgemm, f32_nn_t) {
  using precision = float;
  using OpClass   = cutlass::arch::OpClassSimt;
  using SmArch    = cutlass::arch::Sm80;
  using RingOp = cuasr::min_plus<precision>;

  using Srgemm = cuasr::gemm::device::Srgemm<                           //
      RingOp,                                                           //
      precision, cutlass::layout::ColumnMajor,                   //
      precision, cutlass::layout::ColumnMajor,                   //
      precision, cutlass::layout::RowMajor,
      precision, OpClass, SmArch>;

  EXPECT_TRUE(cuasr::test::gemm::device::TestAllGemm<Srgemm>());
}

///////////////////////////////////////////////////////////////////////////////

TEST(SM80_default_max_plus_srgemm, f64_tt_n) {
  using precision = double;
  using OpClass   = cutlass::arch::OpClassSimt;
  using SmArch    = cutlass::arch::Sm80;
  using RingOp = cuasr::max_plus<precision>;

  using Srgemm = cuasr::gemm::device::Srgemm<                           //
      RingOp,                                                           //
      precision, cutlass::layout::RowMajor,                   //
      precision, cutlass::layout::RowMajor,                   //
      precision, cutlass::layout::ColumnMajor,
      precision, OpClass, SmArch>;

  EXPECT_TRUE(cuasr::test::gemm::device::TestAllGemm<Srgemm>());
}

///////////////////////////////////////////////////////////////////////////////

TEST(SM80_default_max_plus_srgemm, f64_tt_t) {
  using precision = double;
  using OpClass   = cutlass::arch::OpClassSimt;
  using SmArch    = cutlass::arch::Sm80;
  using RingOp = cuasr::max_plus<precision>;

  using Srgemm = cuasr::gemm::device::Srgemm<                           //
      RingOp,                                                           //
      precision, cutlass::layout::RowMajor,                   //
      precision, cutlass::layout::RowMajor,                   //
      precision, cutlass::layout::RowMajor,
      precision, OpClass, SmArch>;

  EXPECT_TRUE(cuasr::test::gemm::device::TestAllGemm<Srgemm>());
}

///////////////////////////////////////////////////////////////////////////////

TEST(SM80_default_max_plus_srgemm, f64_tn_n) {
  using precision = double;
  using OpClass   = cutlass::arch::OpClassSimt;
  using SmArch    = cutlass::arch::Sm80;
  using RingOp = cuasr::max_plus<precision>;

  using Srgemm = cuasr::gemm::device::Srgemm<                           //
      RingOp,                                                           //
      precision, cutlass::layout::RowMajor,                   //
      precision, cutlass::layout::ColumnMajor,                   //
      precision, cutlass::layout::ColumnMajor,
      precision, OpClass, SmArch>;

  EXPECT_TRUE(cuasr::test::gemm::device::TestAllGemm<Srgemm>());
}

///////////////////////////////////////////////////////////////////////////////

TEST(SM80_default_max_plus_srgemm, f64_tn_t) {
  using precision = double;
  using OpClass   = cutlass::arch::OpClassSimt;
  using SmArch    = cutlass::arch::Sm80;
  using RingOp = cuasr::max_plus<precision>;

  using Srgemm = cuasr::gemm::device::Srgemm<                           //
      RingOp,                                                           //
      precision, cutlass::layout::RowMajor,                   //
      precision, cutlass::layout::ColumnMajor,                   //
      precision, cutlass::layout::RowMajor,
      precision, OpClass, SmArch>;

  EXPECT_TRUE(cuasr::test::gemm::device::TestAllGemm<Srgemm>());
}

///////////////////////////////////////////////////////////////////////////////

TEST(SM80_default_max_plus_srgemm, f64_nt_n) {
  using precision = double;
  using OpClass   = cutlass::arch::OpClassSimt;
  using SmArch    = cutlass::arch::Sm80;
  using RingOp = cuasr::max_plus<precision>;

  using Srgemm = cuasr::gemm::device::Srgemm<                           //
      RingOp,                                                           //
      precision, cutlass::layout::ColumnMajor,                   //
      precision, cutlass::layout::RowMajor,                   //
      precision, cutlass::layout::ColumnMajor,
      precision, OpClass, SmArch>;

  EXPECT_TRUE(cuasr::test::gemm::device::TestAllGemm<Srgemm>());
}

///////////////////////////////////////////////////////////////////////////////

TEST(SM80_default_max_plus_srgemm, f64_nt_t) {
  using precision = double;
  using OpClass   = cutlass::arch::OpClassSimt;
  using SmArch    = cutlass::arch::Sm80;
  using RingOp = cuasr::max_plus<precision>;

  using Srgemm = cuasr::gemm::device::Srgemm<                           //
      RingOp,                                                           //
      precision, cutlass::layout::ColumnMajor,                   //
      precision, cutlass::layout::RowMajor,                   //
      precision, cutlass::layout::RowMajor,
      precision, OpClass, SmArch>;

  EXPECT_TRUE(cuasr::test::gemm::device::TestAllGemm<Srgemm>());
}

///////////////////////////////////////////////////////////////////////////////

TEST(SM80_default_max_plus_srgemm, f64_nn_n) {
  using precision = double;
  using OpClass   = cutlass::arch::OpClassSimt;
  using SmArch    = cutlass::arch::Sm80;
  using RingOp = cuasr::max_plus<precision>;

  using Srgemm = cuasr::gemm::device::Srgemm<                           //
      RingOp,                                                           //
      precision, cutlass::layout::ColumnMajor,                   //
      precision, cutlass::layout::ColumnMajor,                   //
      precision, cutlass::layout::ColumnMajor,
      precision, OpClass, SmArch>;

  EXPECT_TRUE(cuasr::test::gemm::device::TestAllGemm<Srgemm>());
}

///////////////////////////////////////////////////////////////////////////////

TEST(SM80_default_max_plus_srgemm, f64_nn_t) {
  using precision = double;
  using OpClass   = cutlass::arch::OpClassSimt;
  using SmArch    = cutlass::arch::Sm80;
  using RingOp = cuasr::max_plus<precision>;

  using Srgemm = cuasr::gemm::device::Srgemm<                           //
      RingOp,                                                           //
      precision, cutlass::layout::ColumnMajor,                   //
      precision, cutlass::layout::ColumnMajor,                   //
      precision, cutlass::layout::RowMajor,
      precision, OpClass, SmArch>;

  EXPECT_TRUE(cuasr::test::gemm::device::TestAllGemm<Srgemm>());
}

///////////////////////////////////////////////////////////////////////////////

TEST(SM80_default_max_plus_srgemm, f32_tt_n) {
  using precision = float;
  using OpClass   = cutlass::arch::OpClassSimt;
  using SmArch    = cutlass::arch::Sm80;
  using RingOp = cuasr::max_plus<precision>;

  using Srgemm = cuasr::gemm::device::Srgemm<                           //
      RingOp,                                                           //
      precision, cutlass::layout::RowMajor,                   //
      precision, cutlass::layout::RowMajor,                   //
      precision, cutlass::layout::ColumnMajor,
      precision, OpClass, SmArch>;

  EXPECT_TRUE(cuasr::test::gemm::device::TestAllGemm<Srgemm>());
}

///////////////////////////////////////////////////////////////////////////////

TEST(SM80_default_max_plus_srgemm, f32_tt_t) {
  using precision = float;
  using OpClass   = cutlass::arch::OpClassSimt;
  using SmArch    = cutlass::arch::Sm80;
  using RingOp = cuasr::max_plus<precision>;

  using Srgemm = cuasr::gemm::device::Srgemm<                           //
      RingOp,                                                           //
      precision, cutlass::layout::RowMajor,                   //
      precision, cutlass::layout::RowMajor,                   //
      precision, cutlass::layout::RowMajor,
      precision, OpClass, SmArch>;

  EXPECT_TRUE(cuasr::test::gemm::device::TestAllGemm<Srgemm>());
}

///////////////////////////////////////////////////////////////////////////////

TEST(SM80_default_max_plus_srgemm, f32_tn_n) {
  using precision = float;
  using OpClass   = cutlass::arch::OpClassSimt;
  using SmArch    = cutlass::arch::Sm80;
  using RingOp = cuasr::max_plus<precision>;

  using Srgemm = cuasr::gemm::device::Srgemm<                           //
      RingOp,                                                           //
      precision, cutlass::layout::RowMajor,                   //
      precision, cutlass::layout::ColumnMajor,                   //
      precision, cutlass::layout::ColumnMajor,
      precision, OpClass, SmArch>;

  EXPECT_TRUE(cuasr::test::gemm::device::TestAllGemm<Srgemm>());
}

///////////////////////////////////////////////////////////////////////////////

TEST(SM80_default_max_plus_srgemm, f32_tn_t) {
  using precision = float;
  using OpClass   = cutlass::arch::OpClassSimt;
  using SmArch    = cutlass::arch::Sm80;
  using RingOp = cuasr::max_plus<precision>;

  using Srgemm = cuasr::gemm::device::Srgemm<                           //
      RingOp,                                                           //
      precision, cutlass::layout::RowMajor,                   //
      precision, cutlass::layout::ColumnMajor,                   //
      precision, cutlass::layout::RowMajor,
      precision, OpClass, SmArch>;

  EXPECT_TRUE(cuasr::test::gemm::device::TestAllGemm<Srgemm>());
}

///////////////////////////////////////////////////////////////////////////////

TEST(SM80_default_max_plus_srgemm, f32_nt_n) {
  using precision = float;
  using OpClass   = cutlass::arch::OpClassSimt;
  using SmArch    = cutlass::arch::Sm80;
  using RingOp = cuasr::max_plus<precision>;

  using Srgemm = cuasr::gemm::device::Srgemm<                           //
      RingOp,                                                           //
      precision, cutlass::layout::ColumnMajor,                   //
      precision, cutlass::layout::RowMajor,                   //
      precision, cutlass::layout::ColumnMajor,
      precision, OpClass, SmArch>;

  EXPECT_TRUE(cuasr::test::gemm::device::TestAllGemm<Srgemm>());
}

///////////////////////////////////////////////////////////////////////////////

TEST(SM80_default_max_plus_srgemm, f32_nt_t) {
  using precision = float;
  using OpClass   = cutlass::arch::OpClassSimt;
  using SmArch    = cutlass::arch::Sm80;
  using RingOp = cuasr::max_plus<precision>;

  using Srgemm = cuasr::gemm::device::Srgemm<                           //
      RingOp,                                                           //
      precision, cutlass::layout::ColumnMajor,                   //
      precision, cutlass::layout::RowMajor,                   //
      precision, cutlass::layout::RowMajor,
      precision, OpClass, SmArch>;

  EXPECT_TRUE(cuasr::test::gemm::device::TestAllGemm<Srgemm>());
}

///////////////////////////////////////////////////////////////////////////////

TEST(SM80_default_max_plus_srgemm, f32_nn_n) {
  using precision = float;
  using OpClass   = cutlass::arch::OpClassSimt;
  using SmArch    = cutlass::arch::Sm80;
  using RingOp = cuasr::max_plus<precision>;

  using Srgemm = cuasr::gemm::device::Srgemm<                           //
      RingOp,                                                           //
      precision, cutlass::layout::ColumnMajor,                   //
      precision, cutlass::layout::ColumnMajor,                   //
      precision, cutlass::layout::ColumnMajor,
      precision, OpClass, SmArch>;

  EXPECT_TRUE(cuasr::test::gemm::device::TestAllGemm<Srgemm>());
}

///////////////////////////////////////////////////////////////////////////////

TEST(SM80_default_max_plus_srgemm, f32_nn_t) {
  using precision = float;
  using OpClass   = cutlass::arch::OpClassSimt;
  using SmArch    = cutlass::arch::Sm80;
  using RingOp = cuasr::max_plus<precision>;

  using Srgemm = cuasr::gemm::device::Srgemm<                           //
      RingOp,                                                           //
      precision, cutlass::layout::ColumnMajor,                   //
      precision, cutlass::layout::ColumnMajor,                   //
      precision, cutlass::layout::RowMajor,
      precision, OpClass, SmArch>;

  EXPECT_TRUE(cuasr::test::gemm::device::TestAllGemm<Srgemm>());
}

///////////////////////////////////////////////////////////////////////////////

TEST(SM80_default_min_max_srgemm, f64_tt_n) {
  using precision = double;
  using OpClass   = cutlass::arch::OpClassSimt;
  using SmArch    = cutlass::arch::Sm80;
  using RingOp = cuasr::min_max<precision>;

  using Srgemm = cuasr::gemm::device::Srgemm<                           //
      RingOp,                                                           //
      precision, cutlass::layout::RowMajor,                   //
      precision, cutlass::layout::RowMajor,                   //
      precision, cutlass::layout::ColumnMajor,
      precision, OpClass, SmArch>;

  EXPECT_TRUE(cuasr::test::gemm::device::TestAllGemm<Srgemm>());
}

///////////////////////////////////////////////////////////////////////////////

TEST(SM80_default_min_max_srgemm, f64_tt_t) {
  using precision = double;
  using OpClass   = cutlass::arch::OpClassSimt;
  using SmArch    = cutlass::arch::Sm80;
  using RingOp = cuasr::min_max<precision>;

  using Srgemm = cuasr::gemm::device::Srgemm<                           //
      RingOp,                                                           //
      precision, cutlass::layout::RowMajor,                   //
      precision, cutlass::layout::RowMajor,                   //
      precision, cutlass::layout::RowMajor,
      precision, OpClass, SmArch>;

  EXPECT_TRUE(cuasr::test::gemm::device::TestAllGemm<Srgemm>());
}

///////////////////////////////////////////////////////////////////////////////

TEST(SM80_default_min_max_srgemm, f64_tn_n) {
  using precision = double;
  using OpClass   = cutlass::arch::OpClassSimt;
  using SmArch    = cutlass::arch::Sm80;
  using RingOp = cuasr::min_max<precision>;

  using Srgemm = cuasr::gemm::device::Srgemm<                           //
      RingOp,                                                           //
      precision, cutlass::layout::RowMajor,                   //
      precision, cutlass::layout::ColumnMajor,                   //
      precision, cutlass::layout::ColumnMajor,
      precision, OpClass, SmArch>;

  EXPECT_TRUE(cuasr::test::gemm::device::TestAllGemm<Srgemm>());
}

///////////////////////////////////////////////////////////////////////////////

TEST(SM80_default_min_max_srgemm, f64_tn_t) {
  using precision = double;
  using OpClass   = cutlass::arch::OpClassSimt;
  using SmArch    = cutlass::arch::Sm80;
  using RingOp = cuasr::min_max<precision>;

  using Srgemm = cuasr::gemm::device::Srgemm<                           //
      RingOp,                                                           //
      precision, cutlass::layout::RowMajor,                   //
      precision, cutlass::layout::ColumnMajor,                   //
      precision, cutlass::layout::RowMajor,
      precision, OpClass, SmArch>;

  EXPECT_TRUE(cuasr::test::gemm::device::TestAllGemm<Srgemm>());
}

///////////////////////////////////////////////////////////////////////////////

TEST(SM80_default_min_max_srgemm, f64_nt_n) {
  using precision = double;
  using OpClass   = cutlass::arch::OpClassSimt;
  using SmArch    = cutlass::arch::Sm80;
  using RingOp = cuasr::min_max<precision>;

  using Srgemm = cuasr::gemm::device::Srgemm<                           //
      RingOp,                                                           //
      precision, cutlass::layout::ColumnMajor,                   //
      precision, cutlass::layout::RowMajor,                   //
      precision, cutlass::layout::ColumnMajor,
      precision, OpClass, SmArch>;

  EXPECT_TRUE(cuasr::test::gemm::device::TestAllGemm<Srgemm>());
}

///////////////////////////////////////////////////////////////////////////////

TEST(SM80_default_min_max_srgemm, f64_nt_t) {
  using precision = double;
  using OpClass   = cutlass::arch::OpClassSimt;
  using SmArch    = cutlass::arch::Sm80;
  using RingOp = cuasr::min_max<precision>;

  using Srgemm = cuasr::gemm::device::Srgemm<                           //
      RingOp,                                                           //
      precision, cutlass::layout::ColumnMajor,                   //
      precision, cutlass::layout::RowMajor,                   //
      precision, cutlass::layout::RowMajor,
      precision, OpClass, SmArch>;

  EXPECT_TRUE(cuasr::test::gemm::device::TestAllGemm<Srgemm>());
}

///////////////////////////////////////////////////////////////////////////////

TEST(SM80_default_min_max_srgemm, f64_nn_n) {
  using precision = double;
  using OpClass   = cutlass::arch::OpClassSimt;
  using SmArch    = cutlass::arch::Sm80;
  using RingOp = cuasr::min_max<precision>;

  using Srgemm = cuasr::gemm::device::Srgemm<                           //
      RingOp,                                                           //
      precision, cutlass::layout::ColumnMajor,                   //
      precision, cutlass::layout::ColumnMajor,                   //
      precision, cutlass::layout::ColumnMajor,
      precision, OpClass, SmArch>;

  EXPECT_TRUE(cuasr::test::gemm::device::TestAllGemm<Srgemm>());
}

///////////////////////////////////////////////////////////////////////////////

TEST(SM80_default_min_max_srgemm, f64_nn_t) {
  using precision = double;
  using OpClass   = cutlass::arch::OpClassSimt;
  using SmArch    = cutlass::arch::Sm80;
  using RingOp = cuasr::min_max<precision>;

  using Srgemm = cuasr::gemm::device::Srgemm<                           //
      RingOp,                                                           //
      precision, cutlass::layout::ColumnMajor,                   //
      precision, cutlass::layout::ColumnMajor,                   //
      precision, cutlass::layout::RowMajor,
      precision, OpClass, SmArch>;

  EXPECT_TRUE(cuasr::test::gemm::device::TestAllGemm<Srgemm>());
}

///////////////////////////////////////////////////////////////////////////////

TEST(SM80_default_min_max_srgemm, f32_tt_n) {
  using precision = float;
  using OpClass   = cutlass::arch::OpClassSimt;
  using SmArch    = cutlass::arch::Sm80;
  using RingOp = cuasr::min_max<precision>;

  using Srgemm = cuasr::gemm::device::Srgemm<                           //
      RingOp,                                                           //
      precision, cutlass::layout::RowMajor,                   //
      precision, cutlass::layout::RowMajor,                   //
      precision, cutlass::layout::ColumnMajor,
      precision, OpClass, SmArch>;

  EXPECT_TRUE(cuasr::test::gemm::device::TestAllGemm<Srgemm>());
}

///////////////////////////////////////////////////////////////////////////////

TEST(SM80_default_min_max_srgemm, f32_tt_t) {
  using precision = float;
  using OpClass   = cutlass::arch::OpClassSimt;
  using SmArch    = cutlass::arch::Sm80;
  using RingOp = cuasr::min_max<precision>;

  using Srgemm = cuasr::gemm::device::Srgemm<                           //
      RingOp,                                                           //
      precision, cutlass::layout::RowMajor,                   //
      precision, cutlass::layout::RowMajor,                   //
      precision, cutlass::layout::RowMajor,
      precision, OpClass, SmArch>;

  EXPECT_TRUE(cuasr::test::gemm::device::TestAllGemm<Srgemm>());
}

///////////////////////////////////////////////////////////////////////////////

TEST(SM80_default_min_max_srgemm, f32_tn_n) {
  using precision = float;
  using OpClass   = cutlass::arch::OpClassSimt;
  using SmArch    = cutlass::arch::Sm80;
  using RingOp = cuasr::min_max<precision>;

  using Srgemm = cuasr::gemm::device::Srgemm<                           //
      RingOp,                                                           //
      precision, cutlass::layout::RowMajor,                   //
      precision, cutlass::layout::ColumnMajor,                   //
      precision, cutlass::layout::ColumnMajor,
      precision, OpClass, SmArch>;

  EXPECT_TRUE(cuasr::test::gemm::device::TestAllGemm<Srgemm>());
}

///////////////////////////////////////////////////////////////////////////////

TEST(SM80_default_min_max_srgemm, f32_tn_t) {
  using precision = float;
  using OpClass   = cutlass::arch::OpClassSimt;
  using SmArch    = cutlass::arch::Sm80;
  using RingOp = cuasr::min_max<precision>;

  using Srgemm = cuasr::gemm::device::Srgemm<                           //
      RingOp,                                                           //
      precision, cutlass::layout::RowMajor,                   //
      precision, cutlass::layout::ColumnMajor,                   //
      precision, cutlass::layout::RowMajor,
      precision, OpClass, SmArch>;

  EXPECT_TRUE(cuasr::test::gemm::device::TestAllGemm<Srgemm>());
}

///////////////////////////////////////////////////////////////////////////////

TEST(SM80_default_min_max_srgemm, f32_nt_n) {
  using precision = float;
  using OpClass   = cutlass::arch::OpClassSimt;
  using SmArch    = cutlass::arch::Sm80;
  using RingOp = cuasr::min_max<precision>;

  using Srgemm = cuasr::gemm::device::Srgemm<                           //
      RingOp,                                                           //
      precision, cutlass::layout::ColumnMajor,                   //
      precision, cutlass::layout::RowMajor,                   //
      precision, cutlass::layout::ColumnMajor,
      precision, OpClass, SmArch>;

  EXPECT_TRUE(cuasr::test::gemm::device::TestAllGemm<Srgemm>());
}

///////////////////////////////////////////////////////////////////////////////

TEST(SM80_default_min_max_srgemm, f32_nt_t) {
  using precision = float;
  using OpClass   = cutlass::arch::OpClassSimt;
  using SmArch    = cutlass::arch::Sm80;
  using RingOp = cuasr::min_max<precision>;

  using Srgemm = cuasr::gemm::device::Srgemm<                           //
      RingOp,                                                           //
      precision, cutlass::layout::ColumnMajor,                   //
      precision, cutlass::layout::RowMajor,                   //
      precision, cutlass::layout::RowMajor,
      precision, OpClass, SmArch>;

  EXPECT_TRUE(cuasr::test::gemm::device::TestAllGemm<Srgemm>());
}

///////////////////////////////////////////////////////////////////////////////

TEST(SM80_default_min_max_srgemm, f32_nn_n) {
  using precision = float;
  using OpClass   = cutlass::arch::OpClassSimt;
  using SmArch    = cutlass::arch::Sm80;
  using RingOp = cuasr::min_max<precision>;

  using Srgemm = cuasr::gemm::device::Srgemm<                           //
      RingOp,                                                           //
      precision, cutlass::layout::ColumnMajor,                   //
      precision, cutlass::layout::ColumnMajor,                   //
      precision, cutlass::layout::ColumnMajor,
      precision, OpClass, SmArch>;

  EXPECT_TRUE(cuasr::test::gemm::device::TestAllGemm<Srgemm>());
}

///////////////////////////////////////////////////////////////////////////////

TEST(SM80_default_min_max_srgemm, f32_nn_t) {
  using precision = float;
  using OpClass   = cutlass::arch::OpClassSimt;
  using SmArch    = cutlass::arch::Sm80;
  using RingOp = cuasr::min_max<precision>;

  using Srgemm = cuasr::gemm::device::Srgemm<                           //
      RingOp,                                                           //
      precision, cutlass::layout::ColumnMajor,                   //
      precision, cutlass::layout::ColumnMajor,                   //
      precision, cutlass::layout::RowMajor,
      precision, OpClass, SmArch>;

  EXPECT_TRUE(cuasr::test::gemm::device::TestAllGemm<Srgemm>());
}

///////////////////////////////////////////////////////////////////////////////

TEST(SM80_default_max_min_srgemm, f64_tt_n) {
  using precision = double;
  using OpClass   = cutlass::arch::OpClassSimt;
  using SmArch    = cutlass::arch::Sm80;
  using RingOp = cuasr::max_min<precision>;

  using Srgemm = cuasr::gemm::device::Srgemm<                           //
      RingOp,                                                           //
      precision, cutlass::layout::RowMajor,                   //
      precision, cutlass::layout::RowMajor,                   //
      precision, cutlass::layout::ColumnMajor,
      precision, OpClass, SmArch>;

  EXPECT_TRUE(cuasr::test::gemm::device::TestAllGemm<Srgemm>());
}

///////////////////////////////////////////////////////////////////////////////

TEST(SM80_default_max_min_srgemm, f64_tt_t) {
  using precision = double;
  using OpClass   = cutlass::arch::OpClassSimt;
  using SmArch    = cutlass::arch::Sm80;
  using RingOp = cuasr::max_min<precision>;

  using Srgemm = cuasr::gemm::device::Srgemm<                           //
      RingOp,                                                           //
      precision, cutlass::layout::RowMajor,                   //
      precision, cutlass::layout::RowMajor,                   //
      precision, cutlass::layout::RowMajor,
      precision, OpClass, SmArch>;

  EXPECT_TRUE(cuasr::test::gemm::device::TestAllGemm<Srgemm>());
}

///////////////////////////////////////////////////////////////////////////////

TEST(SM80_default_max_min_srgemm, f64_tn_n) {
  using precision = double;
  using OpClass   = cutlass::arch::OpClassSimt;
  using SmArch    = cutlass::arch::Sm80;
  using RingOp = cuasr::max_min<precision>;

  using Srgemm = cuasr::gemm::device::Srgemm<                           //
      RingOp,                                                           //
      precision, cutlass::layout::RowMajor,                   //
      precision, cutlass::layout::ColumnMajor,                   //
      precision, cutlass::layout::ColumnMajor,
      precision, OpClass, SmArch>;

  EXPECT_TRUE(cuasr::test::gemm::device::TestAllGemm<Srgemm>());
}

///////////////////////////////////////////////////////////////////////////////

TEST(SM80_default_max_min_srgemm, f64_tn_t) {
  using precision = double;
  using OpClass   = cutlass::arch::OpClassSimt;
  using SmArch    = cutlass::arch::Sm80;
  using RingOp = cuasr::max_min<precision>;

  using Srgemm = cuasr::gemm::device::Srgemm<                           //
      RingOp,                                                           //
      precision, cutlass::layout::RowMajor,                   //
      precision, cutlass::layout::ColumnMajor,                   //
      precision, cutlass::layout::RowMajor,
      precision, OpClass, SmArch>;

  EXPECT_TRUE(cuasr::test::gemm::device::TestAllGemm<Srgemm>());
}

///////////////////////////////////////////////////////////////////////////////

TEST(SM80_default_max_min_srgemm, f64_nt_n) {
  using precision = double;
  using OpClass   = cutlass::arch::OpClassSimt;
  using SmArch    = cutlass::arch::Sm80;
  using RingOp = cuasr::max_min<precision>;

  using Srgemm = cuasr::gemm::device::Srgemm<                           //
      RingOp,                                                           //
      precision, cutlass::layout::ColumnMajor,                   //
      precision, cutlass::layout::RowMajor,                   //
      precision, cutlass::layout::ColumnMajor,
      precision, OpClass, SmArch>;

  EXPECT_TRUE(cuasr::test::gemm::device::TestAllGemm<Srgemm>());
}

///////////////////////////////////////////////////////////////////////////////

TEST(SM80_default_max_min_srgemm, f64_nt_t) {
  using precision = double;
  using OpClass   = cutlass::arch::OpClassSimt;
  using SmArch    = cutlass::arch::Sm80;
  using RingOp = cuasr::max_min<precision>;

  using Srgemm = cuasr::gemm::device::Srgemm<                           //
      RingOp,                                                           //
      precision, cutlass::layout::ColumnMajor,                   //
      precision, cutlass::layout::RowMajor,                   //
      precision, cutlass::layout::RowMajor,
      precision, OpClass, SmArch>;

  EXPECT_TRUE(cuasr::test::gemm::device::TestAllGemm<Srgemm>());
}

///////////////////////////////////////////////////////////////////////////////

TEST(SM80_default_max_min_srgemm, f64_nn_n) {
  using precision = double;
  using OpClass   = cutlass::arch::OpClassSimt;
  using SmArch    = cutlass::arch::Sm80;
  using RingOp = cuasr::max_min<precision>;

  using Srgemm = cuasr::gemm::device::Srgemm<                           //
      RingOp,                                                           //
      precision, cutlass::layout::ColumnMajor,                   //
      precision, cutlass::layout::ColumnMajor,                   //
      precision, cutlass::layout::ColumnMajor,
      precision, OpClass, SmArch>;

  EXPECT_TRUE(cuasr::test::gemm::device::TestAllGemm<Srgemm>());
}

///////////////////////////////////////////////////////////////////////////////

TEST(SM80_default_max_min_srgemm, f64_nn_t) {
  using precision = double;
  using OpClass   = cutlass::arch::OpClassSimt;
  using SmArch    = cutlass::arch::Sm80;
  using RingOp = cuasr::max_min<precision>;

  using Srgemm = cuasr::gemm::device::Srgemm<                           //
      RingOp,                                                           //
      precision, cutlass::layout::ColumnMajor,                   //
      precision, cutlass::layout::ColumnMajor,                   //
      precision, cutlass::layout::RowMajor,
      precision, OpClass, SmArch>;

  EXPECT_TRUE(cuasr::test::gemm::device::TestAllGemm<Srgemm>());
}

///////////////////////////////////////////////////////////////////////////////

TEST(SM80_default_max_min_srgemm, f32_tt_n) {
  using precision = float;
  using OpClass   = cutlass::arch::OpClassSimt;
  using SmArch    = cutlass::arch::Sm80;
  using RingOp = cuasr::max_min<precision>;

  using Srgemm = cuasr::gemm::device::Srgemm<                           //
      RingOp,                                                           //
      precision, cutlass::layout::RowMajor,                   //
      precision, cutlass::layout::RowMajor,                   //
      precision, cutlass::layout::ColumnMajor,
      precision, OpClass, SmArch>;

  EXPECT_TRUE(cuasr::test::gemm::device::TestAllGemm<Srgemm>());
}

///////////////////////////////////////////////////////////////////////////////

TEST(SM80_default_max_min_srgemm, f32_tt_t) {
  using precision = float;
  using OpClass   = cutlass::arch::OpClassSimt;
  using SmArch    = cutlass::arch::Sm80;
  using RingOp = cuasr::max_min<precision>;

  using Srgemm = cuasr::gemm::device::Srgemm<                           //
      RingOp,                                                           //
      precision, cutlass::layout::RowMajor,                   //
      precision, cutlass::layout::RowMajor,                   //
      precision, cutlass::layout::RowMajor,
      precision, OpClass, SmArch>;

  EXPECT_TRUE(cuasr::test::gemm::device::TestAllGemm<Srgemm>());
}

///////////////////////////////////////////////////////////////////////////////

TEST(SM80_default_max_min_srgemm, f32_tn_n) {
  using precision = float;
  using OpClass   = cutlass::arch::OpClassSimt;
  using SmArch    = cutlass::arch::Sm80;
  using RingOp = cuasr::max_min<precision>;

  using Srgemm = cuasr::gemm::device::Srgemm<                           //
      RingOp,                                                           //
      precision, cutlass::layout::RowMajor,                   //
      precision, cutlass::layout::ColumnMajor,                   //
      precision, cutlass::layout::ColumnMajor,
      precision, OpClass, SmArch>;

  EXPECT_TRUE(cuasr::test::gemm::device::TestAllGemm<Srgemm>());
}

///////////////////////////////////////////////////////////////////////////////

TEST(SM80_default_max_min_srgemm, f32_tn_t) {
  using precision = float;
  using OpClass   = cutlass::arch::OpClassSimt;
  using SmArch    = cutlass::arch::Sm80;
  using RingOp = cuasr::max_min<precision>;

  using Srgemm = cuasr::gemm::device::Srgemm<                           //
      RingOp,                                                           //
      precision, cutlass::layout::RowMajor,                   //
      precision, cutlass::layout::ColumnMajor,                   //
      precision, cutlass::layout::RowMajor,
      precision, OpClass, SmArch>;

  EXPECT_TRUE(cuasr::test::gemm::device::TestAllGemm<Srgemm>());
}

///////////////////////////////////////////////////////////////////////////////

TEST(SM80_default_max_min_srgemm, f32_nt_n) {
  using precision = float;
  using OpClass   = cutlass::arch::OpClassSimt;
  using SmArch    = cutlass::arch::Sm80;
  using RingOp = cuasr::max_min<precision>;

  using Srgemm = cuasr::gemm::device::Srgemm<                           //
      RingOp,                                                           //
      precision, cutlass::layout::ColumnMajor,                   //
      precision, cutlass::layout::RowMajor,                   //
      precision, cutlass::layout::ColumnMajor,
      precision, OpClass, SmArch>;

  EXPECT_TRUE(cuasr::test::gemm::device::TestAllGemm<Srgemm>());
}

///////////////////////////////////////////////////////////////////////////////

TEST(SM80_default_max_min_srgemm, f32_nt_t) {
  using precision = float;
  using OpClass   = cutlass::arch::OpClassSimt;
  using SmArch    = cutlass::arch::Sm80;
  using RingOp = cuasr::max_min<precision>;

  using Srgemm = cuasr::gemm::device::Srgemm<                           //
      RingOp,                                                           //
      precision, cutlass::layout::ColumnMajor,                   //
      precision, cutlass::layout::RowMajor,                   //
      precision, cutlass::layout::RowMajor,
      precision, OpClass, SmArch>;

  EXPECT_TRUE(cuasr::test::gemm::device::TestAllGemm<Srgemm>());
}

///////////////////////////////////////////////////////////////////////////////

TEST(SM80_default_max_min_srgemm, f32_nn_n) {
  using precision = float;
  using OpClass   = cutlass::arch::OpClassSimt;
  using SmArch    = cutlass::arch::Sm80;
  using RingOp = cuasr::max_min<precision>;

  using Srgemm = cuasr::gemm::device::Srgemm<                           //
      RingOp,                                                           //
      precision, cutlass::layout::ColumnMajor,                   //
      precision, cutlass::layout::ColumnMajor,                   //
      precision, cutlass::layout::ColumnMajor,
      precision, OpClass, SmArch>;

  EXPECT_TRUE(cuasr::test::gemm::device::TestAllGemm<Srgemm>());
}

///////////////////////////////////////////////////////////////////////////////

TEST(SM80_default_max_min_srgemm, f32_nn_t) {
  using precision = float;
  using OpClass   = cutlass::arch::OpClassSimt;
  using SmArch    = cutlass::arch::Sm80;
  using RingOp = cuasr::max_min<precision>;

  using Srgemm = cuasr::gemm::device::Srgemm<                           //
      RingOp,                                                           //
      precision, cutlass::layout::ColumnMajor,                   //
      precision, cutlass::layout::ColumnMajor,                   //
      precision, cutlass::layout::RowMajor,
      precision, OpClass, SmArch>;

  EXPECT_TRUE(cuasr::test::gemm::device::TestAllGemm<Srgemm>());
}

///////////////////////////////////////////////////////////////////////////////

TEST(SM80_default_min_mult_srgemm, f64_tt_n) {
  using precision = double;
  using OpClass   = cutlass::arch::OpClassSimt;
  using SmArch    = cutlass::arch::Sm80;
  using RingOp = cuasr::min_mult<precision>;

  using Srgemm = cuasr::gemm::device::Srgemm<                           //
      RingOp,                                                           //
      precision, cutlass::layout::RowMajor,                   //
      precision, cutlass::layout::RowMajor,                   //
      precision, cutlass::layout::ColumnMajor,
      precision, OpClass, SmArch>;

  EXPECT_TRUE(cuasr::test::gemm::device::TestAllGemm<Srgemm>());
}

///////////////////////////////////////////////////////////////////////////////

TEST(SM80_default_min_mult_srgemm, f64_tt_t) {
  using precision = double;
  using OpClass   = cutlass::arch::OpClassSimt;
  using SmArch    = cutlass::arch::Sm80;
  using RingOp = cuasr::min_mult<precision>;

  using Srgemm = cuasr::gemm::device::Srgemm<                           //
      RingOp,                                                           //
      precision, cutlass::layout::RowMajor,                   //
      precision, cutlass::layout::RowMajor,                   //
      precision, cutlass::layout::RowMajor,
      precision, OpClass, SmArch>;

  EXPECT_TRUE(cuasr::test::gemm::device::TestAllGemm<Srgemm>());
}

///////////////////////////////////////////////////////////////////////////////

TEST(SM80_default_min_mult_srgemm, f64_tn_n) {
  using precision = double;
  using OpClass   = cutlass::arch::OpClassSimt;
  using SmArch    = cutlass::arch::Sm80;
  using RingOp = cuasr::min_mult<precision>;

  using Srgemm = cuasr::gemm::device::Srgemm<                           //
      RingOp,                                                           //
      precision, cutlass::layout::RowMajor,                   //
      precision, cutlass::layout::ColumnMajor,                   //
      precision, cutlass::layout::ColumnMajor,
      precision, OpClass, SmArch>;

  EXPECT_TRUE(cuasr::test::gemm::device::TestAllGemm<Srgemm>());
}

///////////////////////////////////////////////////////////////////////////////

TEST(SM80_default_min_mult_srgemm, f64_tn_t) {
  using precision = double;
  using OpClass   = cutlass::arch::OpClassSimt;
  using SmArch    = cutlass::arch::Sm80;
  using RingOp = cuasr::min_mult<precision>;

  using Srgemm = cuasr::gemm::device::Srgemm<                           //
      RingOp,                                                           //
      precision, cutlass::layout::RowMajor,                   //
      precision, cutlass::layout::ColumnMajor,                   //
      precision, cutlass::layout::RowMajor,
      precision, OpClass, SmArch>;

  EXPECT_TRUE(cuasr::test::gemm::device::TestAllGemm<Srgemm>());
}

///////////////////////////////////////////////////////////////////////////////

TEST(SM80_default_min_mult_srgemm, f64_nt_n) {
  using precision = double;
  using OpClass   = cutlass::arch::OpClassSimt;
  using SmArch    = cutlass::arch::Sm80;
  using RingOp = cuasr::min_mult<precision>;

  using Srgemm = cuasr::gemm::device::Srgemm<                           //
      RingOp,                                                           //
      precision, cutlass::layout::ColumnMajor,                   //
      precision, cutlass::layout::RowMajor,                   //
      precision, cutlass::layout::ColumnMajor,
      precision, OpClass, SmArch>;

  EXPECT_TRUE(cuasr::test::gemm::device::TestAllGemm<Srgemm>());
}

///////////////////////////////////////////////////////////////////////////////

TEST(SM80_default_min_mult_srgemm, f64_nt_t) {
  using precision = double;
  using OpClass   = cutlass::arch::OpClassSimt;
  using SmArch    = cutlass::arch::Sm80;
  using RingOp = cuasr::min_mult<precision>;

  using Srgemm = cuasr::gemm::device::Srgemm<                           //
      RingOp,                                                           //
      precision, cutlass::layout::ColumnMajor,                   //
      precision, cutlass::layout::RowMajor,                   //
      precision, cutlass::layout::RowMajor,
      precision, OpClass, SmArch>;

  EXPECT_TRUE(cuasr::test::gemm::device::TestAllGemm<Srgemm>());
}

///////////////////////////////////////////////////////////////////////////////

TEST(SM80_default_min_mult_srgemm, f64_nn_n) {
  using precision = double;
  using OpClass   = cutlass::arch::OpClassSimt;
  using SmArch    = cutlass::arch::Sm80;
  using RingOp = cuasr::min_mult<precision>;

  using Srgemm = cuasr::gemm::device::Srgemm<                           //
      RingOp,                                                           //
      precision, cutlass::layout::ColumnMajor,                   //
      precision, cutlass::layout::ColumnMajor,                   //
      precision, cutlass::layout::ColumnMajor,
      precision, OpClass, SmArch>;

  EXPECT_TRUE(cuasr::test::gemm::device::TestAllGemm<Srgemm>());
}

///////////////////////////////////////////////////////////////////////////////

TEST(SM80_default_min_mult_srgemm, f64_nn_t) {
  using precision = double;
  using OpClass   = cutlass::arch::OpClassSimt;
  using SmArch    = cutlass::arch::Sm80;
  using RingOp = cuasr::min_mult<precision>;

  using Srgemm = cuasr::gemm::device::Srgemm<                           //
      RingOp,                                                           //
      precision, cutlass::layout::ColumnMajor,                   //
      precision, cutlass::layout::ColumnMajor,                   //
      precision, cutlass::layout::RowMajor,
      precision, OpClass, SmArch>;

  EXPECT_TRUE(cuasr::test::gemm::device::TestAllGemm<Srgemm>());
}

///////////////////////////////////////////////////////////////////////////////

TEST(SM80_default_min_mult_srgemm, f32_tt_n) {
  using precision = float;
  using OpClass   = cutlass::arch::OpClassSimt;
  using SmArch    = cutlass::arch::Sm80;
  using RingOp = cuasr::min_mult<precision>;

  using Srgemm = cuasr::gemm::device::Srgemm<                           //
      RingOp,                                                           //
      precision, cutlass::layout::RowMajor,                   //
      precision, cutlass::layout::RowMajor,                   //
      precision, cutlass::layout::ColumnMajor,
      precision, OpClass, SmArch>;

  EXPECT_TRUE(cuasr::test::gemm::device::TestAllGemm<Srgemm>());
}

///////////////////////////////////////////////////////////////////////////////

TEST(SM80_default_min_mult_srgemm, f32_tt_t) {
  using precision = float;
  using OpClass   = cutlass::arch::OpClassSimt;
  using SmArch    = cutlass::arch::Sm80;
  using RingOp = cuasr::min_mult<precision>;

  using Srgemm = cuasr::gemm::device::Srgemm<                           //
      RingOp,                                                           //
      precision, cutlass::layout::RowMajor,                   //
      precision, cutlass::layout::RowMajor,                   //
      precision, cutlass::layout::RowMajor,
      precision, OpClass, SmArch>;

  EXPECT_TRUE(cuasr::test::gemm::device::TestAllGemm<Srgemm>());
}

///////////////////////////////////////////////////////////////////////////////

TEST(SM80_default_min_mult_srgemm, f32_tn_n) {
  using precision = float;
  using OpClass   = cutlass::arch::OpClassSimt;
  using SmArch    = cutlass::arch::Sm80;
  using RingOp = cuasr::min_mult<precision>;

  using Srgemm = cuasr::gemm::device::Srgemm<                           //
      RingOp,                                                           //
      precision, cutlass::layout::RowMajor,                   //
      precision, cutlass::layout::ColumnMajor,                   //
      precision, cutlass::layout::ColumnMajor,
      precision, OpClass, SmArch>;

  EXPECT_TRUE(cuasr::test::gemm::device::TestAllGemm<Srgemm>());
}

///////////////////////////////////////////////////////////////////////////////

TEST(SM80_default_min_mult_srgemm, f32_tn_t) {
  using precision = float;
  using OpClass   = cutlass::arch::OpClassSimt;
  using SmArch    = cutlass::arch::Sm80;
  using RingOp = cuasr::min_mult<precision>;

  using Srgemm = cuasr::gemm::device::Srgemm<                           //
      RingOp,                                                           //
      precision, cutlass::layout::RowMajor,                   //
      precision, cutlass::layout::ColumnMajor,                   //
      precision, cutlass::layout::RowMajor,
      precision, OpClass, SmArch>;

  EXPECT_TRUE(cuasr::test::gemm::device::TestAllGemm<Srgemm>());
}

///////////////////////////////////////////////////////////////////////////////

TEST(SM80_default_min_mult_srgemm, f32_nt_n) {
  using precision = float;
  using OpClass   = cutlass::arch::OpClassSimt;
  using SmArch    = cutlass::arch::Sm80;
  using RingOp = cuasr::min_mult<precision>;

  using Srgemm = cuasr::gemm::device::Srgemm<                           //
      RingOp,                                                           //
      precision, cutlass::layout::ColumnMajor,                   //
      precision, cutlass::layout::RowMajor,                   //
      precision, cutlass::layout::ColumnMajor,
      precision, OpClass, SmArch>;

  EXPECT_TRUE(cuasr::test::gemm::device::TestAllGemm<Srgemm>());
}

///////////////////////////////////////////////////////////////////////////////

TEST(SM80_default_min_mult_srgemm, f32_nt_t) {
  using precision = float;
  using OpClass   = cutlass::arch::OpClassSimt;
  using SmArch    = cutlass::arch::Sm80;
  using RingOp = cuasr::min_mult<precision>;

  using Srgemm = cuasr::gemm::device::Srgemm<                           //
      RingOp,                                                           //
      precision, cutlass::layout::ColumnMajor,                   //
      precision, cutlass::layout::RowMajor,                   //
      precision, cutlass::layout::RowMajor,
      precision, OpClass, SmArch>;

  EXPECT_TRUE(cuasr::test::gemm::device::TestAllGemm<Srgemm>());
}

///////////////////////////////////////////////////////////////////////////////

TEST(SM80_default_min_mult_srgemm, f32_nn_n) {
  using precision = float;
  using OpClass   = cutlass::arch::OpClassSimt;
  using SmArch    = cutlass::arch::Sm80;
  using RingOp = cuasr::min_mult<precision>;

  using Srgemm = cuasr::gemm::device::Srgemm<                           //
      RingOp,                                                           //
      precision, cutlass::layout::ColumnMajor,                   //
      precision, cutlass::layout::ColumnMajor,                   //
      precision, cutlass::layout::ColumnMajor,
      precision, OpClass, SmArch>;

  EXPECT_TRUE(cuasr::test::gemm::device::TestAllGemm<Srgemm>());
}

///////////////////////////////////////////////////////////////////////////////

TEST(SM80_default_min_mult_srgemm, f32_nn_t) {
  using precision = float;
  using OpClass   = cutlass::arch::OpClassSimt;
  using SmArch    = cutlass::arch::Sm80;
  using RingOp = cuasr::min_mult<precision>;

  using Srgemm = cuasr::gemm::device::Srgemm<                           //
      RingOp,                                                           //
      precision, cutlass::layout::ColumnMajor,                   //
      precision, cutlass::layout::ColumnMajor,                   //
      precision, cutlass::layout::RowMajor,
      precision, OpClass, SmArch>;

  EXPECT_TRUE(cuasr::test::gemm::device::TestAllGemm<Srgemm>());
}

///////////////////////////////////////////////////////////////////////////////

TEST(SM80_default_max_mult_srgemm, f64_tt_n) {
  using precision = double;
  using OpClass   = cutlass::arch::OpClassSimt;
  using SmArch    = cutlass::arch::Sm80;
  using RingOp = cuasr::max_mult<precision>;

  using Srgemm = cuasr::gemm::device::Srgemm<                           //
      RingOp,                                                           //
      precision, cutlass::layout::RowMajor,                   //
      precision, cutlass::layout::RowMajor,                   //
      precision, cutlass::layout::ColumnMajor,
      precision, OpClass, SmArch>;

  EXPECT_TRUE(cuasr::test::gemm::device::TestAllGemm<Srgemm>());
}

///////////////////////////////////////////////////////////////////////////////

TEST(SM80_default_max_mult_srgemm, f64_tt_t) {
  using precision = double;
  using OpClass   = cutlass::arch::OpClassSimt;
  using SmArch    = cutlass::arch::Sm80;
  using RingOp = cuasr::max_mult<precision>;

  using Srgemm = cuasr::gemm::device::Srgemm<                           //
      RingOp,                                                           //
      precision, cutlass::layout::RowMajor,                   //
      precision, cutlass::layout::RowMajor,                   //
      precision, cutlass::layout::RowMajor,
      precision, OpClass, SmArch>;

  EXPECT_TRUE(cuasr::test::gemm::device::TestAllGemm<Srgemm>());
}

///////////////////////////////////////////////////////////////////////////////

TEST(SM80_default_max_mult_srgemm, f64_tn_n) {
  using precision = double;
  using OpClass   = cutlass::arch::OpClassSimt;
  using SmArch    = cutlass::arch::Sm80;
  using RingOp = cuasr::max_mult<precision>;

  using Srgemm = cuasr::gemm::device::Srgemm<                           //
      RingOp,                                                           //
      precision, cutlass::layout::RowMajor,                   //
      precision, cutlass::layout::ColumnMajor,                   //
      precision, cutlass::layout::ColumnMajor,
      precision, OpClass, SmArch>;

  EXPECT_TRUE(cuasr::test::gemm::device::TestAllGemm<Srgemm>());
}

///////////////////////////////////////////////////////////////////////////////

TEST(SM80_default_max_mult_srgemm, f64_tn_t) {
  using precision = double;
  using OpClass   = cutlass::arch::OpClassSimt;
  using SmArch    = cutlass::arch::Sm80;
  using RingOp = cuasr::max_mult<precision>;

  using Srgemm = cuasr::gemm::device::Srgemm<                           //
      RingOp,                                                           //
      precision, cutlass::layout::RowMajor,                   //
      precision, cutlass::layout::ColumnMajor,                   //
      precision, cutlass::layout::RowMajor,
      precision, OpClass, SmArch>;

  EXPECT_TRUE(cuasr::test::gemm::device::TestAllGemm<Srgemm>());
}

///////////////////////////////////////////////////////////////////////////////

TEST(SM80_default_max_mult_srgemm, f64_nt_n) {
  using precision = double;
  using OpClass   = cutlass::arch::OpClassSimt;
  using SmArch    = cutlass::arch::Sm80;
  using RingOp = cuasr::max_mult<precision>;

  using Srgemm = cuasr::gemm::device::Srgemm<                           //
      RingOp,                                                           //
      precision, cutlass::layout::ColumnMajor,                   //
      precision, cutlass::layout::RowMajor,                   //
      precision, cutlass::layout::ColumnMajor,
      precision, OpClass, SmArch>;

  EXPECT_TRUE(cuasr::test::gemm::device::TestAllGemm<Srgemm>());
}

///////////////////////////////////////////////////////////////////////////////

TEST(SM80_default_max_mult_srgemm, f64_nt_t) {
  using precision = double;
  using OpClass   = cutlass::arch::OpClassSimt;
  using SmArch    = cutlass::arch::Sm80;
  using RingOp = cuasr::max_mult<precision>;

  using Srgemm = cuasr::gemm::device::Srgemm<                           //
      RingOp,                                                           //
      precision, cutlass::layout::ColumnMajor,                   //
      precision, cutlass::layout::RowMajor,                   //
      precision, cutlass::layout::RowMajor,
      precision, OpClass, SmArch>;

  EXPECT_TRUE(cuasr::test::gemm::device::TestAllGemm<Srgemm>());
}

///////////////////////////////////////////////////////////////////////////////

TEST(SM80_default_max_mult_srgemm, f64_nn_n) {
  using precision = double;
  using OpClass   = cutlass::arch::OpClassSimt;
  using SmArch    = cutlass::arch::Sm80;
  using RingOp = cuasr::max_mult<precision>;

  using Srgemm = cuasr::gemm::device::Srgemm<                           //
      RingOp,                                                           //
      precision, cutlass::layout::ColumnMajor,                   //
      precision, cutlass::layout::ColumnMajor,                   //
      precision, cutlass::layout::ColumnMajor,
      precision, OpClass, SmArch>;

  EXPECT_TRUE(cuasr::test::gemm::device::TestAllGemm<Srgemm>());
}

///////////////////////////////////////////////////////////////////////////////

TEST(SM80_default_max_mult_srgemm, f64_nn_t) {
  using precision = double;
  using OpClass   = cutlass::arch::OpClassSimt;
  using SmArch    = cutlass::arch::Sm80;
  using RingOp = cuasr::max_mult<precision>;

  using Srgemm = cuasr::gemm::device::Srgemm<                           //
      RingOp,                                                           //
      precision, cutlass::layout::ColumnMajor,                   //
      precision, cutlass::layout::ColumnMajor,                   //
      precision, cutlass::layout::RowMajor,
      precision, OpClass, SmArch>;

  EXPECT_TRUE(cuasr::test::gemm::device::TestAllGemm<Srgemm>());
}

///////////////////////////////////////////////////////////////////////////////

TEST(SM80_default_max_mult_srgemm, f32_tt_n) {
  using precision = float;
  using OpClass   = cutlass::arch::OpClassSimt;
  using SmArch    = cutlass::arch::Sm80;
  using RingOp = cuasr::max_mult<precision>;

  using Srgemm = cuasr::gemm::device::Srgemm<                           //
      RingOp,                                                           //
      precision, cutlass::layout::RowMajor,                   //
      precision, cutlass::layout::RowMajor,                   //
      precision, cutlass::layout::ColumnMajor,
      precision, OpClass, SmArch>;

  EXPECT_TRUE(cuasr::test::gemm::device::TestAllGemm<Srgemm>());
}

///////////////////////////////////////////////////////////////////////////////

TEST(SM80_default_max_mult_srgemm, f32_tt_t) {
  using precision = float;
  using OpClass   = cutlass::arch::OpClassSimt;
  using SmArch    = cutlass::arch::Sm80;
  using RingOp = cuasr::max_mult<precision>;

  using Srgemm = cuasr::gemm::device::Srgemm<                           //
      RingOp,                                                           //
      precision, cutlass::layout::RowMajor,                   //
      precision, cutlass::layout::RowMajor,                   //
      precision, cutlass::layout::RowMajor,
      precision, OpClass, SmArch>;

  EXPECT_TRUE(cuasr::test::gemm::device::TestAllGemm<Srgemm>());
}

///////////////////////////////////////////////////////////////////////////////

TEST(SM80_default_max_mult_srgemm, f32_tn_n) {
  using precision = float;
  using OpClass   = cutlass::arch::OpClassSimt;
  using SmArch    = cutlass::arch::Sm80;
  using RingOp = cuasr::max_mult<precision>;

  using Srgemm = cuasr::gemm::device::Srgemm<                           //
      RingOp,                                                           //
      precision, cutlass::layout::RowMajor,                   //
      precision, cutlass::layout::ColumnMajor,                   //
      precision, cutlass::layout::ColumnMajor,
      precision, OpClass, SmArch>;

  EXPECT_TRUE(cuasr::test::gemm::device::TestAllGemm<Srgemm>());
}

///////////////////////////////////////////////////////////////////////////////

TEST(SM80_default_max_mult_srgemm, f32_tn_t) {
  using precision = float;
  using OpClass   = cutlass::arch::OpClassSimt;
  using SmArch    = cutlass::arch::Sm80;
  using RingOp = cuasr::max_mult<precision>;

  using Srgemm = cuasr::gemm::device::Srgemm<                           //
      RingOp,                                                           //
      precision, cutlass::layout::RowMajor,                   //
      precision, cutlass::layout::ColumnMajor,                   //
      precision, cutlass::layout::RowMajor,
      precision, OpClass, SmArch>;

  EXPECT_TRUE(cuasr::test::gemm::device::TestAllGemm<Srgemm>());
}

///////////////////////////////////////////////////////////////////////////////

TEST(SM80_default_max_mult_srgemm, f32_nt_n) {
  using precision = float;
  using OpClass   = cutlass::arch::OpClassSimt;
  using SmArch    = cutlass::arch::Sm80;
  using RingOp = cuasr::max_mult<precision>;

  using Srgemm = cuasr::gemm::device::Srgemm<                           //
      RingOp,                                                           //
      precision, cutlass::layout::ColumnMajor,                   //
      precision, cutlass::layout::RowMajor,                   //
      precision, cutlass::layout::ColumnMajor,
      precision, OpClass, SmArch>;

  EXPECT_TRUE(cuasr::test::gemm::device::TestAllGemm<Srgemm>());
}

///////////////////////////////////////////////////////////////////////////////

TEST(SM80_default_max_mult_srgemm, f32_nt_t) {
  using precision = float;
  using OpClass   = cutlass::arch::OpClassSimt;
  using SmArch    = cutlass::arch::Sm80;
  using RingOp = cuasr::max_mult<precision>;

  using Srgemm = cuasr::gemm::device::Srgemm<                           //
      RingOp,                                                           //
      precision, cutlass::layout::ColumnMajor,                   //
      precision, cutlass::layout::RowMajor,                   //
      precision, cutlass::layout::RowMajor,
      precision, OpClass, SmArch>;

  EXPECT_TRUE(cuasr::test::gemm::device::TestAllGemm<Srgemm>());
}

///////////////////////////////////////////////////////////////////////////////

TEST(SM80_default_max_mult_srgemm, f32_nn_n) {
  using precision = float;
  using OpClass   = cutlass::arch::OpClassSimt;
  using SmArch    = cutlass::arch::Sm80;
  using RingOp = cuasr::max_mult<precision>;

  using Srgemm = cuasr::gemm::device::Srgemm<                           //
      RingOp,                                                           //
      precision, cutlass::layout::ColumnMajor,                   //
      precision, cutlass::layout::ColumnMajor,                   //
      precision, cutlass::layout::ColumnMajor,
      precision, OpClass, SmArch>;

  EXPECT_TRUE(cuasr::test::gemm::device::TestAllGemm<Srgemm>());
}

///////////////////////////////////////////////////////////////////////////////

TEST(SM80_default_max_mult_srgemm, f32_nn_t) {
  using precision = float;
  using OpClass   = cutlass::arch::OpClassSimt;
  using SmArch    = cutlass::arch::Sm80;
  using RingOp = cuasr::max_mult<precision>;

  using Srgemm = cuasr::gemm::device::Srgemm<                           //
      RingOp,                                                           //
      precision, cutlass::layout::ColumnMajor,                   //
      precision, cutlass::layout::ColumnMajor,                   //
      precision, cutlass::layout::RowMajor,
      precision, OpClass, SmArch>;

  EXPECT_TRUE(cuasr::test::gemm::device::TestAllGemm<Srgemm>());
}

///////////////////////////////////////////////////////////////////////////////

TEST(SM80_default_or_and_srgemm, f64_tt_n) {
  using precision = double;
  using OpClass   = cutlass::arch::OpClassSimt;
  using SmArch    = cutlass::arch::Sm80;
  using RingOp = cuasr::or_and<precision>;

  using Srgemm = cuasr::gemm::device::Srgemm<                           //
      RingOp,                                                           //
      precision, cutlass::layout::RowMajor,                   //
      precision, cutlass::layout::RowMajor,                   //
      precision, cutlass::layout::ColumnMajor,
      precision, OpClass, SmArch>;

  EXPECT_TRUE(cuasr::test::gemm::device::TestAllGemm<Srgemm>());
}

///////////////////////////////////////////////////////////////////////////////

TEST(SM80_default_or_and_srgemm, f64_tt_t) {
  using precision = double;
  using OpClass   = cutlass::arch::OpClassSimt;
  using SmArch    = cutlass::arch::Sm80;
  using RingOp = cuasr::or_and<precision>;

  using Srgemm = cuasr::gemm::device::Srgemm<                           //
      RingOp,                                                           //
      precision, cutlass::layout::RowMajor,                   //
      precision, cutlass::layout::RowMajor,                   //
      precision, cutlass::layout::RowMajor,
      precision, OpClass, SmArch>;

  EXPECT_TRUE(cuasr::test::gemm::device::TestAllGemm<Srgemm>());
}

///////////////////////////////////////////////////////////////////////////////

TEST(SM80_default_or_and_srgemm, f64_tn_n) {
  using precision = double;
  using OpClass   = cutlass::arch::OpClassSimt;
  using SmArch    = cutlass::arch::Sm80;
  using RingOp = cuasr::or_and<precision>;

  using Srgemm = cuasr::gemm::device::Srgemm<                           //
      RingOp,                                                           //
      precision, cutlass::layout::RowMajor,                   //
      precision, cutlass::layout::ColumnMajor,                   //
      precision, cutlass::layout::ColumnMajor,
      precision, OpClass, SmArch>;

  EXPECT_TRUE(cuasr::test::gemm::device::TestAllGemm<Srgemm>());
}

///////////////////////////////////////////////////////////////////////////////

TEST(SM80_default_or_and_srgemm, f64_tn_t) {
  using precision = double;
  using OpClass   = cutlass::arch::OpClassSimt;
  using SmArch    = cutlass::arch::Sm80;
  using RingOp = cuasr::or_and<precision>;

  using Srgemm = cuasr::gemm::device::Srgemm<                           //
      RingOp,                                                           //
      precision, cutlass::layout::RowMajor,                   //
      precision, cutlass::layout::ColumnMajor,                   //
      precision, cutlass::layout::RowMajor,
      precision, OpClass, SmArch>;

  EXPECT_TRUE(cuasr::test::gemm::device::TestAllGemm<Srgemm>());
}

///////////////////////////////////////////////////////////////////////////////

TEST(SM80_default_or_and_srgemm, f64_nt_n) {
  using precision = double;
  using OpClass   = cutlass::arch::OpClassSimt;
  using SmArch    = cutlass::arch::Sm80;
  using RingOp = cuasr::or_and<precision>;

  using Srgemm = cuasr::gemm::device::Srgemm<                           //
      RingOp,                                                           //
      precision, cutlass::layout::ColumnMajor,                   //
      precision, cutlass::layout::RowMajor,                   //
      precision, cutlass::layout::ColumnMajor,
      precision, OpClass, SmArch>;

  EXPECT_TRUE(cuasr::test::gemm::device::TestAllGemm<Srgemm>());
}

///////////////////////////////////////////////////////////////////////////////

TEST(SM80_default_or_and_srgemm, f64_nt_t) {
  using precision = double;
  using OpClass   = cutlass::arch::OpClassSimt;
  using SmArch    = cutlass::arch::Sm80;
  using RingOp = cuasr::or_and<precision>;

  using Srgemm = cuasr::gemm::device::Srgemm<                           //
      RingOp,                                                           //
      precision, cutlass::layout::ColumnMajor,                   //
      precision, cutlass::layout::RowMajor,                   //
      precision, cutlass::layout::RowMajor,
      precision, OpClass, SmArch>;

  EXPECT_TRUE(cuasr::test::gemm::device::TestAllGemm<Srgemm>());
}

///////////////////////////////////////////////////////////////////////////////

TEST(SM80_default_or_and_srgemm, f64_nn_n) {
  using precision = double;
  using OpClass   = cutlass::arch::OpClassSimt;
  using SmArch    = cutlass::arch::Sm80;
  using RingOp = cuasr::or_and<precision>;

  using Srgemm = cuasr::gemm::device::Srgemm<                           //
      RingOp,                                                           //
      precision, cutlass::layout::ColumnMajor,                   //
      precision, cutlass::layout::ColumnMajor,                   //
      precision, cutlass::layout::ColumnMajor,
      precision, OpClass, SmArch>;

  EXPECT_TRUE(cuasr::test::gemm::device::TestAllGemm<Srgemm>());
}

///////////////////////////////////////////////////////////////////////////////

TEST(SM80_default_or_and_srgemm, f64_nn_t) {
  using precision = double;
  using OpClass   = cutlass::arch::OpClassSimt;
  using SmArch    = cutlass::arch::Sm80;
  using RingOp = cuasr::or_and<precision>;

  using Srgemm = cuasr::gemm::device::Srgemm<                           //
      RingOp,                                                           //
      precision, cutlass::layout::ColumnMajor,                   //
      precision, cutlass::layout::ColumnMajor,                   //
      precision, cutlass::layout::RowMajor,
      precision, OpClass, SmArch>;

  EXPECT_TRUE(cuasr::test::gemm::device::TestAllGemm<Srgemm>());
}

///////////////////////////////////////////////////////////////////////////////

TEST(SM80_default_or_and_srgemm, f32_tt_n) {
  using precision = float;
  using OpClass   = cutlass::arch::OpClassSimt;
  using SmArch    = cutlass::arch::Sm80;
  using RingOp = cuasr::or_and<precision>;

  using Srgemm = cuasr::gemm::device::Srgemm<                           //
      RingOp,                                                           //
      precision, cutlass::layout::RowMajor,                   //
      precision, cutlass::layout::RowMajor,                   //
      precision, cutlass::layout::ColumnMajor,
      precision, OpClass, SmArch>;

  EXPECT_TRUE(cuasr::test::gemm::device::TestAllGemm<Srgemm>());
}

///////////////////////////////////////////////////////////////////////////////

TEST(SM80_default_or_and_srgemm, f32_tt_t) {
  using precision = float;
  using OpClass   = cutlass::arch::OpClassSimt;
  using SmArch    = cutlass::arch::Sm80;
  using RingOp = cuasr::or_and<precision>;

  using Srgemm = cuasr::gemm::device::Srgemm<                           //
      RingOp,                                                           //
      precision, cutlass::layout::RowMajor,                   //
      precision, cutlass::layout::RowMajor,                   //
      precision, cutlass::layout::RowMajor,
      precision, OpClass, SmArch>;

  EXPECT_TRUE(cuasr::test::gemm::device::TestAllGemm<Srgemm>());
}

///////////////////////////////////////////////////////////////////////////////

TEST(SM80_default_or_and_srgemm, f32_tn_n) {
  using precision = float;
  using OpClass   = cutlass::arch::OpClassSimt;
  using SmArch    = cutlass::arch::Sm80;
  using RingOp = cuasr::or_and<precision>;

  using Srgemm = cuasr::gemm::device::Srgemm<                           //
      RingOp,                                                           //
      precision, cutlass::layout::RowMajor,                   //
      precision, cutlass::layout::ColumnMajor,                   //
      precision, cutlass::layout::ColumnMajor,
      precision, OpClass, SmArch>;

  EXPECT_TRUE(cuasr::test::gemm::device::TestAllGemm<Srgemm>());
}

///////////////////////////////////////////////////////////////////////////////

TEST(SM80_default_or_and_srgemm, f32_tn_t) {
  using precision = float;
  using OpClass   = cutlass::arch::OpClassSimt;
  using SmArch    = cutlass::arch::Sm80;
  using RingOp = cuasr::or_and<precision>;

  using Srgemm = cuasr::gemm::device::Srgemm<                           //
      RingOp,                                                           //
      precision, cutlass::layout::RowMajor,                   //
      precision, cutlass::layout::ColumnMajor,                   //
      precision, cutlass::layout::RowMajor,
      precision, OpClass, SmArch>;

  EXPECT_TRUE(cuasr::test::gemm::device::TestAllGemm<Srgemm>());
}

///////////////////////////////////////////////////////////////////////////////

TEST(SM80_default_or_and_srgemm, f32_nt_n) {
  using precision = float;
  using OpClass   = cutlass::arch::OpClassSimt;
  using SmArch    = cutlass::arch::Sm80;
  using RingOp = cuasr::or_and<precision>;

  using Srgemm = cuasr::gemm::device::Srgemm<                           //
      RingOp,                                                           //
      precision, cutlass::layout::ColumnMajor,                   //
      precision, cutlass::layout::RowMajor,                   //
      precision, cutlass::layout::ColumnMajor,
      precision, OpClass, SmArch>;

  EXPECT_TRUE(cuasr::test::gemm::device::TestAllGemm<Srgemm>());
}

///////////////////////////////////////////////////////////////////////////////

TEST(SM80_default_or_and_srgemm, f32_nt_t) {
  using precision = float;
  using OpClass   = cutlass::arch::OpClassSimt;
  using SmArch    = cutlass::arch::Sm80;
  using RingOp = cuasr::or_and<precision>;

  using Srgemm = cuasr::gemm::device::Srgemm<                           //
      RingOp,                                                           //
      precision, cutlass::layout::ColumnMajor,                   //
      precision, cutlass::layout::RowMajor,                   //
      precision, cutlass::layout::RowMajor,
      precision, OpClass, SmArch>;

  EXPECT_TRUE(cuasr::test::gemm::device::TestAllGemm<Srgemm>());
}

///////////////////////////////////////////////////////////////////////////////

TEST(SM80_default_or_and_srgemm, f32_nn_n) {
  using precision = float;
  using OpClass   = cutlass::arch::OpClassSimt;
  using SmArch    = cutlass::arch::Sm80;
  using RingOp = cuasr::or_and<precision>;

  using Srgemm = cuasr::gemm::device::Srgemm<                           //
      RingOp,                                                           //
      precision, cutlass::layout::ColumnMajor,                   //
      precision, cutlass::layout::ColumnMajor,                   //
      precision, cutlass::layout::ColumnMajor,
      precision, OpClass, SmArch>;

  EXPECT_TRUE(cuasr::test::gemm::device::TestAllGemm<Srgemm>());
}

///////////////////////////////////////////////////////////////////////////////

TEST(SM80_default_or_and_srgemm, f32_nn_t) {
  using precision = float;
  using OpClass   = cutlass::arch::OpClassSimt;
  using SmArch    = cutlass::arch::Sm80;
  using RingOp = cuasr::or_and<precision>;

  using Srgemm = cuasr::gemm::device::Srgemm<                           //
      RingOp,                                                           //
      precision, cutlass::layout::ColumnMajor,                   //
      precision, cutlass::layout::ColumnMajor,                   //
      precision, cutlass::layout::RowMajor,
      precision, OpClass, SmArch>;

  EXPECT_TRUE(cuasr::test::gemm::device::TestAllGemm<Srgemm>());
}

