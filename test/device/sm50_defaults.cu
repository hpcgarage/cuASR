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

TEST(SM50_default_plus_mult_d_srgemm, tt_n) {
  using precision = double;
  using OpClass   = cutlass::arch::OpClassSimt;
  using SmArch    = cutlass::arch::Sm50;
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

TEST(SM50_default_plus_mult_d_srgemm, tt_t) {
  using precision = double;
  using OpClass   = cutlass::arch::OpClassSimt;
  using SmArch    = cutlass::arch::Sm50;
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

TEST(SM50_default_plus_mult_d_srgemm, tn_n) {
  using precision = double;
  using OpClass   = cutlass::arch::OpClassSimt;
  using SmArch    = cutlass::arch::Sm50;
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

TEST(SM50_default_plus_mult_d_srgemm, tn_t) {
  using precision = double;
  using OpClass   = cutlass::arch::OpClassSimt;
  using SmArch    = cutlass::arch::Sm50;
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

TEST(SM50_default_plus_mult_d_srgemm, nt_n) {
  using precision = double;
  using OpClass   = cutlass::arch::OpClassSimt;
  using SmArch    = cutlass::arch::Sm50;
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

TEST(SM50_default_plus_mult_d_srgemm, nt_t) {
  using precision = double;
  using OpClass   = cutlass::arch::OpClassSimt;
  using SmArch    = cutlass::arch::Sm50;
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

TEST(SM50_default_plus_mult_d_srgemm, nn_n) {
  using precision = double;
  using OpClass   = cutlass::arch::OpClassSimt;
  using SmArch    = cutlass::arch::Sm50;
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

TEST(SM50_default_plus_mult_d_srgemm, nn_t) {
  using precision = double;
  using OpClass   = cutlass::arch::OpClassSimt;
  using SmArch    = cutlass::arch::Sm50;
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

TEST(SM50_default_plus_mult_s_srgemm, tt_n) {
  using precision = float;
  using OpClass   = cutlass::arch::OpClassSimt;
  using SmArch    = cutlass::arch::Sm50;
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

TEST(SM50_default_plus_mult_s_srgemm, tt_t) {
  using precision = float;
  using OpClass   = cutlass::arch::OpClassSimt;
  using SmArch    = cutlass::arch::Sm50;
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

TEST(SM50_default_plus_mult_s_srgemm, tn_n) {
  using precision = float;
  using OpClass   = cutlass::arch::OpClassSimt;
  using SmArch    = cutlass::arch::Sm50;
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

TEST(SM50_default_plus_mult_s_srgemm, tn_t) {
  using precision = float;
  using OpClass   = cutlass::arch::OpClassSimt;
  using SmArch    = cutlass::arch::Sm50;
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

TEST(SM50_default_plus_mult_s_srgemm, nt_n) {
  using precision = float;
  using OpClass   = cutlass::arch::OpClassSimt;
  using SmArch    = cutlass::arch::Sm50;
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

TEST(SM50_default_plus_mult_s_srgemm, nt_t) {
  using precision = float;
  using OpClass   = cutlass::arch::OpClassSimt;
  using SmArch    = cutlass::arch::Sm50;
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

TEST(SM50_default_plus_mult_s_srgemm, nn_n) {
  using precision = float;
  using OpClass   = cutlass::arch::OpClassSimt;
  using SmArch    = cutlass::arch::Sm50;
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

TEST(SM50_default_plus_mult_s_srgemm, nn_t) {
  using precision = float;
  using OpClass   = cutlass::arch::OpClassSimt;
  using SmArch    = cutlass::arch::Sm50;
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

TEST(SM50_default_min_plus_d_srgemm, tt_n) {
  using precision = double;
  using OpClass   = cutlass::arch::OpClassSimt;
  using SmArch    = cutlass::arch::Sm50;
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

TEST(SM50_default_min_plus_d_srgemm, tt_t) {
  using precision = double;
  using OpClass   = cutlass::arch::OpClassSimt;
  using SmArch    = cutlass::arch::Sm50;
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

TEST(SM50_default_min_plus_d_srgemm, tn_n) {
  using precision = double;
  using OpClass   = cutlass::arch::OpClassSimt;
  using SmArch    = cutlass::arch::Sm50;
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

TEST(SM50_default_min_plus_d_srgemm, tn_t) {
  using precision = double;
  using OpClass   = cutlass::arch::OpClassSimt;
  using SmArch    = cutlass::arch::Sm50;
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

TEST(SM50_default_min_plus_d_srgemm, nt_n) {
  using precision = double;
  using OpClass   = cutlass::arch::OpClassSimt;
  using SmArch    = cutlass::arch::Sm50;
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

TEST(SM50_default_min_plus_d_srgemm, nt_t) {
  using precision = double;
  using OpClass   = cutlass::arch::OpClassSimt;
  using SmArch    = cutlass::arch::Sm50;
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

TEST(SM50_default_min_plus_d_srgemm, nn_n) {
  using precision = double;
  using OpClass   = cutlass::arch::OpClassSimt;
  using SmArch    = cutlass::arch::Sm50;
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

TEST(SM50_default_min_plus_d_srgemm, nn_t) {
  using precision = double;
  using OpClass   = cutlass::arch::OpClassSimt;
  using SmArch    = cutlass::arch::Sm50;
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

TEST(SM50_default_min_plus_s_srgemm, tt_n) {
  using precision = float;
  using OpClass   = cutlass::arch::OpClassSimt;
  using SmArch    = cutlass::arch::Sm50;
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

TEST(SM50_default_min_plus_s_srgemm, tt_t) {
  using precision = float;
  using OpClass   = cutlass::arch::OpClassSimt;
  using SmArch    = cutlass::arch::Sm50;
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

TEST(SM50_default_min_plus_s_srgemm, tn_n) {
  using precision = float;
  using OpClass   = cutlass::arch::OpClassSimt;
  using SmArch    = cutlass::arch::Sm50;
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

TEST(SM50_default_min_plus_s_srgemm, tn_t) {
  using precision = float;
  using OpClass   = cutlass::arch::OpClassSimt;
  using SmArch    = cutlass::arch::Sm50;
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

TEST(SM50_default_min_plus_s_srgemm, nt_n) {
  using precision = float;
  using OpClass   = cutlass::arch::OpClassSimt;
  using SmArch    = cutlass::arch::Sm50;
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

TEST(SM50_default_min_plus_s_srgemm, nt_t) {
  using precision = float;
  using OpClass   = cutlass::arch::OpClassSimt;
  using SmArch    = cutlass::arch::Sm50;
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

TEST(SM50_default_min_plus_s_srgemm, nn_n) {
  using precision = float;
  using OpClass   = cutlass::arch::OpClassSimt;
  using SmArch    = cutlass::arch::Sm50;
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

TEST(SM50_default_min_plus_s_srgemm, nn_t) {
  using precision = float;
  using OpClass   = cutlass::arch::OpClassSimt;
  using SmArch    = cutlass::arch::Sm50;
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

TEST(SM50_default_max_plus_d_srgemm, tt_n) {
  using precision = double;
  using OpClass   = cutlass::arch::OpClassSimt;
  using SmArch    = cutlass::arch::Sm50;
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

TEST(SM50_default_max_plus_d_srgemm, tt_t) {
  using precision = double;
  using OpClass   = cutlass::arch::OpClassSimt;
  using SmArch    = cutlass::arch::Sm50;
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

TEST(SM50_default_max_plus_d_srgemm, tn_n) {
  using precision = double;
  using OpClass   = cutlass::arch::OpClassSimt;
  using SmArch    = cutlass::arch::Sm50;
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

TEST(SM50_default_max_plus_d_srgemm, tn_t) {
  using precision = double;
  using OpClass   = cutlass::arch::OpClassSimt;
  using SmArch    = cutlass::arch::Sm50;
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

TEST(SM50_default_max_plus_d_srgemm, nt_n) {
  using precision = double;
  using OpClass   = cutlass::arch::OpClassSimt;
  using SmArch    = cutlass::arch::Sm50;
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

TEST(SM50_default_max_plus_d_srgemm, nt_t) {
  using precision = double;
  using OpClass   = cutlass::arch::OpClassSimt;
  using SmArch    = cutlass::arch::Sm50;
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

TEST(SM50_default_max_plus_d_srgemm, nn_n) {
  using precision = double;
  using OpClass   = cutlass::arch::OpClassSimt;
  using SmArch    = cutlass::arch::Sm50;
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

TEST(SM50_default_max_plus_d_srgemm, nn_t) {
  using precision = double;
  using OpClass   = cutlass::arch::OpClassSimt;
  using SmArch    = cutlass::arch::Sm50;
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

TEST(SM50_default_max_plus_s_srgemm, tt_n) {
  using precision = float;
  using OpClass   = cutlass::arch::OpClassSimt;
  using SmArch    = cutlass::arch::Sm50;
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

TEST(SM50_default_max_plus_s_srgemm, tt_t) {
  using precision = float;
  using OpClass   = cutlass::arch::OpClassSimt;
  using SmArch    = cutlass::arch::Sm50;
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

TEST(SM50_default_max_plus_s_srgemm, tn_n) {
  using precision = float;
  using OpClass   = cutlass::arch::OpClassSimt;
  using SmArch    = cutlass::arch::Sm50;
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

TEST(SM50_default_max_plus_s_srgemm, tn_t) {
  using precision = float;
  using OpClass   = cutlass::arch::OpClassSimt;
  using SmArch    = cutlass::arch::Sm50;
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

TEST(SM50_default_max_plus_s_srgemm, nt_n) {
  using precision = float;
  using OpClass   = cutlass::arch::OpClassSimt;
  using SmArch    = cutlass::arch::Sm50;
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

TEST(SM50_default_max_plus_s_srgemm, nt_t) {
  using precision = float;
  using OpClass   = cutlass::arch::OpClassSimt;
  using SmArch    = cutlass::arch::Sm50;
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

TEST(SM50_default_max_plus_s_srgemm, nn_n) {
  using precision = float;
  using OpClass   = cutlass::arch::OpClassSimt;
  using SmArch    = cutlass::arch::Sm50;
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

TEST(SM50_default_max_plus_s_srgemm, nn_t) {
  using precision = float;
  using OpClass   = cutlass::arch::OpClassSimt;
  using SmArch    = cutlass::arch::Sm50;
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

TEST(SM50_default_min_max_d_srgemm, tt_n) {
  using precision = double;
  using OpClass   = cutlass::arch::OpClassSimt;
  using SmArch    = cutlass::arch::Sm50;
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

TEST(SM50_default_min_max_d_srgemm, tt_t) {
  using precision = double;
  using OpClass   = cutlass::arch::OpClassSimt;
  using SmArch    = cutlass::arch::Sm50;
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

TEST(SM50_default_min_max_d_srgemm, tn_n) {
  using precision = double;
  using OpClass   = cutlass::arch::OpClassSimt;
  using SmArch    = cutlass::arch::Sm50;
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

TEST(SM50_default_min_max_d_srgemm, tn_t) {
  using precision = double;
  using OpClass   = cutlass::arch::OpClassSimt;
  using SmArch    = cutlass::arch::Sm50;
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

TEST(SM50_default_min_max_d_srgemm, nt_n) {
  using precision = double;
  using OpClass   = cutlass::arch::OpClassSimt;
  using SmArch    = cutlass::arch::Sm50;
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

TEST(SM50_default_min_max_d_srgemm, nt_t) {
  using precision = double;
  using OpClass   = cutlass::arch::OpClassSimt;
  using SmArch    = cutlass::arch::Sm50;
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

TEST(SM50_default_min_max_d_srgemm, nn_n) {
  using precision = double;
  using OpClass   = cutlass::arch::OpClassSimt;
  using SmArch    = cutlass::arch::Sm50;
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

TEST(SM50_default_min_max_d_srgemm, nn_t) {
  using precision = double;
  using OpClass   = cutlass::arch::OpClassSimt;
  using SmArch    = cutlass::arch::Sm50;
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

TEST(SM50_default_min_max_s_srgemm, tt_n) {
  using precision = float;
  using OpClass   = cutlass::arch::OpClassSimt;
  using SmArch    = cutlass::arch::Sm50;
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

TEST(SM50_default_min_max_s_srgemm, tt_t) {
  using precision = float;
  using OpClass   = cutlass::arch::OpClassSimt;
  using SmArch    = cutlass::arch::Sm50;
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

TEST(SM50_default_min_max_s_srgemm, tn_n) {
  using precision = float;
  using OpClass   = cutlass::arch::OpClassSimt;
  using SmArch    = cutlass::arch::Sm50;
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

TEST(SM50_default_min_max_s_srgemm, tn_t) {
  using precision = float;
  using OpClass   = cutlass::arch::OpClassSimt;
  using SmArch    = cutlass::arch::Sm50;
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

TEST(SM50_default_min_max_s_srgemm, nt_n) {
  using precision = float;
  using OpClass   = cutlass::arch::OpClassSimt;
  using SmArch    = cutlass::arch::Sm50;
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

TEST(SM50_default_min_max_s_srgemm, nt_t) {
  using precision = float;
  using OpClass   = cutlass::arch::OpClassSimt;
  using SmArch    = cutlass::arch::Sm50;
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

TEST(SM50_default_min_max_s_srgemm, nn_n) {
  using precision = float;
  using OpClass   = cutlass::arch::OpClassSimt;
  using SmArch    = cutlass::arch::Sm50;
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

TEST(SM50_default_min_max_s_srgemm, nn_t) {
  using precision = float;
  using OpClass   = cutlass::arch::OpClassSimt;
  using SmArch    = cutlass::arch::Sm50;
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

TEST(SM50_default_max_min_d_srgemm, tt_n) {
  using precision = double;
  using OpClass   = cutlass::arch::OpClassSimt;
  using SmArch    = cutlass::arch::Sm50;
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

TEST(SM50_default_max_min_d_srgemm, tt_t) {
  using precision = double;
  using OpClass   = cutlass::arch::OpClassSimt;
  using SmArch    = cutlass::arch::Sm50;
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

TEST(SM50_default_max_min_d_srgemm, tn_n) {
  using precision = double;
  using OpClass   = cutlass::arch::OpClassSimt;
  using SmArch    = cutlass::arch::Sm50;
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

TEST(SM50_default_max_min_d_srgemm, tn_t) {
  using precision = double;
  using OpClass   = cutlass::arch::OpClassSimt;
  using SmArch    = cutlass::arch::Sm50;
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

TEST(SM50_default_max_min_d_srgemm, nt_n) {
  using precision = double;
  using OpClass   = cutlass::arch::OpClassSimt;
  using SmArch    = cutlass::arch::Sm50;
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

TEST(SM50_default_max_min_d_srgemm, nt_t) {
  using precision = double;
  using OpClass   = cutlass::arch::OpClassSimt;
  using SmArch    = cutlass::arch::Sm50;
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

TEST(SM50_default_max_min_d_srgemm, nn_n) {
  using precision = double;
  using OpClass   = cutlass::arch::OpClassSimt;
  using SmArch    = cutlass::arch::Sm50;
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

TEST(SM50_default_max_min_d_srgemm, nn_t) {
  using precision = double;
  using OpClass   = cutlass::arch::OpClassSimt;
  using SmArch    = cutlass::arch::Sm50;
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

TEST(SM50_default_max_min_s_srgemm, tt_n) {
  using precision = float;
  using OpClass   = cutlass::arch::OpClassSimt;
  using SmArch    = cutlass::arch::Sm50;
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

TEST(SM50_default_max_min_s_srgemm, tt_t) {
  using precision = float;
  using OpClass   = cutlass::arch::OpClassSimt;
  using SmArch    = cutlass::arch::Sm50;
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

TEST(SM50_default_max_min_s_srgemm, tn_n) {
  using precision = float;
  using OpClass   = cutlass::arch::OpClassSimt;
  using SmArch    = cutlass::arch::Sm50;
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

TEST(SM50_default_max_min_s_srgemm, tn_t) {
  using precision = float;
  using OpClass   = cutlass::arch::OpClassSimt;
  using SmArch    = cutlass::arch::Sm50;
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

TEST(SM50_default_max_min_s_srgemm, nt_n) {
  using precision = float;
  using OpClass   = cutlass::arch::OpClassSimt;
  using SmArch    = cutlass::arch::Sm50;
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

TEST(SM50_default_max_min_s_srgemm, nt_t) {
  using precision = float;
  using OpClass   = cutlass::arch::OpClassSimt;
  using SmArch    = cutlass::arch::Sm50;
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

TEST(SM50_default_max_min_s_srgemm, nn_n) {
  using precision = float;
  using OpClass   = cutlass::arch::OpClassSimt;
  using SmArch    = cutlass::arch::Sm50;
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

TEST(SM50_default_max_min_s_srgemm, nn_t) {
  using precision = float;
  using OpClass   = cutlass::arch::OpClassSimt;
  using SmArch    = cutlass::arch::Sm50;
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

TEST(SM50_default_min_mult_d_srgemm, tt_n) {
  using precision = double;
  using OpClass   = cutlass::arch::OpClassSimt;
  using SmArch    = cutlass::arch::Sm50;
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

TEST(SM50_default_min_mult_d_srgemm, tt_t) {
  using precision = double;
  using OpClass   = cutlass::arch::OpClassSimt;
  using SmArch    = cutlass::arch::Sm50;
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

TEST(SM50_default_min_mult_d_srgemm, tn_n) {
  using precision = double;
  using OpClass   = cutlass::arch::OpClassSimt;
  using SmArch    = cutlass::arch::Sm50;
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

TEST(SM50_default_min_mult_d_srgemm, tn_t) {
  using precision = double;
  using OpClass   = cutlass::arch::OpClassSimt;
  using SmArch    = cutlass::arch::Sm50;
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

TEST(SM50_default_min_mult_d_srgemm, nt_n) {
  using precision = double;
  using OpClass   = cutlass::arch::OpClassSimt;
  using SmArch    = cutlass::arch::Sm50;
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

TEST(SM50_default_min_mult_d_srgemm, nt_t) {
  using precision = double;
  using OpClass   = cutlass::arch::OpClassSimt;
  using SmArch    = cutlass::arch::Sm50;
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

TEST(SM50_default_min_mult_d_srgemm, nn_n) {
  using precision = double;
  using OpClass   = cutlass::arch::OpClassSimt;
  using SmArch    = cutlass::arch::Sm50;
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

TEST(SM50_default_min_mult_d_srgemm, nn_t) {
  using precision = double;
  using OpClass   = cutlass::arch::OpClassSimt;
  using SmArch    = cutlass::arch::Sm50;
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

TEST(SM50_default_min_mult_s_srgemm, tt_n) {
  using precision = float;
  using OpClass   = cutlass::arch::OpClassSimt;
  using SmArch    = cutlass::arch::Sm50;
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

TEST(SM50_default_min_mult_s_srgemm, tt_t) {
  using precision = float;
  using OpClass   = cutlass::arch::OpClassSimt;
  using SmArch    = cutlass::arch::Sm50;
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

TEST(SM50_default_min_mult_s_srgemm, tn_n) {
  using precision = float;
  using OpClass   = cutlass::arch::OpClassSimt;
  using SmArch    = cutlass::arch::Sm50;
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

TEST(SM50_default_min_mult_s_srgemm, tn_t) {
  using precision = float;
  using OpClass   = cutlass::arch::OpClassSimt;
  using SmArch    = cutlass::arch::Sm50;
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

TEST(SM50_default_min_mult_s_srgemm, nt_n) {
  using precision = float;
  using OpClass   = cutlass::arch::OpClassSimt;
  using SmArch    = cutlass::arch::Sm50;
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

TEST(SM50_default_min_mult_s_srgemm, nt_t) {
  using precision = float;
  using OpClass   = cutlass::arch::OpClassSimt;
  using SmArch    = cutlass::arch::Sm50;
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

TEST(SM50_default_min_mult_s_srgemm, nn_n) {
  using precision = float;
  using OpClass   = cutlass::arch::OpClassSimt;
  using SmArch    = cutlass::arch::Sm50;
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

TEST(SM50_default_min_mult_s_srgemm, nn_t) {
  using precision = float;
  using OpClass   = cutlass::arch::OpClassSimt;
  using SmArch    = cutlass::arch::Sm50;
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

TEST(SM50_default_max_mult_d_srgemm, tt_n) {
  using precision = double;
  using OpClass   = cutlass::arch::OpClassSimt;
  using SmArch    = cutlass::arch::Sm50;
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

TEST(SM50_default_max_mult_d_srgemm, tt_t) {
  using precision = double;
  using OpClass   = cutlass::arch::OpClassSimt;
  using SmArch    = cutlass::arch::Sm50;
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

TEST(SM50_default_max_mult_d_srgemm, tn_n) {
  using precision = double;
  using OpClass   = cutlass::arch::OpClassSimt;
  using SmArch    = cutlass::arch::Sm50;
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

TEST(SM50_default_max_mult_d_srgemm, tn_t) {
  using precision = double;
  using OpClass   = cutlass::arch::OpClassSimt;
  using SmArch    = cutlass::arch::Sm50;
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

TEST(SM50_default_max_mult_d_srgemm, nt_n) {
  using precision = double;
  using OpClass   = cutlass::arch::OpClassSimt;
  using SmArch    = cutlass::arch::Sm50;
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

TEST(SM50_default_max_mult_d_srgemm, nt_t) {
  using precision = double;
  using OpClass   = cutlass::arch::OpClassSimt;
  using SmArch    = cutlass::arch::Sm50;
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

TEST(SM50_default_max_mult_d_srgemm, nn_n) {
  using precision = double;
  using OpClass   = cutlass::arch::OpClassSimt;
  using SmArch    = cutlass::arch::Sm50;
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

TEST(SM50_default_max_mult_d_srgemm, nn_t) {
  using precision = double;
  using OpClass   = cutlass::arch::OpClassSimt;
  using SmArch    = cutlass::arch::Sm50;
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

TEST(SM50_default_max_mult_s_srgemm, tt_n) {
  using precision = float;
  using OpClass   = cutlass::arch::OpClassSimt;
  using SmArch    = cutlass::arch::Sm50;
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

TEST(SM50_default_max_mult_s_srgemm, tt_t) {
  using precision = float;
  using OpClass   = cutlass::arch::OpClassSimt;
  using SmArch    = cutlass::arch::Sm50;
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

TEST(SM50_default_max_mult_s_srgemm, tn_n) {
  using precision = float;
  using OpClass   = cutlass::arch::OpClassSimt;
  using SmArch    = cutlass::arch::Sm50;
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

TEST(SM50_default_max_mult_s_srgemm, tn_t) {
  using precision = float;
  using OpClass   = cutlass::arch::OpClassSimt;
  using SmArch    = cutlass::arch::Sm50;
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

TEST(SM50_default_max_mult_s_srgemm, nt_n) {
  using precision = float;
  using OpClass   = cutlass::arch::OpClassSimt;
  using SmArch    = cutlass::arch::Sm50;
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

TEST(SM50_default_max_mult_s_srgemm, nt_t) {
  using precision = float;
  using OpClass   = cutlass::arch::OpClassSimt;
  using SmArch    = cutlass::arch::Sm50;
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

TEST(SM50_default_max_mult_s_srgemm, nn_n) {
  using precision = float;
  using OpClass   = cutlass::arch::OpClassSimt;
  using SmArch    = cutlass::arch::Sm50;
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

TEST(SM50_default_max_mult_s_srgemm, nn_t) {
  using precision = float;
  using OpClass   = cutlass::arch::OpClassSimt;
  using SmArch    = cutlass::arch::Sm50;
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

TEST(SM50_default_or_and_d_srgemm, tt_n) {
  using precision = double;
  using OpClass   = cutlass::arch::OpClassSimt;
  using SmArch    = cutlass::arch::Sm50;
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

TEST(SM50_default_or_and_d_srgemm, tt_t) {
  using precision = double;
  using OpClass   = cutlass::arch::OpClassSimt;
  using SmArch    = cutlass::arch::Sm50;
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

TEST(SM50_default_or_and_d_srgemm, tn_n) {
  using precision = double;
  using OpClass   = cutlass::arch::OpClassSimt;
  using SmArch    = cutlass::arch::Sm50;
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

TEST(SM50_default_or_and_d_srgemm, tn_t) {
  using precision = double;
  using OpClass   = cutlass::arch::OpClassSimt;
  using SmArch    = cutlass::arch::Sm50;
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

TEST(SM50_default_or_and_d_srgemm, nt_n) {
  using precision = double;
  using OpClass   = cutlass::arch::OpClassSimt;
  using SmArch    = cutlass::arch::Sm50;
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

TEST(SM50_default_or_and_d_srgemm, nt_t) {
  using precision = double;
  using OpClass   = cutlass::arch::OpClassSimt;
  using SmArch    = cutlass::arch::Sm50;
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

TEST(SM50_default_or_and_d_srgemm, nn_n) {
  using precision = double;
  using OpClass   = cutlass::arch::OpClassSimt;
  using SmArch    = cutlass::arch::Sm50;
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

TEST(SM50_default_or_and_d_srgemm, nn_t) {
  using precision = double;
  using OpClass   = cutlass::arch::OpClassSimt;
  using SmArch    = cutlass::arch::Sm50;
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

TEST(SM50_default_or_and_s_srgemm, tt_n) {
  using precision = float;
  using OpClass   = cutlass::arch::OpClassSimt;
  using SmArch    = cutlass::arch::Sm50;
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

TEST(SM50_default_or_and_s_srgemm, tt_t) {
  using precision = float;
  using OpClass   = cutlass::arch::OpClassSimt;
  using SmArch    = cutlass::arch::Sm50;
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

TEST(SM50_default_or_and_s_srgemm, tn_n) {
  using precision = float;
  using OpClass   = cutlass::arch::OpClassSimt;
  using SmArch    = cutlass::arch::Sm50;
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

TEST(SM50_default_or_and_s_srgemm, tn_t) {
  using precision = float;
  using OpClass   = cutlass::arch::OpClassSimt;
  using SmArch    = cutlass::arch::Sm50;
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

TEST(SM50_default_or_and_s_srgemm, nt_n) {
  using precision = float;
  using OpClass   = cutlass::arch::OpClassSimt;
  using SmArch    = cutlass::arch::Sm50;
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

TEST(SM50_default_or_and_s_srgemm, nt_t) {
  using precision = float;
  using OpClass   = cutlass::arch::OpClassSimt;
  using SmArch    = cutlass::arch::Sm50;
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

TEST(SM50_default_or_and_s_srgemm, nn_n) {
  using precision = float;
  using OpClass   = cutlass::arch::OpClassSimt;
  using SmArch    = cutlass::arch::Sm50;
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

TEST(SM50_default_or_and_s_srgemm, nn_t) {
  using precision = float;
  using OpClass   = cutlass::arch::OpClassSimt;
  using SmArch    = cutlass::arch::Sm50;
  using RingOp = cuasr::or_and<precision>;

  using Srgemm = cuasr::gemm::device::Srgemm<                           //
      RingOp,                                                           //
      precision, cutlass::layout::ColumnMajor,                   //
      precision, cutlass::layout::ColumnMajor,                   //
      precision, cutlass::layout::RowMajor,
      precision, OpClass, SmArch>;

  EXPECT_TRUE(cuasr::test::gemm::device::TestAllGemm<Srgemm>());
}
