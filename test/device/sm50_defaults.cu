/***************************************************************************************************
* Copyright (c) 2021, Vijay Thakkar (thakkarv@gatech.edu).
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

TEST(SM50_device_plus_multiplies_dsrgemm_tt_n, default) {
  using precision = double;
  using OpClass   = cutlass::arch::OpClassSimt;
  using SmArch    = cutlass::arch::Sm50;

  using AddOp            = cuasr::plus<precision>;
  using MultOp           = cuasr::multiplies<precision>;

  using Srgemm = cuasr::gemm::device::Srgemm<                           //
      AddOp, MultOp,                                                    //
      precision, cutlass::layout::RowMajor,                   //
      precision, cutlass::layout::RowMajor,                   //
      precision, cutlass::layout::ColumnMajor,
      precision, OpClass, SmArch>;

  EXPECT_TRUE(cuasr::test::gemm::device::TestAllGemm<Srgemm>());
}


///////////////////////////////////////////////////////////////////////////////

TEST(SM50_device_plus_multiplies_dsrgemm_tt_t, default) {
  using precision = double;
  using OpClass   = cutlass::arch::OpClassSimt;
  using SmArch    = cutlass::arch::Sm50;

  using AddOp            = cuasr::plus<precision>;
  using MultOp           = cuasr::multiplies<precision>;

  using Srgemm = cuasr::gemm::device::Srgemm<                           //
      AddOp, MultOp,                                                    //
      precision, cutlass::layout::RowMajor,                   //
      precision, cutlass::layout::RowMajor,                   //
      precision, cutlass::layout::RowMajor,
      precision, OpClass, SmArch>;

  EXPECT_TRUE(cuasr::test::gemm::device::TestAllGemm<Srgemm>());
}


///////////////////////////////////////////////////////////////////////////////

TEST(SM50_device_plus_multiplies_dsrgemm_tn_n, default) {
  using precision = double;
  using OpClass   = cutlass::arch::OpClassSimt;
  using SmArch    = cutlass::arch::Sm50;

  using AddOp            = cuasr::plus<precision>;
  using MultOp           = cuasr::multiplies<precision>;

  using Srgemm = cuasr::gemm::device::Srgemm<                           //
      AddOp, MultOp,                                                    //
      precision, cutlass::layout::RowMajor,                   //
      precision, cutlass::layout::ColumnMajor,                   //
      precision, cutlass::layout::ColumnMajor,
      precision, OpClass, SmArch>;

  EXPECT_TRUE(cuasr::test::gemm::device::TestAllGemm<Srgemm>());
}


///////////////////////////////////////////////////////////////////////////////

TEST(SM50_device_plus_multiplies_dsrgemm_tn_t, default) {
  using precision = double;
  using OpClass   = cutlass::arch::OpClassSimt;
  using SmArch    = cutlass::arch::Sm50;

  using AddOp            = cuasr::plus<precision>;
  using MultOp           = cuasr::multiplies<precision>;

  using Srgemm = cuasr::gemm::device::Srgemm<                           //
      AddOp, MultOp,                                                    //
      precision, cutlass::layout::RowMajor,                   //
      precision, cutlass::layout::ColumnMajor,                   //
      precision, cutlass::layout::RowMajor,
      precision, OpClass, SmArch>;

  EXPECT_TRUE(cuasr::test::gemm::device::TestAllGemm<Srgemm>());
}


///////////////////////////////////////////////////////////////////////////////

TEST(SM50_device_plus_multiplies_dsrgemm_nt_n, default) {
  using precision = double;
  using OpClass   = cutlass::arch::OpClassSimt;
  using SmArch    = cutlass::arch::Sm50;

  using AddOp            = cuasr::plus<precision>;
  using MultOp           = cuasr::multiplies<precision>;

  using Srgemm = cuasr::gemm::device::Srgemm<                           //
      AddOp, MultOp,                                                    //
      precision, cutlass::layout::ColumnMajor,                   //
      precision, cutlass::layout::RowMajor,                   //
      precision, cutlass::layout::ColumnMajor,
      precision, OpClass, SmArch>;

  EXPECT_TRUE(cuasr::test::gemm::device::TestAllGemm<Srgemm>());
}


///////////////////////////////////////////////////////////////////////////////

TEST(SM50_device_plus_multiplies_dsrgemm_nt_t, default) {
  using precision = double;
  using OpClass   = cutlass::arch::OpClassSimt;
  using SmArch    = cutlass::arch::Sm50;

  using AddOp            = cuasr::plus<precision>;
  using MultOp           = cuasr::multiplies<precision>;

  using Srgemm = cuasr::gemm::device::Srgemm<                           //
      AddOp, MultOp,                                                    //
      precision, cutlass::layout::ColumnMajor,                   //
      precision, cutlass::layout::RowMajor,                   //
      precision, cutlass::layout::RowMajor,
      precision, OpClass, SmArch>;

  EXPECT_TRUE(cuasr::test::gemm::device::TestAllGemm<Srgemm>());
}


///////////////////////////////////////////////////////////////////////////////

TEST(SM50_device_plus_multiplies_dsrgemm_nn_n, default) {
  using precision = double;
  using OpClass   = cutlass::arch::OpClassSimt;
  using SmArch    = cutlass::arch::Sm50;

  using AddOp            = cuasr::plus<precision>;
  using MultOp           = cuasr::multiplies<precision>;

  using Srgemm = cuasr::gemm::device::Srgemm<                           //
      AddOp, MultOp,                                                    //
      precision, cutlass::layout::ColumnMajor,                   //
      precision, cutlass::layout::ColumnMajor,                   //
      precision, cutlass::layout::ColumnMajor,
      precision, OpClass, SmArch>;

  EXPECT_TRUE(cuasr::test::gemm::device::TestAllGemm<Srgemm>());
}


///////////////////////////////////////////////////////////////////////////////

TEST(SM50_device_plus_multiplies_dsrgemm_nn_t, default) {
  using precision = double;
  using OpClass   = cutlass::arch::OpClassSimt;
  using SmArch    = cutlass::arch::Sm50;

  using AddOp            = cuasr::plus<precision>;
  using MultOp           = cuasr::multiplies<precision>;

  using Srgemm = cuasr::gemm::device::Srgemm<                           //
      AddOp, MultOp,                                                    //
      precision, cutlass::layout::ColumnMajor,                   //
      precision, cutlass::layout::ColumnMajor,                   //
      precision, cutlass::layout::RowMajor,
      precision, OpClass, SmArch>;

  EXPECT_TRUE(cuasr::test::gemm::device::TestAllGemm<Srgemm>());
}


///////////////////////////////////////////////////////////////////////////////

TEST(SM50_device_plus_multiplies_ssrgemm_tt_n, default) {
  using precision = float;
  using OpClass   = cutlass::arch::OpClassSimt;
  using SmArch    = cutlass::arch::Sm50;

  using AddOp            = cuasr::plus<precision>;
  using MultOp           = cuasr::multiplies<precision>;

  using Srgemm = cuasr::gemm::device::Srgemm<                           //
      AddOp, MultOp,                                                    //
      precision, cutlass::layout::RowMajor,                   //
      precision, cutlass::layout::RowMajor,                   //
      precision, cutlass::layout::ColumnMajor,
      precision, OpClass, SmArch>;

  EXPECT_TRUE(cuasr::test::gemm::device::TestAllGemm<Srgemm>());
}


///////////////////////////////////////////////////////////////////////////////

TEST(SM50_device_plus_multiplies_ssrgemm_tt_t, default) {
  using precision = float;
  using OpClass   = cutlass::arch::OpClassSimt;
  using SmArch    = cutlass::arch::Sm50;

  using AddOp            = cuasr::plus<precision>;
  using MultOp           = cuasr::multiplies<precision>;

  using Srgemm = cuasr::gemm::device::Srgemm<                           //
      AddOp, MultOp,                                                    //
      precision, cutlass::layout::RowMajor,                   //
      precision, cutlass::layout::RowMajor,                   //
      precision, cutlass::layout::RowMajor,
      precision, OpClass, SmArch>;

  EXPECT_TRUE(cuasr::test::gemm::device::TestAllGemm<Srgemm>());
}


///////////////////////////////////////////////////////////////////////////////

TEST(SM50_device_plus_multiplies_ssrgemm_tn_n, default) {
  using precision = float;
  using OpClass   = cutlass::arch::OpClassSimt;
  using SmArch    = cutlass::arch::Sm50;

  using AddOp            = cuasr::plus<precision>;
  using MultOp           = cuasr::multiplies<precision>;

  using Srgemm = cuasr::gemm::device::Srgemm<                           //
      AddOp, MultOp,                                                    //
      precision, cutlass::layout::RowMajor,                   //
      precision, cutlass::layout::ColumnMajor,                   //
      precision, cutlass::layout::ColumnMajor,
      precision, OpClass, SmArch>;

  EXPECT_TRUE(cuasr::test::gemm::device::TestAllGemm<Srgemm>());
}


///////////////////////////////////////////////////////////////////////////////

TEST(SM50_device_plus_multiplies_ssrgemm_tn_t, default) {
  using precision = float;
  using OpClass   = cutlass::arch::OpClassSimt;
  using SmArch    = cutlass::arch::Sm50;

  using AddOp            = cuasr::plus<precision>;
  using MultOp           = cuasr::multiplies<precision>;

  using Srgemm = cuasr::gemm::device::Srgemm<                           //
      AddOp, MultOp,                                                    //
      precision, cutlass::layout::RowMajor,                   //
      precision, cutlass::layout::ColumnMajor,                   //
      precision, cutlass::layout::RowMajor,
      precision, OpClass, SmArch>;

  EXPECT_TRUE(cuasr::test::gemm::device::TestAllGemm<Srgemm>());
}


///////////////////////////////////////////////////////////////////////////////

TEST(SM50_device_plus_multiplies_ssrgemm_nt_n, default) {
  using precision = float;
  using OpClass   = cutlass::arch::OpClassSimt;
  using SmArch    = cutlass::arch::Sm50;

  using AddOp            = cuasr::plus<precision>;
  using MultOp           = cuasr::multiplies<precision>;

  using Srgemm = cuasr::gemm::device::Srgemm<                           //
      AddOp, MultOp,                                                    //
      precision, cutlass::layout::ColumnMajor,                   //
      precision, cutlass::layout::RowMajor,                   //
      precision, cutlass::layout::ColumnMajor,
      precision, OpClass, SmArch>;

  EXPECT_TRUE(cuasr::test::gemm::device::TestAllGemm<Srgemm>());
}


///////////////////////////////////////////////////////////////////////////////

TEST(SM50_device_plus_multiplies_ssrgemm_nt_t, default) {
  using precision = float;
  using OpClass   = cutlass::arch::OpClassSimt;
  using SmArch    = cutlass::arch::Sm50;

  using AddOp            = cuasr::plus<precision>;
  using MultOp           = cuasr::multiplies<precision>;

  using Srgemm = cuasr::gemm::device::Srgemm<                           //
      AddOp, MultOp,                                                    //
      precision, cutlass::layout::ColumnMajor,                   //
      precision, cutlass::layout::RowMajor,                   //
      precision, cutlass::layout::RowMajor,
      precision, OpClass, SmArch>;

  EXPECT_TRUE(cuasr::test::gemm::device::TestAllGemm<Srgemm>());
}


///////////////////////////////////////////////////////////////////////////////

TEST(SM50_device_plus_multiplies_ssrgemm_nn_n, default) {
  using precision = float;
  using OpClass   = cutlass::arch::OpClassSimt;
  using SmArch    = cutlass::arch::Sm50;

  using AddOp            = cuasr::plus<precision>;
  using MultOp           = cuasr::multiplies<precision>;

  using Srgemm = cuasr::gemm::device::Srgemm<                           //
      AddOp, MultOp,                                                    //
      precision, cutlass::layout::ColumnMajor,                   //
      precision, cutlass::layout::ColumnMajor,                   //
      precision, cutlass::layout::ColumnMajor,
      precision, OpClass, SmArch>;

  EXPECT_TRUE(cuasr::test::gemm::device::TestAllGemm<Srgemm>());
}


///////////////////////////////////////////////////////////////////////////////

TEST(SM50_device_plus_multiplies_ssrgemm_nn_t, default) {
  using precision = float;
  using OpClass   = cutlass::arch::OpClassSimt;
  using SmArch    = cutlass::arch::Sm50;

  using AddOp            = cuasr::plus<precision>;
  using MultOp           = cuasr::multiplies<precision>;

  using Srgemm = cuasr::gemm::device::Srgemm<                           //
      AddOp, MultOp,                                                    //
      precision, cutlass::layout::ColumnMajor,                   //
      precision, cutlass::layout::ColumnMajor,                   //
      precision, cutlass::layout::RowMajor,
      precision, OpClass, SmArch>;

  EXPECT_TRUE(cuasr::test::gemm::device::TestAllGemm<Srgemm>());
}


///////////////////////////////////////////////////////////////////////////////

TEST(SM50_device_minimum_plus_dsrgemm_tt_n, default) {
  using precision = double;
  using OpClass   = cutlass::arch::OpClassSimt;
  using SmArch    = cutlass::arch::Sm50;

  using AddOp            = cuasr::minimum<precision>;
  using MultOp           = cuasr::plus<precision>;

  using Srgemm = cuasr::gemm::device::Srgemm<                           //
      AddOp, MultOp,                                                    //
      precision, cutlass::layout::RowMajor,                   //
      precision, cutlass::layout::RowMajor,                   //
      precision, cutlass::layout::ColumnMajor,
      precision, OpClass, SmArch>;

  EXPECT_TRUE(cuasr::test::gemm::device::TestAllGemm<Srgemm>());
}


///////////////////////////////////////////////////////////////////////////////

TEST(SM50_device_minimum_plus_dsrgemm_tt_t, default) {
  using precision = double;
  using OpClass   = cutlass::arch::OpClassSimt;
  using SmArch    = cutlass::arch::Sm50;

  using AddOp            = cuasr::minimum<precision>;
  using MultOp           = cuasr::plus<precision>;

  using Srgemm = cuasr::gemm::device::Srgemm<                           //
      AddOp, MultOp,                                                    //
      precision, cutlass::layout::RowMajor,                   //
      precision, cutlass::layout::RowMajor,                   //
      precision, cutlass::layout::RowMajor,
      precision, OpClass, SmArch>;

  EXPECT_TRUE(cuasr::test::gemm::device::TestAllGemm<Srgemm>());
}


///////////////////////////////////////////////////////////////////////////////

TEST(SM50_device_minimum_plus_dsrgemm_tn_n, default) {
  using precision = double;
  using OpClass   = cutlass::arch::OpClassSimt;
  using SmArch    = cutlass::arch::Sm50;

  using AddOp            = cuasr::minimum<precision>;
  using MultOp           = cuasr::plus<precision>;

  using Srgemm = cuasr::gemm::device::Srgemm<                           //
      AddOp, MultOp,                                                    //
      precision, cutlass::layout::RowMajor,                   //
      precision, cutlass::layout::ColumnMajor,                   //
      precision, cutlass::layout::ColumnMajor,
      precision, OpClass, SmArch>;

  EXPECT_TRUE(cuasr::test::gemm::device::TestAllGemm<Srgemm>());
}


///////////////////////////////////////////////////////////////////////////////

TEST(SM50_device_minimum_plus_dsrgemm_tn_t, default) {
  using precision = double;
  using OpClass   = cutlass::arch::OpClassSimt;
  using SmArch    = cutlass::arch::Sm50;

  using AddOp            = cuasr::minimum<precision>;
  using MultOp           = cuasr::plus<precision>;

  using Srgemm = cuasr::gemm::device::Srgemm<                           //
      AddOp, MultOp,                                                    //
      precision, cutlass::layout::RowMajor,                   //
      precision, cutlass::layout::ColumnMajor,                   //
      precision, cutlass::layout::RowMajor,
      precision, OpClass, SmArch>;

  EXPECT_TRUE(cuasr::test::gemm::device::TestAllGemm<Srgemm>());
}


///////////////////////////////////////////////////////////////////////////////

TEST(SM50_device_minimum_plus_dsrgemm_nt_n, default) {
  using precision = double;
  using OpClass   = cutlass::arch::OpClassSimt;
  using SmArch    = cutlass::arch::Sm50;

  using AddOp            = cuasr::minimum<precision>;
  using MultOp           = cuasr::plus<precision>;

  using Srgemm = cuasr::gemm::device::Srgemm<                           //
      AddOp, MultOp,                                                    //
      precision, cutlass::layout::ColumnMajor,                   //
      precision, cutlass::layout::RowMajor,                   //
      precision, cutlass::layout::ColumnMajor,
      precision, OpClass, SmArch>;

  EXPECT_TRUE(cuasr::test::gemm::device::TestAllGemm<Srgemm>());
}


///////////////////////////////////////////////////////////////////////////////

TEST(SM50_device_minimum_plus_dsrgemm_nt_t, default) {
  using precision = double;
  using OpClass   = cutlass::arch::OpClassSimt;
  using SmArch    = cutlass::arch::Sm50;

  using AddOp            = cuasr::minimum<precision>;
  using MultOp           = cuasr::plus<precision>;

  using Srgemm = cuasr::gemm::device::Srgemm<                           //
      AddOp, MultOp,                                                    //
      precision, cutlass::layout::ColumnMajor,                   //
      precision, cutlass::layout::RowMajor,                   //
      precision, cutlass::layout::RowMajor,
      precision, OpClass, SmArch>;

  EXPECT_TRUE(cuasr::test::gemm::device::TestAllGemm<Srgemm>());
}


///////////////////////////////////////////////////////////////////////////////

TEST(SM50_device_minimum_plus_dsrgemm_nn_n, default) {
  using precision = double;
  using OpClass   = cutlass::arch::OpClassSimt;
  using SmArch    = cutlass::arch::Sm50;

  using AddOp            = cuasr::minimum<precision>;
  using MultOp           = cuasr::plus<precision>;

  using Srgemm = cuasr::gemm::device::Srgemm<                           //
      AddOp, MultOp,                                                    //
      precision, cutlass::layout::ColumnMajor,                   //
      precision, cutlass::layout::ColumnMajor,                   //
      precision, cutlass::layout::ColumnMajor,
      precision, OpClass, SmArch>;

  EXPECT_TRUE(cuasr::test::gemm::device::TestAllGemm<Srgemm>());
}


///////////////////////////////////////////////////////////////////////////////

TEST(SM50_device_minimum_plus_dsrgemm_nn_t, default) {
  using precision = double;
  using OpClass   = cutlass::arch::OpClassSimt;
  using SmArch    = cutlass::arch::Sm50;

  using AddOp            = cuasr::minimum<precision>;
  using MultOp           = cuasr::plus<precision>;

  using Srgemm = cuasr::gemm::device::Srgemm<                           //
      AddOp, MultOp,                                                    //
      precision, cutlass::layout::ColumnMajor,                   //
      precision, cutlass::layout::ColumnMajor,                   //
      precision, cutlass::layout::RowMajor,
      precision, OpClass, SmArch>;

  EXPECT_TRUE(cuasr::test::gemm::device::TestAllGemm<Srgemm>());
}


///////////////////////////////////////////////////////////////////////////////

TEST(SM50_device_minimum_plus_ssrgemm_tt_n, default) {
  using precision = float;
  using OpClass   = cutlass::arch::OpClassSimt;
  using SmArch    = cutlass::arch::Sm50;

  using AddOp            = cuasr::minimum<precision>;
  using MultOp           = cuasr::plus<precision>;

  using Srgemm = cuasr::gemm::device::Srgemm<                           //
      AddOp, MultOp,                                                    //
      precision, cutlass::layout::RowMajor,                   //
      precision, cutlass::layout::RowMajor,                   //
      precision, cutlass::layout::ColumnMajor,
      precision, OpClass, SmArch>;

  EXPECT_TRUE(cuasr::test::gemm::device::TestAllGemm<Srgemm>());
}


///////////////////////////////////////////////////////////////////////////////

TEST(SM50_device_minimum_plus_ssrgemm_tt_t, default) {
  using precision = float;
  using OpClass   = cutlass::arch::OpClassSimt;
  using SmArch    = cutlass::arch::Sm50;

  using AddOp            = cuasr::minimum<precision>;
  using MultOp           = cuasr::plus<precision>;

  using Srgemm = cuasr::gemm::device::Srgemm<                           //
      AddOp, MultOp,                                                    //
      precision, cutlass::layout::RowMajor,                   //
      precision, cutlass::layout::RowMajor,                   //
      precision, cutlass::layout::RowMajor,
      precision, OpClass, SmArch>;

  EXPECT_TRUE(cuasr::test::gemm::device::TestAllGemm<Srgemm>());
}


///////////////////////////////////////////////////////////////////////////////

TEST(SM50_device_minimum_plus_ssrgemm_tn_n, default) {
  using precision = float;
  using OpClass   = cutlass::arch::OpClassSimt;
  using SmArch    = cutlass::arch::Sm50;

  using AddOp            = cuasr::minimum<precision>;
  using MultOp           = cuasr::plus<precision>;

  using Srgemm = cuasr::gemm::device::Srgemm<                           //
      AddOp, MultOp,                                                    //
      precision, cutlass::layout::RowMajor,                   //
      precision, cutlass::layout::ColumnMajor,                   //
      precision, cutlass::layout::ColumnMajor,
      precision, OpClass, SmArch>;

  EXPECT_TRUE(cuasr::test::gemm::device::TestAllGemm<Srgemm>());
}


///////////////////////////////////////////////////////////////////////////////

TEST(SM50_device_minimum_plus_ssrgemm_tn_t, default) {
  using precision = float;
  using OpClass   = cutlass::arch::OpClassSimt;
  using SmArch    = cutlass::arch::Sm50;

  using AddOp            = cuasr::minimum<precision>;
  using MultOp           = cuasr::plus<precision>;

  using Srgemm = cuasr::gemm::device::Srgemm<                           //
      AddOp, MultOp,                                                    //
      precision, cutlass::layout::RowMajor,                   //
      precision, cutlass::layout::ColumnMajor,                   //
      precision, cutlass::layout::RowMajor,
      precision, OpClass, SmArch>;

  EXPECT_TRUE(cuasr::test::gemm::device::TestAllGemm<Srgemm>());
}


///////////////////////////////////////////////////////////////////////////////

TEST(SM50_device_minimum_plus_ssrgemm_nt_n, default) {
  using precision = float;
  using OpClass   = cutlass::arch::OpClassSimt;
  using SmArch    = cutlass::arch::Sm50;

  using AddOp            = cuasr::minimum<precision>;
  using MultOp           = cuasr::plus<precision>;

  using Srgemm = cuasr::gemm::device::Srgemm<                           //
      AddOp, MultOp,                                                    //
      precision, cutlass::layout::ColumnMajor,                   //
      precision, cutlass::layout::RowMajor,                   //
      precision, cutlass::layout::ColumnMajor,
      precision, OpClass, SmArch>;

  EXPECT_TRUE(cuasr::test::gemm::device::TestAllGemm<Srgemm>());
}


///////////////////////////////////////////////////////////////////////////////

TEST(SM50_device_minimum_plus_ssrgemm_nt_t, default) {
  using precision = float;
  using OpClass   = cutlass::arch::OpClassSimt;
  using SmArch    = cutlass::arch::Sm50;

  using AddOp            = cuasr::minimum<precision>;
  using MultOp           = cuasr::plus<precision>;

  using Srgemm = cuasr::gemm::device::Srgemm<                           //
      AddOp, MultOp,                                                    //
      precision, cutlass::layout::ColumnMajor,                   //
      precision, cutlass::layout::RowMajor,                   //
      precision, cutlass::layout::RowMajor,
      precision, OpClass, SmArch>;

  EXPECT_TRUE(cuasr::test::gemm::device::TestAllGemm<Srgemm>());
}


///////////////////////////////////////////////////////////////////////////////

TEST(SM50_device_minimum_plus_ssrgemm_nn_n, default) {
  using precision = float;
  using OpClass   = cutlass::arch::OpClassSimt;
  using SmArch    = cutlass::arch::Sm50;

  using AddOp            = cuasr::minimum<precision>;
  using MultOp           = cuasr::plus<precision>;

  using Srgemm = cuasr::gemm::device::Srgemm<                           //
      AddOp, MultOp,                                                    //
      precision, cutlass::layout::ColumnMajor,                   //
      precision, cutlass::layout::ColumnMajor,                   //
      precision, cutlass::layout::ColumnMajor,
      precision, OpClass, SmArch>;

  EXPECT_TRUE(cuasr::test::gemm::device::TestAllGemm<Srgemm>());
}


///////////////////////////////////////////////////////////////////////////////

TEST(SM50_device_minimum_plus_ssrgemm_nn_t, default) {
  using precision = float;
  using OpClass   = cutlass::arch::OpClassSimt;
  using SmArch    = cutlass::arch::Sm50;

  using AddOp            = cuasr::minimum<precision>;
  using MultOp           = cuasr::plus<precision>;

  using Srgemm = cuasr::gemm::device::Srgemm<                           //
      AddOp, MultOp,                                                    //
      precision, cutlass::layout::ColumnMajor,                   //
      precision, cutlass::layout::ColumnMajor,                   //
      precision, cutlass::layout::RowMajor,
      precision, OpClass, SmArch>;

  EXPECT_TRUE(cuasr::test::gemm::device::TestAllGemm<Srgemm>());
}


///////////////////////////////////////////////////////////////////////////////

TEST(SM50_device_maximum_plus_dsrgemm_tt_n, default) {
  using precision = double;
  using OpClass   = cutlass::arch::OpClassSimt;
  using SmArch    = cutlass::arch::Sm50;

  using AddOp            = cuasr::maximum<precision>;
  using MultOp           = cuasr::plus<precision>;

  using Srgemm = cuasr::gemm::device::Srgemm<                           //
      AddOp, MultOp,                                                    //
      precision, cutlass::layout::RowMajor,                   //
      precision, cutlass::layout::RowMajor,                   //
      precision, cutlass::layout::ColumnMajor,
      precision, OpClass, SmArch>;

  EXPECT_TRUE(cuasr::test::gemm::device::TestAllGemm<Srgemm>());
}


///////////////////////////////////////////////////////////////////////////////

TEST(SM50_device_maximum_plus_dsrgemm_tt_t, default) {
  using precision = double;
  using OpClass   = cutlass::arch::OpClassSimt;
  using SmArch    = cutlass::arch::Sm50;

  using AddOp            = cuasr::maximum<precision>;
  using MultOp           = cuasr::plus<precision>;

  using Srgemm = cuasr::gemm::device::Srgemm<                           //
      AddOp, MultOp,                                                    //
      precision, cutlass::layout::RowMajor,                   //
      precision, cutlass::layout::RowMajor,                   //
      precision, cutlass::layout::RowMajor,
      precision, OpClass, SmArch>;

  EXPECT_TRUE(cuasr::test::gemm::device::TestAllGemm<Srgemm>());
}


///////////////////////////////////////////////////////////////////////////////

TEST(SM50_device_maximum_plus_dsrgemm_tn_n, default) {
  using precision = double;
  using OpClass   = cutlass::arch::OpClassSimt;
  using SmArch    = cutlass::arch::Sm50;

  using AddOp            = cuasr::maximum<precision>;
  using MultOp           = cuasr::plus<precision>;

  using Srgemm = cuasr::gemm::device::Srgemm<                           //
      AddOp, MultOp,                                                    //
      precision, cutlass::layout::RowMajor,                   //
      precision, cutlass::layout::ColumnMajor,                   //
      precision, cutlass::layout::ColumnMajor,
      precision, OpClass, SmArch>;

  EXPECT_TRUE(cuasr::test::gemm::device::TestAllGemm<Srgemm>());
}


///////////////////////////////////////////////////////////////////////////////

TEST(SM50_device_maximum_plus_dsrgemm_tn_t, default) {
  using precision = double;
  using OpClass   = cutlass::arch::OpClassSimt;
  using SmArch    = cutlass::arch::Sm50;

  using AddOp            = cuasr::maximum<precision>;
  using MultOp           = cuasr::plus<precision>;

  using Srgemm = cuasr::gemm::device::Srgemm<                           //
      AddOp, MultOp,                                                    //
      precision, cutlass::layout::RowMajor,                   //
      precision, cutlass::layout::ColumnMajor,                   //
      precision, cutlass::layout::RowMajor,
      precision, OpClass, SmArch>;

  EXPECT_TRUE(cuasr::test::gemm::device::TestAllGemm<Srgemm>());
}


///////////////////////////////////////////////////////////////////////////////

TEST(SM50_device_maximum_plus_dsrgemm_nt_n, default) {
  using precision = double;
  using OpClass   = cutlass::arch::OpClassSimt;
  using SmArch    = cutlass::arch::Sm50;

  using AddOp            = cuasr::maximum<precision>;
  using MultOp           = cuasr::plus<precision>;

  using Srgemm = cuasr::gemm::device::Srgemm<                           //
      AddOp, MultOp,                                                    //
      precision, cutlass::layout::ColumnMajor,                   //
      precision, cutlass::layout::RowMajor,                   //
      precision, cutlass::layout::ColumnMajor,
      precision, OpClass, SmArch>;

  EXPECT_TRUE(cuasr::test::gemm::device::TestAllGemm<Srgemm>());
}


///////////////////////////////////////////////////////////////////////////////

TEST(SM50_device_maximum_plus_dsrgemm_nt_t, default) {
  using precision = double;
  using OpClass   = cutlass::arch::OpClassSimt;
  using SmArch    = cutlass::arch::Sm50;

  using AddOp            = cuasr::maximum<precision>;
  using MultOp           = cuasr::plus<precision>;

  using Srgemm = cuasr::gemm::device::Srgemm<                           //
      AddOp, MultOp,                                                    //
      precision, cutlass::layout::ColumnMajor,                   //
      precision, cutlass::layout::RowMajor,                   //
      precision, cutlass::layout::RowMajor,
      precision, OpClass, SmArch>;

  EXPECT_TRUE(cuasr::test::gemm::device::TestAllGemm<Srgemm>());
}


///////////////////////////////////////////////////////////////////////////////

TEST(SM50_device_maximum_plus_dsrgemm_nn_n, default) {
  using precision = double;
  using OpClass   = cutlass::arch::OpClassSimt;
  using SmArch    = cutlass::arch::Sm50;

  using AddOp            = cuasr::maximum<precision>;
  using MultOp           = cuasr::plus<precision>;

  using Srgemm = cuasr::gemm::device::Srgemm<                           //
      AddOp, MultOp,                                                    //
      precision, cutlass::layout::ColumnMajor,                   //
      precision, cutlass::layout::ColumnMajor,                   //
      precision, cutlass::layout::ColumnMajor,
      precision, OpClass, SmArch>;

  EXPECT_TRUE(cuasr::test::gemm::device::TestAllGemm<Srgemm>());
}


///////////////////////////////////////////////////////////////////////////////

TEST(SM50_device_maximum_plus_dsrgemm_nn_t, default) {
  using precision = double;
  using OpClass   = cutlass::arch::OpClassSimt;
  using SmArch    = cutlass::arch::Sm50;

  using AddOp            = cuasr::maximum<precision>;
  using MultOp           = cuasr::plus<precision>;

  using Srgemm = cuasr::gemm::device::Srgemm<                           //
      AddOp, MultOp,                                                    //
      precision, cutlass::layout::ColumnMajor,                   //
      precision, cutlass::layout::ColumnMajor,                   //
      precision, cutlass::layout::RowMajor,
      precision, OpClass, SmArch>;

  EXPECT_TRUE(cuasr::test::gemm::device::TestAllGemm<Srgemm>());
}


///////////////////////////////////////////////////////////////////////////////

TEST(SM50_device_maximum_plus_ssrgemm_tt_n, default) {
  using precision = float;
  using OpClass   = cutlass::arch::OpClassSimt;
  using SmArch    = cutlass::arch::Sm50;

  using AddOp            = cuasr::maximum<precision>;
  using MultOp           = cuasr::plus<precision>;

  using Srgemm = cuasr::gemm::device::Srgemm<                           //
      AddOp, MultOp,                                                    //
      precision, cutlass::layout::RowMajor,                   //
      precision, cutlass::layout::RowMajor,                   //
      precision, cutlass::layout::ColumnMajor,
      precision, OpClass, SmArch>;

  EXPECT_TRUE(cuasr::test::gemm::device::TestAllGemm<Srgemm>());
}


///////////////////////////////////////////////////////////////////////////////

TEST(SM50_device_maximum_plus_ssrgemm_tt_t, default) {
  using precision = float;
  using OpClass   = cutlass::arch::OpClassSimt;
  using SmArch    = cutlass::arch::Sm50;

  using AddOp            = cuasr::maximum<precision>;
  using MultOp           = cuasr::plus<precision>;

  using Srgemm = cuasr::gemm::device::Srgemm<                           //
      AddOp, MultOp,                                                    //
      precision, cutlass::layout::RowMajor,                   //
      precision, cutlass::layout::RowMajor,                   //
      precision, cutlass::layout::RowMajor,
      precision, OpClass, SmArch>;

  EXPECT_TRUE(cuasr::test::gemm::device::TestAllGemm<Srgemm>());
}


///////////////////////////////////////////////////////////////////////////////

TEST(SM50_device_maximum_plus_ssrgemm_tn_n, default) {
  using precision = float;
  using OpClass   = cutlass::arch::OpClassSimt;
  using SmArch    = cutlass::arch::Sm50;

  using AddOp            = cuasr::maximum<precision>;
  using MultOp           = cuasr::plus<precision>;

  using Srgemm = cuasr::gemm::device::Srgemm<                           //
      AddOp, MultOp,                                                    //
      precision, cutlass::layout::RowMajor,                   //
      precision, cutlass::layout::ColumnMajor,                   //
      precision, cutlass::layout::ColumnMajor,
      precision, OpClass, SmArch>;

  EXPECT_TRUE(cuasr::test::gemm::device::TestAllGemm<Srgemm>());
}


///////////////////////////////////////////////////////////////////////////////

TEST(SM50_device_maximum_plus_ssrgemm_tn_t, default) {
  using precision = float;
  using OpClass   = cutlass::arch::OpClassSimt;
  using SmArch    = cutlass::arch::Sm50;

  using AddOp            = cuasr::maximum<precision>;
  using MultOp           = cuasr::plus<precision>;

  using Srgemm = cuasr::gemm::device::Srgemm<                           //
      AddOp, MultOp,                                                    //
      precision, cutlass::layout::RowMajor,                   //
      precision, cutlass::layout::ColumnMajor,                   //
      precision, cutlass::layout::RowMajor,
      precision, OpClass, SmArch>;

  EXPECT_TRUE(cuasr::test::gemm::device::TestAllGemm<Srgemm>());
}


///////////////////////////////////////////////////////////////////////////////

TEST(SM50_device_maximum_plus_ssrgemm_nt_n, default) {
  using precision = float;
  using OpClass   = cutlass::arch::OpClassSimt;
  using SmArch    = cutlass::arch::Sm50;

  using AddOp            = cuasr::maximum<precision>;
  using MultOp           = cuasr::plus<precision>;

  using Srgemm = cuasr::gemm::device::Srgemm<                           //
      AddOp, MultOp,                                                    //
      precision, cutlass::layout::ColumnMajor,                   //
      precision, cutlass::layout::RowMajor,                   //
      precision, cutlass::layout::ColumnMajor,
      precision, OpClass, SmArch>;

  EXPECT_TRUE(cuasr::test::gemm::device::TestAllGemm<Srgemm>());
}


///////////////////////////////////////////////////////////////////////////////

TEST(SM50_device_maximum_plus_ssrgemm_nt_t, default) {
  using precision = float;
  using OpClass   = cutlass::arch::OpClassSimt;
  using SmArch    = cutlass::arch::Sm50;

  using AddOp            = cuasr::maximum<precision>;
  using MultOp           = cuasr::plus<precision>;

  using Srgemm = cuasr::gemm::device::Srgemm<                           //
      AddOp, MultOp,                                                    //
      precision, cutlass::layout::ColumnMajor,                   //
      precision, cutlass::layout::RowMajor,                   //
      precision, cutlass::layout::RowMajor,
      precision, OpClass, SmArch>;

  EXPECT_TRUE(cuasr::test::gemm::device::TestAllGemm<Srgemm>());
}


///////////////////////////////////////////////////////////////////////////////

TEST(SM50_device_maximum_plus_ssrgemm_nn_n, default) {
  using precision = float;
  using OpClass   = cutlass::arch::OpClassSimt;
  using SmArch    = cutlass::arch::Sm50;

  using AddOp            = cuasr::maximum<precision>;
  using MultOp           = cuasr::plus<precision>;

  using Srgemm = cuasr::gemm::device::Srgemm<                           //
      AddOp, MultOp,                                                    //
      precision, cutlass::layout::ColumnMajor,                   //
      precision, cutlass::layout::ColumnMajor,                   //
      precision, cutlass::layout::ColumnMajor,
      precision, OpClass, SmArch>;

  EXPECT_TRUE(cuasr::test::gemm::device::TestAllGemm<Srgemm>());
}


///////////////////////////////////////////////////////////////////////////////

TEST(SM50_device_maximum_plus_ssrgemm_nn_t, default) {
  using precision = float;
  using OpClass   = cutlass::arch::OpClassSimt;
  using SmArch    = cutlass::arch::Sm50;

  using AddOp            = cuasr::maximum<precision>;
  using MultOp           = cuasr::plus<precision>;

  using Srgemm = cuasr::gemm::device::Srgemm<                           //
      AddOp, MultOp,                                                    //
      precision, cutlass::layout::ColumnMajor,                   //
      precision, cutlass::layout::ColumnMajor,                   //
      precision, cutlass::layout::RowMajor,
      precision, OpClass, SmArch>;

  EXPECT_TRUE(cuasr::test::gemm::device::TestAllGemm<Srgemm>());
}


///////////////////////////////////////////////////////////////////////////////

TEST(SM50_device_minimum_maximum_dsrgemm_tt_n, default) {
  using precision = double;
  using OpClass   = cutlass::arch::OpClassSimt;
  using SmArch    = cutlass::arch::Sm50;

  using AddOp            = cuasr::minimum<precision>;
  using MultOp           = cuasr::maximum<precision>;

  using Srgemm = cuasr::gemm::device::Srgemm<                           //
      AddOp, MultOp,                                                    //
      precision, cutlass::layout::RowMajor,                   //
      precision, cutlass::layout::RowMajor,                   //
      precision, cutlass::layout::ColumnMajor,
      precision, OpClass, SmArch>;

  EXPECT_TRUE(cuasr::test::gemm::device::TestAllGemm<Srgemm>());
}


///////////////////////////////////////////////////////////////////////////////

TEST(SM50_device_minimum_maximum_dsrgemm_tt_t, default) {
  using precision = double;
  using OpClass   = cutlass::arch::OpClassSimt;
  using SmArch    = cutlass::arch::Sm50;

  using AddOp            = cuasr::minimum<precision>;
  using MultOp           = cuasr::maximum<precision>;

  using Srgemm = cuasr::gemm::device::Srgemm<                           //
      AddOp, MultOp,                                                    //
      precision, cutlass::layout::RowMajor,                   //
      precision, cutlass::layout::RowMajor,                   //
      precision, cutlass::layout::RowMajor,
      precision, OpClass, SmArch>;

  EXPECT_TRUE(cuasr::test::gemm::device::TestAllGemm<Srgemm>());
}


///////////////////////////////////////////////////////////////////////////////

TEST(SM50_device_minimum_maximum_dsrgemm_tn_n, default) {
  using precision = double;
  using OpClass   = cutlass::arch::OpClassSimt;
  using SmArch    = cutlass::arch::Sm50;

  using AddOp            = cuasr::minimum<precision>;
  using MultOp           = cuasr::maximum<precision>;

  using Srgemm = cuasr::gemm::device::Srgemm<                           //
      AddOp, MultOp,                                                    //
      precision, cutlass::layout::RowMajor,                   //
      precision, cutlass::layout::ColumnMajor,                   //
      precision, cutlass::layout::ColumnMajor,
      precision, OpClass, SmArch>;

  EXPECT_TRUE(cuasr::test::gemm::device::TestAllGemm<Srgemm>());
}


///////////////////////////////////////////////////////////////////////////////

TEST(SM50_device_minimum_maximum_dsrgemm_tn_t, default) {
  using precision = double;
  using OpClass   = cutlass::arch::OpClassSimt;
  using SmArch    = cutlass::arch::Sm50;

  using AddOp            = cuasr::minimum<precision>;
  using MultOp           = cuasr::maximum<precision>;

  using Srgemm = cuasr::gemm::device::Srgemm<                           //
      AddOp, MultOp,                                                    //
      precision, cutlass::layout::RowMajor,                   //
      precision, cutlass::layout::ColumnMajor,                   //
      precision, cutlass::layout::RowMajor,
      precision, OpClass, SmArch>;

  EXPECT_TRUE(cuasr::test::gemm::device::TestAllGemm<Srgemm>());
}


///////////////////////////////////////////////////////////////////////////////

TEST(SM50_device_minimum_maximum_dsrgemm_nt_n, default) {
  using precision = double;
  using OpClass   = cutlass::arch::OpClassSimt;
  using SmArch    = cutlass::arch::Sm50;

  using AddOp            = cuasr::minimum<precision>;
  using MultOp           = cuasr::maximum<precision>;

  using Srgemm = cuasr::gemm::device::Srgemm<                           //
      AddOp, MultOp,                                                    //
      precision, cutlass::layout::ColumnMajor,                   //
      precision, cutlass::layout::RowMajor,                   //
      precision, cutlass::layout::ColumnMajor,
      precision, OpClass, SmArch>;

  EXPECT_TRUE(cuasr::test::gemm::device::TestAllGemm<Srgemm>());
}


///////////////////////////////////////////////////////////////////////////////

TEST(SM50_device_minimum_maximum_dsrgemm_nt_t, default) {
  using precision = double;
  using OpClass   = cutlass::arch::OpClassSimt;
  using SmArch    = cutlass::arch::Sm50;

  using AddOp            = cuasr::minimum<precision>;
  using MultOp           = cuasr::maximum<precision>;

  using Srgemm = cuasr::gemm::device::Srgemm<                           //
      AddOp, MultOp,                                                    //
      precision, cutlass::layout::ColumnMajor,                   //
      precision, cutlass::layout::RowMajor,                   //
      precision, cutlass::layout::RowMajor,
      precision, OpClass, SmArch>;

  EXPECT_TRUE(cuasr::test::gemm::device::TestAllGemm<Srgemm>());
}


///////////////////////////////////////////////////////////////////////////////

TEST(SM50_device_minimum_maximum_dsrgemm_nn_n, default) {
  using precision = double;
  using OpClass   = cutlass::arch::OpClassSimt;
  using SmArch    = cutlass::arch::Sm50;

  using AddOp            = cuasr::minimum<precision>;
  using MultOp           = cuasr::maximum<precision>;

  using Srgemm = cuasr::gemm::device::Srgemm<                           //
      AddOp, MultOp,                                                    //
      precision, cutlass::layout::ColumnMajor,                   //
      precision, cutlass::layout::ColumnMajor,                   //
      precision, cutlass::layout::ColumnMajor,
      precision, OpClass, SmArch>;

  EXPECT_TRUE(cuasr::test::gemm::device::TestAllGemm<Srgemm>());
}


///////////////////////////////////////////////////////////////////////////////

TEST(SM50_device_minimum_maximum_dsrgemm_nn_t, default) {
  using precision = double;
  using OpClass   = cutlass::arch::OpClassSimt;
  using SmArch    = cutlass::arch::Sm50;

  using AddOp            = cuasr::minimum<precision>;
  using MultOp           = cuasr::maximum<precision>;

  using Srgemm = cuasr::gemm::device::Srgemm<                           //
      AddOp, MultOp,                                                    //
      precision, cutlass::layout::ColumnMajor,                   //
      precision, cutlass::layout::ColumnMajor,                   //
      precision, cutlass::layout::RowMajor,
      precision, OpClass, SmArch>;

  EXPECT_TRUE(cuasr::test::gemm::device::TestAllGemm<Srgemm>());
}


///////////////////////////////////////////////////////////////////////////////

TEST(SM50_device_minimum_maximum_ssrgemm_tt_n, default) {
  using precision = float;
  using OpClass   = cutlass::arch::OpClassSimt;
  using SmArch    = cutlass::arch::Sm50;

  using AddOp            = cuasr::minimum<precision>;
  using MultOp           = cuasr::maximum<precision>;

  using Srgemm = cuasr::gemm::device::Srgemm<                           //
      AddOp, MultOp,                                                    //
      precision, cutlass::layout::RowMajor,                   //
      precision, cutlass::layout::RowMajor,                   //
      precision, cutlass::layout::ColumnMajor,
      precision, OpClass, SmArch>;

  EXPECT_TRUE(cuasr::test::gemm::device::TestAllGemm<Srgemm>());
}


///////////////////////////////////////////////////////////////////////////////

TEST(SM50_device_minimum_maximum_ssrgemm_tt_t, default) {
  using precision = float;
  using OpClass   = cutlass::arch::OpClassSimt;
  using SmArch    = cutlass::arch::Sm50;

  using AddOp            = cuasr::minimum<precision>;
  using MultOp           = cuasr::maximum<precision>;

  using Srgemm = cuasr::gemm::device::Srgemm<                           //
      AddOp, MultOp,                                                    //
      precision, cutlass::layout::RowMajor,                   //
      precision, cutlass::layout::RowMajor,                   //
      precision, cutlass::layout::RowMajor,
      precision, OpClass, SmArch>;

  EXPECT_TRUE(cuasr::test::gemm::device::TestAllGemm<Srgemm>());
}


///////////////////////////////////////////////////////////////////////////////

TEST(SM50_device_minimum_maximum_ssrgemm_tn_n, default) {
  using precision = float;
  using OpClass   = cutlass::arch::OpClassSimt;
  using SmArch    = cutlass::arch::Sm50;

  using AddOp            = cuasr::minimum<precision>;
  using MultOp           = cuasr::maximum<precision>;

  using Srgemm = cuasr::gemm::device::Srgemm<                           //
      AddOp, MultOp,                                                    //
      precision, cutlass::layout::RowMajor,                   //
      precision, cutlass::layout::ColumnMajor,                   //
      precision, cutlass::layout::ColumnMajor,
      precision, OpClass, SmArch>;

  EXPECT_TRUE(cuasr::test::gemm::device::TestAllGemm<Srgemm>());
}


///////////////////////////////////////////////////////////////////////////////

TEST(SM50_device_minimum_maximum_ssrgemm_tn_t, default) {
  using precision = float;
  using OpClass   = cutlass::arch::OpClassSimt;
  using SmArch    = cutlass::arch::Sm50;

  using AddOp            = cuasr::minimum<precision>;
  using MultOp           = cuasr::maximum<precision>;

  using Srgemm = cuasr::gemm::device::Srgemm<                           //
      AddOp, MultOp,                                                    //
      precision, cutlass::layout::RowMajor,                   //
      precision, cutlass::layout::ColumnMajor,                   //
      precision, cutlass::layout::RowMajor,
      precision, OpClass, SmArch>;

  EXPECT_TRUE(cuasr::test::gemm::device::TestAllGemm<Srgemm>());
}


///////////////////////////////////////////////////////////////////////////////

TEST(SM50_device_minimum_maximum_ssrgemm_nt_n, default) {
  using precision = float;
  using OpClass   = cutlass::arch::OpClassSimt;
  using SmArch    = cutlass::arch::Sm50;

  using AddOp            = cuasr::minimum<precision>;
  using MultOp           = cuasr::maximum<precision>;

  using Srgemm = cuasr::gemm::device::Srgemm<                           //
      AddOp, MultOp,                                                    //
      precision, cutlass::layout::ColumnMajor,                   //
      precision, cutlass::layout::RowMajor,                   //
      precision, cutlass::layout::ColumnMajor,
      precision, OpClass, SmArch>;

  EXPECT_TRUE(cuasr::test::gemm::device::TestAllGemm<Srgemm>());
}


///////////////////////////////////////////////////////////////////////////////

TEST(SM50_device_minimum_maximum_ssrgemm_nt_t, default) {
  using precision = float;
  using OpClass   = cutlass::arch::OpClassSimt;
  using SmArch    = cutlass::arch::Sm50;

  using AddOp            = cuasr::minimum<precision>;
  using MultOp           = cuasr::maximum<precision>;

  using Srgemm = cuasr::gemm::device::Srgemm<                           //
      AddOp, MultOp,                                                    //
      precision, cutlass::layout::ColumnMajor,                   //
      precision, cutlass::layout::RowMajor,                   //
      precision, cutlass::layout::RowMajor,
      precision, OpClass, SmArch>;

  EXPECT_TRUE(cuasr::test::gemm::device::TestAllGemm<Srgemm>());
}


///////////////////////////////////////////////////////////////////////////////

TEST(SM50_device_minimum_maximum_ssrgemm_nn_n, default) {
  using precision = float;
  using OpClass   = cutlass::arch::OpClassSimt;
  using SmArch    = cutlass::arch::Sm50;

  using AddOp            = cuasr::minimum<precision>;
  using MultOp           = cuasr::maximum<precision>;

  using Srgemm = cuasr::gemm::device::Srgemm<                           //
      AddOp, MultOp,                                                    //
      precision, cutlass::layout::ColumnMajor,                   //
      precision, cutlass::layout::ColumnMajor,                   //
      precision, cutlass::layout::ColumnMajor,
      precision, OpClass, SmArch>;

  EXPECT_TRUE(cuasr::test::gemm::device::TestAllGemm<Srgemm>());
}


///////////////////////////////////////////////////////////////////////////////

TEST(SM50_device_minimum_maximum_ssrgemm_nn_t, default) {
  using precision = float;
  using OpClass   = cutlass::arch::OpClassSimt;
  using SmArch    = cutlass::arch::Sm50;

  using AddOp            = cuasr::minimum<precision>;
  using MultOp           = cuasr::maximum<precision>;

  using Srgemm = cuasr::gemm::device::Srgemm<                           //
      AddOp, MultOp,                                                    //
      precision, cutlass::layout::ColumnMajor,                   //
      precision, cutlass::layout::ColumnMajor,                   //
      precision, cutlass::layout::RowMajor,
      precision, OpClass, SmArch>;

  EXPECT_TRUE(cuasr::test::gemm::device::TestAllGemm<Srgemm>());
}


///////////////////////////////////////////////////////////////////////////////

TEST(SM50_device_maximum_minimum_dsrgemm_tt_n, default) {
  using precision = double;
  using OpClass   = cutlass::arch::OpClassSimt;
  using SmArch    = cutlass::arch::Sm50;

  using AddOp            = cuasr::maximum<precision>;
  using MultOp           = cuasr::minimum<precision>;

  using Srgemm = cuasr::gemm::device::Srgemm<                           //
      AddOp, MultOp,                                                    //
      precision, cutlass::layout::RowMajor,                   //
      precision, cutlass::layout::RowMajor,                   //
      precision, cutlass::layout::ColumnMajor,
      precision, OpClass, SmArch>;

  EXPECT_TRUE(cuasr::test::gemm::device::TestAllGemm<Srgemm>());
}


///////////////////////////////////////////////////////////////////////////////

TEST(SM50_device_maximum_minimum_dsrgemm_tt_t, default) {
  using precision = double;
  using OpClass   = cutlass::arch::OpClassSimt;
  using SmArch    = cutlass::arch::Sm50;

  using AddOp            = cuasr::maximum<precision>;
  using MultOp           = cuasr::minimum<precision>;

  using Srgemm = cuasr::gemm::device::Srgemm<                           //
      AddOp, MultOp,                                                    //
      precision, cutlass::layout::RowMajor,                   //
      precision, cutlass::layout::RowMajor,                   //
      precision, cutlass::layout::RowMajor,
      precision, OpClass, SmArch>;

  EXPECT_TRUE(cuasr::test::gemm::device::TestAllGemm<Srgemm>());
}


///////////////////////////////////////////////////////////////////////////////

TEST(SM50_device_maximum_minimum_dsrgemm_tn_n, default) {
  using precision = double;
  using OpClass   = cutlass::arch::OpClassSimt;
  using SmArch    = cutlass::arch::Sm50;

  using AddOp            = cuasr::maximum<precision>;
  using MultOp           = cuasr::minimum<precision>;

  using Srgemm = cuasr::gemm::device::Srgemm<                           //
      AddOp, MultOp,                                                    //
      precision, cutlass::layout::RowMajor,                   //
      precision, cutlass::layout::ColumnMajor,                   //
      precision, cutlass::layout::ColumnMajor,
      precision, OpClass, SmArch>;

  EXPECT_TRUE(cuasr::test::gemm::device::TestAllGemm<Srgemm>());
}


///////////////////////////////////////////////////////////////////////////////

TEST(SM50_device_maximum_minimum_dsrgemm_tn_t, default) {
  using precision = double;
  using OpClass   = cutlass::arch::OpClassSimt;
  using SmArch    = cutlass::arch::Sm50;

  using AddOp            = cuasr::maximum<precision>;
  using MultOp           = cuasr::minimum<precision>;

  using Srgemm = cuasr::gemm::device::Srgemm<                           //
      AddOp, MultOp,                                                    //
      precision, cutlass::layout::RowMajor,                   //
      precision, cutlass::layout::ColumnMajor,                   //
      precision, cutlass::layout::RowMajor,
      precision, OpClass, SmArch>;

  EXPECT_TRUE(cuasr::test::gemm::device::TestAllGemm<Srgemm>());
}


///////////////////////////////////////////////////////////////////////////////

TEST(SM50_device_maximum_minimum_dsrgemm_nt_n, default) {
  using precision = double;
  using OpClass   = cutlass::arch::OpClassSimt;
  using SmArch    = cutlass::arch::Sm50;

  using AddOp            = cuasr::maximum<precision>;
  using MultOp           = cuasr::minimum<precision>;

  using Srgemm = cuasr::gemm::device::Srgemm<                           //
      AddOp, MultOp,                                                    //
      precision, cutlass::layout::ColumnMajor,                   //
      precision, cutlass::layout::RowMajor,                   //
      precision, cutlass::layout::ColumnMajor,
      precision, OpClass, SmArch>;

  EXPECT_TRUE(cuasr::test::gemm::device::TestAllGemm<Srgemm>());
}


///////////////////////////////////////////////////////////////////////////////

TEST(SM50_device_maximum_minimum_dsrgemm_nt_t, default) {
  using precision = double;
  using OpClass   = cutlass::arch::OpClassSimt;
  using SmArch    = cutlass::arch::Sm50;

  using AddOp            = cuasr::maximum<precision>;
  using MultOp           = cuasr::minimum<precision>;

  using Srgemm = cuasr::gemm::device::Srgemm<                           //
      AddOp, MultOp,                                                    //
      precision, cutlass::layout::ColumnMajor,                   //
      precision, cutlass::layout::RowMajor,                   //
      precision, cutlass::layout::RowMajor,
      precision, OpClass, SmArch>;

  EXPECT_TRUE(cuasr::test::gemm::device::TestAllGemm<Srgemm>());
}


///////////////////////////////////////////////////////////////////////////////

TEST(SM50_device_maximum_minimum_dsrgemm_nn_n, default) {
  using precision = double;
  using OpClass   = cutlass::arch::OpClassSimt;
  using SmArch    = cutlass::arch::Sm50;

  using AddOp            = cuasr::maximum<precision>;
  using MultOp           = cuasr::minimum<precision>;

  using Srgemm = cuasr::gemm::device::Srgemm<                           //
      AddOp, MultOp,                                                    //
      precision, cutlass::layout::ColumnMajor,                   //
      precision, cutlass::layout::ColumnMajor,                   //
      precision, cutlass::layout::ColumnMajor,
      precision, OpClass, SmArch>;

  EXPECT_TRUE(cuasr::test::gemm::device::TestAllGemm<Srgemm>());
}


///////////////////////////////////////////////////////////////////////////////

TEST(SM50_device_maximum_minimum_dsrgemm_nn_t, default) {
  using precision = double;
  using OpClass   = cutlass::arch::OpClassSimt;
  using SmArch    = cutlass::arch::Sm50;

  using AddOp            = cuasr::maximum<precision>;
  using MultOp           = cuasr::minimum<precision>;

  using Srgemm = cuasr::gemm::device::Srgemm<                           //
      AddOp, MultOp,                                                    //
      precision, cutlass::layout::ColumnMajor,                   //
      precision, cutlass::layout::ColumnMajor,                   //
      precision, cutlass::layout::RowMajor,
      precision, OpClass, SmArch>;

  EXPECT_TRUE(cuasr::test::gemm::device::TestAllGemm<Srgemm>());
}


///////////////////////////////////////////////////////////////////////////////

TEST(SM50_device_maximum_minimum_ssrgemm_tt_n, default) {
  using precision = float;
  using OpClass   = cutlass::arch::OpClassSimt;
  using SmArch    = cutlass::arch::Sm50;

  using AddOp            = cuasr::maximum<precision>;
  using MultOp           = cuasr::minimum<precision>;

  using Srgemm = cuasr::gemm::device::Srgemm<                           //
      AddOp, MultOp,                                                    //
      precision, cutlass::layout::RowMajor,                   //
      precision, cutlass::layout::RowMajor,                   //
      precision, cutlass::layout::ColumnMajor,
      precision, OpClass, SmArch>;

  EXPECT_TRUE(cuasr::test::gemm::device::TestAllGemm<Srgemm>());
}


///////////////////////////////////////////////////////////////////////////////

TEST(SM50_device_maximum_minimum_ssrgemm_tt_t, default) {
  using precision = float;
  using OpClass   = cutlass::arch::OpClassSimt;
  using SmArch    = cutlass::arch::Sm50;

  using AddOp            = cuasr::maximum<precision>;
  using MultOp           = cuasr::minimum<precision>;

  using Srgemm = cuasr::gemm::device::Srgemm<                           //
      AddOp, MultOp,                                                    //
      precision, cutlass::layout::RowMajor,                   //
      precision, cutlass::layout::RowMajor,                   //
      precision, cutlass::layout::RowMajor,
      precision, OpClass, SmArch>;

  EXPECT_TRUE(cuasr::test::gemm::device::TestAllGemm<Srgemm>());
}


///////////////////////////////////////////////////////////////////////////////

TEST(SM50_device_maximum_minimum_ssrgemm_tn_n, default) {
  using precision = float;
  using OpClass   = cutlass::arch::OpClassSimt;
  using SmArch    = cutlass::arch::Sm50;

  using AddOp            = cuasr::maximum<precision>;
  using MultOp           = cuasr::minimum<precision>;

  using Srgemm = cuasr::gemm::device::Srgemm<                           //
      AddOp, MultOp,                                                    //
      precision, cutlass::layout::RowMajor,                   //
      precision, cutlass::layout::ColumnMajor,                   //
      precision, cutlass::layout::ColumnMajor,
      precision, OpClass, SmArch>;

  EXPECT_TRUE(cuasr::test::gemm::device::TestAllGemm<Srgemm>());
}


///////////////////////////////////////////////////////////////////////////////

TEST(SM50_device_maximum_minimum_ssrgemm_tn_t, default) {
  using precision = float;
  using OpClass   = cutlass::arch::OpClassSimt;
  using SmArch    = cutlass::arch::Sm50;

  using AddOp            = cuasr::maximum<precision>;
  using MultOp           = cuasr::minimum<precision>;

  using Srgemm = cuasr::gemm::device::Srgemm<                           //
      AddOp, MultOp,                                                    //
      precision, cutlass::layout::RowMajor,                   //
      precision, cutlass::layout::ColumnMajor,                   //
      precision, cutlass::layout::RowMajor,
      precision, OpClass, SmArch>;

  EXPECT_TRUE(cuasr::test::gemm::device::TestAllGemm<Srgemm>());
}


///////////////////////////////////////////////////////////////////////////////

TEST(SM50_device_maximum_minimum_ssrgemm_nt_n, default) {
  using precision = float;
  using OpClass   = cutlass::arch::OpClassSimt;
  using SmArch    = cutlass::arch::Sm50;

  using AddOp            = cuasr::maximum<precision>;
  using MultOp           = cuasr::minimum<precision>;

  using Srgemm = cuasr::gemm::device::Srgemm<                           //
      AddOp, MultOp,                                                    //
      precision, cutlass::layout::ColumnMajor,                   //
      precision, cutlass::layout::RowMajor,                   //
      precision, cutlass::layout::ColumnMajor,
      precision, OpClass, SmArch>;

  EXPECT_TRUE(cuasr::test::gemm::device::TestAllGemm<Srgemm>());
}


///////////////////////////////////////////////////////////////////////////////

TEST(SM50_device_maximum_minimum_ssrgemm_nt_t, default) {
  using precision = float;
  using OpClass   = cutlass::arch::OpClassSimt;
  using SmArch    = cutlass::arch::Sm50;

  using AddOp            = cuasr::maximum<precision>;
  using MultOp           = cuasr::minimum<precision>;

  using Srgemm = cuasr::gemm::device::Srgemm<                           //
      AddOp, MultOp,                                                    //
      precision, cutlass::layout::ColumnMajor,                   //
      precision, cutlass::layout::RowMajor,                   //
      precision, cutlass::layout::RowMajor,
      precision, OpClass, SmArch>;

  EXPECT_TRUE(cuasr::test::gemm::device::TestAllGemm<Srgemm>());
}


///////////////////////////////////////////////////////////////////////////////

TEST(SM50_device_maximum_minimum_ssrgemm_nn_n, default) {
  using precision = float;
  using OpClass   = cutlass::arch::OpClassSimt;
  using SmArch    = cutlass::arch::Sm50;

  using AddOp            = cuasr::maximum<precision>;
  using MultOp           = cuasr::minimum<precision>;

  using Srgemm = cuasr::gemm::device::Srgemm<                           //
      AddOp, MultOp,                                                    //
      precision, cutlass::layout::ColumnMajor,                   //
      precision, cutlass::layout::ColumnMajor,                   //
      precision, cutlass::layout::ColumnMajor,
      precision, OpClass, SmArch>;

  EXPECT_TRUE(cuasr::test::gemm::device::TestAllGemm<Srgemm>());
}


///////////////////////////////////////////////////////////////////////////////

TEST(SM50_device_maximum_minimum_ssrgemm_nn_t, default) {
  using precision = float;
  using OpClass   = cutlass::arch::OpClassSimt;
  using SmArch    = cutlass::arch::Sm50;

  using AddOp            = cuasr::maximum<precision>;
  using MultOp           = cuasr::minimum<precision>;

  using Srgemm = cuasr::gemm::device::Srgemm<                           //
      AddOp, MultOp,                                                    //
      precision, cutlass::layout::ColumnMajor,                   //
      precision, cutlass::layout::ColumnMajor,                   //
      precision, cutlass::layout::RowMajor,
      precision, OpClass, SmArch>;

  EXPECT_TRUE(cuasr::test::gemm::device::TestAllGemm<Srgemm>());
}


///////////////////////////////////////////////////////////////////////////////

TEST(SM50_device_minimum_multiplies_dsrgemm_tt_n, default) {
  using precision = double;
  using OpClass   = cutlass::arch::OpClassSimt;
  using SmArch    = cutlass::arch::Sm50;

  using AddOp            = cuasr::minimum<precision>;
  using MultOp           = cuasr::multiplies<precision>;

  using Srgemm = cuasr::gemm::device::Srgemm<                           //
      AddOp, MultOp,                                                    //
      precision, cutlass::layout::RowMajor,                   //
      precision, cutlass::layout::RowMajor,                   //
      precision, cutlass::layout::ColumnMajor,
      precision, OpClass, SmArch>;

  EXPECT_TRUE(cuasr::test::gemm::device::TestAllGemm<Srgemm>());
}


///////////////////////////////////////////////////////////////////////////////

TEST(SM50_device_minimum_multiplies_dsrgemm_tt_t, default) {
  using precision = double;
  using OpClass   = cutlass::arch::OpClassSimt;
  using SmArch    = cutlass::arch::Sm50;

  using AddOp            = cuasr::minimum<precision>;
  using MultOp           = cuasr::multiplies<precision>;

  using Srgemm = cuasr::gemm::device::Srgemm<                           //
      AddOp, MultOp,                                                    //
      precision, cutlass::layout::RowMajor,                   //
      precision, cutlass::layout::RowMajor,                   //
      precision, cutlass::layout::RowMajor,
      precision, OpClass, SmArch>;

  EXPECT_TRUE(cuasr::test::gemm::device::TestAllGemm<Srgemm>());
}


///////////////////////////////////////////////////////////////////////////////

TEST(SM50_device_minimum_multiplies_dsrgemm_tn_n, default) {
  using precision = double;
  using OpClass   = cutlass::arch::OpClassSimt;
  using SmArch    = cutlass::arch::Sm50;

  using AddOp            = cuasr::minimum<precision>;
  using MultOp           = cuasr::multiplies<precision>;

  using Srgemm = cuasr::gemm::device::Srgemm<                           //
      AddOp, MultOp,                                                    //
      precision, cutlass::layout::RowMajor,                   //
      precision, cutlass::layout::ColumnMajor,                   //
      precision, cutlass::layout::ColumnMajor,
      precision, OpClass, SmArch>;

  EXPECT_TRUE(cuasr::test::gemm::device::TestAllGemm<Srgemm>());
}


///////////////////////////////////////////////////////////////////////////////

TEST(SM50_device_minimum_multiplies_dsrgemm_tn_t, default) {
  using precision = double;
  using OpClass   = cutlass::arch::OpClassSimt;
  using SmArch    = cutlass::arch::Sm50;

  using AddOp            = cuasr::minimum<precision>;
  using MultOp           = cuasr::multiplies<precision>;

  using Srgemm = cuasr::gemm::device::Srgemm<                           //
      AddOp, MultOp,                                                    //
      precision, cutlass::layout::RowMajor,                   //
      precision, cutlass::layout::ColumnMajor,                   //
      precision, cutlass::layout::RowMajor,
      precision, OpClass, SmArch>;

  EXPECT_TRUE(cuasr::test::gemm::device::TestAllGemm<Srgemm>());
}


///////////////////////////////////////////////////////////////////////////////

TEST(SM50_device_minimum_multiplies_dsrgemm_nt_n, default) {
  using precision = double;
  using OpClass   = cutlass::arch::OpClassSimt;
  using SmArch    = cutlass::arch::Sm50;

  using AddOp            = cuasr::minimum<precision>;
  using MultOp           = cuasr::multiplies<precision>;

  using Srgemm = cuasr::gemm::device::Srgemm<                           //
      AddOp, MultOp,                                                    //
      precision, cutlass::layout::ColumnMajor,                   //
      precision, cutlass::layout::RowMajor,                   //
      precision, cutlass::layout::ColumnMajor,
      precision, OpClass, SmArch>;

  EXPECT_TRUE(cuasr::test::gemm::device::TestAllGemm<Srgemm>());
}


///////////////////////////////////////////////////////////////////////////////

TEST(SM50_device_minimum_multiplies_dsrgemm_nt_t, default) {
  using precision = double;
  using OpClass   = cutlass::arch::OpClassSimt;
  using SmArch    = cutlass::arch::Sm50;

  using AddOp            = cuasr::minimum<precision>;
  using MultOp           = cuasr::multiplies<precision>;

  using Srgemm = cuasr::gemm::device::Srgemm<                           //
      AddOp, MultOp,                                                    //
      precision, cutlass::layout::ColumnMajor,                   //
      precision, cutlass::layout::RowMajor,                   //
      precision, cutlass::layout::RowMajor,
      precision, OpClass, SmArch>;

  EXPECT_TRUE(cuasr::test::gemm::device::TestAllGemm<Srgemm>());
}


///////////////////////////////////////////////////////////////////////////////

TEST(SM50_device_minimum_multiplies_dsrgemm_nn_n, default) {
  using precision = double;
  using OpClass   = cutlass::arch::OpClassSimt;
  using SmArch    = cutlass::arch::Sm50;

  using AddOp            = cuasr::minimum<precision>;
  using MultOp           = cuasr::multiplies<precision>;

  using Srgemm = cuasr::gemm::device::Srgemm<                           //
      AddOp, MultOp,                                                    //
      precision, cutlass::layout::ColumnMajor,                   //
      precision, cutlass::layout::ColumnMajor,                   //
      precision, cutlass::layout::ColumnMajor,
      precision, OpClass, SmArch>;

  EXPECT_TRUE(cuasr::test::gemm::device::TestAllGemm<Srgemm>());
}


///////////////////////////////////////////////////////////////////////////////

TEST(SM50_device_minimum_multiplies_dsrgemm_nn_t, default) {
  using precision = double;
  using OpClass   = cutlass::arch::OpClassSimt;
  using SmArch    = cutlass::arch::Sm50;

  using AddOp            = cuasr::minimum<precision>;
  using MultOp           = cuasr::multiplies<precision>;

  using Srgemm = cuasr::gemm::device::Srgemm<                           //
      AddOp, MultOp,                                                    //
      precision, cutlass::layout::ColumnMajor,                   //
      precision, cutlass::layout::ColumnMajor,                   //
      precision, cutlass::layout::RowMajor,
      precision, OpClass, SmArch>;

  EXPECT_TRUE(cuasr::test::gemm::device::TestAllGemm<Srgemm>());
}


///////////////////////////////////////////////////////////////////////////////

TEST(SM50_device_minimum_multiplies_ssrgemm_tt_n, default) {
  using precision = float;
  using OpClass   = cutlass::arch::OpClassSimt;
  using SmArch    = cutlass::arch::Sm50;

  using AddOp            = cuasr::minimum<precision>;
  using MultOp           = cuasr::multiplies<precision>;

  using Srgemm = cuasr::gemm::device::Srgemm<                           //
      AddOp, MultOp,                                                    //
      precision, cutlass::layout::RowMajor,                   //
      precision, cutlass::layout::RowMajor,                   //
      precision, cutlass::layout::ColumnMajor,
      precision, OpClass, SmArch>;

  EXPECT_TRUE(cuasr::test::gemm::device::TestAllGemm<Srgemm>());
}


///////////////////////////////////////////////////////////////////////////////

TEST(SM50_device_minimum_multiplies_ssrgemm_tt_t, default) {
  using precision = float;
  using OpClass   = cutlass::arch::OpClassSimt;
  using SmArch    = cutlass::arch::Sm50;

  using AddOp            = cuasr::minimum<precision>;
  using MultOp           = cuasr::multiplies<precision>;

  using Srgemm = cuasr::gemm::device::Srgemm<                           //
      AddOp, MultOp,                                                    //
      precision, cutlass::layout::RowMajor,                   //
      precision, cutlass::layout::RowMajor,                   //
      precision, cutlass::layout::RowMajor,
      precision, OpClass, SmArch>;

  EXPECT_TRUE(cuasr::test::gemm::device::TestAllGemm<Srgemm>());
}


///////////////////////////////////////////////////////////////////////////////

TEST(SM50_device_minimum_multiplies_ssrgemm_tn_n, default) {
  using precision = float;
  using OpClass   = cutlass::arch::OpClassSimt;
  using SmArch    = cutlass::arch::Sm50;

  using AddOp            = cuasr::minimum<precision>;
  using MultOp           = cuasr::multiplies<precision>;

  using Srgemm = cuasr::gemm::device::Srgemm<                           //
      AddOp, MultOp,                                                    //
      precision, cutlass::layout::RowMajor,                   //
      precision, cutlass::layout::ColumnMajor,                   //
      precision, cutlass::layout::ColumnMajor,
      precision, OpClass, SmArch>;

  EXPECT_TRUE(cuasr::test::gemm::device::TestAllGemm<Srgemm>());
}


///////////////////////////////////////////////////////////////////////////////

TEST(SM50_device_minimum_multiplies_ssrgemm_tn_t, default) {
  using precision = float;
  using OpClass   = cutlass::arch::OpClassSimt;
  using SmArch    = cutlass::arch::Sm50;

  using AddOp            = cuasr::minimum<precision>;
  using MultOp           = cuasr::multiplies<precision>;

  using Srgemm = cuasr::gemm::device::Srgemm<                           //
      AddOp, MultOp,                                                    //
      precision, cutlass::layout::RowMajor,                   //
      precision, cutlass::layout::ColumnMajor,                   //
      precision, cutlass::layout::RowMajor,
      precision, OpClass, SmArch>;

  EXPECT_TRUE(cuasr::test::gemm::device::TestAllGemm<Srgemm>());
}


///////////////////////////////////////////////////////////////////////////////

TEST(SM50_device_minimum_multiplies_ssrgemm_nt_n, default) {
  using precision = float;
  using OpClass   = cutlass::arch::OpClassSimt;
  using SmArch    = cutlass::arch::Sm50;

  using AddOp            = cuasr::minimum<precision>;
  using MultOp           = cuasr::multiplies<precision>;

  using Srgemm = cuasr::gemm::device::Srgemm<                           //
      AddOp, MultOp,                                                    //
      precision, cutlass::layout::ColumnMajor,                   //
      precision, cutlass::layout::RowMajor,                   //
      precision, cutlass::layout::ColumnMajor,
      precision, OpClass, SmArch>;

  EXPECT_TRUE(cuasr::test::gemm::device::TestAllGemm<Srgemm>());
}


///////////////////////////////////////////////////////////////////////////////

TEST(SM50_device_minimum_multiplies_ssrgemm_nt_t, default) {
  using precision = float;
  using OpClass   = cutlass::arch::OpClassSimt;
  using SmArch    = cutlass::arch::Sm50;

  using AddOp            = cuasr::minimum<precision>;
  using MultOp           = cuasr::multiplies<precision>;

  using Srgemm = cuasr::gemm::device::Srgemm<                           //
      AddOp, MultOp,                                                    //
      precision, cutlass::layout::ColumnMajor,                   //
      precision, cutlass::layout::RowMajor,                   //
      precision, cutlass::layout::RowMajor,
      precision, OpClass, SmArch>;

  EXPECT_TRUE(cuasr::test::gemm::device::TestAllGemm<Srgemm>());
}


///////////////////////////////////////////////////////////////////////////////

TEST(SM50_device_minimum_multiplies_ssrgemm_nn_n, default) {
  using precision = float;
  using OpClass   = cutlass::arch::OpClassSimt;
  using SmArch    = cutlass::arch::Sm50;

  using AddOp            = cuasr::minimum<precision>;
  using MultOp           = cuasr::multiplies<precision>;

  using Srgemm = cuasr::gemm::device::Srgemm<                           //
      AddOp, MultOp,                                                    //
      precision, cutlass::layout::ColumnMajor,                   //
      precision, cutlass::layout::ColumnMajor,                   //
      precision, cutlass::layout::ColumnMajor,
      precision, OpClass, SmArch>;

  EXPECT_TRUE(cuasr::test::gemm::device::TestAllGemm<Srgemm>());
}


///////////////////////////////////////////////////////////////////////////////

TEST(SM50_device_minimum_multiplies_ssrgemm_nn_t, default) {
  using precision = float;
  using OpClass   = cutlass::arch::OpClassSimt;
  using SmArch    = cutlass::arch::Sm50;

  using AddOp            = cuasr::minimum<precision>;
  using MultOp           = cuasr::multiplies<precision>;

  using Srgemm = cuasr::gemm::device::Srgemm<                           //
      AddOp, MultOp,                                                    //
      precision, cutlass::layout::ColumnMajor,                   //
      precision, cutlass::layout::ColumnMajor,                   //
      precision, cutlass::layout::RowMajor,
      precision, OpClass, SmArch>;

  EXPECT_TRUE(cuasr::test::gemm::device::TestAllGemm<Srgemm>());
}


///////////////////////////////////////////////////////////////////////////////

TEST(SM50_device_maximum_multiplies_dsrgemm_tt_n, default) {
  using precision = double;
  using OpClass   = cutlass::arch::OpClassSimt;
  using SmArch    = cutlass::arch::Sm50;

  using AddOp            = cuasr::maximum<precision>;
  using MultOp           = cuasr::multiplies<precision>;

  using Srgemm = cuasr::gemm::device::Srgemm<                           //
      AddOp, MultOp,                                                    //
      precision, cutlass::layout::RowMajor,                   //
      precision, cutlass::layout::RowMajor,                   //
      precision, cutlass::layout::ColumnMajor,
      precision, OpClass, SmArch>;

  EXPECT_TRUE(cuasr::test::gemm::device::TestAllGemm<Srgemm>());
}


///////////////////////////////////////////////////////////////////////////////

TEST(SM50_device_maximum_multiplies_dsrgemm_tt_t, default) {
  using precision = double;
  using OpClass   = cutlass::arch::OpClassSimt;
  using SmArch    = cutlass::arch::Sm50;

  using AddOp            = cuasr::maximum<precision>;
  using MultOp           = cuasr::multiplies<precision>;

  using Srgemm = cuasr::gemm::device::Srgemm<                           //
      AddOp, MultOp,                                                    //
      precision, cutlass::layout::RowMajor,                   //
      precision, cutlass::layout::RowMajor,                   //
      precision, cutlass::layout::RowMajor,
      precision, OpClass, SmArch>;

  EXPECT_TRUE(cuasr::test::gemm::device::TestAllGemm<Srgemm>());
}


///////////////////////////////////////////////////////////////////////////////

TEST(SM50_device_maximum_multiplies_dsrgemm_tn_n, default) {
  using precision = double;
  using OpClass   = cutlass::arch::OpClassSimt;
  using SmArch    = cutlass::arch::Sm50;

  using AddOp            = cuasr::maximum<precision>;
  using MultOp           = cuasr::multiplies<precision>;

  using Srgemm = cuasr::gemm::device::Srgemm<                           //
      AddOp, MultOp,                                                    //
      precision, cutlass::layout::RowMajor,                   //
      precision, cutlass::layout::ColumnMajor,                   //
      precision, cutlass::layout::ColumnMajor,
      precision, OpClass, SmArch>;

  EXPECT_TRUE(cuasr::test::gemm::device::TestAllGemm<Srgemm>());
}


///////////////////////////////////////////////////////////////////////////////

TEST(SM50_device_maximum_multiplies_dsrgemm_tn_t, default) {
  using precision = double;
  using OpClass   = cutlass::arch::OpClassSimt;
  using SmArch    = cutlass::arch::Sm50;

  using AddOp            = cuasr::maximum<precision>;
  using MultOp           = cuasr::multiplies<precision>;

  using Srgemm = cuasr::gemm::device::Srgemm<                           //
      AddOp, MultOp,                                                    //
      precision, cutlass::layout::RowMajor,                   //
      precision, cutlass::layout::ColumnMajor,                   //
      precision, cutlass::layout::RowMajor,
      precision, OpClass, SmArch>;

  EXPECT_TRUE(cuasr::test::gemm::device::TestAllGemm<Srgemm>());
}


///////////////////////////////////////////////////////////////////////////////

TEST(SM50_device_maximum_multiplies_dsrgemm_nt_n, default) {
  using precision = double;
  using OpClass   = cutlass::arch::OpClassSimt;
  using SmArch    = cutlass::arch::Sm50;

  using AddOp            = cuasr::maximum<precision>;
  using MultOp           = cuasr::multiplies<precision>;

  using Srgemm = cuasr::gemm::device::Srgemm<                           //
      AddOp, MultOp,                                                    //
      precision, cutlass::layout::ColumnMajor,                   //
      precision, cutlass::layout::RowMajor,                   //
      precision, cutlass::layout::ColumnMajor,
      precision, OpClass, SmArch>;

  EXPECT_TRUE(cuasr::test::gemm::device::TestAllGemm<Srgemm>());
}


///////////////////////////////////////////////////////////////////////////////

TEST(SM50_device_maximum_multiplies_dsrgemm_nt_t, default) {
  using precision = double;
  using OpClass   = cutlass::arch::OpClassSimt;
  using SmArch    = cutlass::arch::Sm50;

  using AddOp            = cuasr::maximum<precision>;
  using MultOp           = cuasr::multiplies<precision>;

  using Srgemm = cuasr::gemm::device::Srgemm<                           //
      AddOp, MultOp,                                                    //
      precision, cutlass::layout::ColumnMajor,                   //
      precision, cutlass::layout::RowMajor,                   //
      precision, cutlass::layout::RowMajor,
      precision, OpClass, SmArch>;

  EXPECT_TRUE(cuasr::test::gemm::device::TestAllGemm<Srgemm>());
}


///////////////////////////////////////////////////////////////////////////////

TEST(SM50_device_maximum_multiplies_dsrgemm_nn_n, default) {
  using precision = double;
  using OpClass   = cutlass::arch::OpClassSimt;
  using SmArch    = cutlass::arch::Sm50;

  using AddOp            = cuasr::maximum<precision>;
  using MultOp           = cuasr::multiplies<precision>;

  using Srgemm = cuasr::gemm::device::Srgemm<                           //
      AddOp, MultOp,                                                    //
      precision, cutlass::layout::ColumnMajor,                   //
      precision, cutlass::layout::ColumnMajor,                   //
      precision, cutlass::layout::ColumnMajor,
      precision, OpClass, SmArch>;

  EXPECT_TRUE(cuasr::test::gemm::device::TestAllGemm<Srgemm>());
}


///////////////////////////////////////////////////////////////////////////////

TEST(SM50_device_maximum_multiplies_dsrgemm_nn_t, default) {
  using precision = double;
  using OpClass   = cutlass::arch::OpClassSimt;
  using SmArch    = cutlass::arch::Sm50;

  using AddOp            = cuasr::maximum<precision>;
  using MultOp           = cuasr::multiplies<precision>;

  using Srgemm = cuasr::gemm::device::Srgemm<                           //
      AddOp, MultOp,                                                    //
      precision, cutlass::layout::ColumnMajor,                   //
      precision, cutlass::layout::ColumnMajor,                   //
      precision, cutlass::layout::RowMajor,
      precision, OpClass, SmArch>;

  EXPECT_TRUE(cuasr::test::gemm::device::TestAllGemm<Srgemm>());
}


///////////////////////////////////////////////////////////////////////////////

TEST(SM50_device_maximum_multiplies_ssrgemm_tt_n, default) {
  using precision = float;
  using OpClass   = cutlass::arch::OpClassSimt;
  using SmArch    = cutlass::arch::Sm50;

  using AddOp            = cuasr::maximum<precision>;
  using MultOp           = cuasr::multiplies<precision>;

  using Srgemm = cuasr::gemm::device::Srgemm<                           //
      AddOp, MultOp,                                                    //
      precision, cutlass::layout::RowMajor,                   //
      precision, cutlass::layout::RowMajor,                   //
      precision, cutlass::layout::ColumnMajor,
      precision, OpClass, SmArch>;

  EXPECT_TRUE(cuasr::test::gemm::device::TestAllGemm<Srgemm>());
}


///////////////////////////////////////////////////////////////////////////////

TEST(SM50_device_maximum_multiplies_ssrgemm_tt_t, default) {
  using precision = float;
  using OpClass   = cutlass::arch::OpClassSimt;
  using SmArch    = cutlass::arch::Sm50;

  using AddOp            = cuasr::maximum<precision>;
  using MultOp           = cuasr::multiplies<precision>;

  using Srgemm = cuasr::gemm::device::Srgemm<                           //
      AddOp, MultOp,                                                    //
      precision, cutlass::layout::RowMajor,                   //
      precision, cutlass::layout::RowMajor,                   //
      precision, cutlass::layout::RowMajor,
      precision, OpClass, SmArch>;

  EXPECT_TRUE(cuasr::test::gemm::device::TestAllGemm<Srgemm>());
}


///////////////////////////////////////////////////////////////////////////////

TEST(SM50_device_maximum_multiplies_ssrgemm_tn_n, default) {
  using precision = float;
  using OpClass   = cutlass::arch::OpClassSimt;
  using SmArch    = cutlass::arch::Sm50;

  using AddOp            = cuasr::maximum<precision>;
  using MultOp           = cuasr::multiplies<precision>;

  using Srgemm = cuasr::gemm::device::Srgemm<                           //
      AddOp, MultOp,                                                    //
      precision, cutlass::layout::RowMajor,                   //
      precision, cutlass::layout::ColumnMajor,                   //
      precision, cutlass::layout::ColumnMajor,
      precision, OpClass, SmArch>;

  EXPECT_TRUE(cuasr::test::gemm::device::TestAllGemm<Srgemm>());
}


///////////////////////////////////////////////////////////////////////////////

TEST(SM50_device_maximum_multiplies_ssrgemm_tn_t, default) {
  using precision = float;
  using OpClass   = cutlass::arch::OpClassSimt;
  using SmArch    = cutlass::arch::Sm50;

  using AddOp            = cuasr::maximum<precision>;
  using MultOp           = cuasr::multiplies<precision>;

  using Srgemm = cuasr::gemm::device::Srgemm<                           //
      AddOp, MultOp,                                                    //
      precision, cutlass::layout::RowMajor,                   //
      precision, cutlass::layout::ColumnMajor,                   //
      precision, cutlass::layout::RowMajor,
      precision, OpClass, SmArch>;

  EXPECT_TRUE(cuasr::test::gemm::device::TestAllGemm<Srgemm>());
}


///////////////////////////////////////////////////////////////////////////////

TEST(SM50_device_maximum_multiplies_ssrgemm_nt_n, default) {
  using precision = float;
  using OpClass   = cutlass::arch::OpClassSimt;
  using SmArch    = cutlass::arch::Sm50;

  using AddOp            = cuasr::maximum<precision>;
  using MultOp           = cuasr::multiplies<precision>;

  using Srgemm = cuasr::gemm::device::Srgemm<                           //
      AddOp, MultOp,                                                    //
      precision, cutlass::layout::ColumnMajor,                   //
      precision, cutlass::layout::RowMajor,                   //
      precision, cutlass::layout::ColumnMajor,
      precision, OpClass, SmArch>;

  EXPECT_TRUE(cuasr::test::gemm::device::TestAllGemm<Srgemm>());
}


///////////////////////////////////////////////////////////////////////////////

TEST(SM50_device_maximum_multiplies_ssrgemm_nt_t, default) {
  using precision = float;
  using OpClass   = cutlass::arch::OpClassSimt;
  using SmArch    = cutlass::arch::Sm50;

  using AddOp            = cuasr::maximum<precision>;
  using MultOp           = cuasr::multiplies<precision>;

  using Srgemm = cuasr::gemm::device::Srgemm<                           //
      AddOp, MultOp,                                                    //
      precision, cutlass::layout::ColumnMajor,                   //
      precision, cutlass::layout::RowMajor,                   //
      precision, cutlass::layout::RowMajor,
      precision, OpClass, SmArch>;

  EXPECT_TRUE(cuasr::test::gemm::device::TestAllGemm<Srgemm>());
}


///////////////////////////////////////////////////////////////////////////////

TEST(SM50_device_maximum_multiplies_ssrgemm_nn_n, default) {
  using precision = float;
  using OpClass   = cutlass::arch::OpClassSimt;
  using SmArch    = cutlass::arch::Sm50;

  using AddOp            = cuasr::maximum<precision>;
  using MultOp           = cuasr::multiplies<precision>;

  using Srgemm = cuasr::gemm::device::Srgemm<                           //
      AddOp, MultOp,                                                    //
      precision, cutlass::layout::ColumnMajor,                   //
      precision, cutlass::layout::ColumnMajor,                   //
      precision, cutlass::layout::ColumnMajor,
      precision, OpClass, SmArch>;

  EXPECT_TRUE(cuasr::test::gemm::device::TestAllGemm<Srgemm>());
}


///////////////////////////////////////////////////////////////////////////////

TEST(SM50_device_maximum_multiplies_ssrgemm_nn_t, default) {
  using precision = float;
  using OpClass   = cutlass::arch::OpClassSimt;
  using SmArch    = cutlass::arch::Sm50;

  using AddOp            = cuasr::maximum<precision>;
  using MultOp           = cuasr::multiplies<precision>;

  using Srgemm = cuasr::gemm::device::Srgemm<                           //
      AddOp, MultOp,                                                    //
      precision, cutlass::layout::ColumnMajor,                   //
      precision, cutlass::layout::ColumnMajor,                   //
      precision, cutlass::layout::RowMajor,
      precision, OpClass, SmArch>;

  EXPECT_TRUE(cuasr::test::gemm::device::TestAllGemm<Srgemm>());
}


///////////////////////////////////////////////////////////////////////////////

TEST(SM50_device_binary_or_binary_and_dsrgemm_tt_n, default) {
  using precision = double;
  using OpClass   = cutlass::arch::OpClassSimt;
  using SmArch    = cutlass::arch::Sm50;

  using AddOp            = cuasr::binary_or<precision>;
  using MultOp           = cuasr::binary_and<precision>;

  using Srgemm = cuasr::gemm::device::Srgemm<                           //
      AddOp, MultOp,                                                    //
      precision, cutlass::layout::RowMajor,                   //
      precision, cutlass::layout::RowMajor,                   //
      precision, cutlass::layout::ColumnMajor,
      precision, OpClass, SmArch>;

  EXPECT_TRUE(cuasr::test::gemm::device::TestAllGemm<Srgemm>());
}


///////////////////////////////////////////////////////////////////////////////

TEST(SM50_device_binary_or_binary_and_dsrgemm_tt_t, default) {
  using precision = double;
  using OpClass   = cutlass::arch::OpClassSimt;
  using SmArch    = cutlass::arch::Sm50;

  using AddOp            = cuasr::binary_or<precision>;
  using MultOp           = cuasr::binary_and<precision>;

  using Srgemm = cuasr::gemm::device::Srgemm<                           //
      AddOp, MultOp,                                                    //
      precision, cutlass::layout::RowMajor,                   //
      precision, cutlass::layout::RowMajor,                   //
      precision, cutlass::layout::RowMajor,
      precision, OpClass, SmArch>;

  EXPECT_TRUE(cuasr::test::gemm::device::TestAllGemm<Srgemm>());
}


///////////////////////////////////////////////////////////////////////////////

TEST(SM50_device_binary_or_binary_and_dsrgemm_tn_n, default) {
  using precision = double;
  using OpClass   = cutlass::arch::OpClassSimt;
  using SmArch    = cutlass::arch::Sm50;

  using AddOp            = cuasr::binary_or<precision>;
  using MultOp           = cuasr::binary_and<precision>;

  using Srgemm = cuasr::gemm::device::Srgemm<                           //
      AddOp, MultOp,                                                    //
      precision, cutlass::layout::RowMajor,                   //
      precision, cutlass::layout::ColumnMajor,                   //
      precision, cutlass::layout::ColumnMajor,
      precision, OpClass, SmArch>;

  EXPECT_TRUE(cuasr::test::gemm::device::TestAllGemm<Srgemm>());
}


///////////////////////////////////////////////////////////////////////////////

TEST(SM50_device_binary_or_binary_and_dsrgemm_tn_t, default) {
  using precision = double;
  using OpClass   = cutlass::arch::OpClassSimt;
  using SmArch    = cutlass::arch::Sm50;

  using AddOp            = cuasr::binary_or<precision>;
  using MultOp           = cuasr::binary_and<precision>;

  using Srgemm = cuasr::gemm::device::Srgemm<                           //
      AddOp, MultOp,                                                    //
      precision, cutlass::layout::RowMajor,                   //
      precision, cutlass::layout::ColumnMajor,                   //
      precision, cutlass::layout::RowMajor,
      precision, OpClass, SmArch>;

  EXPECT_TRUE(cuasr::test::gemm::device::TestAllGemm<Srgemm>());
}


///////////////////////////////////////////////////////////////////////////////

TEST(SM50_device_binary_or_binary_and_dsrgemm_nt_n, default) {
  using precision = double;
  using OpClass   = cutlass::arch::OpClassSimt;
  using SmArch    = cutlass::arch::Sm50;

  using AddOp            = cuasr::binary_or<precision>;
  using MultOp           = cuasr::binary_and<precision>;

  using Srgemm = cuasr::gemm::device::Srgemm<                           //
      AddOp, MultOp,                                                    //
      precision, cutlass::layout::ColumnMajor,                   //
      precision, cutlass::layout::RowMajor,                   //
      precision, cutlass::layout::ColumnMajor,
      precision, OpClass, SmArch>;

  EXPECT_TRUE(cuasr::test::gemm::device::TestAllGemm<Srgemm>());
}


///////////////////////////////////////////////////////////////////////////////

TEST(SM50_device_binary_or_binary_and_dsrgemm_nt_t, default) {
  using precision = double;
  using OpClass   = cutlass::arch::OpClassSimt;
  using SmArch    = cutlass::arch::Sm50;

  using AddOp            = cuasr::binary_or<precision>;
  using MultOp           = cuasr::binary_and<precision>;

  using Srgemm = cuasr::gemm::device::Srgemm<                           //
      AddOp, MultOp,                                                    //
      precision, cutlass::layout::ColumnMajor,                   //
      precision, cutlass::layout::RowMajor,                   //
      precision, cutlass::layout::RowMajor,
      precision, OpClass, SmArch>;

  EXPECT_TRUE(cuasr::test::gemm::device::TestAllGemm<Srgemm>());
}


///////////////////////////////////////////////////////////////////////////////

TEST(SM50_device_binary_or_binary_and_dsrgemm_nn_n, default) {
  using precision = double;
  using OpClass   = cutlass::arch::OpClassSimt;
  using SmArch    = cutlass::arch::Sm50;

  using AddOp            = cuasr::binary_or<precision>;
  using MultOp           = cuasr::binary_and<precision>;

  using Srgemm = cuasr::gemm::device::Srgemm<                           //
      AddOp, MultOp,                                                    //
      precision, cutlass::layout::ColumnMajor,                   //
      precision, cutlass::layout::ColumnMajor,                   //
      precision, cutlass::layout::ColumnMajor,
      precision, OpClass, SmArch>;

  EXPECT_TRUE(cuasr::test::gemm::device::TestAllGemm<Srgemm>());
}


///////////////////////////////////////////////////////////////////////////////

TEST(SM50_device_binary_or_binary_and_dsrgemm_nn_t, default) {
  using precision = double;
  using OpClass   = cutlass::arch::OpClassSimt;
  using SmArch    = cutlass::arch::Sm50;

  using AddOp            = cuasr::binary_or<precision>;
  using MultOp           = cuasr::binary_and<precision>;

  using Srgemm = cuasr::gemm::device::Srgemm<                           //
      AddOp, MultOp,                                                    //
      precision, cutlass::layout::ColumnMajor,                   //
      precision, cutlass::layout::ColumnMajor,                   //
      precision, cutlass::layout::RowMajor,
      precision, OpClass, SmArch>;

  EXPECT_TRUE(cuasr::test::gemm::device::TestAllGemm<Srgemm>());
}


///////////////////////////////////////////////////////////////////////////////

TEST(SM50_device_binary_or_binary_and_ssrgemm_tt_n, default) {
  using precision = float;
  using OpClass   = cutlass::arch::OpClassSimt;
  using SmArch    = cutlass::arch::Sm50;

  using AddOp            = cuasr::binary_or<precision>;
  using MultOp           = cuasr::binary_and<precision>;

  using Srgemm = cuasr::gemm::device::Srgemm<                           //
      AddOp, MultOp,                                                    //
      precision, cutlass::layout::RowMajor,                   //
      precision, cutlass::layout::RowMajor,                   //
      precision, cutlass::layout::ColumnMajor,
      precision, OpClass, SmArch>;

  EXPECT_TRUE(cuasr::test::gemm::device::TestAllGemm<Srgemm>());
}


///////////////////////////////////////////////////////////////////////////////

TEST(SM50_device_binary_or_binary_and_ssrgemm_tt_t, default) {
  using precision = float;
  using OpClass   = cutlass::arch::OpClassSimt;
  using SmArch    = cutlass::arch::Sm50;

  using AddOp            = cuasr::binary_or<precision>;
  using MultOp           = cuasr::binary_and<precision>;

  using Srgemm = cuasr::gemm::device::Srgemm<                           //
      AddOp, MultOp,                                                    //
      precision, cutlass::layout::RowMajor,                   //
      precision, cutlass::layout::RowMajor,                   //
      precision, cutlass::layout::RowMajor,
      precision, OpClass, SmArch>;

  EXPECT_TRUE(cuasr::test::gemm::device::TestAllGemm<Srgemm>());
}


///////////////////////////////////////////////////////////////////////////////

TEST(SM50_device_binary_or_binary_and_ssrgemm_tn_n, default) {
  using precision = float;
  using OpClass   = cutlass::arch::OpClassSimt;
  using SmArch    = cutlass::arch::Sm50;

  using AddOp            = cuasr::binary_or<precision>;
  using MultOp           = cuasr::binary_and<precision>;

  using Srgemm = cuasr::gemm::device::Srgemm<                           //
      AddOp, MultOp,                                                    //
      precision, cutlass::layout::RowMajor,                   //
      precision, cutlass::layout::ColumnMajor,                   //
      precision, cutlass::layout::ColumnMajor,
      precision, OpClass, SmArch>;

  EXPECT_TRUE(cuasr::test::gemm::device::TestAllGemm<Srgemm>());
}


///////////////////////////////////////////////////////////////////////////////

TEST(SM50_device_binary_or_binary_and_ssrgemm_tn_t, default) {
  using precision = float;
  using OpClass   = cutlass::arch::OpClassSimt;
  using SmArch    = cutlass::arch::Sm50;

  using AddOp            = cuasr::binary_or<precision>;
  using MultOp           = cuasr::binary_and<precision>;

  using Srgemm = cuasr::gemm::device::Srgemm<                           //
      AddOp, MultOp,                                                    //
      precision, cutlass::layout::RowMajor,                   //
      precision, cutlass::layout::ColumnMajor,                   //
      precision, cutlass::layout::RowMajor,
      precision, OpClass, SmArch>;

  EXPECT_TRUE(cuasr::test::gemm::device::TestAllGemm<Srgemm>());
}


///////////////////////////////////////////////////////////////////////////////

TEST(SM50_device_binary_or_binary_and_ssrgemm_nt_n, default) {
  using precision = float;
  using OpClass   = cutlass::arch::OpClassSimt;
  using SmArch    = cutlass::arch::Sm50;

  using AddOp            = cuasr::binary_or<precision>;
  using MultOp           = cuasr::binary_and<precision>;

  using Srgemm = cuasr::gemm::device::Srgemm<                           //
      AddOp, MultOp,                                                    //
      precision, cutlass::layout::ColumnMajor,                   //
      precision, cutlass::layout::RowMajor,                   //
      precision, cutlass::layout::ColumnMajor,
      precision, OpClass, SmArch>;

  EXPECT_TRUE(cuasr::test::gemm::device::TestAllGemm<Srgemm>());
}


///////////////////////////////////////////////////////////////////////////////

TEST(SM50_device_binary_or_binary_and_ssrgemm_nt_t, default) {
  using precision = float;
  using OpClass   = cutlass::arch::OpClassSimt;
  using SmArch    = cutlass::arch::Sm50;

  using AddOp            = cuasr::binary_or<precision>;
  using MultOp           = cuasr::binary_and<precision>;

  using Srgemm = cuasr::gemm::device::Srgemm<                           //
      AddOp, MultOp,                                                    //
      precision, cutlass::layout::ColumnMajor,                   //
      precision, cutlass::layout::RowMajor,                   //
      precision, cutlass::layout::RowMajor,
      precision, OpClass, SmArch>;

  EXPECT_TRUE(cuasr::test::gemm::device::TestAllGemm<Srgemm>());
}


///////////////////////////////////////////////////////////////////////////////

TEST(SM50_device_binary_or_binary_and_ssrgemm_nn_n, default) {
  using precision = float;
  using OpClass   = cutlass::arch::OpClassSimt;
  using SmArch    = cutlass::arch::Sm50;

  using AddOp            = cuasr::binary_or<precision>;
  using MultOp           = cuasr::binary_and<precision>;

  using Srgemm = cuasr::gemm::device::Srgemm<                           //
      AddOp, MultOp,                                                    //
      precision, cutlass::layout::ColumnMajor,                   //
      precision, cutlass::layout::ColumnMajor,                   //
      precision, cutlass::layout::ColumnMajor,
      precision, OpClass, SmArch>;

  EXPECT_TRUE(cuasr::test::gemm::device::TestAllGemm<Srgemm>());
}


///////////////////////////////////////////////////////////////////////////////

TEST(SM50_device_binary_or_binary_and_ssrgemm_nn_t, default) {
  using precision = float;
  using OpClass   = cutlass::arch::OpClassSimt;
  using SmArch    = cutlass::arch::Sm50;

  using AddOp            = cuasr::binary_or<precision>;
  using MultOp           = cuasr::binary_and<precision>;

  using Srgemm = cuasr::gemm::device::Srgemm<                           //
      AddOp, MultOp,                                                    //
      precision, cutlass::layout::ColumnMajor,                   //
      precision, cutlass::layout::ColumnMajor,                   //
      precision, cutlass::layout::RowMajor,
      precision, OpClass, SmArch>;

  EXPECT_TRUE(cuasr::test::gemm::device::TestAllGemm<Srgemm>());
}

