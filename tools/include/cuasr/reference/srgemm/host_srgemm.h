#pragma once

#include "cutlass/coord.h"
#include "cutlass/functional.h"
#include "cutlass/numeric_conversion.h"
#include "cutlass/numeric_types.h"

#include "cutlass/arch/mma.h"
#include "cutlass/gemm/gemm.h"
#include "cutlass/tensor_view.h"
#include "cutlass/util/host_tensor.h"

namespace cuasr {
namespace reference {
namespace host {

/// Host side SemiRing GEMM for rank-2 tensors for testing.
template <
    typename AdditionOp,
    typename MultiplicationOp,
    typename ElementA,
    typename LayoutA,
    typename ElementB,
    typename LayoutB,
    typename ElementC,
    typename LayoutC,
    typename ScalarType,
    typename ComputeType,
    typename EpilogueOp,
    typename ConvertOp = cutlass::NumericConverter<ElementC, ScalarType>>
struct Srgemm {
public:
  void operator()(
      cutlass::gemm::GemmCoord problem_size,
      ComputeType alpha,
      cutlass::TensorRef<ElementA, LayoutA> tensor_a,
      cutlass::TensorRef<ElementB, LayoutB> tensor_b,
      ComputeType beta,
      cutlass::TensorRef<ElementC, LayoutC> tensor_c,
      cutlass::TensorRef<ElementC, LayoutC> tensor_d,
      ComputeType add_identity) {
    static_assert(
        LayoutA::kRank == 2 && LayoutB::kRank == 2 && LayoutC::kRank == 2,
        "Tensors must be of rank 2");

// use OMP to speed up host side reference GEMM if we can
#pragma omp parallel proc_bind(spread) firstprivate(                                     \
    problem_size, tensor_a, tensor_b, tensor_c, tensor_d, add_identity, do_epilogue_add, \
    alpha, beta)
    {
      int const M = problem_size.m();
      int const N = problem_size.n();
      int const K = problem_size.k();

      // Blocking necessary to speedup reference implementation
      constexpr int Mblock = 32;
      constexpr int Nblock = 32;

      ConvertOp convert_op;
      AdditionOp add_op;
      MultiplicationOp mult_op;

#pragma omp for schedule(static) collapse(2)
      for (int row_block = 0; row_block < M; row_block += Mblock) {
        for (int col_block = 0; col_block < N; col_block += Nblock) {
          // init registers
          ComputeType accum[Mblock][Nblock];
          for (int j = 0; j < Nblock; j++) {
            for (int i = 0; i < Mblock; i++) {
              accum[i][j] = add_identity;
            }
          }

          // main loop over k-dim
          for (int k_block = 0; k_block < K; ++k_block) {
            for (int j = 0; j < Nblock; j++) {
              for (int i = 0; i < Mblock; i++) {
                int row = row_block + i;
                int col = col_block + j;
                if (row < M && col < N) {
                  ElementA a = tensor_a.at(cutlass::MatrixCoord(row, k_block));
                  ElementB b = tensor_b.at(cutlass::MatrixCoord(k_block, col));

                  ComputeType compute_a(static_cast<ComputeType>(a));
                  ComputeType compute_b(static_cast<ComputeType>(b));

                  accum[i][j] = add_op(mult_op(compute_a, compute_b), accum[i][j]);
                }
              }
            }
          }

          // perform epilogue operator
          for (int j = 0; j < Nblock; j++) {
            for (int i = 0; i < Mblock; i++) {
              int row = row_block + i;
              int col = col_block + j;
              cutlass::MatrixCoord coord(row, col);
              if (row < M && col < N) {
                auto c             = tensor_c.at(coord);
                tensor_d.at(coord) = convert_op(     //
                    add_op(                          //
                        mult_op(alpha, accum[i][j]), //
                        mult_op(beta, c)             //
                        )                            //
                );
              }
            }
          }
        }
      } // #pragma omp for
    }   // #pragma omp parallel
  }
};

} // namespace reference
} // namespace host
} // namespace cuasr
