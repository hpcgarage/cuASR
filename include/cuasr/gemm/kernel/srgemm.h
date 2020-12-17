/***************************************************************************************************
 * Copyright (c) 2020, Vijay Thakkar (thakkarv@gatech.edu).  All rights reserved.
 **************************************************************************************************/
/*! \file
    \brief Template for a pipelined Semiring GEMM kernel. Does not compute batching or support split-K.
*/

#pragma once

#include "cutlass/cutlass.h"

#include "cutlass/gemm/gemm.h"
#include "cutlass/matrix_coord.h"
#include "cutlass/semaphore.h"

#include "cuasr/arch/srmma.h"

/////////////////////////////////////////////////////////////////////////////////////////////////

namespace cuasr {
namespace gemm {
namespace kernel {

/////////////////////////////////////////////////////////////////////////////////////////////////

// SemiRing Gemm kernel that support custom thread level MMA and init values.
template <
  typename Srmma_,                ///! Threadblock-scoped matrix multiply-accumulate
  typename AdditionOp_,           ///! Addition operator of the semi-ring
  typename MultiplicationOp_,     ///! Multiplication operator of the semi-ring
  typename Epilogue_,             ///! Epilogue
  typename ThreadblockSwizzle_,   ///! Threadblock swizzling function
  bool SplitKSerial               ///! If true, code supporting split-K via serial reduction is enabled.
>
struct Srgemm {

  using Srmma = Srmma_;
  using Epilogue = Epilogue_;
  using OutputOp = typename Epilogue::OutputOp;
  using ThreadblockSwizzle = ThreadblockSwizzle_;
  using AdditionOp = AdditionOp_;
  using MultiplicationOp = MultiplicationOp_;
  static bool const kSplitKSerial = SplitKSerial;

  /// Warp count (concept: GemmShape)
  using WarpCount = typename Srmma::WarpCount;
  static int const kThreadCount = 32 * WarpCount::kCount;

  /// Parameters structure
  struct Params {
    cutlass::gemm::GemmCoord problem_size;
    cutlass::gemm::GemmCoord grid_tiled_shape;
    typename Srmma::IteratorA::Params params_A;
    typename Srmma::IteratorA::TensorRef ref_A;
    typename Srmma::IteratorB::Params params_B;
    typename Srmma::IteratorB::TensorRef ref_B;
    typename Epilogue::OutputTileIterator::Params params_C;
    typename Epilogue::OutputTileIterator::TensorRef ref_C;
    typename Epilogue::OutputTileIterator::Params params_D;
    typename Epilogue::OutputTileIterator::TensorRef ref_D;
    typename OutputOp::Params output_op;
    int *semaphore;
    int gemm_k_iterations;
    int gemm_k_size;

    //
    // Methods
    //

    CUTLASS_HOST_DEVICE
    Params() { }

    CUTLASS_HOST_DEVICE
    Params(
      cutlass::gemm::GemmCoord const & problem_size,
      cutlass::gemm::GemmCoord const & grid_tiled_shape,
      typename Srmma::IteratorA::TensorRef ref_A,
      typename Srmma::IteratorB::TensorRef ref_B,
      typename Epilogue::OutputTileIterator::TensorRef ref_C,
      typename Epilogue::OutputTileIterator::TensorRef ref_D,
      typename OutputOp::Params output_op = typename OutputOp::Params(),
      int *semaphore = nullptr
    ):
      problem_size(problem_size),
      grid_tiled_shape(grid_tiled_shape),
      params_A(ref_A.layout()),
      ref_A(ref_A),
      params_B(ref_B.layout()),
      ref_B(ref_B),
      params_C(ref_C.layout()),
      ref_C(ref_C),
      params_D(ref_D.layout()),
      ref_D(ref_D),
      output_op(output_op),
      semaphore(semaphore) {

      int total_gemm_k_iterations = (problem_size.k() + Srmma::Shape::kK - 1) / Srmma::Shape::kK;
      int gemm_k_iterations = (total_gemm_k_iterations + grid_tiled_shape.k() - 1) / grid_tiled_shape.k();

      gemm_k_size = gemm_k_iterations * Srmma::Shape::kK;
    }
  };

  /// Shared memory storage structure
  union SharedStorage {
    typename Srmma::SharedStorage main_loop;
    typename Epilogue::SharedStorage epilogue;
  };

  //
  // Methods
  //

  CUTLASS_HOST_DEVICE
  Srgemm() { }

  /// Determines whether kernel satisfies alignment
    static cutlass::Status can_implement(
      cutlass::gemm::GemmCoord const & problem_size,
      typename Srmma::IteratorA::TensorRef ref_A,
      typename Srmma::IteratorB::TensorRef ref_B,
      typename Epilogue::OutputTileIterator::TensorRef ref_C,
      typename Epilogue::OutputTileIterator::TensorRef ref_D) {

    static int const kAlignmentA = Srmma::IteratorA::AccessType::kElements;
    static int const kAlignmentB = Srmma::IteratorB::AccessType::kElements;
    static int const kAlignmentC = Epilogue::OutputTileIterator::kElementsPerAccess;

    if (!TensorRef_aligned(ref_A, kAlignmentA)) {
      return cutlass::Status::kErrorMisalignedOperand;
    }

    if (!TensorRef_aligned(ref_B, kAlignmentB)) {
      return cutlass::Status::kErrorMisalignedOperand;
    }

    if (!TensorRef_aligned(ref_C, kAlignmentC)) {
      return cutlass::Status::kErrorMisalignedOperand;
    }

    if (!TensorRef_aligned(ref_D, kAlignmentC)) {
      return cutlass::Status::kErrorMisalignedOperand;
    }

    if ((problem_size.m() % kAlignmentA) || (problem_size.k() % kAlignmentA) ||
      (problem_size.n() % kAlignmentB) || (problem_size.k() % kAlignmentB) ||
      (problem_size.m() % kAlignmentC) || (problem_size.n() % kAlignmentC)) {

      return cutlass::Status::kErrorMisalignedOperand;
    }

    return cutlass::Status::kSuccess;
  }

  /// Executes one GEMM
  CUTLASS_DEVICE
  void operator()(Params const &params, SharedStorage &shared_storage) {
    constexpr typename OutputOp::ElementCompute kAdditiveIdentity = AdditionOp::Identity;

    // Compute threadblock location
    ThreadblockSwizzle threadblock_swizzle;

    cutlass::gemm::GemmCoord threadblock_tile_offset = threadblock_swizzle.get_tile_offset(params.grid_tiled_shape);

    // Early exit if CTA is out of range
    if (params.grid_tiled_shape.m() <= threadblock_tile_offset.m() ||
      params.grid_tiled_shape.n() <= threadblock_tile_offset.n()) {

      return;
    }

    // Compute initial location in logical coordinates
    cutlass::MatrixCoord tb_offset_A{
      threadblock_tile_offset.m() * Srmma::Shape::kM,
      threadblock_tile_offset.k() * params.gemm_k_size,
    };

    cutlass::MatrixCoord tb_offset_B{
      threadblock_tile_offset.k() * params.gemm_k_size,
      threadblock_tile_offset.n() * Srmma::Shape::kN
    };

    // Problem size is a function of threadblock index in the K dimension
    int problem_size_k = min(
      params.problem_size.k(),
      (threadblock_tile_offset.k() + 1) * params.gemm_k_size);

    // Compute threadblock-scoped matrix multiply-add
    int gemm_k_iterations = (problem_size_k - tb_offset_A.column() + Srmma::Shape::kK - 1) / Srmma::Shape::kK;

    // Compute position within threadblock
    int thread_idx = threadIdx.x;

    // Construct iterators to A and B operands
    typename Srmma::IteratorA iterator_A(
      params.params_A,
      params.ref_A.data(),
      {params.problem_size.m(), problem_size_k},
      thread_idx,
      tb_offset_A);

    typename Srmma::IteratorB iterator_B(
      params.params_B,
      params.ref_B.data(),
      {problem_size_k, params.problem_size.n()},
      thread_idx,
      tb_offset_B);

    int warp_idx = threadIdx.x / 32;
    int lane_idx = threadIdx.x % 32;

    //
    // Main loop
    //

    // Construct thread-scoped matrix multiply
    Srmma srmma_thrblock_op(shared_storage.main_loop, thread_idx, warp_idx, lane_idx, kAdditiveIdentity);

    // need to clear accumulators to additive identity for SemiRing Gemm
    typename Srmma::FragmentC accumulators;
    accumulators.fill(kAdditiveIdentity);

    if (!kSplitKSerial || gemm_k_iterations > 0) {
      // Compute threadblock-scoped matrix multiply-add
      srmma_thrblock_op(gemm_k_iterations, accumulators, iterator_A, iterator_B, accumulators);
    }

    //
    // Epilogue
    //

    OutputOp output_op(params.output_op);

    //
    // Masked tile iterators constructed from members
    //

    threadblock_tile_offset = threadblock_swizzle.get_tile_offset(params.grid_tiled_shape);

    //assume identity swizzle
    cutlass::MatrixCoord threadblock_offset(
      threadblock_tile_offset.m() * Srmma::Shape::kM,
      threadblock_tile_offset.n() * Srmma::Shape::kN
    );

    int block_idx = threadblock_tile_offset.m() + threadblock_tile_offset.n() * params.grid_tiled_shape.m();

    // Construct the semaphore.
    cutlass::Semaphore semaphore(params.semaphore + block_idx, thread_idx);

    // If performing a reduction via split-K, fetch the initial synchronization
    if (kSplitKSerial && params.grid_tiled_shape.k() > 1) {

      // Fetch the synchronization lock initially but do not block.
      semaphore.fetch();

      // Indicate which position in a serial reduction the output operator is currently updating
      output_op.set_k_partition(threadblock_tile_offset.k());
    }

    // Tile iterator loading from source tensor.
    typename Epilogue::OutputTileIterator iterator_C(
      params.params_C,
      params.ref_C.data(),
      params.problem_size.mn(),
      thread_idx,
      threadblock_offset
    );

    // Tile iterator writing to destination tensor.
    typename Epilogue::OutputTileIterator iterator_D(
      params.params_D,
      params.ref_D.data(),
      params.problem_size.mn(),
      thread_idx,
      threadblock_offset
    );

    Epilogue epilogue(
      shared_storage.epilogue,
      thread_idx,
      warp_idx,
      lane_idx);

    // Wait on the semaphore - this latency may have been covered by iterator construction
    if (kSplitKSerial && params.grid_tiled_shape.k() > 1) {

      // For subsequent threadblocks, the source matrix is held in the 'D' tensor.
      if (threadblock_tile_offset.k()) {
        iterator_C = iterator_D;
      }

      semaphore.wait(threadblock_tile_offset.k());

      __threadfence();
    }

    // Execute the epilogue operator to update the destination tensor.
    epilogue(output_op, iterator_D, accumulators, iterator_C);

    //
    // Release the semaphore
    //

    if (kSplitKSerial && params.grid_tiled_shape.k() > 1) {

      int lock = 0;
      if (params.grid_tiled_shape.k() == threadblock_tile_offset.k() + 1) {

        // The final threadblock resets the semaphore for subsequent grids.
        lock = 0;
      }
      else {
        // Otherwise, the semaphore is incremented
        lock = threadblock_tile_offset.k() + 1;
      }

      __threadfence();
      semaphore.release(lock);
    }
  }
};


/////////////////////////////////////////////////////////////////////////////////////////////////

} // namespace kernel
} // namespace gemm
} // namespace cuasr

/////////////////////////////////////////////////////////////////////////////////////////////////
