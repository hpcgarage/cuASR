/***************************************************************************************************
 * Copyright (c) 2020, Vijay Thakkar (thakkarv@gatech.edu).
 **************************************************************************************************/
/*! \file
    \brief Template for a pipelined Semiring GEMM kernel. Does not compute batching or support split-K.
*/

#pragma once

#include "cutlass/cutlass.h"
#include "cutlass/arch/arch.h"
#include "cutlass/device_kernel.h"
#include "cutlass/gemm/threadblock/threadblock_swizzle.h"
#include "cutlass/numeric_types.h"

#include "cuasr/gemm/device/default_srgemm_configuration.h"
#include "cuasr/gemm/kernel/default_srgemm.h"
#include "cuasr/gemm/kernel/srgemm.h"

////////////////////////////////////////////////////////////////////////////////

namespace cuasr {
namespace gemm {
namespace device {

////////////////////////////////////////////////////////////////////////////////

template <
  /// Addition operator of the semi-ring
  typename AdditionOp_,
  /// Multiplication operator of the semi-ring
  typename MultiplicationOp_,
  /// Element type for A matrix operand
  typename ElementA_,
  /// Layout type for A matrix operand
  typename LayoutA_,
  /// Element type for B matrix operand
  typename ElementB_,
  /// Layout type for B matrix operand
  typename LayoutB_,
  /// Element type for C and D matrix operands
  typename ElementC_,
  /// Layout type for C and D matrix operands
  typename LayoutC_,
  /// Element type for internal accumulation
  typename ElementAccumulator_ = ElementC_,
  /// Operator class tag
  typename OperatorClass_ = cutlass::arch::OpClassSimt,
  /// Tag indicating architecture to tune for
  typename ArchTag_ = cutlass::arch::Sm50,
  /// Threadblock-level tile size (concept: GemmShape)
  typename ThreadblockShape_ = typename DefaultSemiRingConfiguration<
      ElementA_, ElementB_, ElementC_, ElementAccumulator_,
      OperatorClass_, AdditionOp_, MultiplicationOp_, ArchTag_>::ThreadblockShape,
  /// Warp-level tile size (concept: GemmShape)
  typename WarpShape_ = typename DefaultSemiRingConfiguration<
      ElementA_, ElementB_, ElementC_, ElementAccumulator_,
      OperatorClass_, AdditionOp_, MultiplicationOp_, ArchTag_>::WarpShape,
  /// Instruction-level tile size (concept: GemmShape)
  typename InstructionShape_ = typename DefaultSemiRingConfiguration<
      ElementA_, ElementB_, ElementC_, ElementAccumulator_,
      OperatorClass_, AdditionOp_, MultiplicationOp_, ArchTag_>::InstructionShape,
  /// Epilogue output operator
  typename EpilogueOutputOp_ = typename DefaultSemiRingConfiguration<
      ElementA_, ElementB_, ElementC_, ElementAccumulator_,
      OperatorClass_, AdditionOp_, MultiplicationOp_, ArchTag_>::EpilogueOutputOp,
  /// Threadblock-level swizzling operator
  typename ThreadblockSwizzle_ =
      typename cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>,
  /// Number of stages used in the pipelined mainloop
  int Stages = DefaultSemiRingConfiguration<
      ElementA_, ElementB_, ElementC_, ElementAccumulator_,
      OperatorClass_, AdditionOp_, MultiplicationOp_, ArchTag_>::kStages,
  /// Access granularity of A matrix in units of elements
  int AlignmentA = DefaultSemiRingConfiguration<
      ElementA_, ElementB_, ElementC_, ElementAccumulator_,
      OperatorClass_, AdditionOp_, MultiplicationOp_, ArchTag_>::kAlignmentA,
  /// Access granularity of B matrix in units of elements
  int AlignmentB = DefaultSemiRingConfiguration<
      ElementA_, ElementB_, ElementC_, ElementAccumulator_,
      OperatorClass_, AdditionOp_, MultiplicationOp_, ArchTag_>::kAlignmentB,
  /// If true, kernel supports split-K with serial reduction
  bool SplitKSerial = false
>
class Srgemm {
 public:

  using ElementA = ElementA_;
  using LayoutA = LayoutA_;
  using TensorRefA = cutlass::TensorRef<ElementA const, LayoutA>;
  using ElementB = ElementB_;
  using LayoutB = LayoutB_;
  using TensorRefB = cutlass::TensorRef<ElementB const, LayoutB>;
  using ElementC = ElementC_;
  using LayoutC = LayoutC_;
  using TensorRefC = cutlass::TensorRef<ElementC const, LayoutC>;
  using TensorRefD = cutlass::TensorRef<ElementC, LayoutC>;
  using ElementAccumulator = ElementAccumulator_;
  using OperatorClass = OperatorClass_;
  using ArchTag = ArchTag_;
  using ThreadblockShape = ThreadblockShape_;
  using WarpShape = WarpShape_;
  using InstructionShape = InstructionShape_;
  using EpilogueOutputOp = EpilogueOutputOp_;
  using ThreadblockSwizzle = ThreadblockSwizzle_;
  using AdditionOp = AdditionOp_;
  using MultiplicationOp = MultiplicationOp_;
  static int const kStages = Stages;
  static int const kAlignmentA = AlignmentA;
  static int const kAlignmentB = AlignmentB;
  static int const kAlignmentC = EpilogueOutputOp::kCount;
  static bool const kSplitKSerial = SplitKSerial;

  /// Define the kernel
  using SrgemmKernel = typename cuasr::gemm::kernel::DefaultSrgemm<
    ElementA,
    LayoutA,
    kAlignmentA,
    ElementB,
    LayoutB,
    kAlignmentB,
    ElementC,
    LayoutC,
    ElementAccumulator,
    OperatorClass,
    ArchTag,
    ThreadblockShape,
    WarpShape,
    InstructionShape,
    AdditionOp,
    MultiplicationOp,
    EpilogueOutputOp,
    ThreadblockSwizzle,
    kStages,
    kSplitKSerial
  >::SrgemmKernel;

  /// Argument structure
  struct Arguments {

    //
    // Data members
    //

    cutlass::gemm::GemmCoord problem_size;
    cutlass::TensorRef<ElementA const, LayoutA> ref_A;
    cutlass::TensorRef<ElementB const, LayoutB> ref_B;
    cutlass::TensorRef<ElementC const, LayoutC> ref_C;
    cutlass::TensorRef<ElementC, LayoutC> ref_D;
    typename EpilogueOutputOp::Params epilogue;
    int split_k_slices;

    //
    // Methods
    //

    /// Default ctor
    CUTLASS_HOST_DEVICE
    Arguments(): problem_size(0, 0, 0), split_k_slices(1) {

    }

    /// Constructs an Arguments structure
    CUTLASS_HOST_DEVICE
    Arguments(
      cutlass::gemm::GemmCoord problem_size_,
      cutlass::TensorRef<ElementA const, LayoutA> ref_A_,
      cutlass::TensorRef<ElementB const, LayoutB> ref_B_,
      cutlass::TensorRef<ElementC const, LayoutC> ref_C_,
      cutlass::TensorRef<ElementC, LayoutC> ref_D_,
      typename EpilogueOutputOp::Params epilogue_,
      int split_k_slices = 1
    ):
      problem_size(problem_size_),
      ref_A(ref_A_),
      ref_B(ref_B_),
      ref_C(ref_C_),
      ref_D(ref_D_),
      epilogue(epilogue_),
      split_k_slices(split_k_slices) {

    }
  };

private:

  /// Kernel parameters object
  typename SrgemmKernel::Params params_;

public:

  /// Constructs the GEMM.
  Srgemm() { }

  /// Determines whether the GEMM can execute the given problem.
  static cutlass::Status can_implement(Arguments const &args) {

    if (!kSplitKSerial && args.split_k_slices > 1) {
      return cutlass::Status::kErrorInvalidProblem;
    }

    cutlass::Status status = SrgemmKernel::can_implement(
      args.problem_size,
      args.ref_A.non_const_ref(),
      args.ref_B.non_const_ref(),
      args.ref_C.non_const_ref(),
      args.ref_D
    );

    if (status != cutlass::Status::kSuccess) {
      return status;
    }

    return cutlass::Status::kSuccess;
  }

  /// Gets the workspace size
  static size_t get_workspace_size(Arguments const &args) {

    if (kSplitKSerial && args.split_k_slices > 1) {

      // Determine grid shape
      ThreadblockSwizzle threadblock_swizzle;

      cutlass::gemm::GemmCoord tiled_shape = threadblock_swizzle.get_tiled_shape(
        args.problem_size,
        {ThreadblockShape::kM, ThreadblockShape::kN, ThreadblockShape::kK},
        args.split_k_slices);

      return sizeof(int) * size_t(tiled_shape.m()) * size_t(tiled_shape.n());
    }

    return 0;
  }

  /// Initializes GEMM state from arguments.
  cutlass::Status initialize(Arguments const &args, void *workspace = nullptr, cudaStream_t stream = nullptr) {

    // Determine grid shape
    ThreadblockSwizzle threadblock_swizzle;

    cutlass::gemm::GemmCoord grid_shape = threadblock_swizzle.get_tiled_shape(
      args.problem_size,
      {ThreadblockShape::kM, ThreadblockShape::kN, ThreadblockShape::kK},
      args.split_k_slices);

    if (kSplitKSerial) {
      if (args.split_k_slices > 1) {
        if (!workspace) {
          return cutlass::Status::kErrorWorkspaceNull;
        }

        size_t bytes = get_workspace_size(args);

        cudaError_t result = cudaMemsetAsync(workspace, 0, bytes, stream);

        if (result != cudaSuccess) {
          return cutlass::Status::kErrorInternal;
        }
      }
    }
    else {

      if (args.split_k_slices > 1) {
        return cutlass::Status::kErrorInvalidProblem;
      }
    }

    // Initialize the Params structure
    params_ = typename SrgemmKernel::Params{
      args.problem_size,
      grid_shape,
      args.ref_A.non_const_ref(),
      args.ref_B.non_const_ref(),
      args.ref_C.non_const_ref(),
      args.ref_D,
      args.epilogue,
      static_cast<int *>(workspace)
    };

    return cutlass::Status::kSuccess;
  }

  /// Lightweight update given a subset of arguments
  cutlass::Status update(Arguments const &args, void *workspace = nullptr) {

    if (kSplitKSerial && args.split_k_slices > 1) {
      if (!workspace) {
        return cutlass::Status::kErrorWorkspaceNull;
      }
    }

    params_.ref_A.reset(args.ref_A.non_const_ref().data());
    params_.ref_B.reset(args.ref_B.non_const_ref().data());
    params_.ref_C.reset(args.ref_C.non_const_ref().data());
    params_.ref_D.reset(args.ref_D.data());
    params_.semaphore = static_cast<int *>(workspace);

    return cutlass::Status::kSuccess;
  }

  /// Runs the kernel using initialized state.
  cutlass::Status run(cudaStream_t stream = nullptr) {

    ThreadblockSwizzle threadblock_swizzle;

    dim3 grid = threadblock_swizzle.get_grid_shape(params_.grid_tiled_shape);
    dim3 block(SrgemmKernel::kThreadCount, 1, 1);

    cudaError_t result;

    int smem_size = int(sizeof(typename SrgemmKernel::SharedStorage));
    if (smem_size >= (48 << 10)) {
      result = cudaFuncSetAttribute(cutlass::Kernel<SrgemmKernel>,
                                    cudaFuncAttributeMaxDynamicSharedMemorySize,
                                    smem_size);

      if (result != cudaSuccess) {
        return cutlass::Status::kErrorInternal;
      }

      result = cudaFuncSetAttribute(
          cutlass::Kernel<SrgemmKernel>,
          cudaFuncAttributePreferredSharedMemoryCarveout, 100);

      if (result != cudaSuccess) {
        return cutlass::Status::kErrorInternal;
      }
    }

    cutlass::Kernel<SrgemmKernel><<<grid, block, smem_size, stream>>>(params_);

    result = cudaGetLastError();

    return result == cudaSuccess ? cutlass::Status::kSuccess : cutlass::Status::kErrorInternal;
  }

  /// Runs the kernel using initialized state.
  cutlass::Status operator()(cudaStream_t stream = nullptr) {
    return run(stream);
  }

  /// Runs the kernel using initialized state.
  cutlass::Status operator()(
    Arguments const &args,
    void *workspace = nullptr,
    cudaStream_t stream = nullptr) {

    cutlass::Status status = initialize(args, workspace);

    if (status == cutlass::Status::kSuccess) {
      status = run(stream);
    }

    return status;
  }
};

template <
    /// Addition operator of the semi-ring
    typename AdditionOp_,
    /// Multiplication operator of the semi-ring
    typename MultiplicationOp_,
    /// Element type for A matrix operand
    typename ElementA_,
    /// Layout type for A matrix operand
    typename LayoutA_,
    /// Element type for B matrix operand
    typename ElementB_,
    /// Layout type for B matrix operand
    typename LayoutB_,
    /// Element type for C and D matrix operands
    typename ElementC_,
    /// Element type for internal accumulation
    typename ElementAccumulator_,
    /// Operator class tag
    typename OperatorClass_,
    /// Tag indicating architecture to tune for
    typename ArchTag_,
    /// Threadblock-level tile size (concept: GemmShape)
    typename ThreadblockShape_,
    /// Warp-level tile size (concept: GemmShape)
    typename WarpShape_,
    /// Instruction-level tile size (concept: GemmShape)
    typename InstructionShape_,
    /// Epilogue output operator
    typename EpilogueOutputOp_,
    /// Threadblock-level swizzling operator
    typename ThreadblockSwizzle_,
    /// Number of stages used in the pipelined mainloop
    int Stages,
    /// Access granularity of A matrix in units of elements
    int AlignmentA,
    /// Access granularity of B matrix in units of elements
    int AlignmentB,
    /// If true, kernel supports split-K as a serial reduction
    bool SplitKSerial>
class Srgemm<AdditionOp_, MultiplicationOp_,
            ElementA_, LayoutA_, ElementB_, LayoutB_, ElementC_,
            cutlass::layout::ColumnMajor,  // partially specialized on LayoutC
            ElementAccumulator_, OperatorClass_, ArchTag_, ThreadblockShape_,
            WarpShape_, InstructionShape_, EpilogueOutputOp_,
            ThreadblockSwizzle_, Stages, AlignmentA, AlignmentB, SplitKSerial
            > {
 public:

  using ElementA = ElementA_;
  using LayoutA = LayoutA_;
  using TensorRefA = cutlass::TensorRef<ElementA const, LayoutA>;
  using ElementB = ElementB_;
  using LayoutB = LayoutB_;
  using TensorRefB = cutlass::TensorRef<ElementB const, LayoutB>;
  using ElementC = ElementC_;
  using LayoutC = cutlass::layout::ColumnMajor;
  using TensorRefC = cutlass::TensorRef<ElementC const, LayoutC>;
  using TensorRefD = cutlass::TensorRef<ElementC, LayoutC>;
  using ElementAccumulator = ElementAccumulator_;
  using OperatorClass = OperatorClass_;
  using ArchTag = ArchTag_;
  using ThreadblockShape = ThreadblockShape_;
  using WarpShape = WarpShape_;
  using InstructionShape = InstructionShape_;
  using EpilogueOutputOp = EpilogueOutputOp_;
  using ThreadblockSwizzle = ThreadblockSwizzle_;
  using AdditionOp = AdditionOp_;
  using MultiplicationOp = MultiplicationOp_;
  static int const kStages = Stages;
  static int const kAlignmentA = AlignmentA;
  static int const kAlignmentB = AlignmentB;
  static bool const kSplitKSerial = SplitKSerial;

  using UnderlyingOperator = Srgemm<
    AdditionOp,
    MultiplicationOp,
    ElementB,
    typename cutlass::layout::LayoutTranspose<LayoutB>::type,
    ElementA,
    typename cutlass::layout::LayoutTranspose<LayoutA>::type,
    ElementC,
    cutlass::layout::RowMajor,
    ElementAccumulator,
    OperatorClass,
    ArchTag,
    ThreadblockShape,
    WarpShape,
    InstructionShape,
    EpilogueOutputOp,
    ThreadblockSwizzle,
    Stages,
    kAlignmentB,
    kAlignmentA,
    SplitKSerial
  >;


  using UnderlyingArguments = typename UnderlyingOperator::Arguments;
  using SrgemmKernel = typename UnderlyingOperator::SrgemmKernel;
  static int const kAlignmentC = UnderlyingOperator::kAlignmentC;

  /// Argument structure
  struct Arguments {

    //
    // Data members
    //

    cutlass::gemm::GemmCoord problem_size;
    cutlass::TensorRef<ElementA const, LayoutA> ref_A;
    cutlass::TensorRef<ElementB const, LayoutB> ref_B;
    cutlass::TensorRef<ElementC const, LayoutC> ref_C;
    cutlass::TensorRef<ElementC, LayoutC> ref_D;
    typename EpilogueOutputOp::Params epilogue;
    int split_k_slices;

    //
    // Methods
    //

    /// Default ctor
    CUTLASS_HOST_DEVICE
    Arguments() { }

    /// Constructs an Arguments structure
    CUTLASS_HOST_DEVICE
    Arguments(
      cutlass::gemm::GemmCoord problem_size_,
      cutlass::TensorRef<ElementA const, LayoutA> ref_A_,
      cutlass::TensorRef<ElementB const, LayoutB> ref_B_,
      cutlass::TensorRef<ElementC const, LayoutC> ref_C_,
      cutlass::TensorRef<ElementC, LayoutC> ref_D_,
      typename EpilogueOutputOp::Params epilogue_,
      int split_k_slices = 1
    ):
      problem_size(problem_size_),
      ref_A(ref_A_),
      ref_B(ref_B_),
      ref_C(ref_C_),
      ref_D(ref_D_),
      epilogue(epilogue_),
      split_k_slices(split_k_slices) { }
  };

private:

  UnderlyingOperator underlying_operator_;

public:

  /// Constructs the GEMM.
  Srgemm() { }

  /// Helper to construct a transposed equivalent for the underying GEMM operator
  static UnderlyingArguments to_underlying_arguments(Arguments const &args) {
    return UnderlyingArguments(
      {args.problem_size.n(), args.problem_size.m(), args.problem_size.k()},
      {args.ref_B.data(), args.ref_B.stride(0)},
      {args.ref_A.data(), args.ref_A.stride(0)},
      {args.ref_C.data(), args.ref_C.stride(0)},
      {args.ref_D.data(), args.ref_D.stride(0)},
      args.epilogue,
      args.split_k_slices
    );
  }

  /// Determines whether the GEMM can execute the given problem.
  static cutlass::Status can_implement(Arguments const &args) {
    return UnderlyingOperator::can_implement(to_underlying_arguments(args));
  }

  /// Gets the workspace size
  static size_t get_workspace_size(Arguments const &args) {
    return UnderlyingOperator::get_workspace_size(to_underlying_arguments(args));
  }

  /// Initializes GEMM state from arguments.
  cutlass::Status initialize(Arguments const &args, void *workspace = nullptr, cudaStream_t stream = nullptr) {
    return underlying_operator_.initialize(to_underlying_arguments(args), workspace);
  }

  /// Lightweight update given a subset of arguments
  cutlass::Status update(Arguments const &args, void *workspace = nullptr) {
    return underlying_operator_.update(to_underlying_arguments(args), workspace);
  }

  /// Runs the kernel using initialized state.
  cutlass::Status run(cudaStream_t stream = nullptr) {
    return underlying_operator_.run(stream);
  }

  /// Runs the kernel using initialized state.
  cutlass::Status operator()(cudaStream_t stream = nullptr) {
    return run(stream);
  }

  /// Runs the kernel using initialized state.
  cutlass::Status operator()(
    Arguments const &args,
    void *workspace = nullptr,
    cudaStream_t stream = nullptr) {

    cutlass::Status status = initialize(args, workspace);
    if (status == cutlass::Status::kSuccess) {
      status = run(stream);
    }
    return status;
  }
};

} // namespace device
} // namespace gemm
} // namespace cuasr
