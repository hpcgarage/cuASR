#pragma once

#include "cutlass/util/distribution.h"
#include "cutlass/util/host_tensor.h"
#include "cutlass/util/reference/host/tensor_compare.h"
#include "cutlass/util/reference/host/tensor_copy.h"
#include "cutlass/util/reference/host/tensor_fill.h"
#include "cutlass/util/reference/host/tensor_norm.h"
#include "cutlass/util/tensor_view_io.h"

#include "cuasr/reference/srgemm/host_srgemm.h"


namespace cuasr {
namespace bench {
namespace device {


namespace {
inline char const *to_string(cutlass::Status status) {
  switch (status) {
    case cutlass::Status::kSuccess:
      return "kSuccess";
    case cutlass::Status::kErrorMisalignedOperand:
      return "kErrorMisalignedOperand";
    case cutlass::Status::kErrorInvalidLayout:
      return "kErrorInvalidLayout";
    case cutlass::Status::kErrorInvalidProblem:
      return "kErrorInvalidProblem";
    case cutlass::Status::kErrorNotSupported:
      return "kErrorNotSupported";
    case cutlass::Status::kErrorWorkspaceNull:
      return "kErrorWorkspaceNull";
    case cutlass::Status::kErrorInternal:
      return "kErrorInternal";
    case cutlass::Status::kInvalid:
      return "kInvalid";
    default:
      break;
  }
  return "invalid";
}
}

// Given a SIMT SRGEMM, sets up host and device tensors for the benchmark loop
template <typename Srgemm>
class BenchHarness {
  using ElementAccumulator = typename Srgemm::ElementAccumulator;
  using ElementCompute =
      typename Srgemm::SrgemmKernel::Epilogue::OutputOp::ElementCompute;

  cutlass::gemm::GemmCoord problem_size;
  cutlass::Distribution::Kind init_A;
  cutlass::Distribution::Kind init_B;
  cutlass::Distribution::Kind init_C;
  uint64_t seed;

  cutlass::HostTensor<typename Srgemm::ElementA, typename Srgemm::LayoutA> tensor_A;
  cutlass::HostTensor<typename Srgemm::ElementB, typename Srgemm::LayoutB> tensor_B;
  cutlass::HostTensor<typename Srgemm::ElementC, typename Srgemm::LayoutC> tensor_C;
  cutlass::HostTensor<typename Srgemm::ElementC, typename Srgemm::LayoutC> tensor_D;
  cutlass::HostTensor<typename Srgemm::ElementC, typename Srgemm::LayoutC> reference_D;

  /// Helper to initialize a tensor view
  template <typename Element, typename Layout>
  bool initialize_tensor(
      cutlass::TensorView<Element, Layout> view,
      cutlass::Distribution::Kind dist_kind,
      uint64_t seed) {
    if (dist_kind == cutlass::Distribution::Uniform) {
      double scope_max, scope_min;
      int bits_input  = cutlass::sizeof_bits<Element>::value;
      int bits_output = cutlass::sizeof_bits<typename Srgemm::ElementC>::value;

      if (bits_input == 1) {
        scope_max = 2;
        scope_min = 0;
      }
      else if (bits_input <= 8) {
        scope_max = 2;
        scope_min = -2;
      }
      else if (bits_output == 16) {
        scope_max = 5;
        scope_min = -5;
      }
      else {
        scope_max = 8;
        scope_min = -8;
      }

      cutlass::reference::host::TensorFillRandomUniform(
          view, seed, scope_max, scope_min, 0);
    }
    else if (dist_kind == cutlass::Distribution::Identity) {
      cutlass::reference::host::TensorFillIdentity(view);
    }
    else if (dist_kind == cutlass::Distribution::Gaussian) {
      cutlass::reference::host::TensorFillRandomGaussian(view, seed, 0, 0.5);
    }
    else if (dist_kind == cutlass::Distribution::Sequential) {
      cutlass::reference::host::BlockFillSequential(view.data(), view.capacity());
    }
    else {
      return false;
    }
    return true;
  }

public:
  BenchHarness() = delete;

  // Methods
  BenchHarness(
      cutlass::gemm::GemmCoord problem_size_,
      cutlass::Distribution::Kind init_A_ = cutlass::Distribution::Uniform,
      cutlass::Distribution::Kind init_B_ = cutlass::Distribution::Uniform,
      cutlass::Distribution::Kind init_C_ = cutlass::Distribution::Uniform,
      uint64_t seed_                      = 2080)
      : problem_size(problem_size_)
      , init_A(init_A_)
      , init_B(init_B_)
      , init_C(init_C_)
      , seed(seed_) {
    this->initialize(problem_size);
  }


  // Initializes data structures on both host-device side for benchmark
  auto initialize(cutlass::gemm::GemmCoord problem_size) -> void {
    // Allocate the GEMM workspace
    tensor_A.resize(problem_size.mk());
    tensor_B.resize(problem_size.kn());
    tensor_C.resize(problem_size.mn());
    tensor_D.resize(problem_size.mn());
    reference_D.resize(problem_size.mn(), false);

    initialize_tensor(tensor_A.host_view(), init_A, seed + 2019);
    initialize_tensor(tensor_B.host_view(), init_B, seed + 2018);
    initialize_tensor(tensor_C.host_view(), init_C, seed + 2017);

    // It is possible to randomly initialize to all zeros, so override this with non-zeros
    // in the upper left corner of each operand.
    tensor_A.host_view().at({ 0, 0 }) = typename Srgemm::ElementA(1);
    tensor_B.host_view().at({ 0, 0 }) = typename Srgemm::ElementB(1);
    tensor_C.host_view().at({ 0, 0 }) = typename Srgemm::ElementC(1);

    cutlass::reference::host::TensorCopy(reference_D.host_view(), tensor_C.host_view());

    tensor_A.sync_device();
    tensor_B.sync_device();
    tensor_C.sync_device();
    tensor_D.sync_device();
  }

  // Runs one loop of the benchmark on initialized tensors
  auto
  run(int split_k_slices   = 1,
      ElementCompute alpha = ElementCompute(Srgemm::MultiplicationOp::Identity),
      ElementCompute beta  = ElementCompute(Srgemm::MultiplicationOp::Identity))
      -> cutlass::Status {
    // Initialize the GEMM operator
    typename Srgemm::Arguments arguments {
      problem_size,                 //
      tensor_A.device_ref(),        //
      tensor_B.device_ref(),        //
      tensor_C.device_ref(),        //
      tensor_D.device_ref(),        //
      { alpha, beta },              //
      Srgemm::AdditionOp::Identity, //
      split_k_slices                //
    };

    Srgemm gemm_op;
    size_t workspace_size = Srgemm::get_workspace_size(arguments);
    cutlass::device_memory::allocation<uint8_t> workspace(workspace_size);
    cutlass::Status status = gemm_op.initialize(arguments, workspace.get());

    // Run the GEMM
    status = gemm_op();
    return status;
  }
};


} // namespace device
} // namespace bench
} // namespace cuasr
