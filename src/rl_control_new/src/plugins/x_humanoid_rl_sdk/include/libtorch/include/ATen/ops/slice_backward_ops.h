#pragma once

// @generated by torchgen/gen.py from Operator.h

#include <tuple>
#include <vector>

// Forward declarations of any types needed in the operator signatures.
// We can't directly include these classes because it will cause circular include dependencies.
// This file is included by TensorBody.h, which defines the Tensor class.
#include <ATen/core/ATen_fwd.h>

namespace at {
namespace _ops {


struct TORCH_API slice_backward {
  using schema = at::Tensor (const at::Tensor &, at::IntArrayRef, int64_t, int64_t, int64_t, int64_t);
  using ptr_schema = schema*;
  // See Note [static constexpr char* members for windows NVCC]
  STATIC_CONSTEXPR_STR_INL_EXCEPT_WIN_CUDA(name, "aten::slice_backward")
  STATIC_CONSTEXPR_STR_INL_EXCEPT_WIN_CUDA(overload_name, "")
  STATIC_CONSTEXPR_STR_INL_EXCEPT_WIN_CUDA(schema_str, "slice_backward(Tensor grad_output, int[] input_sizes, int dim, int start, int end, int step) -> Tensor")
  static at::Tensor call(const at::Tensor & grad_output, at::IntArrayRef input_sizes, int64_t dim, int64_t start, int64_t end, int64_t step);
  static at::Tensor redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & grad_output, at::IntArrayRef input_sizes, int64_t dim, int64_t start, int64_t end, int64_t step);
};

}} // namespace at::_ops
