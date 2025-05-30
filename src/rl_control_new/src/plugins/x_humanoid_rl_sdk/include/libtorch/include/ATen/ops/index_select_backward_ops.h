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


struct TORCH_API index_select_backward {
  using schema = at::Tensor (const at::Tensor &, at::IntArrayRef, int64_t, const at::Tensor &);
  using ptr_schema = schema*;
  // See Note [static constexpr char* members for windows NVCC]
  STATIC_CONSTEXPR_STR_INL_EXCEPT_WIN_CUDA(name, "aten::index_select_backward")
  STATIC_CONSTEXPR_STR_INL_EXCEPT_WIN_CUDA(overload_name, "")
  STATIC_CONSTEXPR_STR_INL_EXCEPT_WIN_CUDA(schema_str, "index_select_backward(Tensor grad, int[] self_sizes, int dim, Tensor index) -> Tensor")
  static at::Tensor call(const at::Tensor & grad, at::IntArrayRef self_sizes, int64_t dim, const at::Tensor & index);
  static at::Tensor redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & grad, at::IntArrayRef self_sizes, int64_t dim, const at::Tensor & index);
};

}} // namespace at::_ops
