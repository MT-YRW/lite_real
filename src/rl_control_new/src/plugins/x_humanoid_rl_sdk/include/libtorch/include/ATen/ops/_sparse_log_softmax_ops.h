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


struct TORCH_API _sparse_log_softmax_int {
  using schema = at::Tensor (const at::Tensor &, int64_t, c10::optional<at::ScalarType>);
  using ptr_schema = schema*;
  // See Note [static constexpr char* members for windows NVCC]
  STATIC_CONSTEXPR_STR_INL_EXCEPT_WIN_CUDA(name, "aten::_sparse_log_softmax")
  STATIC_CONSTEXPR_STR_INL_EXCEPT_WIN_CUDA(overload_name, "int")
  STATIC_CONSTEXPR_STR_INL_EXCEPT_WIN_CUDA(schema_str, "_sparse_log_softmax.int(Tensor self, int dim, ScalarType? dtype=None) -> Tensor")
  static at::Tensor call(const at::Tensor & self, int64_t dim, c10::optional<at::ScalarType> dtype);
  static at::Tensor redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, int64_t dim, c10::optional<at::ScalarType> dtype);
};

struct TORCH_API _sparse_log_softmax_Dimname {
  using schema = at::Tensor (const at::Tensor &, at::Dimname, c10::optional<at::ScalarType>);
  using ptr_schema = schema*;
  // See Note [static constexpr char* members for windows NVCC]
  STATIC_CONSTEXPR_STR_INL_EXCEPT_WIN_CUDA(name, "aten::_sparse_log_softmax")
  STATIC_CONSTEXPR_STR_INL_EXCEPT_WIN_CUDA(overload_name, "Dimname")
  STATIC_CONSTEXPR_STR_INL_EXCEPT_WIN_CUDA(schema_str, "_sparse_log_softmax.Dimname(Tensor self, Dimname dim, *, ScalarType? dtype=None) -> Tensor")
  static at::Tensor call(const at::Tensor & self, at::Dimname dim, c10::optional<at::ScalarType> dtype);
  static at::Tensor redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, at::Dimname dim, c10::optional<at::ScalarType> dtype);
};

struct TORCH_API _sparse_log_softmax {
  using schema = at::Tensor (const at::Tensor &, int64_t, bool);
  using ptr_schema = schema*;
  // See Note [static constexpr char* members for windows NVCC]
  STATIC_CONSTEXPR_STR_INL_EXCEPT_WIN_CUDA(name, "aten::_sparse_log_softmax")
  STATIC_CONSTEXPR_STR_INL_EXCEPT_WIN_CUDA(overload_name, "")
  STATIC_CONSTEXPR_STR_INL_EXCEPT_WIN_CUDA(schema_str, "_sparse_log_softmax(Tensor self, int dim, bool half_to_float) -> Tensor")
  static at::Tensor call(const at::Tensor & self, int64_t dim, bool half_to_float);
  static at::Tensor redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, int64_t dim, bool half_to_float);
};

}} // namespace at::_ops
