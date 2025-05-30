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


struct TORCH_API var_mean {
  using schema = ::std::tuple<at::Tensor,at::Tensor> (const at::Tensor &, bool);
  using ptr_schema = schema*;
  // See Note [static constexpr char* members for windows NVCC]
  STATIC_CONSTEXPR_STR_INL_EXCEPT_WIN_CUDA(name, "aten::var_mean")
  STATIC_CONSTEXPR_STR_INL_EXCEPT_WIN_CUDA(overload_name, "")
  STATIC_CONSTEXPR_STR_INL_EXCEPT_WIN_CUDA(schema_str, "var_mean(Tensor self, bool unbiased=True) -> (Tensor, Tensor)")
  static ::std::tuple<at::Tensor,at::Tensor> call(const at::Tensor & self, bool unbiased);
  static ::std::tuple<at::Tensor,at::Tensor> redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, bool unbiased);
};

struct TORCH_API var_mean_dim {
  using schema = ::std::tuple<at::Tensor,at::Tensor> (const at::Tensor &, at::IntArrayRef, bool, bool);
  using ptr_schema = schema*;
  // See Note [static constexpr char* members for windows NVCC]
  STATIC_CONSTEXPR_STR_INL_EXCEPT_WIN_CUDA(name, "aten::var_mean")
  STATIC_CONSTEXPR_STR_INL_EXCEPT_WIN_CUDA(overload_name, "dim")
  STATIC_CONSTEXPR_STR_INL_EXCEPT_WIN_CUDA(schema_str, "var_mean.dim(Tensor self, int[1] dim, bool unbiased=True, bool keepdim=False) -> (Tensor, Tensor)")
  static ::std::tuple<at::Tensor,at::Tensor> call(const at::Tensor & self, at::IntArrayRef dim, bool unbiased, bool keepdim);
  static ::std::tuple<at::Tensor,at::Tensor> redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, at::IntArrayRef dim, bool unbiased, bool keepdim);
};

struct TORCH_API var_mean_correction {
  using schema = ::std::tuple<at::Tensor,at::Tensor> (const at::Tensor &, at::OptionalIntArrayRef, c10::optional<int64_t>, bool);
  using ptr_schema = schema*;
  // See Note [static constexpr char* members for windows NVCC]
  STATIC_CONSTEXPR_STR_INL_EXCEPT_WIN_CUDA(name, "aten::var_mean")
  STATIC_CONSTEXPR_STR_INL_EXCEPT_WIN_CUDA(overload_name, "correction")
  STATIC_CONSTEXPR_STR_INL_EXCEPT_WIN_CUDA(schema_str, "var_mean.correction(Tensor self, int[1]? dim, *, int? correction, bool keepdim=False) -> (Tensor, Tensor)")
  static ::std::tuple<at::Tensor,at::Tensor> call(const at::Tensor & self, at::OptionalIntArrayRef dim, c10::optional<int64_t> correction, bool keepdim);
  static ::std::tuple<at::Tensor,at::Tensor> redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, at::OptionalIntArrayRef dim, c10::optional<int64_t> correction, bool keepdim);
};

struct TORCH_API var_mean_names_dim {
  using schema = ::std::tuple<at::Tensor,at::Tensor> (const at::Tensor &, at::DimnameList, bool, bool);
  using ptr_schema = schema*;
  // See Note [static constexpr char* members for windows NVCC]
  STATIC_CONSTEXPR_STR_INL_EXCEPT_WIN_CUDA(name, "aten::var_mean")
  STATIC_CONSTEXPR_STR_INL_EXCEPT_WIN_CUDA(overload_name, "names_dim")
  STATIC_CONSTEXPR_STR_INL_EXCEPT_WIN_CUDA(schema_str, "var_mean.names_dim(Tensor self, Dimname[1] dim, bool unbiased=True, bool keepdim=False) -> (Tensor, Tensor)")
  static ::std::tuple<at::Tensor,at::Tensor> call(const at::Tensor & self, at::DimnameList dim, bool unbiased, bool keepdim);
  static ::std::tuple<at::Tensor,at::Tensor> redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, at::DimnameList dim, bool unbiased, bool keepdim);
};

struct TORCH_API var_mean_correction_names {
  using schema = ::std::tuple<at::Tensor,at::Tensor> (const at::Tensor &, at::DimnameList, c10::optional<int64_t>, bool);
  using ptr_schema = schema*;
  // See Note [static constexpr char* members for windows NVCC]
  STATIC_CONSTEXPR_STR_INL_EXCEPT_WIN_CUDA(name, "aten::var_mean")
  STATIC_CONSTEXPR_STR_INL_EXCEPT_WIN_CUDA(overload_name, "correction_names")
  STATIC_CONSTEXPR_STR_INL_EXCEPT_WIN_CUDA(schema_str, "var_mean.correction_names(Tensor self, Dimname[1] dim, *, int? correction, bool keepdim=False) -> (Tensor, Tensor)")
  static ::std::tuple<at::Tensor,at::Tensor> call(const at::Tensor & self, at::DimnameList dim, c10::optional<int64_t> correction, bool keepdim);
  static ::std::tuple<at::Tensor,at::Tensor> redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, at::DimnameList dim, c10::optional<int64_t> correction, bool keepdim);
};

}} // namespace at::_ops
