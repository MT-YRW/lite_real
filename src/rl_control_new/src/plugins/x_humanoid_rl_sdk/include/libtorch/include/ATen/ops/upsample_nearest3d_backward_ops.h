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


struct TORCH_API upsample_nearest3d_backward_vec {
  using schema = at::Tensor (const at::Tensor &, at::OptionalIntArrayRef, at::IntArrayRef, c10::optional<at::ArrayRef<double>>);
  using ptr_schema = schema*;
  // See Note [static constexpr char* members for windows NVCC]
  STATIC_CONSTEXPR_STR_INL_EXCEPT_WIN_CUDA(name, "aten::upsample_nearest3d_backward")
  STATIC_CONSTEXPR_STR_INL_EXCEPT_WIN_CUDA(overload_name, "vec")
  STATIC_CONSTEXPR_STR_INL_EXCEPT_WIN_CUDA(schema_str, "upsample_nearest3d_backward.vec(Tensor grad_output, int[]? output_size, int[] input_size, float[]? scale_factors) -> Tensor")
  static at::Tensor call(const at::Tensor & grad_output, at::OptionalIntArrayRef output_size, at::IntArrayRef input_size, c10::optional<at::ArrayRef<double>> scale_factors);
  static at::Tensor redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & grad_output, at::OptionalIntArrayRef output_size, at::IntArrayRef input_size, c10::optional<at::ArrayRef<double>> scale_factors);
};

struct TORCH_API upsample_nearest3d_backward_grad_input {
  using schema = at::Tensor & (const at::Tensor &, at::IntArrayRef, at::IntArrayRef, c10::optional<double>, c10::optional<double>, c10::optional<double>, at::Tensor &);
  using ptr_schema = schema*;
  // See Note [static constexpr char* members for windows NVCC]
  STATIC_CONSTEXPR_STR_INL_EXCEPT_WIN_CUDA(name, "aten::upsample_nearest3d_backward")
  STATIC_CONSTEXPR_STR_INL_EXCEPT_WIN_CUDA(overload_name, "grad_input")
  STATIC_CONSTEXPR_STR_INL_EXCEPT_WIN_CUDA(schema_str, "upsample_nearest3d_backward.grad_input(Tensor grad_output, int[3] output_size, int[5] input_size, float? scales_d=None, float? scales_h=None, float? scales_w=None, *, Tensor(a!) grad_input) -> Tensor(a!)")
  static at::Tensor & call(const at::Tensor & grad_output, at::IntArrayRef output_size, at::IntArrayRef input_size, c10::optional<double> scales_d, c10::optional<double> scales_h, c10::optional<double> scales_w, at::Tensor & grad_input);
  static at::Tensor & redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & grad_output, at::IntArrayRef output_size, at::IntArrayRef input_size, c10::optional<double> scales_d, c10::optional<double> scales_h, c10::optional<double> scales_w, at::Tensor & grad_input);
};

struct TORCH_API upsample_nearest3d_backward {
  using schema = at::Tensor (const at::Tensor &, at::IntArrayRef, at::IntArrayRef, c10::optional<double>, c10::optional<double>, c10::optional<double>);
  using ptr_schema = schema*;
  // See Note [static constexpr char* members for windows NVCC]
  STATIC_CONSTEXPR_STR_INL_EXCEPT_WIN_CUDA(name, "aten::upsample_nearest3d_backward")
  STATIC_CONSTEXPR_STR_INL_EXCEPT_WIN_CUDA(overload_name, "")
  STATIC_CONSTEXPR_STR_INL_EXCEPT_WIN_CUDA(schema_str, "upsample_nearest3d_backward(Tensor grad_output, int[3] output_size, int[5] input_size, float? scales_d=None, float? scales_h=None, float? scales_w=None) -> Tensor")
  static at::Tensor call(const at::Tensor & grad_output, at::IntArrayRef output_size, at::IntArrayRef input_size, c10::optional<double> scales_d, c10::optional<double> scales_h, c10::optional<double> scales_w);
  static at::Tensor redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & grad_output, at::IntArrayRef output_size, at::IntArrayRef input_size, c10::optional<double> scales_d, c10::optional<double> scales_h, c10::optional<double> scales_w);
};

}} // namespace at::_ops
