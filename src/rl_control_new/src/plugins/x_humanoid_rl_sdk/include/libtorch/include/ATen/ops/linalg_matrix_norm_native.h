#pragma once

// @generated by torchgen/gen.py from NativeFunction.h

#include <c10/core/Scalar.h>
#include <c10/core/Storage.h>
#include <c10/core/TensorOptions.h>
#include <c10/util/Deprecated.h>
#include <c10/util/Optional.h>
#include <c10/core/QScheme.h>
#include <ATen/core/Reduction.h>
#include <ATen/core/Tensor.h>
#include <tuple>
#include <vector>


namespace at {
namespace native {

TORCH_API at::Tensor linalg_matrix_norm(const at::Tensor & self, const at::Scalar & ord, at::IntArrayRef dim={-2,-1}, bool keepdim=false, c10::optional<at::ScalarType> dtype=c10::nullopt);
TORCH_API at::Tensor & linalg_matrix_norm_out(const at::Tensor & self, const at::Scalar & ord, at::IntArrayRef dim, bool keepdim, c10::optional<at::ScalarType> dtype, at::Tensor & out);
TORCH_API at::Tensor linalg_matrix_norm(const at::Tensor & self, c10::string_view ord="fro", at::IntArrayRef dim={-2,-1}, bool keepdim=false, c10::optional<at::ScalarType> dtype=c10::nullopt);
TORCH_API at::Tensor & linalg_matrix_norm_out(const at::Tensor & self, c10::string_view ord, at::IntArrayRef dim, bool keepdim, c10::optional<at::ScalarType> dtype, at::Tensor & out);

} // namespace native
} // namespace at
