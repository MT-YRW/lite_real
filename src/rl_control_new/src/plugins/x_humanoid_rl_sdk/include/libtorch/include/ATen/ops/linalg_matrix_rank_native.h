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

TORCH_API at::Tensor linalg_matrix_rank(const at::Tensor & input, const c10::optional<at::Tensor> & atol={}, const c10::optional<at::Tensor> & rtol={}, bool hermitian=false);
TORCH_API at::Tensor & linalg_matrix_rank_out(const at::Tensor & input, const c10::optional<at::Tensor> & atol, const c10::optional<at::Tensor> & rtol, bool hermitian, at::Tensor & out);
TORCH_API at::Tensor linalg_matrix_rank(const at::Tensor & self, c10::optional<double> atol=c10::nullopt, c10::optional<double> rtol=c10::nullopt, bool hermitian=false);
TORCH_API at::Tensor & linalg_matrix_rank_out(const at::Tensor & self, c10::optional<double> atol, c10::optional<double> rtol, bool hermitian, at::Tensor & out);
TORCH_API at::Tensor linalg_matrix_rank(const at::Tensor & self, double tol, bool hermitian=false);
TORCH_API at::Tensor & linalg_matrix_rank_out(const at::Tensor & self, double tol, bool hermitian, at::Tensor & out);
TORCH_API at::Tensor linalg_matrix_rank(const at::Tensor & input, const at::Tensor & tol, bool hermitian=false);
TORCH_API at::Tensor & linalg_matrix_rank_out(const at::Tensor & input, const at::Tensor & tol, bool hermitian, at::Tensor & out);

} // namespace native
} // namespace at
