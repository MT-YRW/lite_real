#pragma once

// @generated by torchgen/gen.py from Function.h

#include <ATen/Context.h>
#include <ATen/DeviceGuard.h>
#include <ATen/TensorUtils.h>
#include <ATen/TracerMode.h>
#include <ATen/core/Generator.h>
#include <ATen/core/Reduction.h>
#include <ATen/core/Tensor.h>
#include <c10/core/Scalar.h>
#include <c10/core/Storage.h>
#include <c10/core/TensorOptions.h>
#include <c10/util/Deprecated.h>
#include <c10/util/Optional.h>



#include <ATen/ops/linalg_matrix_rank_ops.h>

namespace at {


// aten::linalg_matrix_rank.atol_rtol_tensor(Tensor input, *, Tensor? atol=None, Tensor? rtol=None, bool hermitian=False) -> Tensor
TORCH_API inline at::Tensor linalg_matrix_rank(const at::Tensor & input, const c10::optional<at::Tensor> & atol={}, const c10::optional<at::Tensor> & rtol={}, bool hermitian=false) {
    return at::_ops::linalg_matrix_rank_atol_rtol_tensor::call(input, atol, rtol, hermitian);
}

// aten::linalg_matrix_rank.atol_rtol_tensor_out(Tensor input, *, Tensor? atol=None, Tensor? rtol=None, bool hermitian=False, Tensor(a!) out) -> Tensor(a!)
TORCH_API inline at::Tensor & linalg_matrix_rank_out(at::Tensor & out, const at::Tensor & input, const c10::optional<at::Tensor> & atol={}, const c10::optional<at::Tensor> & rtol={}, bool hermitian=false) {
    return at::_ops::linalg_matrix_rank_atol_rtol_tensor_out::call(input, atol, rtol, hermitian, out);
}

// aten::linalg_matrix_rank.atol_rtol_tensor_out(Tensor input, *, Tensor? atol=None, Tensor? rtol=None, bool hermitian=False, Tensor(a!) out) -> Tensor(a!)
TORCH_API inline at::Tensor & linalg_matrix_rank_outf(const at::Tensor & input, const c10::optional<at::Tensor> & atol, const c10::optional<at::Tensor> & rtol, bool hermitian, at::Tensor & out) {
    return at::_ops::linalg_matrix_rank_atol_rtol_tensor_out::call(input, atol, rtol, hermitian, out);
}

// aten::linalg_matrix_rank.atol_rtol_float(Tensor self, *, float? atol=None, float? rtol=None, bool hermitian=False) -> Tensor
TORCH_API inline at::Tensor linalg_matrix_rank(const at::Tensor & self, c10::optional<double> atol, c10::optional<double> rtol, bool hermitian=false) {
    return at::_ops::linalg_matrix_rank_atol_rtol_float::call(self, atol, rtol, hermitian);
}

// aten::linalg_matrix_rank.atol_rtol_float_out(Tensor self, *, float? atol=None, float? rtol=None, bool hermitian=False, Tensor(a!) out) -> Tensor(a!)
TORCH_API inline at::Tensor & linalg_matrix_rank_out(at::Tensor & out, const at::Tensor & self, c10::optional<double> atol, c10::optional<double> rtol, bool hermitian=false) {
    return at::_ops::linalg_matrix_rank_atol_rtol_float_out::call(self, atol, rtol, hermitian, out);
}

// aten::linalg_matrix_rank.atol_rtol_float_out(Tensor self, *, float? atol=None, float? rtol=None, bool hermitian=False, Tensor(a!) out) -> Tensor(a!)
TORCH_API inline at::Tensor & linalg_matrix_rank_outf(const at::Tensor & self, c10::optional<double> atol, c10::optional<double> rtol, bool hermitian, at::Tensor & out) {
    return at::_ops::linalg_matrix_rank_atol_rtol_float_out::call(self, atol, rtol, hermitian, out);
}

// aten::linalg_matrix_rank(Tensor self, float tol, bool hermitian=False) -> Tensor
TORCH_API inline at::Tensor linalg_matrix_rank(const at::Tensor & self, double tol, bool hermitian=false) {
    return at::_ops::linalg_matrix_rank::call(self, tol, hermitian);
}

// aten::linalg_matrix_rank.out(Tensor self, float tol, bool hermitian=False, *, Tensor(a!) out) -> Tensor(a!)
TORCH_API inline at::Tensor & linalg_matrix_rank_out(at::Tensor & out, const at::Tensor & self, double tol, bool hermitian=false) {
    return at::_ops::linalg_matrix_rank_out::call(self, tol, hermitian, out);
}

// aten::linalg_matrix_rank.out(Tensor self, float tol, bool hermitian=False, *, Tensor(a!) out) -> Tensor(a!)
TORCH_API inline at::Tensor & linalg_matrix_rank_outf(const at::Tensor & self, double tol, bool hermitian, at::Tensor & out) {
    return at::_ops::linalg_matrix_rank_out::call(self, tol, hermitian, out);
}

// aten::linalg_matrix_rank.tol_tensor(Tensor input, Tensor tol, bool hermitian=False) -> Tensor
TORCH_API inline at::Tensor linalg_matrix_rank(const at::Tensor & input, const at::Tensor & tol, bool hermitian=false) {
    return at::_ops::linalg_matrix_rank_tol_tensor::call(input, tol, hermitian);
}

// aten::linalg_matrix_rank.out_tol_tensor(Tensor input, Tensor tol, bool hermitian=False, *, Tensor(a!) out) -> Tensor(a!)
TORCH_API inline at::Tensor & linalg_matrix_rank_out(at::Tensor & out, const at::Tensor & input, const at::Tensor & tol, bool hermitian=false) {
    return at::_ops::linalg_matrix_rank_out_tol_tensor::call(input, tol, hermitian, out);
}

// aten::linalg_matrix_rank.out_tol_tensor(Tensor input, Tensor tol, bool hermitian=False, *, Tensor(a!) out) -> Tensor(a!)
TORCH_API inline at::Tensor & linalg_matrix_rank_outf(const at::Tensor & input, const at::Tensor & tol, bool hermitian, at::Tensor & out) {
    return at::_ops::linalg_matrix_rank_out_tol_tensor::call(input, tol, hermitian, out);
}

}
