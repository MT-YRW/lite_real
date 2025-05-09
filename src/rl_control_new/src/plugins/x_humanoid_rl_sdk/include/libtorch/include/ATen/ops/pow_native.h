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
#include <ATen/ops/pow_meta.h>

namespace at {
namespace native {

struct TORCH_API structured_pow_Tensor_Tensor_out : public at::meta::structured_pow_Tensor_Tensor {
void impl(const at::Tensor & self, const at::Tensor & exponent, const at::Tensor & out);
};
struct TORCH_API structured_pow_Scalar_out : public at::meta::structured_pow_Scalar {
void impl(const at::Scalar & self, const at::Tensor & exponent, const at::Tensor & out);
};
struct TORCH_API structured_pow_Tensor_Scalar_out : public at::meta::structured_pow_Tensor_Scalar {
void impl(const at::Tensor & self, const at::Scalar & exponent, const at::Tensor & out);
};
TORCH_API at::Tensor pow_sparse_scalar(const at::Tensor & self, const at::Scalar & exponent);
TORCH_API at::Tensor & pow_out_sparse_scalar(const at::Tensor & self, const at::Scalar & exponent, at::Tensor & out);

} // namespace native
} // namespace at
