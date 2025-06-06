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
#include <ATen/ops/bitwise_left_shift_meta.h>

namespace at {
namespace native {

struct TORCH_API structured_bitwise_left_shift_out : public at::meta::structured_bitwise_left_shift_Tensor {
void impl(const at::Tensor & self, const at::Tensor & other, const at::Tensor & out);
};
TORCH_API at::Tensor bitwise_left_shift(const at::Tensor & self, const at::Scalar & other);
TORCH_API at::Tensor & bitwise_left_shift_out(const at::Tensor & self, const at::Scalar & other, at::Tensor & out);
TORCH_API at::Tensor & bitwise_left_shift_(at::Tensor & self, const at::Scalar & other);
TORCH_API at::Tensor bitwise_left_shift(const at::Scalar & self, const at::Tensor & other);

} // namespace native
} // namespace at
