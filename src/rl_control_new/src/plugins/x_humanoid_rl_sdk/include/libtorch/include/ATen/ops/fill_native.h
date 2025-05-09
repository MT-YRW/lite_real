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

TORCH_API at::Tensor fill(const at::Tensor & self, const at::Scalar & value);
TORCH_API at::Tensor & fill_(at::Tensor & self, const at::Scalar & value);
TORCH_API at::Tensor & fill_sparse_csr_(at::Tensor & self, const at::Scalar & value);
TORCH_API at::Tensor & fill_meta_(at::Tensor & self, const at::Scalar & value);
TORCH_API at::Tensor & fill_quantized_(at::Tensor & self, const at::Scalar & value);
TORCH_API at::Tensor fill(const at::Tensor & self, const at::Tensor & value);
TORCH_API at::Tensor & fill_(at::Tensor & self, const at::Tensor & value);
TORCH_API at::Tensor & fill_meta_(at::Tensor & self, const at::Tensor & value);
TORCH_API at::Tensor & fill_quantized_(at::Tensor & self, const at::Tensor & value);

} // namespace native
} // namespace at
