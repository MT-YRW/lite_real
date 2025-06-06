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

TORCH_API at::Tensor abs(const at::Tensor & self);
TORCH_API at::Tensor & abs_(at::Tensor & self);
TORCH_API at::Tensor & abs_out(const at::Tensor & self, at::Tensor & out);
TORCH_API at::Tensor abs_sparse(const at::Tensor & self);
TORCH_API at::Tensor & abs_sparse_out(const at::Tensor & self, at::Tensor & out);
TORCH_API at::Tensor & abs_sparse_(at::Tensor & self);
TORCH_API at::Tensor abs_sparse_csr(const at::Tensor & self);
TORCH_API at::Tensor & abs_sparse_csr_out(const at::Tensor & self, at::Tensor & out);
TORCH_API at::Tensor & abs_sparse_csr_(at::Tensor & self);

} // namespace native
} // namespace at
