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

TORCH_API at::Tensor triplet_margin_loss(const at::Tensor & anchor, const at::Tensor & positive, const at::Tensor & negative, double margin=1.0, double p=2, double eps=1e-06, bool swap=false, int64_t reduction=at::Reduction::Mean);

} // namespace native
} // namespace at
