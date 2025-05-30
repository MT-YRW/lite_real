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
#include <ATen/ops/_log_softmax_meta.h>

namespace at {
namespace native {

struct TORCH_API structured_log_softmax_cpu_out : public at::meta::structured__log_softmax {
void impl(const at::Tensor & self, int64_t dim, bool half_to_float, const at::Tensor & out);
};
struct TORCH_API structured_log_softmax_cuda_out : public at::meta::structured__log_softmax {
void impl(const at::Tensor & self, int64_t dim, bool half_to_float, const at::Tensor & out);
};

} // namespace native
} // namespace at
