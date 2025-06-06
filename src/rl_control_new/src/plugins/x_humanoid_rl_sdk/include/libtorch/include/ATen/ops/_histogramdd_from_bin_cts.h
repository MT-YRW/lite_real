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



#include <ATen/ops/_histogramdd_from_bin_cts_ops.h>

namespace at {


// aten::_histogramdd_from_bin_cts(Tensor self, int[] bins, *, float[]? range=None, Tensor? weight=None, bool density=False) -> Tensor
TORCH_API inline at::Tensor _histogramdd_from_bin_cts(const at::Tensor & self, at::IntArrayRef bins, c10::optional<at::ArrayRef<double>> range=c10::nullopt, const c10::optional<at::Tensor> & weight={}, bool density=false) {
    return at::_ops::_histogramdd_from_bin_cts::call(self, bins, range, weight, density);
}

}
