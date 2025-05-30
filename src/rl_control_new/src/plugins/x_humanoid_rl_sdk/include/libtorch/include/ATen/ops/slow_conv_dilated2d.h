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



#include <ATen/ops/slow_conv_dilated2d_ops.h>

namespace at {


// aten::slow_conv_dilated2d(Tensor self, Tensor weight, int[2] kernel_size, Tensor? bias=None, int[2] stride=1, int[2] padding=0, int[2] dilation=1) -> Tensor
TORCH_API inline at::Tensor slow_conv_dilated2d(const at::Tensor & self, const at::Tensor & weight, at::IntArrayRef kernel_size, const c10::optional<at::Tensor> & bias={}, at::IntArrayRef stride=1, at::IntArrayRef padding=0, at::IntArrayRef dilation=1) {
    return at::_ops::slow_conv_dilated2d::call(self, weight, kernel_size, bias, stride, padding, dilation);
}

}
