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



#include <ATen/ops/_copy_from_and_resize_ops.h>

namespace at {


// aten::_copy_from_and_resize(Tensor self, Tensor dst) -> Tensor
TORCH_API inline at::Tensor _copy_from_and_resize(const at::Tensor & self, const at::Tensor & dst) {
    return at::_ops::_copy_from_and_resize::call(self, dst);
}

}
