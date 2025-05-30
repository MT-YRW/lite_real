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



#include <ATen/ops/reflection_pad2d_ops.h>

namespace at {


// aten::reflection_pad2d.out(Tensor self, int[4] padding, *, Tensor(a!) out) -> Tensor(a!)
TORCH_API inline at::Tensor & reflection_pad2d_out(at::Tensor & out, const at::Tensor & self, at::IntArrayRef padding) {
    return at::_ops::reflection_pad2d_out::call(self, padding, out);
}

// aten::reflection_pad2d.out(Tensor self, int[4] padding, *, Tensor(a!) out) -> Tensor(a!)
TORCH_API inline at::Tensor & reflection_pad2d_outf(const at::Tensor & self, at::IntArrayRef padding, at::Tensor & out) {
    return at::_ops::reflection_pad2d_out::call(self, padding, out);
}

// aten::reflection_pad2d(Tensor self, int[4] padding) -> Tensor
TORCH_API inline at::Tensor reflection_pad2d(const at::Tensor & self, at::IntArrayRef padding) {
    return at::_ops::reflection_pad2d::call(self, padding);
}

}
