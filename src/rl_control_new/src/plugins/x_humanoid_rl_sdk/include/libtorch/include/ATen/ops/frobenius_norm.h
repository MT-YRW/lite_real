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



#include <ATen/ops/frobenius_norm_ops.h>

namespace at {


// aten::frobenius_norm(Tensor self) -> Tensor
TORCH_API inline at::Tensor frobenius_norm(const at::Tensor & self) {
    return at::_ops::frobenius_norm::call(self);
}

// aten::frobenius_norm.dim(Tensor self, int[1] dim, bool keepdim=False) -> Tensor
TORCH_API inline at::Tensor frobenius_norm(const at::Tensor & self, at::IntArrayRef dim, bool keepdim=false) {
    return at::_ops::frobenius_norm_dim::call(self, dim, keepdim);
}

// aten::frobenius_norm.out(Tensor self, int[1] dim, bool keepdim=False, *, Tensor(a!) out) -> Tensor(a!)
TORCH_API inline at::Tensor & frobenius_norm_out(at::Tensor & out, const at::Tensor & self, at::IntArrayRef dim, bool keepdim=false) {
    return at::_ops::frobenius_norm_out::call(self, dim, keepdim, out);
}

// aten::frobenius_norm.out(Tensor self, int[1] dim, bool keepdim=False, *, Tensor(a!) out) -> Tensor(a!)
TORCH_API inline at::Tensor & frobenius_norm_outf(const at::Tensor & self, at::IntArrayRef dim, bool keepdim, at::Tensor & out) {
    return at::_ops::frobenius_norm_out::call(self, dim, keepdim, out);
}

}
