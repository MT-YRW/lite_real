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



#include <ATen/ops/fill_ops.h>

namespace at {


// aten::fill.Scalar(Tensor self, Scalar value) -> Tensor
TORCH_API inline at::Tensor fill(const at::Tensor & self, const at::Scalar & value) {
    return at::_ops::fill_Scalar::call(self, value);
}

// aten::fill.Tensor(Tensor self, Tensor value) -> Tensor
TORCH_API inline at::Tensor fill(const at::Tensor & self, const at::Tensor & value) {
    return at::_ops::fill_Tensor::call(self, value);
}

// aten::fill_.Scalar(Tensor(a!) self, Scalar value) -> Tensor(a!)
TORCH_API inline at::Tensor & fill_(at::Tensor & self, const at::Scalar & value) {
    return at::_ops::fill__Scalar::call(self, value);
}

// aten::fill_.Tensor(Tensor(a!) self, Tensor value) -> Tensor(a!)
TORCH_API inline at::Tensor & fill_(at::Tensor & self, const at::Tensor & value) {
    return at::_ops::fill__Tensor::call(self, value);
}

// aten::fill.Scalar_out(Tensor self, Scalar value, *, Tensor(a!) out) -> Tensor(a!)
TORCH_API inline at::Tensor & fill_out(at::Tensor & out, const at::Tensor & self, const at::Scalar & value) {
    return at::_ops::fill_Scalar_out::call(self, value, out);
}

// aten::fill.Scalar_out(Tensor self, Scalar value, *, Tensor(a!) out) -> Tensor(a!)
TORCH_API inline at::Tensor & fill_outf(const at::Tensor & self, const at::Scalar & value, at::Tensor & out) {
    return at::_ops::fill_Scalar_out::call(self, value, out);
}

// aten::fill.Tensor_out(Tensor self, Tensor value, *, Tensor(a!) out) -> Tensor(a!)
TORCH_API inline at::Tensor & fill_out(at::Tensor & out, const at::Tensor & self, const at::Tensor & value) {
    return at::_ops::fill_Tensor_out::call(self, value, out);
}

// aten::fill.Tensor_out(Tensor self, Tensor value, *, Tensor(a!) out) -> Tensor(a!)
TORCH_API inline at::Tensor & fill_outf(const at::Tensor & self, const at::Tensor & value, at::Tensor & out) {
    return at::_ops::fill_Tensor_out::call(self, value, out);
}

}
