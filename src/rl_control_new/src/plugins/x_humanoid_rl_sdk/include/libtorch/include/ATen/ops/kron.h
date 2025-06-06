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



#include <ATen/ops/kron_ops.h>

namespace at {


// aten::kron(Tensor self, Tensor other) -> Tensor
TORCH_API inline at::Tensor kron(const at::Tensor & self, const at::Tensor & other) {
    return at::_ops::kron::call(self, other);
}

// aten::kron.out(Tensor self, Tensor other, *, Tensor(a!) out) -> Tensor(a!)
TORCH_API inline at::Tensor & kron_out(at::Tensor & out, const at::Tensor & self, const at::Tensor & other) {
    return at::_ops::kron_out::call(self, other, out);
}

// aten::kron.out(Tensor self, Tensor other, *, Tensor(a!) out) -> Tensor(a!)
TORCH_API inline at::Tensor & kron_outf(const at::Tensor & self, const at::Tensor & other, at::Tensor & out) {
    return at::_ops::kron_out::call(self, other, out);
}

}
