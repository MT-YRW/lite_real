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



#include <ATen/ops/glu_backward_ops.h>

namespace at {


// aten::glu_backward.grad_input(Tensor grad_output, Tensor self, int dim, *, Tensor(a!) grad_input) -> Tensor(a!)
TORCH_API inline at::Tensor & glu_backward_out(at::Tensor & grad_input, const at::Tensor & grad_output, const at::Tensor & self, int64_t dim) {
    return at::_ops::glu_backward_grad_input::call(grad_output, self, dim, grad_input);
}

// aten::glu_backward.grad_input(Tensor grad_output, Tensor self, int dim, *, Tensor(a!) grad_input) -> Tensor(a!)
TORCH_API inline at::Tensor & glu_backward_outf(const at::Tensor & grad_output, const at::Tensor & self, int64_t dim, at::Tensor & grad_input) {
    return at::_ops::glu_backward_grad_input::call(grad_output, self, dim, grad_input);
}

// aten::glu_backward(Tensor grad_output, Tensor self, int dim) -> Tensor
TORCH_API inline at::Tensor glu_backward(const at::Tensor & grad_output, const at::Tensor & self, int64_t dim) {
    return at::_ops::glu_backward::call(grad_output, self, dim);
}

}
