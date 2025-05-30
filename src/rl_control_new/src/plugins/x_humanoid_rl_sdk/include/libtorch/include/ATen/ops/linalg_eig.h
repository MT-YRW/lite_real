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



#include <ATen/ops/linalg_eig_ops.h>

namespace at {


// aten::linalg_eig(Tensor self) -> (Tensor eigenvalues, Tensor eigenvectors)
TORCH_API inline ::std::tuple<at::Tensor,at::Tensor> linalg_eig(const at::Tensor & self) {
    return at::_ops::linalg_eig::call(self);
}

// aten::linalg_eig.out(Tensor self, *, Tensor(a!) eigenvalues, Tensor(b!) eigenvectors) -> (Tensor(a!) eigenvalues, Tensor(b!) eigenvectors)
TORCH_API inline ::std::tuple<at::Tensor &,at::Tensor &> linalg_eig_out(at::Tensor & eigenvalues, at::Tensor & eigenvectors, const at::Tensor & self) {
    return at::_ops::linalg_eig_out::call(self, eigenvalues, eigenvectors);
}

// aten::linalg_eig.out(Tensor self, *, Tensor(a!) eigenvalues, Tensor(b!) eigenvectors) -> (Tensor(a!) eigenvalues, Tensor(b!) eigenvectors)
TORCH_API inline ::std::tuple<at::Tensor &,at::Tensor &> linalg_eig_outf(const at::Tensor & self, at::Tensor & eigenvalues, at::Tensor & eigenvectors) {
    return at::_ops::linalg_eig_out::call(self, eigenvalues, eigenvectors);
}

}
