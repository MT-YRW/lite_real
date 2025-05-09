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



#include <ATen/ops/copy_sparse_to_sparse_ops.h>

namespace at {


// aten::copy_sparse_to_sparse_(Tensor(a!) self, Tensor src, bool non_blocking=False) -> Tensor(a!)
TORCH_API inline at::Tensor & copy_sparse_to_sparse_(at::Tensor & self, const at::Tensor & src, bool non_blocking=false) {
    return at::_ops::copy_sparse_to_sparse_::call(self, src, non_blocking);
}

// aten::copy_sparse_to_sparse.out(Tensor self, Tensor src, bool non_blocking=False, *, Tensor(a!) out) -> Tensor(a!)
TORCH_API inline at::Tensor & copy_sparse_to_sparse_out(at::Tensor & out, const at::Tensor & self, const at::Tensor & src, bool non_blocking=false) {
    return at::_ops::copy_sparse_to_sparse_out::call(self, src, non_blocking, out);
}

// aten::copy_sparse_to_sparse.out(Tensor self, Tensor src, bool non_blocking=False, *, Tensor(a!) out) -> Tensor(a!)
TORCH_API inline at::Tensor & copy_sparse_to_sparse_outf(const at::Tensor & self, const at::Tensor & src, bool non_blocking, at::Tensor & out) {
    return at::_ops::copy_sparse_to_sparse_out::call(self, src, non_blocking, out);
}

// aten::copy_sparse_to_sparse.functional(Tensor self, Tensor src, bool non_blocking=False) -> Tensor
TORCH_API inline at::Tensor copy_sparse_to_sparse_functional(const at::Tensor & self, const at::Tensor & src, bool non_blocking=false) {
    return at::_ops::copy_sparse_to_sparse_functional::call(self, src, non_blocking);
}

}
