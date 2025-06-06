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



#include <ATen/ops/lstsq_ops.h>

namespace at {


// aten::lstsq.X(Tensor self, Tensor A, *, Tensor(a!) X, Tensor(b!) qr) -> (Tensor(a!) solution, Tensor(b!) QR)
TORCH_API inline ::std::tuple<at::Tensor &,at::Tensor &> lstsq_out(at::Tensor & X, at::Tensor & qr, const at::Tensor & self, const at::Tensor & A) {
    return at::_ops::lstsq_X::call(self, A, X, qr);
}

// aten::lstsq.X(Tensor self, Tensor A, *, Tensor(a!) X, Tensor(b!) qr) -> (Tensor(a!) solution, Tensor(b!) QR)
TORCH_API inline ::std::tuple<at::Tensor &,at::Tensor &> lstsq_outf(const at::Tensor & self, const at::Tensor & A, at::Tensor & X, at::Tensor & qr) {
    return at::_ops::lstsq_X::call(self, A, X, qr);
}

// aten::lstsq(Tensor self, Tensor A) -> (Tensor solution, Tensor QR)
TORCH_API inline ::std::tuple<at::Tensor,at::Tensor> lstsq(const at::Tensor & self, const at::Tensor & A) {
    return at::_ops::lstsq::call(self, A);
}

}
