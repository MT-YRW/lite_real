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



#include <ATen/ops/linalg_lu_factor_ops.h>

namespace at {


// aten::linalg_lu_factor(Tensor A, *, bool pivot=True) -> (Tensor LU, Tensor pivots)
TORCH_API inline ::std::tuple<at::Tensor,at::Tensor> linalg_lu_factor(const at::Tensor & A, bool pivot=true) {
    return at::_ops::linalg_lu_factor::call(A, pivot);
}

// aten::linalg_lu_factor.out(Tensor A, *, bool pivot=True, Tensor(a!) LU, Tensor(b!) pivots) -> (Tensor(a!) LU, Tensor(b!) pivots)
TORCH_API inline ::std::tuple<at::Tensor &,at::Tensor &> linalg_lu_factor_out(at::Tensor & LU, at::Tensor & pivots, const at::Tensor & A, bool pivot=true) {
    return at::_ops::linalg_lu_factor_out::call(A, pivot, LU, pivots);
}

// aten::linalg_lu_factor.out(Tensor A, *, bool pivot=True, Tensor(a!) LU, Tensor(b!) pivots) -> (Tensor(a!) LU, Tensor(b!) pivots)
TORCH_API inline ::std::tuple<at::Tensor &,at::Tensor &> linalg_lu_factor_outf(const at::Tensor & A, bool pivot, at::Tensor & LU, at::Tensor & pivots) {
    return at::_ops::linalg_lu_factor_out::call(A, pivot, LU, pivots);
}

}
