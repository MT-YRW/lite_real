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



#include <ATen/ops/linalg_ldl_solve_ops.h>

namespace at {


// aten::linalg_ldl_solve(Tensor LD, Tensor pivots, Tensor B, *, bool hermitian=False) -> Tensor
TORCH_API inline at::Tensor linalg_ldl_solve(const at::Tensor & LD, const at::Tensor & pivots, const at::Tensor & B, bool hermitian=false) {
    return at::_ops::linalg_ldl_solve::call(LD, pivots, B, hermitian);
}

// aten::linalg_ldl_solve.out(Tensor LD, Tensor pivots, Tensor B, *, bool hermitian=False, Tensor(a!) out) -> Tensor(a!)
TORCH_API inline at::Tensor & linalg_ldl_solve_out(at::Tensor & out, const at::Tensor & LD, const at::Tensor & pivots, const at::Tensor & B, bool hermitian=false) {
    return at::_ops::linalg_ldl_solve_out::call(LD, pivots, B, hermitian, out);
}

// aten::linalg_ldl_solve.out(Tensor LD, Tensor pivots, Tensor B, *, bool hermitian=False, Tensor(a!) out) -> Tensor(a!)
TORCH_API inline at::Tensor & linalg_ldl_solve_outf(const at::Tensor & LD, const at::Tensor & pivots, const at::Tensor & B, bool hermitian, at::Tensor & out) {
    return at::_ops::linalg_ldl_solve_out::call(LD, pivots, B, hermitian, out);
}

}
