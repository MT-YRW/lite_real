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



#include <ATen/ops/linalg_matrix_exp_ops.h>

namespace at {


// aten::linalg_matrix_exp(Tensor self) -> Tensor
TORCH_API inline at::Tensor linalg_matrix_exp(const at::Tensor & self) {
    return at::_ops::linalg_matrix_exp::call(self);
}

}
