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



#include <ATen/ops/_nested_from_padded_and_nested_example_ops.h>

namespace at {


// aten::_nested_from_padded_and_nested_example(Tensor padded, Tensor nt_example) -> Tensor
TORCH_API inline at::Tensor _nested_from_padded_and_nested_example(const at::Tensor & padded, const at::Tensor & nt_example) {
    return at::_ops::_nested_from_padded_and_nested_example::call(padded, nt_example);
}

}
