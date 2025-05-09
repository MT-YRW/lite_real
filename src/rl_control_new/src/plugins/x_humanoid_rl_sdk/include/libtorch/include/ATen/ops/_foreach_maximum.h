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



#include <ATen/ops/_foreach_maximum_ops.h>

namespace at {


// aten::_foreach_maximum.List(Tensor[] tensors1, Tensor[] tensors2) -> Tensor[]
TORCH_API inline ::std::vector<at::Tensor> _foreach_maximum(at::TensorList tensors1, at::TensorList tensors2) {
    return at::_ops::_foreach_maximum_List::call(tensors1, tensors2);
}

}
