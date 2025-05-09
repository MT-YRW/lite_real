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



#include <ATen/ops/_transform_bias_rescale_qkv_ops.h>

namespace at {


// aten::_transform_bias_rescale_qkv(Tensor qkv, Tensor qkv_bias, int num_heads) -> (Tensor, Tensor, Tensor)
TORCH_API inline ::std::tuple<at::Tensor,at::Tensor,at::Tensor> _transform_bias_rescale_qkv(const at::Tensor & qkv, const at::Tensor & qkv_bias, int64_t num_heads) {
    return at::_ops::_transform_bias_rescale_qkv::call(qkv, qkv_bias, num_heads);
}

}
