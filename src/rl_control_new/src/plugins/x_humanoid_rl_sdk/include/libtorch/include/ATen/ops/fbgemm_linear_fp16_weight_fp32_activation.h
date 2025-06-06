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



#include <ATen/ops/fbgemm_linear_fp16_weight_fp32_activation_ops.h>

namespace at {


// aten::fbgemm_linear_fp16_weight_fp32_activation(Tensor input, Tensor packed_weight, Tensor bias) -> Tensor
TORCH_API inline at::Tensor fbgemm_linear_fp16_weight_fp32_activation(const at::Tensor & input, const at::Tensor & packed_weight, const at::Tensor & bias) {
    return at::_ops::fbgemm_linear_fp16_weight_fp32_activation::call(input, packed_weight, bias);
}

}
