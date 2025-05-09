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



#include <ATen/ops/_transformer_encoder_layer_fwd_ops.h>

namespace at {


// aten::_transformer_encoder_layer_fwd(Tensor src, int embed_dim, int num_heads, Tensor qkv_weight, Tensor qkv_bias, Tensor proj_weight, Tensor proj_bias, bool use_gelu, bool norm_first, float eps, Tensor norm_weight_1, Tensor norm_bias_1, Tensor norm_weight_2, Tensor norm_bias_2, Tensor ffn_weight_1, Tensor ffn_bias_1, Tensor ffn_weight_2, Tensor ffn_bias_2, Tensor? mask=None) -> Tensor
TORCH_API inline at::Tensor _transformer_encoder_layer_fwd(const at::Tensor & src, int64_t embed_dim, int64_t num_heads, const at::Tensor & qkv_weight, const at::Tensor & qkv_bias, const at::Tensor & proj_weight, const at::Tensor & proj_bias, bool use_gelu, bool norm_first, double eps, const at::Tensor & norm_weight_1, const at::Tensor & norm_bias_1, const at::Tensor & norm_weight_2, const at::Tensor & norm_bias_2, const at::Tensor & ffn_weight_1, const at::Tensor & ffn_bias_1, const at::Tensor & ffn_weight_2, const at::Tensor & ffn_bias_2, const c10::optional<at::Tensor> & mask={}) {
    return at::_ops::_transformer_encoder_layer_fwd::call(src, embed_dim, num_heads, qkv_weight, qkv_bias, proj_weight, proj_bias, use_gelu, norm_first, eps, norm_weight_1, norm_bias_1, norm_weight_2, norm_bias_2, ffn_weight_1, ffn_bias_1, ffn_weight_2, ffn_bias_2, mask);
}

}
