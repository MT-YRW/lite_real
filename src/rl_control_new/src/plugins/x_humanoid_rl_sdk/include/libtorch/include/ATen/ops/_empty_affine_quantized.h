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



#include <ATen/ops/_empty_affine_quantized_ops.h>

namespace at {


// aten::_empty_affine_quantized(int[] size, *, ScalarType? dtype=None, Layout? layout=None, Device? device=None, bool? pin_memory=None, float scale=1, int zero_point=0, MemoryFormat? memory_format=contiguous_format) -> Tensor
TORCH_API inline at::Tensor _empty_affine_quantized(at::IntArrayRef size, at::TensorOptions options={}, double scale=1, int64_t zero_point=0, c10::optional<at::MemoryFormat> memory_format=MemoryFormat::Contiguous) {
    return at::_ops::_empty_affine_quantized::call(size, optTypeMetaToScalarType(options.dtype_opt()), options.layout_opt(), options.device_opt(), options.pinned_memory_opt(), scale, zero_point, c10::impl::check_tensor_options_and_extract_memory_format(options, memory_format));
}

// aten::_empty_affine_quantized(int[] size, *, ScalarType? dtype=None, Layout? layout=None, Device? device=None, bool? pin_memory=None, float scale=1, int zero_point=0, MemoryFormat? memory_format=contiguous_format) -> Tensor
TORCH_API inline at::Tensor _empty_affine_quantized(at::IntArrayRef size, c10::optional<at::ScalarType> dtype, c10::optional<at::Layout> layout, c10::optional<at::Device> device, c10::optional<bool> pin_memory, double scale, int64_t zero_point, c10::optional<at::MemoryFormat> memory_format) {
    return at::_ops::_empty_affine_quantized::call(size, dtype, layout, device, pin_memory, scale, zero_point, memory_format);
}

}
