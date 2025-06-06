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



#include <ATen/ops/empty_ops.h>

namespace at {


// aten::empty.names(int[] size, *, Dimname[]? names, ScalarType? dtype=None, Layout? layout=None, Device? device=None, bool? pin_memory=None, MemoryFormat? memory_format=None) -> Tensor
TORCH_API inline at::Tensor empty(at::IntArrayRef size, c10::optional<at::DimnameList> names, at::TensorOptions options={}, c10::optional<at::MemoryFormat> memory_format=c10::nullopt) {
    return at::_ops::empty_names::call(size, names, optTypeMetaToScalarType(options.dtype_opt()), options.layout_opt(), options.device_opt(), options.pinned_memory_opt(), c10::impl::check_tensor_options_and_extract_memory_format(options, memory_format));
}

// aten::empty.names(int[] size, *, Dimname[]? names, ScalarType? dtype=None, Layout? layout=None, Device? device=None, bool? pin_memory=None, MemoryFormat? memory_format=None) -> Tensor
TORCH_API inline at::Tensor empty(at::IntArrayRef size, c10::optional<at::DimnameList> names, c10::optional<at::ScalarType> dtype, c10::optional<at::Layout> layout, c10::optional<at::Device> device, c10::optional<bool> pin_memory, c10::optional<at::MemoryFormat> memory_format) {
    return at::_ops::empty_names::call(size, names, dtype, layout, device, pin_memory, memory_format);
}

// aten::empty.memory_format(int[] size, *, ScalarType? dtype=None, Layout? layout=None, Device? device=None, bool? pin_memory=None, MemoryFormat? memory_format=None) -> Tensor
TORCH_API inline at::Tensor empty(at::IntArrayRef size, at::TensorOptions options={}, c10::optional<at::MemoryFormat> memory_format=c10::nullopt) {
    return at::_ops::empty_memory_format::call(size, optTypeMetaToScalarType(options.dtype_opt()), options.layout_opt(), options.device_opt(), options.pinned_memory_opt(), c10::impl::check_tensor_options_and_extract_memory_format(options, memory_format));
}

// aten::empty.memory_format(int[] size, *, ScalarType? dtype=None, Layout? layout=None, Device? device=None, bool? pin_memory=None, MemoryFormat? memory_format=None) -> Tensor
TORCH_API inline at::Tensor empty(at::IntArrayRef size, c10::optional<at::ScalarType> dtype, c10::optional<at::Layout> layout, c10::optional<at::Device> device, c10::optional<bool> pin_memory, c10::optional<at::MemoryFormat> memory_format) {
    return at::_ops::empty_memory_format::call(size, dtype, layout, device, pin_memory, memory_format);
}

// aten::empty.out(int[] size, *, MemoryFormat? memory_format=None, Tensor(a!) out) -> Tensor(a!)
TORCH_API inline at::Tensor & empty_out(at::Tensor & out, at::IntArrayRef size, c10::optional<at::MemoryFormat> memory_format=c10::nullopt) {
    return at::_ops::empty_out::call(size, memory_format, out);
}

// aten::empty.out(int[] size, *, MemoryFormat? memory_format=None, Tensor(a!) out) -> Tensor(a!)
TORCH_API inline at::Tensor & empty_outf(at::IntArrayRef size, c10::optional<at::MemoryFormat> memory_format, at::Tensor & out) {
    return at::_ops::empty_out::call(size, memory_format, out);
}

}
