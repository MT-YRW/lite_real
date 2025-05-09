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



#include <ATen/ops/stft_ops.h>

namespace at {


// aten::stft(Tensor self, int n_fft, int? hop_length=None, int? win_length=None, Tensor? window=None, bool normalized=False, bool? onesided=None, bool? return_complex=None) -> Tensor
TORCH_API inline at::Tensor stft(const at::Tensor & self, int64_t n_fft, c10::optional<int64_t> hop_length, c10::optional<int64_t> win_length, const c10::optional<at::Tensor> & window, bool normalized, c10::optional<bool> onesided=c10::nullopt, c10::optional<bool> return_complex=c10::nullopt) {
    return at::_ops::stft::call(self, n_fft, hop_length, win_length, window, normalized, onesided, return_complex);
}

// aten::stft.center(Tensor self, int n_fft, int? hop_length=None, int? win_length=None, Tensor? window=None, bool center=True, str pad_mode="reflect", bool normalized=False, bool? onesided=None, bool? return_complex=None) -> Tensor
TORCH_API inline at::Tensor stft(const at::Tensor & self, int64_t n_fft, c10::optional<int64_t> hop_length=c10::nullopt, c10::optional<int64_t> win_length=c10::nullopt, const c10::optional<at::Tensor> & window={}, bool center=true, c10::string_view pad_mode="reflect", bool normalized=false, c10::optional<bool> onesided=c10::nullopt, c10::optional<bool> return_complex=c10::nullopt) {
    return at::_ops::stft_center::call(self, n_fft, hop_length, win_length, window, center, pad_mode, normalized, onesided, return_complex);
}

}
