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



#include <ATen/ops/uniform_ops.h>

namespace at {


// aten::uniform.out(Tensor self, float from=0, float to=1, *, Generator? generator=None, Tensor(a!) out) -> Tensor(a!)
TORCH_API inline at::Tensor & uniform_out(at::Tensor & out, const at::Tensor & self, double from=0, double to=1, c10::optional<at::Generator> generator=c10::nullopt) {
    return at::_ops::uniform_out::call(self, from, to, generator, out);
}

// aten::uniform.out(Tensor self, float from=0, float to=1, *, Generator? generator=None, Tensor(a!) out) -> Tensor(a!)
TORCH_API inline at::Tensor & uniform_outf(const at::Tensor & self, double from, double to, c10::optional<at::Generator> generator, at::Tensor & out) {
    return at::_ops::uniform_out::call(self, from, to, generator, out);
}

// aten::uniform.functional(Tensor self, float from=0, float to=1, *, Generator? generator=None) -> Tensor
TORCH_API inline at::Tensor uniform_functional(const at::Tensor & self, double from=0, double to=1, c10::optional<at::Generator> generator=c10::nullopt) {
    return at::_ops::uniform_functional::call(self, from, to, generator);
}

}
