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



#include <ATen/ops/_foreach_addcdiv_ops.h>

namespace at {


// aten::_foreach_addcdiv_.Scalar(Tensor(a!)[] self, Tensor[] tensor1, Tensor[] tensor2, Scalar value=1) -> ()
TORCH_API inline void _foreach_addcdiv_(at::TensorList self, at::TensorList tensor1, at::TensorList tensor2, const at::Scalar & value=1) {
    return at::_ops::_foreach_addcdiv__Scalar::call(self, tensor1, tensor2, value);
}

// aten::_foreach_addcdiv_.ScalarList(Tensor(a!)[] self, Tensor[] tensor1, Tensor[] tensor2, Scalar[] scalars) -> ()
TORCH_API inline void _foreach_addcdiv_(at::TensorList self, at::TensorList tensor1, at::TensorList tensor2, at::ArrayRef<at::Scalar> scalars) {
    return at::_ops::_foreach_addcdiv__ScalarList::call(self, tensor1, tensor2, scalars);
}

// aten::_foreach_addcdiv.Scalar(Tensor[] input, Tensor[] tensor1, Tensor[] tensor2, Scalar value=1) -> Tensor[]
TORCH_API inline ::std::vector<at::Tensor> _foreach_addcdiv(at::TensorList input, at::TensorList tensor1, at::TensorList tensor2, const at::Scalar & value=1) {
    return at::_ops::_foreach_addcdiv_Scalar::call(input, tensor1, tensor2, value);
}

// aten::_foreach_addcdiv.ScalarList(Tensor[] input, Tensor[] tensor1, Tensor[] tensor2, Scalar[] scalars) -> Tensor[]
TORCH_API inline ::std::vector<at::Tensor> _foreach_addcdiv(at::TensorList input, at::TensorList tensor1, at::TensorList tensor2, at::ArrayRef<at::Scalar> scalars) {
    return at::_ops::_foreach_addcdiv_ScalarList::call(input, tensor1, tensor2, scalars);
}

// aten::_foreach_addcdiv.Scalar_out(Tensor[] self, Tensor[] tensor1, Tensor[] tensor2, Scalar value=1, *, Tensor(a!)[] out) -> ()
TORCH_API inline void _foreach_addcdiv_out(at::TensorList out, at::TensorList self, at::TensorList tensor1, at::TensorList tensor2, const at::Scalar & value=1) {
    return at::_ops::_foreach_addcdiv_Scalar_out::call(self, tensor1, tensor2, value, out);
}

// aten::_foreach_addcdiv.Scalar_out(Tensor[] self, Tensor[] tensor1, Tensor[] tensor2, Scalar value=1, *, Tensor(a!)[] out) -> ()
TORCH_API inline void _foreach_addcdiv_outf(at::TensorList self, at::TensorList tensor1, at::TensorList tensor2, const at::Scalar & value, at::TensorList out) {
    return at::_ops::_foreach_addcdiv_Scalar_out::call(self, tensor1, tensor2, value, out);
}

// aten::_foreach_addcdiv.Scalar_functional(Tensor[] self, Tensor[] tensor1, Tensor[] tensor2, Scalar value=1) -> Tensor[] self_out
TORCH_API inline ::std::vector<at::Tensor> _foreach_addcdiv_functional(at::TensorList self, at::TensorList tensor1, at::TensorList tensor2, const at::Scalar & value=1) {
    return at::_ops::_foreach_addcdiv_Scalar_functional::call(self, tensor1, tensor2, value);
}

// aten::_foreach_addcdiv.ScalarList_out(Tensor[] self, Tensor[] tensor1, Tensor[] tensor2, Scalar[] scalars, *, Tensor(a!)[] out) -> ()
TORCH_API inline void _foreach_addcdiv_out(at::TensorList out, at::TensorList self, at::TensorList tensor1, at::TensorList tensor2, at::ArrayRef<at::Scalar> scalars) {
    return at::_ops::_foreach_addcdiv_ScalarList_out::call(self, tensor1, tensor2, scalars, out);
}

// aten::_foreach_addcdiv.ScalarList_out(Tensor[] self, Tensor[] tensor1, Tensor[] tensor2, Scalar[] scalars, *, Tensor(a!)[] out) -> ()
TORCH_API inline void _foreach_addcdiv_outf(at::TensorList self, at::TensorList tensor1, at::TensorList tensor2, at::ArrayRef<at::Scalar> scalars, at::TensorList out) {
    return at::_ops::_foreach_addcdiv_ScalarList_out::call(self, tensor1, tensor2, scalars, out);
}

// aten::_foreach_addcdiv.ScalarList_functional(Tensor[] self, Tensor[] tensor1, Tensor[] tensor2, Scalar[] scalars) -> Tensor[] self_out
TORCH_API inline ::std::vector<at::Tensor> _foreach_addcdiv_functional(at::TensorList self, at::TensorList tensor1, at::TensorList tensor2, at::ArrayRef<at::Scalar> scalars) {
    return at::_ops::_foreach_addcdiv_ScalarList_functional::call(self, tensor1, tensor2, scalars);
}

}
