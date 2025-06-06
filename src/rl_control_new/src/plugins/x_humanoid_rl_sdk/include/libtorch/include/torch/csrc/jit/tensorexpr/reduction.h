#pragma once

#include <torch/csrc/jit/tensorexpr/expr.h>
#include <torch/csrc/jit/tensorexpr/ir.h>
#include <torch/csrc/jit/tensorexpr/ir_printer.h>
#include <torch/csrc/jit/tensorexpr/types.h>

#include <functional>
#include <vector>

namespace torch {
namespace jit {
namespace tensorexpr {

using ParameterList = const std::vector<VarHandle>;
using ReduceInteraction = std::function<ExprHandle(ExprHandle, ExprHandle)>;

// A Reducer is a user interface describing a particular reduction
// operation. It has three components: An initialization value, a way of
// interacting each value with the accumulation, and a method for obtaining the
// current value to be reduced. It is materialized into a ReduceOp when loop
// variables are known.
class TORCH_API Reducer {
 public:
  Reducer(ExprHandle init, ReduceInteraction& interaction)
      : init_(init.node()), interaction_(interaction) {}

  template <typename RI>
  Reducer(ExprHandle init, RI interaction) : init_(init.node()) {
    interaction_ = interaction;
  }
  virtual ~Reducer() = default;

  ExprPtr initializer() const {
    return init_;
  }

  ExprHandle operator()(
      BufHandle result_buf,
      ExprHandle body,
      const std::vector<ExprHandle>& output,
      const std::vector<VarHandle>& inner) const;

  ReduceOpPtr operator()(
      BufPtr result_buf,
      ExprPtr body,
      const std::vector<ExprPtr>& output,
      const std::vector<VarPtr>& inner) const;

  // Polymorphic handling of Body functions with a variety of parameters.
  static ExprHandle getReduceBody(
      const std::function<ExprHandle(ParameterList&)>& func,
      const std::vector<VarHandle>& vars) {
    return func(vars);
  }

  static ExprHandle getReduceBody(
      const std::function<ExprHandle(const VarHandle&)>& func,
      const std::vector<VarHandle>& vars) {
    if (vars.size() != 1) {
      throw malformed_input("mismatch between reduce body and arg size (1)");
    }

    return func(vars[0]);
  }

  static ExprHandle getReduceBody(
      const std::function<ExprHandle(const VarHandle&, const VarHandle&)>& func,
      const std::vector<VarHandle>& vars) {
    if (vars.size() != 2) {
      throw malformed_input("mismatch between reduce body and arg size (2)");
    }
    return func(vars[0], vars[1]);
  }

  static ExprHandle getReduceBody(
      const std::function<
          ExprHandle(const VarHandle&, const VarHandle&, const VarHandle&)>&
          func,
      const std::vector<VarHandle>& vars) {
    if (vars.size() != 3) {
      throw malformed_input("mismatch between reduce body and arg size (3)");
    }
    return func(vars[0], vars[1], vars[2]);
  }

  static ExprHandle getReduceBody(
      const std::function<ExprHandle(
          const VarHandle&,
          const VarHandle&,
          const VarHandle&,
          const VarHandle&)>& func,
      const std::vector<VarHandle>& vars) {
    if (vars.size() != 4) {
      throw malformed_input("mismatch between reduce body and arg size (4)");
    }
    return func(vars[0], vars[1], vars[2], vars[3]);
  }

  // Completes the reduction operator by applying the interaction function to
  // the accumulation and the body expression.
  static ExprPtr complete(
      BufPtr accumulator,
      ReduceInteraction interaction,
      ExprHandle body,
      const std::vector<ExprPtr>& output_args,
      const std::vector<VarPtr>& reduce_args) {
    ExprHandle accum =
        ExprHandle(alloc<Load>(body.dtype(), accumulator, output_args));
    auto e = interaction(accum, body);
    return e.node();
  }
  static ExprHandle complete(
      BufHandle accumulator,
      ReduceInteraction interaction,
      ExprHandle body,
      const std::vector<ExprHandle>& output_args,
      const std::vector<VarHandle>& reduce_args) {
    ExprHandle accum = Load::make(body.dtype(), accumulator, output_args);
    auto e = interaction(accum, body);
    return e;
  }

 private:
  ExprPtr init_;
  ReduceInteraction interaction_;
};

// An expression representing a Reduction operation (e.g. Sum, Max) broken into
// it's component parts: initialization, accumulation var, acquisition of value
// to be reduced and interaction.
//
// This is intended to be expanded in the loopnest and not make it to codegen.
class TORCH_API ReduceOp : public ExprNode<ReduceOp> {
 public:
  // NOLINTNEXTLINE(cppcoreguidelines-pro-type-member-init)
  ReduceOp(
      ExprPtr body,
      std::vector<VarPtr> reduce_args,
      const Reducer& reducer)
      : ExprNodeBase(body->dtype()),
        body_(body),
        reduce_args_(std::move(reduce_args)),
        reducer_(reducer) {}
  static ExprHandle make(
      ExprHandle body,
      std::vector<VarHandle> reduce_args,
      const Reducer& reducer);

  // return the body expression which obtains the value to be reduced.
  ExprPtr body() const {
    return body_;
  }

  // Returns the original Reducer factory that can create ReduceOps.
  const Reducer& reducer() const {
    return reducer_;
  }

  // returns variables associated with the axes of reduction.
  const std::vector<VarPtr>& reduce_args() const {
    return reduce_args_;
  }

 private:
  ExprPtr body_;
  std::vector<VarPtr> reduce_args_;
  const Reducer reducer_;
};

class Sum : public Reducer {
 public:
  Sum()
      : Reducer(ExprHandle(0), [](ExprHandle a, ExprHandle b) {
          return a + b;
        }) {}
};

inline ExprHandle maximumVal(ScalarType type) {
  switch (type) {
#define MAX_BY_TYPE_CASE(Type, Name) \
  case ScalarType::Name:             \
    return ExprHandle(std::numeric_limits<Type>::max());
    AT_FORALL_SCALAR_TYPES_AND3(Bool, Half, BFloat16, MAX_BY_TYPE_CASE)
#undef MAX_BY_TYPE_CASE
    default:
      throw unsupported_dtype();
  }
  return ExprHandle();
}

inline ExprHandle minimumVal(ScalarType type) {
  switch (type) {
#define MAX_BY_TYPE_CASE(Type, Name) \
  case ScalarType::Name:             \
    return ExprHandle(std::numeric_limits<Type>::min());
    AT_FORALL_SCALAR_TYPES_AND3(Bool, Half, BFloat16, MAX_BY_TYPE_CASE)
#undef MAX_BY_TYPE_CASE
    default:
      throw unsupported_dtype();
  }
}

class Maximum : public Reducer {
 public:
  // TODO possible to remove this arg by deferring the init value until we
  // know the dtype of the body.
  Maximum(Dtype dtype)
      : Reducer(
            minimumVal(dtype.scalar_type()),
            [](ExprHandle a, ExprHandle b) { return Max::make(a, b, true); }) {}
  Maximum(ExprHandle initializer)
      : Reducer(initializer, [](ExprHandle a, ExprHandle b) {
          return Max::make(a, b, true);
        }) {}
};

class Minimum : public Reducer {
 public:
  Minimum(Dtype dtype)
      : Reducer(
            maximumVal(dtype.scalar_type()),
            [](ExprHandle a, ExprHandle b) { return Min::make(a, b, true); }) {}
  Minimum(ExprHandle initializer)
      : Reducer(initializer, [](ExprHandle a, ExprHandle b) {
          return Min::make(a, b, true);
        }) {}
};

class ReductionExpander : public IRMutator {
 public:
  StmtPtr expand(StmtPtr s) {
    return s->accept_mutator(this);
  }

  ExprPtr mutate(ReduceOpPtr v) override {
    return v->body();
  }
};

} // namespace tensorexpr
} // namespace jit
} // namespace torch
