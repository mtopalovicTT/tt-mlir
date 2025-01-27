// SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0
#include "binary.h"
#include "tt/runtime/detail/logger.h"
#include "tt/runtime/detail/ttnn.h"
#include "tt/runtime/ttnn/operations/eltwise/binary/utils.h"
#include "tt/runtime/ttnn/operations/utils.h"
#include "ttnn/operations/eltwise/binary/binary_composite.hpp"

namespace tt::runtime::ttnn::operations::binary {

static void runEltwiseBinaryOP(
    const ::tt::target::ttnn::EltwiseOp *op, ProgramTensorPool &tensorPool,
    std::function<::ttnn::Tensor(
        const ::ttnn::Tensor &, const ::ttnn::Tensor &,
        const std::optional<const ::ttnn::DataType> &,
        const std::optional<::tt::tt_metal::MemoryConfig> &,
        std::optional<::ttnn::Tensor>,
        std::optional<::ttnn::operations::unary::FusedActivations>,
        std::optional<::ttnn::operations::unary::UnaryWithParam>)>
        ttnnOp) {

  ::ttnn::Tensor *lhs = nullptr;
  ::ttnn::Tensor *rhs = nullptr;
  getEltwiseBinaryOPInputTensors(op, tensorPool, &lhs, &rhs);

  ::ttnn::DataType outputDataType = utils::getDataType(op->out());
  ::tt::tt_metal::MemoryConfig outputMemoryConfig =
      utils::createMemoryConfig(op->out());

  ::ttnn::Tensor out = ttnnOp(*lhs, *rhs, outputDataType, outputMemoryConfig,
                              std::nullopt, std::nullopt, std::nullopt);
  tensorPool.insert_or_assign(op->out()->global_id(), out);
}

void run(const ::tt::target::ttnn::EltwiseOp *op, ProgramContext &context) {
  ProgramTensorPool &tensorPool = context.getTensorPool();
  switch (op->type()) {
  /* Eltwise Binary */
  case ::tt::target::ttnn::EltwiseOpType::Add: {
    runEltwiseBinaryOP(op, tensorPool, ::ttnn::add);
    break;
  }
  case ::tt::target::ttnn::EltwiseOpType::LogicalAnd: {
    runEltwiseBinaryOP(op, tensorPool, ::ttnn::logical_and);
    break;
  }
  case ::tt::target::ttnn::EltwiseOpType::LogicalOr: {
    runEltwiseBinaryOP(op, tensorPool, ::ttnn::logical_or);
    break;
  }
  case ::tt::target::ttnn::EltwiseOpType::Multiply: {
    runEltwiseBinaryOP(op, tensorPool, ::ttnn::multiply);
    break;
  }
  case ::tt::target::ttnn::EltwiseOpType::Subtract: {
    runEltwiseBinaryOP(op, tensorPool, ::ttnn::subtract);
    break;
  }
  case ::tt::target::ttnn::EltwiseOpType::Equal: {
    runEltwiseBinaryOP(op, tensorPool, ::ttnn::eq);
    break;
  }
  case ::tt::target::ttnn::EltwiseOpType::NotEqual: {
    runEltwiseBinaryOP(op, tensorPool, ::ttnn::ne);
    break;
  }
  case ::tt::target::ttnn::EltwiseOpType::GreaterEqual: {
    runEltwiseBinaryOP(op, tensorPool, ::ttnn::ge);
    break;
  }
  case ::tt::target::ttnn::EltwiseOpType::GreaterThan: {
    runEltwiseBinaryOP(op, tensorPool, ::ttnn::gt);
    break;
  }
  case ::tt::target::ttnn::EltwiseOpType::LessEqual: {
    runEltwiseBinaryOP(op, tensorPool, ::ttnn::le);
    break;
  }
  case ::tt::target::ttnn::EltwiseOpType::LessThan: {
    runEltwiseBinaryOP(op, tensorPool, ::ttnn::lt);
    break;
  }
  case ::tt::target::ttnn::EltwiseOpType::Div: {
    runEltwiseBinaryOP(op, tensorPool, ::ttnn::divide);
    break;
  }
  default:
    throw std::invalid_argument("Unsupported Eltwise Binary operation");
  }
}

} // namespace tt::runtime::ttnn::operations::binary
