// SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "linear.h"
#include "tt/runtime/detail/logger.h"
#include "tt/runtime/ttnn/operations/utils.h"
#include <optional>

namespace tt::runtime::ttnn::operations::linear {

void run(const ::tt::target::ttnn::LinearOp *op, ProgramContext &context) {
  ProgramTensorPool &tensorPool = context.getTensorPool();
  const ::ttnn::Tensor &lhs = tensorPool.at(op->in0()->global_id());
  const ::ttnn::Tensor &rhs = tensorPool.at(op->in1()->global_id());
  DEBUG_ASSERT(lhs.is_allocated());
  DEBUG_ASSERT(rhs.is_allocated());

  std::optional<::ttnn::Tensor> bias =
    op->bias() ? std::make_optional(tensorPool.at(op->bias()->global_id()))
               : std::nullopt;

  ::ttnn::DataType outputDataType = utils::getDataType(op->out());
  ::tt::tt_metal::MemoryConfig outputMemoryConfig =
      utils::createMemoryConfig(op->out());

  const std::optional<const ::tt::tt_metal::MemoryConfig> memoryConfig =
      std::make_optional(outputMemoryConfig);

  const std::optional<const ::ttnn::DataType> dtype =
      std::make_optional(outputDataType);

  ::ttnn::Tensor out = ::ttnn::linear(
      lhs, rhs, bias, /*transposeA*/ false, /*transposeB*/ false, memoryConfig, dtype,
      /*programConfig*/ std::nullopt, /*activation*/ std::nullopt,
      /*computeKernelConfig*/ std::nullopt, /*coreGrid*/ std::nullopt);

  tensorPool.insert_or_assign(op->out()->global_id(), out);
}
} // namespace tt::runtime::ttnn::operations::linear
