// SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "embedding.h"
#include "tt/runtime/detail/ttnn.h"
#include "tt/runtime/ttnn/operations/utils.h"

namespace tt::runtime::ttnn::operations::embedding {
void run(const ::tt::target::ttnn::EmbeddingOp *op, ProgramContext &context) {

  ProgramTensorPool &tensorPool = context.tensorPool;
  const ::ttnn::Tensor &input = tensorPool.at(op->input()->global_id());
  const ::ttnn::Tensor &weight = tensorPool.at(op->weight()->global_id());
  // default params for embedding op
  std::optional<int> padToken = std::nullopt;
  ::tt::tt_metal::Layout layout = ::ttnn::ROW_MAJOR_LAYOUT;
  auto embeddingsType = ::ttnn::operations::embedding::EmbeddingsType::GENERIC;
  ::ttnn::DataType outputDataType = utils::getDataType(op->output());
  ::ttnn::MemoryConfig outputMemoryConfig =
      utils::createMemoryConfig(op->output());
  ::ttnn::Tensor out =
      ::ttnn::embedding(input, weight, padToken, layout, embeddingsType,
                        outputDataType, outputMemoryConfig);
  tensorPool.insert_or_assign(op->output()->global_id(), out);
}
} // namespace tt::runtime::ttnn::operations::embedding