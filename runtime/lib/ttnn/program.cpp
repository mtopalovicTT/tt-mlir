// SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0
#include "operations/ccl/all_gather.h"
#include "operations/context/get_device.h"
#include "operations/conv/conv2d.h"
#include "operations/creation/empty.h"
#include "operations/creation/full.h"
#include "operations/data_movement/concat.h"
#include "operations/data_movement/reshape.h"
#include "operations/data_movement/slice.h"
#include "operations/data_movement/transpose.h"
#include "operations/deletion/dealloc.h"
#include "operations/eltwise/binary/binary.h"
#include "operations/eltwise/binary/binary_composite.h"
#include "operations/eltwise/unary/unary.h"
#include "operations/eltwise/unary/unary_composite.h"
#include "operations/embedding/embedding.h"
#include "operations/layout/from_device.h"
#include "operations/layout/to_device.h"
#include "operations/layout/to_layout.h"
#include "operations/layout/to_memory_config.h"
#include "operations/layout/typecast.h"
#include "operations/matmul/matmul.h"
#include "operations/normalization/softmax.h"
#include "operations/pool/maxpool2d.h"
#include "operations/reduction/reduction.h"
#include "tt/runtime/detail/debug.h"
#include "tt/runtime/detail/logger.h"
#include "tt/runtime/ttnn/types.h"
#include "ttmlir/Target/TTNN/program_generated.h"

namespace tt::runtime::ttnn {
using LogType = ::tt::runtime::logger::LogType;

class ProgramExecutor {
public:
  ProgramExecutor(const TensorMap &liveTensors,
                  const std::unordered_set<uint32_t> &programInputs,
                  const std::unordered_set<uint32_t> &programOutputs,
                  ::ttnn::MeshDevice *meshDevice)
      : context(ProgramContext(liveTensors, programInputs, programOutputs,
                               meshDevice)) {}

  void execute(const ::tt::target::ttnn::Program *program) {
    for (const ::tt::target::ttnn::Operation *op : *program->operations()) {
      LOG_DEBUG(LogType::LogRuntimeTTNN,
                "Executing operation: ", op->debug_info()->c_str());
      runOperation(op);
    }
  }

  ProgramContext &getContext() { return context; }

private:
  ProgramContext context;
  void runOperation(const ::tt::target::ttnn::Operation *op);
  void runEltwiseOperation(const ::tt::target::ttnn::EltwiseOp *op);
};

void ProgramExecutor::runEltwiseOperation(
    const ::tt::target::ttnn::EltwiseOp *op) {
  auto runUnaryOp = [&]() {
    if (operations::unary::composite::isUnaryCompositeOp(op)) {
      return operations::unary::composite::run(op, context);
    }
    return operations::unary::run(op, context);
  };

  auto runBinaryOp = [&]() {
    if (operations::binary::composite::isBinaryCompositeOp(op)) {
      return operations::binary::composite::run(op, context);
    }
    return operations::binary::run(op, context);
  };

  if (operations::unary::isUnaryOp(op)) {
    return runUnaryOp();
  }

  if (operations::binary::isBinaryOp(op)) {
    return runBinaryOp();
  }

  throw std::invalid_argument("Unsupported Eltwise operation");
}

void ProgramExecutor::runOperation(const ::tt::target::ttnn::Operation *op) {
  switch (op->type_type()) {
  case ::tt::target::ttnn::OpType::GetDeviceOp: {
    auto childOp = op->type_as_GetDeviceOp();
    operations::context::run(childOp, context);
    break;
  }
  case ::tt::target::ttnn::OpType::ToMemoryConfigOp: {
    auto childOp = op->type_as_ToMemoryConfigOp();
    operations::layout::run(childOp, context);
    break;
  }
  case ::tt::target::ttnn::OpType::ToLayoutOp: {
    auto childOp = op->type_as_ToLayoutOp();
    operations::layout::run(childOp, context);
    break;
  }
  case ::tt::target::ttnn::OpType::TypecastOp: {
    auto childOp = op->type_as_TypecastOp();
    operations::layout::run(childOp, context);
    break;
  }
  case ::tt::target::ttnn::OpType::ToDeviceOp: {
    auto childOp = op->type_as_ToDeviceOp();
    operations::layout::run(childOp, context);
    break;
  }
  case ::tt::target::ttnn::OpType::FromDeviceOp: {
    auto childOp = op->type_as_FromDeviceOp();
    operations::layout::run(childOp, context);
    break;
  }
  case ::tt::target::ttnn::OpType::EmptyOp: {
    auto childOp = op->type_as_EmptyOp();
    operations::creation::run(childOp, context);
    break;
  }
  case ::tt::target::ttnn::OpType::FullOp: {
    auto childOp = op->type_as_FullOp();
    operations::creation::run(childOp, context);
    break;
  }
  case ::tt::target::ttnn::OpType::EltwiseOp: {
    auto childOp = op->type_as_EltwiseOp();
    runEltwiseOperation(childOp);
    break;
  }
  // ANCHOR: adding_an_op_matmul_runtime_program
  case ::tt::target::ttnn::OpType::MatmulOp: {
    auto childOp = op->type_as_MatmulOp();
    operations::matmul::run(childOp, context);
    break;
  }
  // ANCHOR_END: adding_an_op_matmul_runtime_program
  case ::tt::target::ttnn::OpType::ReductionOp: {
    auto childOp = op->type_as_ReductionOp();
    operations::reduction::run(childOp, context);
    break;
  }
  case ::tt::target::ttnn::OpType::EmbeddingOp: {
    auto childOp = op->type_as_EmbeddingOp();
    operations::embedding::run(childOp, context);
    break;
  }
  case ::tt::target::ttnn::OpType::SoftmaxOp: {
    auto childOp = op->type_as_SoftmaxOp();
    operations::normalization::run(childOp, context);
    break;
  }
  case ::tt::target::ttnn::OpType::TransposeOp: {
    auto childOp = op->type_as_TransposeOp();
    operations::data_movement::run(childOp, context);
    break;
  }
  case ::tt::target::ttnn::OpType::ConcatOp: {
    auto childOp = op->type_as_ConcatOp();
    operations::data_movement::run(childOp, context);
    break;
  }
  case ::tt::target::ttnn::OpType::ReshapeOp: {
    auto childOp = op->type_as_ReshapeOp();
    operations::data_movement::run(childOp, context);
    break;
  }
  case ::tt::target::ttnn::OpType::SliceOp: {
    auto childOp = op->type_as_SliceOp();
    operations::data_movement::run(childOp, context);
    break;
  }
  case ::tt::target::ttnn::OpType::Conv2dOp: {
    auto childOp = op->type_as_Conv2dOp();
    operations::conv::run(childOp, context);
    break;
  }
  case ::tt::target::ttnn::OpType::DeallocOp: {
    auto childOp = op->type_as_DeallocOp();
    return operations::deletion::run(childOp, context);
    break;
  }
  case ::tt::target::ttnn::OpType::MaxPool2dOp: {
    auto childOp = op->type_as_MaxPool2dOp();
    operations::pool::run(childOp, context);
    break;
  }
  case ::tt::target::ttnn::OpType::AllGatherOp: {
    auto childOp = op->type_as_AllGatherOp();
    operations::ccl::run(childOp, context);
    break;
  }
  default: {
    throw std::runtime_error("Unsupported operation type");
  }
  }

  if (auto callback = debug::Hooks::get().getOperatorCallback(); callback) {
    (*callback)(static_cast<const void *>(&context),
                static_cast<const void *>(op));
  }
}

// Nop is single input, output tensor where input is returned as output.
static bool handleNopProgram(::tt::target::ttnn::Program const *program,
                             std::vector<::ttnn::Tensor *> const &inputs,
                             std::vector<::ttnn::Tensor *> const &outputs) {

  bool isNop = program->inputs()->size() == 1 &&
               program->outputs()->size() == 1 &&
               program->inputs()->Get(0)->global_id() ==
                   program->outputs()->Get(0)->global_id();

  if (isNop) {
    void *src = ::tt::tt_metal::get_raw_host_data_ptr(*inputs.at(0));
    void *dst = ::tt::tt_metal::get_raw_host_data_ptr(*outputs.at(0));
    std::uint32_t size = outputs[0]->volume() * outputs[0]->element_size();
    std::memcpy(dst, src, size);
  }
  return isNop;
}

void runProgram(::ttnn::MeshDevice &meshDevice,
                ::tt::target::ttnn::Program const *program,
                std::vector<::ttnn::Tensor *> const &inputs,
                std::vector<::ttnn::Tensor *> const &outputs) {
  if (handleNopProgram(program, inputs, outputs)) {
    return;
  }
  TensorMap liveTensors;
  std::unordered_set<uint32_t> programInputs;
  int inputIndex = 0;
  LOG_ASSERT(program->inputs()->size() == inputs.size(),
             "Program input size mismatch: ", program->inputs()->size(),
             " != ", inputs.size());
  for (::tt::target::TensorRef const *input : *program->inputs()) {
    auto [iter, inserted] =
        liveTensors.try_emplace(input->global_id(), inputs[inputIndex++]);
    LOG_ASSERT(inserted, "Duplicate input tensor");
    programInputs.emplace(input->global_id());
  }

  int outputIndex = 0;
  std::unordered_set<uint32_t> programOutputs;
  LOG_ASSERT(program->outputs()->size() == outputs.size());
  for (::tt::target::TensorRef const *output : *program->outputs()) {
    auto [iter, inserted] =
        liveTensors.try_emplace(output->global_id(), outputs[outputIndex++]);
    LOG_ASSERT(inserted, "Duplicate output tensor");
    programOutputs.emplace(output->global_id());
  }
  ProgramExecutor executor(liveTensors, programInputs, programOutputs,
                           &meshDevice);
  executor.execute(program);
}

} // namespace tt::runtime::ttnn
