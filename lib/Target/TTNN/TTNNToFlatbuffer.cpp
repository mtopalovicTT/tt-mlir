// SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttmlir/Target/TTNN/TTNNToFlatbuffer.h"

#include "ttmlir/Dialect/TT/IR/TT.h"
#include "ttmlir/Dialect/TT/IR/TTOpsTypes.h"
#include "ttmlir/Dialect/TTKernel/IR/TTKernel.h"
#include "ttmlir/Dialect/TTKernel/IR/TTKernelOps.h"
#include "ttmlir/Dialect/TTKernel/IR/TTKernelOpsTypes.h"
#include "ttmlir/Dialect/TTNN/IR/TTNNOps.h"
#include "ttmlir/Dialect/TTNN/IR/TTNNOpsAttrs.h"
#include "ttmlir/Dialect/TTNN/IR/TTNNOpsTypes.h"
#include "ttmlir/Dialect/TTNN/Transforms/Passes.h"
#include "ttmlir/Dialect/TTNN/Transforms/TTNNToCpp.h"
#include "ttmlir/Target/Common/Target.h"
#include "ttmlir/Target/TTNN/Target.h"
#include "ttmlir/Target/TTNN/binary_generated.h"
#include "ttmlir/Target/TTNN/program_generated.h"
#include "ttmlir/Target/TTNN/utils.h"
#include "ttmlir/Target/Utils/FlatbufferObjectCache.h"
#include "ttmlir/Target/Utils/FuncOpToProgram.h"
#include "ttmlir/Target/Utils/MLIRToFlatbuffer.h"
#include "ttmlir/Version.h"

#include "mlir/Dialect/EmitC/IR/EmitC.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Support/LogicalResult.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/raw_ostream.h"

#include <cassert>
#include <fstream>
#include <optional>

namespace mlir::tt::ttnn {

constexpr uint64_t kHostAllocatedSize = 0;
constexpr uint64_t kHostAllocatedAddress = 0;

#define GEN_PASS_DEF_TTNNSERIALIZETOBINARY
#include "ttmlir/Dialect/TTNN/Transforms/Passes.h.inc"

::tt::target::Dim2dRange toFlatbuffer(CoreRangeAttr coreRange) {
  auto offset = coreRange.getOffset();
  auto size = coreRange.getSize();
  return ::tt::target::Dim2dRange(::tt::target::Dim2d(offset[0], offset[1]),
                                  ::tt::target::Dim2d(size[0], size[1]));
}

::flatbuffers::Offset<::tt::target::ShardSpec>
shardSpecToFlatbuffer(FlatbufferObjectCache &cache,
                      ::mlir::tt::ttnn::ShardSpecAttr shardSpec) {
  llvm::ArrayRef<int64_t> shardShapeArr = shardSpec.getShardShape().getShape();
  std::vector<int64_t> shardShapeVec(shardShapeArr.begin(),
                                     shardShapeArr.end());
  auto shardShape = cache.fbb->CreateVector<int64_t>(shardShapeVec);
  return ::tt::target::CreateShardSpec(*cache.fbb, shardShape);
}

::flatbuffers::Offset<::tt::target::MemoryConfigDesc>
memoryConfigToFlatbuffer(FlatbufferObjectCache &cache,
                         ::mlir::tt::ttnn::MemoryConfigAttr memoryConfig) {
  ::tt::target::TensorMemoryLayout tensorMemoryLayout =
      ::tt::mlir::ttnn::utils::toTargetTensorMemoryLayout(
          memoryConfig.getTensorMemoryLayout().getValue());
  ::tt::target::BufferType bufferType =
      ::tt::mlir::ttnn::utils::toTargetBufferType(
          memoryConfig.getBufferType().getValue());
  auto shardSpec =
      cache.getOrCreate(memoryConfig.getShardSpec(), shardSpecToFlatbuffer);
  ::flatbuffers::Offset<::tt::target::MemoryConfigDesc> memoryConfigDesc =
      ::tt::target::CreateMemoryConfigDesc(*cache.fbb, tensorMemoryLayout,
                                           bufferType, shardSpec);
  return memoryConfigDesc;
}

::flatbuffers::Offset<::tt::target::DeviceRef>
createDeviceRef(FlatbufferObjectCache &cache, Value device) {
  auto deviceType = mlir::cast<DeviceType>(device.getType());
  auto chipIds = deviceType.getDesc().getChipIds();
  assert(chipIds.size() == 1 && "expected single chip");
  return ::tt::target::CreateDeviceRef(*cache.fbb, chipIds[0]);
}

template <typename OpT>
::flatbuffers::Offset<::tt::target::ttnn::Operation>
createOperation(FlatbufferObjectCache &cache, ::flatbuffers::Offset<OpT> op,
                std::string const &debugString) {
  return CreateOperationDirect(
      *cache.fbb, ::tt::target::ttnn::OpTypeTraits<OpT>::enum_value, op.Union(),
      debugString.c_str());
}

::flatbuffers::Offset<::tt::target::ttnn::GetDeviceOp>
createOp(FlatbufferObjectCache &cache, GetDeviceOp op) {
  auto result = op.getResult();
  auto resultType = mlir::cast<DeviceType>(result.getType());
  auto meshShape = resultType.getDesc().getMeshShape();
  auto meshVolume = ttmlir::utils::volume(meshShape);
  if (meshVolume > 1) {
    // Only support creating meshes along batch dim for now
    assert(meshShape.size() == 3 && "expected 3D mesh shape");
    assert(meshShape[1] == 1 && "expected non-batch dim to be 1");
    assert(meshShape[2] == 1 && "expected non-batch dim to be 1");
  }
  ::tt::target::Dim2d mesh(1, meshVolume);
  auto chipIds = toFlatbuffer(cache, resultType.getDesc().getChipIds());
  auto out = cache.getOrCreate(result, createDeviceRef);
  return ::tt::target::ttnn::CreateGetDeviceOp(*cache.fbb, &mesh, chipIds, out);
}

::flatbuffers::Offset<::tt::target::ttnn::ToMemoryConfigOp>
createOp(FlatbufferObjectCache &cache, ToMemoryConfigOp op) {
  auto input =
      cache.at<::tt::target::TensorRef>(getOperandThroughDPSOps(op.getInput()));

  auto memoryConfigDesc =
      cache.getOrCreate(op.getMemoryConfig(), memoryConfigToFlatbuffer);

  auto output = cache.getOrCreate(op.getResult(), tensorValueToFlatbuffer,
                                  kHostAllocatedAddress, kHostAllocatedSize);
  return ::tt::target::ttnn::CreateToMemoryConfigOp(*cache.fbb, input,
                                                    memoryConfigDesc, output);
}

::flatbuffers::Offset<::tt::target::ttnn::ToLayoutOp>
createOp(FlatbufferObjectCache &cache, ToLayoutOp op) {
  auto input =
      cache.at<::tt::target::TensorRef>(getOperandThroughDPSOps(op.getInput()));
  ::tt::target::TensorLayout layout =
      ::tt::mlir::ttnn::utils::toTargetTensorLayout(op.getLayout());
  auto output = cache.getOrCreate(op.getResult(), tensorValueToFlatbuffer,
                                  kHostAllocatedAddress, kHostAllocatedSize);

  std::optional<::mlir::tt::DataType> dtype = op.getDtype();
  std::optional<::mlir::tt::ttnn::MemoryConfigAttr> memoryConfig =
      op.getMemoryConfig();
  ::mlir::Value device = op.getDevice();
  if (device) {
    device = getOperandThroughDPSOps(device);
  }
  return ::tt::target::ttnn::CreateToLayoutOp(
      *cache.fbb, input, layout,
      dtype.has_value()
          ? ::flatbuffers::Optional<::tt::target::DataType>(
                ::tt::mlir::ttnn::utils::toTargetDataType(dtype.value()))
          : ::flatbuffers::nullopt,
      memoryConfig.has_value()
          ? cache.getOrCreate(memoryConfig.value(), memoryConfigToFlatbuffer)
          : 0,
      device ? cache.at<::tt::target::DeviceRef>(device) : 0, output);
}

::flatbuffers::Offset<::tt::target::ttnn::TypecastOp>
createOp(FlatbufferObjectCache &cache, TypecastOp op) {
  auto input =
      cache.at<::tt::target::TensorRef>(getOperandThroughDPSOps(op.getInput()));
  ::tt::target::DataType dtype =
      ::tt::mlir::ttnn::utils::toTargetDataType(op.getDtype());
  auto output = cache.getOrCreate(op.getResult(), tensorValueToFlatbuffer,
                                  kHostAllocatedAddress, kHostAllocatedSize);

  return ::tt::target::ttnn::CreateTypecastOp(*cache.fbb, input, dtype, output);
}

::flatbuffers::Offset<::tt::target::ttnn::ToDeviceOp>
createOp(FlatbufferObjectCache &cache, ToDeviceOp op) {
  auto input =
      cache.at<::tt::target::TensorRef>(getOperandThroughDPSOps(op.getInput()));
  auto device = getOperandThroughDPSOps(op.getDevice());

  auto output = cache.getOrCreate(op.getResult(), tensorValueToFlatbuffer,
                                  kHostAllocatedAddress, kHostAllocatedSize);

  if (!op.getMemoryConfig()) {
    return ::tt::target::ttnn::CreateToDeviceOp(
        *cache.fbb, input, cache.at<::tt::target::DeviceRef>(device),
        /* memoryConfigDesc */ 0, output);
  }

  auto memoryConfigDesc =
      cache.getOrCreate(op.getMemoryConfig().value(), memoryConfigToFlatbuffer);

  return ::tt::target::ttnn::CreateToDeviceOp(
      *cache.fbb, input, cache.at<::tt::target::DeviceRef>(device),
      memoryConfigDesc, output);
}

::flatbuffers::Offset<::tt::target::ttnn::FromDeviceOp>
createOp(FlatbufferObjectCache &cache, FromDeviceOp op) {
  auto input =
      cache.at<::tt::target::TensorRef>(getOperandThroughDPSOps(op.getInput()));

  auto output = cache.getOrCreate(op.getResult(), tensorValueToFlatbuffer,
                                  kHostAllocatedAddress, kHostAllocatedSize);

  return ::tt::target::ttnn::CreateFromDeviceOp(*cache.fbb, input, output);
}

::flatbuffers::Offset<::tt::target::ttnn::EmptyOp>
createOp(FlatbufferObjectCache &cache, EmptyOp op) {
  ::llvm::ArrayRef<int64_t> shape = op.getShape().getShape();
  ::tt::target::DataType dtype =
      ::tt::mlir::ttnn::utils::toTargetDataType(op.getDtype().value());
  ::tt::target::TensorLayout layout =
      ::tt::mlir::ttnn::utils::toTargetTensorLayout(op.getLayout().value());

  uint32_t numShards = 1;
  ::tt::target::DistributedTensorConfig distributionType =
      ::tt::target::DistributedTensorConfig::NONE;
  ::flatbuffers::Offset<void> distribution = 0;
  flatbuffers::Offset<::tt::target::DistributionStrategy> strategy =
      ::tt::target::CreateDistributionStrategy(*cache.fbb, distributionType,
                                               distribution);
  auto output = getOperandThroughDPSOps(op.getResult());

  // If the device is not set, we create on host
  //
  if (!op.getDevice()) {
    return ::tt::target::ttnn::CreateEmptyOp(
        *cache.fbb, cache.fbb->CreateVector<int64_t>(shape), dtype, layout,
        numShards, /* device */ 0, /* memcfg */ 0, strategy,
        cache.getOrCreate(output, tensorValueToFlatbuffer,
                          kHostAllocatedAddress, kHostAllocatedSize));
  }

  auto device = getOperandThroughDPSOps(op.getDevice());

  auto memoryConfigDesc =
      cache.getOrCreate(*op.getMemoryConfig(), memoryConfigToFlatbuffer);

  return ::tt::target::ttnn::CreateEmptyOp(
      *cache.fbb, cache.fbb->CreateVector<int64_t>(shape), dtype, layout,
      numShards, cache.at<::tt::target::DeviceRef>(device), memoryConfigDesc,
      strategy,
      cache.getOrCreate(output, tensorValueToFlatbuffer, kHostAllocatedAddress,
                        kHostAllocatedSize));
}

::flatbuffers::Offset<::tt::target::ttnn::FullOp>
createOp(FlatbufferObjectCache &cache, FullOp op) {
  auto device = getOperandThroughDPSOps(op.getDevice());
  auto fillValue = op.getFillValue().convertToFloat();
  auto output = getOperandThroughDPSOps(op.getResult());
  uint32_t numShards = 1;
  ::tt::target::DistributedTensorConfig distributionType =
      ::tt::target::DistributedTensorConfig::NONE;
  ::flatbuffers::Offset<void> distribution = 0;
  flatbuffers::Offset<::tt::target::DistributionStrategy> strategy =
      ::tt::target::CreateDistributionStrategy(*cache.fbb, distributionType,
                                               distribution);
  return ::tt::target::ttnn::CreateFullOp(
      *cache.fbb, cache.at<::tt::target::DeviceRef>(device), fillValue,
      numShards, strategy,
      cache.getOrCreate(output, tensorValueToFlatbuffer, kHostAllocatedAddress,
                        kHostAllocatedSize));
}

// ANCHOR: adding_an_op_matmul_serialize_to_binary
::flatbuffers::Offset<::tt::target::ttnn::MatmulOp>
createOp(FlatbufferObjectCache &cache, MatmulOp op) {
  auto in0 =
      cache.at<::tt::target::TensorRef>(getOperandThroughDPSOps(op.getA()));
  auto in1 =
      cache.at<::tt::target::TensorRef>(getOperandThroughDPSOps(op.getB()));
  auto output = cache.at<::tt::target::TensorRef>(
      getOperandThroughDPSOps(op.getResult()));
  return ::tt::target::ttnn::CreateMatmulOp(*cache.fbb, in0, in1, output);
}
// ANCHOR_END: adding_an_op_matmul_serialize_to_binary

::flatbuffers::Offset<::tt::target::ttnn::Conv2dOp>
createOp(FlatbufferObjectCache &cache, Conv2dOp op) {
  auto in0 =
      cache.at<::tt::target::TensorRef>(getOperandThroughDPSOps(op.getInput()));
  auto in1 = cache.at<::tt::target::TensorRef>(
      getOperandThroughDPSOps(op.getWeight()));
  auto in2 = op.getODSOperands(2).empty()
                 ? flatbuffers::Offset<::tt::target::TensorRef>()
                 : cache.at<::tt::target::TensorRef>(
                       getOperandThroughDPSOps(op.getBias()));
  auto output = cache.at<::tt::target::TensorRef>(
      getOperandThroughDPSOps(op.getResult()));

  auto device = getOperandThroughDPSOps(op.getDevice());
  return ::tt::target::ttnn::CreateConv2dOp(
      *cache.fbb, in0, in1, in2, output,
      cache.at<::tt::target::DeviceRef>(device), op.getInChannels(),
      op.getOutChannels(), op.getBatchSize(), op.getInputHeight(),
      op.getInputWidth(), op.getKernelHeight(), op.getKernelWidth(),
      op.getStrideHeight(), op.getStrideWidth(), op.getPaddingHeight(),
      op.getPaddingWidth(), op.getDilationHeight(), op.getDilationWidth(),
      op.getGroups());
}

::flatbuffers::Offset<::tt::target::ttnn::AllGatherOp>
createOp(FlatbufferObjectCache &cache, AllGatherOp op) {
  auto input =
      cache.at<::tt::target::TensorRef>(getOperandThroughDPSOps(op.getInput()));
  auto output = cache.getOrCreate(op.getResult(), tensorValueToFlatbuffer,
                                  kHostAllocatedAddress, kHostAllocatedSize);
  return ::tt::target::ttnn::CreateAllGatherOp(*cache.fbb, input, output,
                                               op.getDim(), op.getNumLinks());
}

template <typename EltwiseOp>
::flatbuffers::Offset<::tt::target::ttnn::EltwiseOp>
createEltwiseOp(FlatbufferObjectCache &cache, EltwiseOp op) {
  ::tt::target::ttnn::EltwiseOpType type;
  ::tt::target::ttnn::EltwiseOpParams paramsType =
      ::tt::target::ttnn::EltwiseOpParams::NONE;
  ::flatbuffers::Offset<void> params = 0;
  if constexpr (std::is_same_v<EltwiseOp, AbsOp>) {
    type = ::tt::target::ttnn::EltwiseOpType::Abs;
  } else if constexpr (std::is_same_v<EltwiseOp, AddOp>) {
    type = ::tt::target::ttnn::EltwiseOpType::Add;
  } else if constexpr (std::is_same_v<EltwiseOp, CbrtOp>) {
    type = ::tt::target::ttnn::EltwiseOpType::Cbrt;
  } else if constexpr (std::is_same_v<EltwiseOp, FloorOp>) {
    type = ::tt::target::ttnn::EltwiseOpType::Floor;
  } else if constexpr (std::is_same_v<EltwiseOp, IsFiniteOp>) {
    type = ::tt::target::ttnn::EltwiseOpType::IsFinite;
  } else if constexpr (std::is_same_v<EltwiseOp, LogicalAndOp>) {
    type = ::tt::target::ttnn::EltwiseOpType::LogicalAnd;
  } else if constexpr (std::is_same_v<EltwiseOp, LogicalNotOp>) {
    type = ::tt::target::ttnn::EltwiseOpType::LogicalNot;
  } else if constexpr (std::is_same_v<EltwiseOp, LogicalOrOp>) {
    type = ::tt::target::ttnn::EltwiseOpType::LogicalOr;
  } else if constexpr (std::is_same_v<EltwiseOp, MultiplyOp>) {
    type = ::tt::target::ttnn::EltwiseOpType::Multiply;
  } else if constexpr (std::is_same_v<EltwiseOp, NegOp>) {
    type = ::tt::target::ttnn::EltwiseOpType::Neg;
  } else if constexpr (std::is_same_v<EltwiseOp, SubtractOp>) {
    type = ::tt::target::ttnn::EltwiseOpType::Subtract;
  } else if constexpr (std::is_same_v<EltwiseOp, EqualOp>) {
    type = ::tt::target::ttnn::EltwiseOpType::Equal;
  } else if constexpr (std::is_same_v<EltwiseOp, NotEqualOp>) {
    type = ::tt::target::ttnn::EltwiseOpType::NotEqual;
  } else if constexpr (std::is_same_v<EltwiseOp, GreaterEqualOp>) {
    type = ::tt::target::ttnn::EltwiseOpType::GreaterEqual;
  } else if constexpr (std::is_same_v<EltwiseOp, GreaterThanOp>) {
    type = ::tt::target::ttnn::EltwiseOpType::GreaterThan;
  } else if constexpr (std::is_same_v<EltwiseOp, LessEqualOp>) {
    type = ::tt::target::ttnn::EltwiseOpType::LessEqual;
  } else if constexpr (std::is_same_v<EltwiseOp, LessThanOp>) {
    type = ::tt::target::ttnn::EltwiseOpType::LessThan;
  } else if constexpr (std::is_same_v<EltwiseOp, MaximumOp>) {
    type = ::tt::target::ttnn::EltwiseOpType::Maximum;
  } else if constexpr (std::is_same_v<EltwiseOp, MinimumOp>) {
    type = ::tt::target::ttnn::EltwiseOpType::Minimum;
  } else if constexpr (std::is_same_v<EltwiseOp, ReluOp>) {
    type = ::tt::target::ttnn::EltwiseOpType::Relu;
  } else if constexpr (std::is_same_v<EltwiseOp, SqrtOp>) {
    type = ::tt::target::ttnn::EltwiseOpType::Sqrt;
  } else if constexpr (std::is_same_v<EltwiseOp, RsqrtOp>) {
    type = ::tt::target::ttnn::EltwiseOpType::Rsqrt;
  } else if constexpr (std::is_same_v<EltwiseOp, SignOp>) {
    type = ::tt::target::ttnn::EltwiseOpType::Sign;
  } else if constexpr (std::is_same_v<EltwiseOp, ReciprocalOp>) {
    type = ::tt::target::ttnn::EltwiseOpType::Reciprocal;
  } else if constexpr (std::is_same_v<EltwiseOp, DivOp>) {
    type = ::tt::target::ttnn::EltwiseOpType::Div;
  } else if constexpr (std::is_same_v<EltwiseOp, SigmoidOp>) {
    type = ::tt::target::ttnn::EltwiseOpType::Sigmoid;
  } else if constexpr (std::is_same_v<EltwiseOp, Log1pOp>) {
    type = ::tt::target::ttnn::EltwiseOpType::Log1p;
  } else if constexpr (std::is_same_v<EltwiseOp, ExpOp>) {
    type = ::tt::target::ttnn::EltwiseOpType::Exp;
  } else if constexpr (std::is_same_v<EltwiseOp, CeilOp>) {
    type = ::tt::target::ttnn::EltwiseOpType::Ceil;
  } else if constexpr (std::is_same_v<EltwiseOp, CosOp>) {
    type = ::tt::target::ttnn::EltwiseOpType::Cos;
  } else if constexpr (std::is_same_v<EltwiseOp, SinOp>) {
    type = ::tt::target::ttnn::EltwiseOpType::Sin;
  } else if constexpr (std::is_same_v<EltwiseOp, LogOp>) {
    type = ::tt::target::ttnn::EltwiseOpType::Log;
  } else if constexpr (std::is_same_v<EltwiseOp, Expm1Op>) {
    type = ::tt::target::ttnn::EltwiseOpType::Expm1;
  } else if constexpr (std::is_same_v<EltwiseOp, RemainderOp>) {
    type = ::tt::target::ttnn::EltwiseOpType::Remainder;
  } else {
    llvm_unreachable("unhandled EltwiseOp");
  }
  std::vector<::flatbuffers::Offset<::tt::target::TensorRef>> ins;
  for (auto input : op.getInputs()) {
    ins.push_back(
        cache.at<::tt::target::TensorRef>(getOperandThroughDPSOps(input)));
  }
  assert(op.getOutputs().size() == 1);
  return ::tt::target::ttnn::CreateEltwiseOpDirect(
      *cache.fbb, type, &ins,
      cache.at<::tt::target::TensorRef>(
          getOperandThroughDPSOps(op.getOutputs().front())),
      paramsType, params);
}

template <typename ReductionOp>
::flatbuffers::Offset<::tt::target::ttnn::ReductionOp>
createReductionOp(FlatbufferObjectCache &cache, ReductionOp op) {
  ::tt::target::ttnn::ReductionOpType type;
  if constexpr (std::is_same_v<ReductionOp, SumOp>) {
    type = ::tt::target::ttnn::ReductionOpType::Sum;
  } else if constexpr (std::is_same_v<ReductionOp, MeanOp>) {
    type = ::tt::target::ttnn::ReductionOpType::Mean;
  } else if constexpr (std::is_same_v<ReductionOp, MaxOp>) {
    type = ::tt::target::ttnn::ReductionOpType::Max;
  } else {
    llvm_unreachable("unhandled ReductionOp");
  }

  auto in =
      cache.at<::tt::target::TensorRef>(getOperandThroughDPSOps(op.getInput()));
  auto output = cache.getOrCreate(op.getResult(), tensorValueToFlatbuffer,
                                  kHostAllocatedAddress, kHostAllocatedSize);
  auto dim_arg =
      arrayAttrToFlatbuffer<mlir::IntegerAttr, int>(cache, op.getDimArg());

  return ::tt::target::ttnn::CreateReductionOp(*cache.fbb, type, in, output,
                                               dim_arg, op.getKeepDim());
}

template <typename TransposeOp>
::flatbuffers::Offset<::tt::target::ttnn::TransposeOp>
createTransposeOp(FlatbufferObjectCache &cache, TransposeOp op) {
  auto in =
      cache.at<::tt::target::TensorRef>(getOperandThroughDPSOps(op.getInput()));
  auto out = cache.getOrCreate(op.getResult(), tensorValueToFlatbuffer,
                               kHostAllocatedAddress, kHostAllocatedSize);
  int32_t dim0 = op.getDim0();
  int32_t dim1 = op.getDim1();

  return ::tt::target::ttnn::CreateTransposeOp(*cache.fbb, in, out, dim0, dim1);
}

template <typename ConcatOp>
::flatbuffers::Offset<::tt::target::ttnn::ConcatOp>
createConcatOp(FlatbufferObjectCache &cache, ConcatOp op) {
  std::vector<::flatbuffers::Offset<::tt::target::TensorRef>> ins;
  for (auto input : op.getInputs()) {
    ins.push_back(
        cache.at<::tt::target::TensorRef>(getOperandThroughDPSOps(input)));
  }
  auto out = cache.at<::tt::target::TensorRef>(
      getOperandThroughDPSOps(op.getResult()));
  int32_t dim = op.getDim();

  return ::tt::target::ttnn::CreateConcatOpDirect(*cache.fbb, &ins, out, dim);
}

template <typename EmbeddingOp>
::flatbuffers::Offset<::tt::target::ttnn::EmbeddingOp>
createEmbeddingOp(FlatbufferObjectCache &cache, EmbeddingOp op) {
  auto in0 =
      cache.at<::tt::target::TensorRef>(getOperandThroughDPSOps(op.getInput()));
  auto in1 = cache.at<::tt::target::TensorRef>(
      getOperandThroughDPSOps(op.getWeight()));
  auto output = cache.getOrCreate(op.getResult(), tensorValueToFlatbuffer,
                                  kHostAllocatedAddress, kHostAllocatedSize);
  return ::tt::target::ttnn::CreateEmbeddingOp(*cache.fbb, in0, in1, output);
}

template <typename ReshapeOp>
::flatbuffers::Offset<::tt::target::ttnn::ReshapeOp>
createReshapeOp(FlatbufferObjectCache &cache, ReshapeOp op) {
  auto in =
      cache.at<::tt::target::TensorRef>(getOperandThroughDPSOps(op.getInput()));
  auto shape =
      arrayAttrToFlatbuffer<mlir::IntegerAttr, int>(cache, op.getShape());
  auto out = cache.getOrCreate(op.getResult(), tensorValueToFlatbuffer,
                               kHostAllocatedAddress, kHostAllocatedSize);

  return ::tt::target::ttnn::CreateReshapeOp(*cache.fbb, in, out, shape);
}

template <typename SliceOp>
::flatbuffers::Offset<::tt::target::ttnn::SliceOp>
createSliceOp(FlatbufferObjectCache &cache, SliceOp op) {
  auto in =
      cache.at<::tt::target::TensorRef>(getOperandThroughDPSOps(op.getInput()));
  auto out = cache.at<::tt::target::TensorRef>(
      getOperandThroughDPSOps(op.getResult()));
  auto begins =
      arrayAttrToFlatbuffer<mlir::IntegerAttr, int64_t>(cache, op.getBegins());
  auto ends =
      arrayAttrToFlatbuffer<mlir::IntegerAttr, int64_t>(cache, op.getEnds());
  auto step =
      arrayAttrToFlatbuffer<mlir::IntegerAttr, int64_t>(cache, op.getStep());

  return ::tt::target::ttnn::CreateSliceOp(*cache.fbb, in, out, begins, ends,
                                           step);
}

template <typename MaxPool2dOp>
::flatbuffers::Offset<::tt::target::ttnn::MaxPool2dOp>
createMaxPool2dOp(FlatbufferObjectCache &cache, MaxPool2dOp op) {
  auto in =
      cache.at<::tt::target::TensorRef>(getOperandThroughDPSOps(op.getInput()));
  auto out = cache.at<::tt::target::TensorRef>(
      getOperandThroughDPSOps(op.getResult()));

  auto device = getOperandThroughDPSOps(op.getDevice());
  return ::tt::target::ttnn::CreateMaxPool2dOp(
      *cache.fbb, in, out, cache.at<::tt::target::DeviceRef>(device),
      op.getBatchSize(), op.getInputHeight(), op.getInputWidth(),
      op.getChannels(), op.getKernelHeight(), op.getKernelWidth(),
      op.getStrideHeight(), op.getStrideWidth(), op.getDilationHeight(),
      op.getDilationWidth(), op.getCeilMode(), op.getPaddingHeight(),
      op.getPaddingWidth());
}

template <typename SoftmaxOp>
::flatbuffers::Offset<::tt::target::ttnn::SoftmaxOp>
createSoftmaxOp(FlatbufferObjectCache &cache, SoftmaxOp op) {
  auto in =
      cache.at<::tt::target::TensorRef>(getOperandThroughDPSOps(op.getInput()));
  auto out = cache.getOrCreate(op.getResult(), tensorValueToFlatbuffer,
                               kHostAllocatedAddress, kHostAllocatedSize);
  int32_t dimension = op.getDimension();

  return ::tt::target::ttnn::CreateSoftmaxOp(*cache.fbb, in, out, dimension);
}

template <typename DeallocOp>
::flatbuffers::Offset<::tt::target::ttnn::DeallocOp>
createDeallocOp(FlatbufferObjectCache &cache, DeallocOp op) {
  auto in =
      cache.at<::tt::target::TensorRef>(getOperandThroughDPSOps(op.getInput()));
  return ::tt::target::ttnn::CreateDeallocOp(*cache.fbb, in);
}

::flatbuffers::Offset<::tt::target::ttnn::Operation>
emitTTNNOperation(FlatbufferObjectCache &cache, Operation *op,
                  std::string const &debugString) {
  if (auto getDeviceOp = dyn_cast<GetDeviceOp>(op); getDeviceOp) {
    return createOperation(cache, createOp(cache, getDeviceOp), debugString);
  }
  if (auto toMemoryConfigOp = dyn_cast<ToMemoryConfigOp>(op);
      toMemoryConfigOp) {
    return createOperation(cache, createOp(cache, toMemoryConfigOp),
                           debugString);
  }
  if (auto toLayoutOp = dyn_cast<ToLayoutOp>(op); toLayoutOp) {
    return createOperation(cache, createOp(cache, toLayoutOp), debugString);
  }
  if (auto typecastOp = dyn_cast<TypecastOp>(op); typecastOp) {
    return createOperation(cache, createOp(cache, typecastOp), debugString);
  }
  if (auto toDeviceOp = dyn_cast<ToDeviceOp>(op); toDeviceOp) {
    return createOperation(cache, createOp(cache, toDeviceOp), debugString);
  }
  if (auto fromDeviceOp = dyn_cast<FromDeviceOp>(op); fromDeviceOp) {
    return createOperation(cache, createOp(cache, fromDeviceOp), debugString);
  }
  if (auto emptyOp = dyn_cast<EmptyOp>(op); emptyOp) {
    return createOperation(cache, createOp(cache, emptyOp), debugString);
  }
  if (auto fullOp = dyn_cast<FullOp>(op); fullOp) {
    return createOperation(cache, createOp(cache, fullOp), debugString);
  }
  if (auto absOp = dyn_cast<AbsOp>(op); absOp) {
    return createOperation(cache, createEltwiseOp(cache, absOp), debugString);
  }
  if (auto addOp = dyn_cast<AddOp>(op); addOp) {
    return createOperation(cache, createEltwiseOp(cache, addOp), debugString);
  }
  if (auto floorOp = dyn_cast<FloorOp>(op); floorOp) {
    return createOperation(cache, createEltwiseOp(cache, floorOp), debugString);
  }
  if (auto isFiniteOp = dyn_cast<IsFiniteOp>(op); isFiniteOp) {
    return createOperation(cache, createEltwiseOp(cache, isFiniteOp),
                           debugString);
  }
  if (auto andOp = dyn_cast<LogicalAndOp>(op); andOp) {
    return createOperation(cache, createEltwiseOp(cache, andOp), debugString);
  }
  if (auto cbrtOp = dyn_cast<CbrtOp>(op); cbrtOp) {
    return createOperation(cache, createEltwiseOp(cache, cbrtOp), debugString);
  }
  if (auto notOp = dyn_cast<LogicalNotOp>(op); notOp) {
    return createOperation(cache, createEltwiseOp(cache, notOp), debugString);
  }
  if (auto orOp = dyn_cast<LogicalOrOp>(op); orOp) {
    return createOperation(cache, createEltwiseOp(cache, orOp), debugString);
  }
  if (auto multiplyOp = dyn_cast<MultiplyOp>(op); multiplyOp) {
    return createOperation(cache, createEltwiseOp(cache, multiplyOp),
                           debugString);
  }
  if (auto negOp = dyn_cast<NegOp>(op); negOp) {
    return createOperation(cache, createEltwiseOp(cache, negOp), debugString);
  }
  if (auto subtractOp = dyn_cast<SubtractOp>(op); subtractOp) {
    return createOperation(cache, createEltwiseOp(cache, subtractOp),
                           debugString);
  }
  if (auto eqOp = dyn_cast<EqualOp>(op); eqOp) {
    return createOperation(cache, createEltwiseOp(cache, eqOp), debugString);
  }
  if (auto neOp = dyn_cast<NotEqualOp>(op); neOp) {
    return createOperation(cache, createEltwiseOp(cache, neOp), debugString);
  }
  if (auto geOp = dyn_cast<GreaterEqualOp>(op); geOp) {
    return createOperation(cache, createEltwiseOp(cache, geOp), debugString);
  }
  if (auto gtOp = dyn_cast<GreaterThanOp>(op); gtOp) {
    return createOperation(cache, createEltwiseOp(cache, gtOp), debugString);
  }
  if (auto leOp = dyn_cast<LessEqualOp>(op); leOp) {
    return createOperation(cache, createEltwiseOp(cache, leOp), debugString);
  }
  if (auto ltOp = dyn_cast<LessThanOp>(op); ltOp) {
    return createOperation(cache, createEltwiseOp(cache, ltOp), debugString);
  }
  if (auto maximumOp = dyn_cast<MaximumOp>(op); maximumOp) {
    return createOperation(cache, createEltwiseOp(cache, maximumOp),
                           debugString);
  }
  if (auto minimumOp = dyn_cast<MinimumOp>(op); minimumOp) {
    return createOperation(cache, createEltwiseOp(cache, minimumOp),
                           debugString);
  }
  if (auto reluOp = dyn_cast<ReluOp>(op); reluOp) {
    return createOperation(cache, createEltwiseOp(cache, reluOp), debugString);
  }
  if (auto sqrtOp = dyn_cast<SqrtOp>(op); sqrtOp) {
    return createOperation(cache, createEltwiseOp(cache, sqrtOp), debugString);
  }
  if (auto rsqrtOp = dyn_cast<RsqrtOp>(op); rsqrtOp) {
    return createOperation(cache, createEltwiseOp(cache, rsqrtOp), debugString);
  }
  if (auto signOp = dyn_cast<SignOp>(op); signOp) {
    return createOperation(cache, createEltwiseOp(cache, signOp), debugString);
  }
  if (auto expOp = dyn_cast<ExpOp>(op); expOp) {
    return createOperation(cache, createEltwiseOp(cache, expOp), debugString);
  }
  if (auto logOp = dyn_cast<LogOp>(op); logOp) {
    return createOperation(cache, createEltwiseOp(cache, logOp), debugString);
  }
  if (auto expm1Op = dyn_cast<Expm1Op>(op); expm1Op) {
    return createOperation(cache, createEltwiseOp(cache, expm1Op), debugString);
  }
  if (auto sigmoidOp = dyn_cast<SigmoidOp>(op); sigmoidOp) {
    return createOperation(cache, createEltwiseOp(cache, sigmoidOp),
                           debugString);
  }
  if (auto log1pOp = dyn_cast<Log1pOp>(op); log1pOp) {
    return createOperation(cache, createEltwiseOp(cache, log1pOp), debugString);
  }
  if (auto reciprocalOp = dyn_cast<ReciprocalOp>(op); reciprocalOp) {
    return createOperation(cache, createEltwiseOp(cache, reciprocalOp),
                           debugString);
  }
  if (auto divOp = dyn_cast<DivOp>(op); divOp) {
    return createOperation(cache, createEltwiseOp(cache, divOp), debugString);
  }
  if (auto remainderOp = dyn_cast<RemainderOp>(op); remainderOp) {
    return createOperation(cache, createEltwiseOp(cache, remainderOp),
                           debugString);
  }
  if (auto matmulOp = dyn_cast<MatmulOp>(op); matmulOp) {
    return createOperation(cache, createOp(cache, matmulOp), debugString);
  }
  if (auto sumOp = dyn_cast<SumOp>(op); sumOp) {
    return createOperation(cache, createReductionOp(cache, sumOp), debugString);
  }
  if (auto meanOp = dyn_cast<MeanOp>(op); meanOp) {
    return createOperation(cache, createReductionOp(cache, meanOp),
                           debugString);
  }
  if (auto maxOp = dyn_cast<MaxOp>(op); maxOp) {
    return createOperation(cache, createReductionOp(cache, maxOp), debugString);
  }
  if (auto embeddingOp = dyn_cast<EmbeddingOp>(op); embeddingOp) {
    return createOperation(cache, createEmbeddingOp(cache, embeddingOp),
                           debugString);
  }
  if (auto softmaxOp = dyn_cast<SoftmaxOp>(op); softmaxOp) {
    return createOperation(cache, createSoftmaxOp(cache, softmaxOp),
                           debugString);
  }
  if (auto transposeOp = dyn_cast<TransposeOp>(op); transposeOp) {
    return createOperation(cache, createTransposeOp(cache, transposeOp),
                           debugString);
  }
  if (auto conv2dOp = dyn_cast<Conv2dOp>(op); conv2dOp) {
    return createOperation(cache, createOp(cache, conv2dOp), debugString);
  }
  if (auto allGatherOp = dyn_cast<AllGatherOp>(op); allGatherOp) {
    return createOperation(cache, createOp(cache, allGatherOp), debugString);
  }
  if (auto concatOp = dyn_cast<ConcatOp>(op); concatOp) {
    return createOperation(cache, createConcatOp(cache, concatOp), debugString);
  }
  if (auto reshapeOp = dyn_cast<ReshapeOp>(op); reshapeOp) {
    return createOperation(cache, createReshapeOp(cache, reshapeOp),
                           debugString);
  }
  if (auto sliceOp = dyn_cast<SliceOp>(op); sliceOp) {
    return createOperation(cache, createSliceOp(cache, sliceOp), debugString);
  }
  if (auto max_pool2dOp = dyn_cast<MaxPool2dOp>(op); max_pool2dOp) {
    return createOperation(cache, createMaxPool2dOp(cache, max_pool2dOp),
                           debugString);
  }
  if (auto deallocOp = dyn_cast<DeallocOp>(op); deallocOp) {
    return createOperation(cache, createDeallocOp(cache, deallocOp),
                           debugString);
  }
  if (auto ceilOp = dyn_cast<CeilOp>(op); ceilOp) {
    return createOperation(cache, createEltwiseOp(cache, ceilOp), debugString);
  }
  if (auto cosOp = dyn_cast<CosOp>(op); cosOp) {
    return createOperation(cache, createEltwiseOp(cache, cosOp), debugString);
  }
  if (auto sinOp = dyn_cast<SinOp>(op); sinOp) {
    return createOperation(cache, createEltwiseOp(cache, sinOp), debugString);
  }

  llvm_unreachable("unhandled op in emitTTNNOperation");
}

std::shared_ptr<void>
ttnnToFlatbuffer(Operation *op,
                 std::unordered_map<std::string, GoldenTensor> goldenMap) {
  ModuleOp module = dyn_cast<ModuleOp>(op);
  assert(module && "Expected ModuleOp as top level operation");

  ::flatbuffers::FlatBufferBuilder fbb;
  FlatbufferObjectCache cache(&fbb);

  ::ttmlir::Version ttmlirVersion = ::ttmlir::getVersion();
  ::tt::target::Version binaryVersion(ttmlirVersion.major, ttmlirVersion.minor,
                                      ttmlirVersion.patch);

  auto systemDesc =
      toFlatbuffer(cache, mlir::cast<tt::SystemDescAttr>(
                              module->getAttr(tt::SystemDescAttr::name)));

  auto mlir = toDebugInfo(fbb, "ttnn", module);
  std::string cpp;
  llvm::raw_string_ostream os(cpp);
  auto result = mlir::tt::ttnn::emitTTNNAsCpp(module, os);
  (void)result;

  std::vector<::flatbuffers::Offset<::tt::target::GoldenKV>> goldenKVList;
  goldenKVList.reserve(goldenMap.size());

  for (auto element : goldenMap) {
    std::vector<std::uint8_t> dataTensor = element.second.convertDataToVector();
    auto goldenTensor = ::tt::target::CreateGoldenTensorDirect(
        fbb, element.second.name.c_str(), &element.second.shape,
        &element.second.strides, element.second.dtype, &dataTensor);
    auto goldenKV = ::tt::target::CreateGoldenKVDirect(
        fbb, element.first.c_str(), goldenTensor);
    goldenKVList.push_back(goldenKV);
  }

  auto goldenInfo = ::tt::target::CreateGoldenInfoDirect(fbb, &goldenKVList);
  auto debugInfo =
      ::tt::target::CreateDebugInfoDirect(fbb, mlir, cpp.c_str(), goldenInfo);

  std::vector<::flatbuffers::Offset<::tt::target::ttnn::Program>> programs;
  module->walk([&](func::FuncOp func) {
    Program<::tt::target::ttnn::Operation> program =
        funcOpToProgram<::tt::target::ttnn::Operation>(cache, func,
                                                       emitTTNNOperation);
    programs.push_back(::tt::target::ttnn::CreateProgramDirect(
        fbb, program.name, &program.inputs, &program.outputs, &program.ops,
        debugInfo));
  });

  auto binary = ::tt::target::ttnn::CreateTTNNBinaryDirect(
      fbb, &binaryVersion, ::ttmlir::getGitHash(), systemDesc, &programs);

  ::tt::target::ttnn::FinishSizePrefixedTTNNBinaryBuffer(fbb, binary);
  ::flatbuffers::Verifier verifier(fbb.GetBufferPointer(), fbb.GetSize());
  ::tt::target::ttnn::VerifySizePrefixedTTNNBinaryBuffer(verifier);

  uint8_t *buf = fbb.GetBufferPointer();
  std::size_t size = fbb.GetSize();

  std::shared_ptr<void> bufferPtr =
      std::shared_ptr<void>(std::malloc(size), std::free);
  std::memcpy(bufferPtr.get(), buf, size);
  return bufferPtr;
}

LogicalResult translateTTNNToFlatbuffer(
    Operation *op, llvm::raw_ostream &os,
    std::unordered_map<std::string, GoldenTensor> goldenMap) {
  std::shared_ptr<void> data = ttnnToFlatbuffer(op, goldenMap);
  std::size_t size = ::flatbuffers::GetSizePrefixedBufferLength(
      static_cast<const uint8_t *>(data.get()));
  os.write(reinterpret_cast<char const *>(data.get()), size);
  return success();
}
} // namespace mlir::tt::ttnn
