// SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <cstddef>
#include <llvm/Support/Casting.h>
#include <mlir/IR/AttrTypeSubElements.h>
#include <stdexcept>

#include "TTNNWrapper.hpp"
#include "TTNNWrapperLib_Impl.hpp"
#include "impl/buffers/buffer.hpp"
#include "ttmlir/Dialect/TT/IR/TTOpsTypes.h"
#include "ttnn/tensor/types.hpp"

namespace mlir::tt {

namespace detail {

::tt::tt_metal::DataType getDataType(const mlir::MemRefType &memref) {

  // what's better way to to this?
  // auto dataTypeAttr =
  //     llvm::dyn_cast_or_null<mlir::tt::DataTypeAttr>(memref.getElementType());
  // if (!dataTypeAttr) {
  //   throw std::runtime_error("Invalid element type");
  // }

  // switch (dataTypeAttr.getValue()) {
  // case tt::DataType::BFloat16:
  //   return ::tt::tt_metal::DataType::BFLOAT16;
  // case tt::DataType::Float32:
  //   return ::tt::tt_metal::DataType::FLOAT32;
  // case tt::DataType::BFP_BFloat8:
  //   return ::tt::tt_metal::DataType::BFLOAT8_B;
  // case tt::DataType::BFP_BFloat4:
  //   return ::tt::tt_metal::DataType::BFLOAT4_B;
  // case tt::DataType::UInt8:
  //   return ::tt::tt_metal::DataType::UINT8;
  // case tt::DataType::UInt16:
  //   return ::tt::tt_metal::DataType::UINT16;
  // default:
  //   throw std::runtime_error("Invalid element type");
  // }

  mlir::Type element_type = memref.getElementType();
  // what's better way to to this?
  // auto data_type = mlir::cast<DataType>(element_type);
  std::string ret_value;
  llvm::raw_string_ostream os(ret_value);
  element_type.print(os);
  std::string data_type_str = os.str();
  if (data_type_str == "f32") {
    return ::tt::tt_metal::DataType::FLOAT32;
  }
  if (data_type_str == "bf16") {
    return ::tt::tt_metal::DataType::BFLOAT16;
  }
  if (data_type_str == "bfp_bf8") {
    return ::tt::tt_metal::DataType::BFLOAT8_B;
  }
  if (data_type_str == "bfp_bf4") {
    return ::tt::tt_metal::DataType::BFLOAT4_B;
  }
  if (data_type_str == "u32") {
    return ::tt::tt_metal::DataType::UINT32;
  }
  if (data_type_str == "u16") {
    return ::tt::tt_metal::DataType::UINT16;
  }
  if (data_type_str == "u8") {
    return ::tt::tt_metal::DataType::UINT8;
  }
  throw std::runtime_error("Invalid element type");
}

::ttnn::SimpleShape getTensorShape(const mlir::MemRefType &memref) {
  std::vector<uint32_t> shape;
  for (auto i = 0; i < memref.getRank(); i++) {
    shape.push_back(memref.getShape()[i]);
  }
  return ::ttnn::SimpleShape(shape);
}

const std::array<uint32_t, 2>
getShardShape(const mlir::tt::LayoutAttr &layout) {
  const auto layoutShardTile = layout.getShardShape(false);

  if (layoutShardTile.size() != 2) {
    llvm::errs() << "ERROR: layout_shard_tile.size() != 2\n";
    return {0, 0};
  }

  std::array<uint32_t, 2> shardShape;
  shardShape[0] = layoutShardTile[0];
  shardShape[1] = layoutShardTile[1];
  return shardShape;
}

::tt::tt_metal::Layout getTensorLayout(const mlir::tt::LayoutAttr &layout) {
  return layout.isTiled() ? ::tt::tt_metal::Layout::TILE
                          : ::tt::tt_metal::Layout::ROW_MAJOR;
}

::tt::tt_metal::CoreRangeSet
getCoreRangeSet(const mlir::tt::LayoutAttr &layout) {
  // TODO(mbezulj): handle more complex grid shapes
  // assuming grid shape is one rect starting at (0,0)

  const auto layoutGrid = layout.getGrid();

  const auto layoutGridShape = layoutGrid.getShape();
  if (layoutGridShape.size() != 2) {
    llvm::errs() << "ERROR: layout_grid.getShape().size() == 2\n";
    return {};
  }

  return ::tt::tt_metal::CoreRangeSet(::tt::tt_metal::CoreRange(
      ::tt::tt_metal::CoreCoord(0, layoutGridShape[0]),
      ::tt::tt_metal::CoreCoord(0, layoutGridShape[1])));
}

std::optional<::tt::tt_metal::ShardSpec>
layout_get_shard_spec(const mlir::tt::LayoutAttr &layout) {
  // ::tt::tt_metal::ShardOrientation is not part of LayoutAttr;
  // defaulting to ROW_MAJOR. TODO: figure out if we need to expose this
  return isShardedMemoryLayout(layout.getMemLayout())
             ? std::make_optional(::tt::tt_metal::ShardSpec(
                   getCoreRangeSet(layout), getShardShape(layout),
                   ::tt::tt_metal::ShardOrientation::ROW_MAJOR, false))
             : std::nullopt;
}

::tt::tt_metal::BufferType getBufferType(const mlir::MemRefType &memref) {
  auto memorySpace =
      mlir::cast<tt::MemorySpaceAttr>(memref.getMemorySpace()).getValue();

  switch (memorySpace) {
  case tt::MemorySpace::DeviceDRAM:
    return ::tt::tt_metal::BufferType::DRAM;
  case tt::MemorySpace::DeviceL1:
    return ::tt::tt_metal::BufferType::L1;
  default: // TODO(mbezulj): handle other memory spaces
    return ::tt::tt_metal::BufferType::DRAM;
  }
}

::tt::tt_metal::TensorMemoryLayout
getTensorMemoryLayout(const mlir::tt::LayoutAttr &layout) {
  auto tensorMemoryLayout = layout.getMemLayout();

  switch (tensorMemoryLayout) {
  case tt::TensorMemoryLayout::Interleaved:
    return ::tt::tt_metal::TensorMemoryLayout::INTERLEAVED;
  case tt::TensorMemoryLayout::SingleBank:
    return ::tt::tt_metal::TensorMemoryLayout::SINGLE_BANK;
  case tt::TensorMemoryLayout::HeightSharded:
    return ::tt::tt_metal::TensorMemoryLayout::HEIGHT_SHARDED;
  case tt::TensorMemoryLayout::WidthSharded:
    return ::tt::tt_metal::TensorMemoryLayout::WIDTH_SHARDED;
  case tt::TensorMemoryLayout::BlockSharded:
    return ::tt::tt_metal::TensorMemoryLayout::BLOCK_SHARDED;
  default:
    return ::tt::tt_metal::TensorMemoryLayout::INTERLEAVED;
  }
}

::tt::tt_metal::MemoryConfig
getMemoryConfig(const mlir::tt::LayoutAttr &layout) {

  auto tensorMemoryLayout = getTensorMemoryLayout(layout);
  auto bufferType = getBufferType(layout.getMemref());

  auto shardSpec = layout_get_shard_spec(layout);
  return ::tt::tt_metal::MemoryConfig(tensorMemoryLayout, bufferType,
                                      shardSpec);
}

} // namespace detail

bool ReluIsLegal(const mlir::tt::LayoutAttr &inputLayout,
                 const mlir::tt::LayoutAttr &outputLayout) {

  // const auto inputShape = detail::getTensorShape(inputLayout.getMemref());
  // const auto inputDataType = detail::getDataType(inputLayout.getMemref());
  // const auto inputLayoutType = detail::getTensorLayout(inputLayout);
  // const auto inputMemoryConfig = detail::getMemoryConfig(inputLayout);

  // const auto outputShape = detail::getTensorShape(outputLayout.getMemref());
  // const auto outputDataType = detail::getDataType(outputLayout.getMemref());
  // const auto outputLayoutType = detail::getTensorLayout(outputLayout);
  // const auto outputMemoryConfig = detail::getMemoryConfig(outputLayout);

  // std::cout << "inputShape: " << inputShape << std::endl;
  // std::cout << "inputDataType: " << inputDataType << std::endl;
  // std::cout << "inputLayoutType: " << inputLayoutType << std::endl;
  // std::cout << "inputMemoryConfig: " << inputMemoryConfig << std::endl;

  // std::cout << "outputShape: " << outputShape << std::endl;
  // std::cout << "outputDataType: " << outputDataType << std::endl;
  // std::cout << "outputLayoutType: " << outputLayoutType << std::endl;
  // std::cout << "outputMemoryConfig: " << outputMemoryConfig << std::endl;

  return true;
}

size_t ReluGetOpL1Usage(const mlir::tt::LayoutAttr &inputLayout,
                        const mlir::tt::LayoutAttr &outputLayout) {
  throw std::runtime_error("Not implemented");
}

void calculus(mlir::tt::LayoutAttr layout) {
  throw std::runtime_error("Not implemented");
}

void print_tensor_shape(const mlir::MemRefType &memref) {
  const auto shape = detail::getTensorShape(memref);
  std::cout << shape << std::endl;
}

} // namespace mlir::tt