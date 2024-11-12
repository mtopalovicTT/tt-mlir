// SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <numeric>

#include "mlir/IR/Builders.h"
#include "mlir/IR/DialectImplementation.h"
#include "ttmlir/Dialect/TT/IR/TTOpsTypes.h"
#include "ttmlir/Dialect/TTNN/IR/TTNNOpsAttrs.h"
#include "ttmlir/Dialect/TTNN/Utils/Utils.h"
#include "ttmlir/Utils.h"
#include "llvm/ADT/TypeSwitch.h"

using namespace mlir::tt::ttnn;

inline bool isSystemBufferType(BufferType bufferType) {
  return bufferType == BufferType::SystemMemory;
}

inline bool isDeviceBufferType(BufferType bufferType) {
  return bufferType == BufferType::DRAM || bufferType == BufferType::L1;
}

inline bool isL1BufferType(BufferType bufferType) {
  return bufferType == BufferType::L1;
}

bool TensorConfigAttr::isTiled() const {
  return ::mlir::isa<::mlir::tt::TileType>(getElementType());
}

// Get stride given tensor logical shape
llvm::SmallVector<int64_t>
TensorConfigAttr::getStride(ArrayRef<int64_t> logicalShape) const {

  llvm::SmallVector<int64_t> stride(logicalShape.size());
  AffineMap linearMap = getLinear();

  // Calculate the physical shape of the tensor.
  // Given tensor (6x15x10) and linear (d0, d1, d2) -> (d0 * 15 + d1, d2)
  // The physical shape is (90, 10)
  auto physicalShape = ttmlir::utils::evalShape(linearMap, logicalShape);

  // Origin point in the logical space (0, 0)
  SmallVector<AffineExpr> originPoint(logicalShape.size(),
                                      getAffineConstantExpr(0, getContext()));

  size_t prevDimElems = 1;

  // Iterates through physical dimensions (starting from the inner one).
  for (int i = linearMap.getNumResults() - 1; i >= 0; i--) {
    AffineExpr expr = linearMap.getResult(i);

    // Get coordinate of the i-th dimension (in physical space) of the origin
    // (in logical space).
    AffineExpr constantExpr = expr.replaceDims(originPoint);
    std::int64_t valueAtZero =
        llvm::cast<AffineConstantExpr>(constantExpr).getValue();

    for (size_t j = 0; j < logicalShape.size(); j++) {
      if (!expr.isFunctionOfDim(j)) {
        continue;
      }

      // Move from the origin point by one in the j-th dimension,
      // and get the coordinate of the i-th dimension (in physical space).
      auto newPoint = originPoint;
      newPoint[j] = getAffineConstantExpr(1, getContext());
      constantExpr = expr.replaceDims(newPoint);
      std::int64_t valueAtOne =
          llvm::cast<AffineConstantExpr>(constantExpr).getValue();

      // One step in the j-th dimension, jumps delta * prevDimElems elements in
      // the physical space.
      int64_t delta = valueAtOne - valueAtZero;
      stride[j] = prevDimElems * delta;
    }

    prevDimElems *= physicalShape[i];
  }

  return stride;
}

// Get the buffer type (DRAM/L1/SystemMemory)
BufferType TensorConfigAttr::getBufferType() const {
  return mlir::cast<BufferTypeAttr>(getMemref().getMemorySpace()).getValue();
}

// Get element type
mlir::Type TensorConfigAttr::getElementType() const {
  return getMemref().getElementType();
}

mlir::tt::DataType TensorConfigAttr::getDataTypeFromMemRef() const {
  Type elementType = getElementType();
  DataType dtype = DataType::Float32;
  if (llvm::isa<TileType>(elementType)) {
    auto tileType = mlir::cast<TileType>(elementType);
    dtype = tileType.getDataType();
  } else {
    dtype = elementTypeToDataType(elementType);
  }
  return dtype;
}

// Get element type
//
// This function returns element type of the tensor. If the element type
// is TileType, then we return the scalar element type.
//
// /return The element type of the tensor.
mlir::Type TensorConfigAttr::getScalarElementType() const {
  auto elementType = getElementType();
  if (mlir::isa<TileType>(elementType)) {
    return mlir::cast<TileType>(elementType).getElementType();
  }
  return elementType;
}

// Gets the size of shard in bytes
//
// This function returns the size of the shard in bytes.
// Size is calculated by multiplying shard shape with element size.
//
// /return The size of the shard in bytes.
uint64_t TensorConfigAttr::getElementSizeBytes() const {
  mlir::Type elementType = getElementType();
  if (mlir::isa<TileType>(elementType)) {
    auto tileType = mlir::cast<TileType>(elementType);
    return tileType.getSizeBytes();
  }
  return elementType.getIntOrFloatBitWidth() / 8;
}

// Get shard shape
//
// This function returns the shape of the shard. If element type is TileType
// and convertTileToScalar is true, then the shape is converted to scalar shape.
// Example: TIleType(32x32) and shard size is 2x2, then the scalar shape is
// 64x64
//
// /param convertTileToScalar If true, convert tile shape to scalar shape.
// /return The shape of the shard.
llvm::SmallVector<int64_t>
TensorConfigAttr::getShardShape(bool convertTileToScalar) const {
  SmallVector<int64_t> shardShape(getMemref().getShape());
  auto elementType = getElementType();
  if (mlir::isa<TileType>(elementType) && convertTileToScalar) {
    return mlir::cast<TileType>(elementType).getScalarShape(shardShape);
  }
  return shardShape;
}

// Get size of tensor in tiles
//
// This function returns the size of the tensor in tiles.
// Size is calculate by pluging the tensor shape into the linear map.
// This result is then divided by the tile shape.
//
// /param tensorShape The shape of the tensor
// /return The size of the tensor in tiles.
llvm::SmallVector<int64_t>
TensorConfigAttr::getTiledShape(llvm::ArrayRef<int64_t> tensorShape) const {
  assert(isTiled() && "Expected a tiled layout");

  // Affine map in form of (d0, d1, d2) -> (d0 * 15 + d1, d2)
  mlir::AffineMap linear = getLinear();
  uint32_t rank = linear.getNumResults();
  assert(rank >= 2 && "Expected at least two results in linear map");
  mlir::AffineExpr y = linear.getResult(rank - 2);
  mlir::AffineExpr x = linear.getResult(rank - 1);

  TileType tileType = mlir::cast<TileType>(getElementType());
  int64_t tileH = tileType.getHeight();
  int64_t tileW = tileType.getWidth();

  // Construct new affine map with where x and y are divided by tile width and
  // height respectively. Example:
  // (d0, d1, d2) -> (d0 * 15 + d1) / 32, d2 / 32
  mlir::AffineMap tiled =
      linear.replace(mlir::DenseMap<mlir::AffineExpr, mlir::AffineExpr>{
          {y, y.floorDiv(tileH)}, {x, x.floorDiv(tileW)}});

  // Get tiled shape by evaluating the affine map with tensor shape.
  return ttmlir::utils::evalShape(tiled, tensorShape);
}

// Get the size of shard in bytes
//
// This function returns the size of the shard in bytes.
// Size is calculated by multiplying shard shape with element size.
// Element size for TileType is tile width * tile height * element size.
// For scalar types, element size is element bit width / 8.
//
// /return The size of the shard in bytes.
uint64_t TensorConfigAttr::getShardSizeInBytes() const {
  MemRefType ty = getMemref();
  auto shape = ty.getShape();
  uint64_t size = getElementSizeBytes();
  return std::accumulate(shape.begin(), shape.end(), size,
                         std::multiplies<uint64_t>());
}

// Check if the tensor memory layout is sharded
bool TensorConfigAttr::hasShardedTensorMemoryLayout() const {
  return (getMemLayout() == TensorMemoryLayout::HeightSharded ||
          getMemLayout() == TensorMemoryLayout::WidthSharded ||
          getMemLayout() == TensorMemoryLayout::BlockSharded);
}

// Check if the tensor memory layout is sharded in L1 memory
bool TensorConfigAttr::hasShardedL1TensorMemoryLayout() const {
  return isL1BufferType(getBufferType()) and
         (getMemLayout() == TensorMemoryLayout::HeightSharded ||
          getMemLayout() == TensorMemoryLayout::WidthSharded ||
          getMemLayout() == TensorMemoryLayout::BlockSharded);
}

bool TensorConfigAttr::hasInterleavedL1TensorMemoryLayout() const {
  return isL1BufferType(getBufferType()) and
         (getMemLayout() == TensorMemoryLayout::Interleaved);
}

// Get new identity affine map i.e (d0, d1) -> (d0, d1)
//
// This function returns a new identity affine map
// with the same number of dimensions as the linear map.
//
// /return The new identity affine map.
mlir::AffineMap TensorConfigAttr::getIdentityTileLinearMap() const {
  assert(isTiled() && "Expected a tiled layout");

  return mlir::AffineMap::getMultiDimIdentityMap(getLinear().getNumResults(),
                                                 getContext());
}

// Takes phyisical memory map and replaces the symbols with the shard shape
//
// This function takes a physical memory map and replaces the symbols with the
// shard shape
//
// /param physicalMemoryMap The physical memory map (d0, d1)[s0, s1]
// /return New memory map with symbols replaced with shard shape.
mlir::AffineMap TensorConfigAttr::replaceMemoryMapSymbolsWithShardShape(
    AffineMap physicalMemoryMap) const {
  mlir::SmallVector<int64_t> shardShape =
      getShardShape(false /*convertTileToScalar*/);
  assert(physicalMemoryMap.getNumSymbols() == shardShape.size() &&
         "Physical memory map must have same number of symbols as logical "
         "shard rank");

  SmallVector<AffineExpr> symReplacements;
  for (unsigned i = 0; i < physicalMemoryMap.getNumSymbols(); ++i) {
    symReplacements.push_back(
        getAffineConstantExpr(shardShape[i], getContext()));
  }

  SmallVector<AffineExpr> dimReplacements;
  for (unsigned i = 0; i < physicalMemoryMap.getNumDims(); ++i) {
    dimReplacements.push_back(getAffineDimExpr(i, getContext()));
  }

  return physicalMemoryMap.replaceDimsAndSymbols(
      dimReplacements, symReplacements, physicalMemoryMap.getNumDims(), 0);
}

// TODO (milant) figure out what is going on here...
int64_t TensorConfigAttr::getTensorSizeInBytes(ArrayRef<int64_t> tensorShape,
                                               DeviceAttr device) const {
  SmallVector<int64_t> shape = isTiled() ? getTiledShape(tensorShape)
                                         : SmallVector<int64_t>(tensorShape);
  MemorySpace memorySpace = utils::toTTMemorySpace(getBufferType());
  AffineMap linearMap = isTiled() ? getIdentityTileLinearMap() : getLinear();
  mlir::SmallVector<std::int64_t> linearShape =
      ttmlir::utils::evalShape(linearMap, shape);
  AffineMap memoryMap = replaceMemoryMapSymbolsWithShardShape(
      device.getMapForMemorySpace(memorySpace));
  mlir::SmallVector<std::int64_t> physicalMemory =
      ttmlir::utils::evalShape(memoryMap, linearShape);
  std::int64_t elementSize = getElementSizeBytes();
  uint64_t sizeBytes =
      physicalMemory[MemoryMapResultIdx::ShardOffset] * elementSize;
  return sizeBytes;
}

// Construct a new TensorConfigAttr
//
// This function creates a new TensorConfigAttr with the given parameters.
// The element type, buffer type and memory layout are preserved.
//
// /param context The MLIR context.
// /param tensorShape The shape of the tensor (i.e 6x10x10)
// /param grid The grid where the tensor will be placed (i.e 2x3)
// /param collapseIntervals The intervals to collapse (i.e. {{0, -1}})
// /return The constructed TensorConfigAttr
TensorConfigAttr TensorConfigAttr::withGrid(
    ::mlir::MLIRContext *context, ArrayRef<int64_t> tensorShape, GridAttr grid,
    ArrayRef<std::pair<std::int64_t, std::int64_t>> collapseIntervals) {
  return get(context, tensorShape, getElementType(), getBufferType(), grid,
             getMemLayout(), collapseIntervals);
}

// Construct a new TensorConfigAttr
//
// This function creates a new TensorConfigAttr with the given parameters.
// The shape of the tensor, buffer type, element type and memory layout are
// preserved.
//
// /param context The MLIR context.
// /param grid The grid where the tensor will be placed.
// /param collapseIntervals The intervals to collapse (i.e. {{0, -1}})
// /return The constructed TensorConfigAttr
TensorConfigAttr TensorConfigAttr::withGrid(
    ::mlir::MLIRContext *context, RankedTensorType ty, GridAttr grid,
    ArrayRef<std::pair<std::int64_t, std::int64_t>> collapseIntervals) {
  assert(ty);
  return TensorConfigAttr::withGrid(context, ty.getShape(), grid,
                                    collapseIntervals);
}

// Construct a new TensorConfigAttr
//
// This function creates a deep copy of the current TensorConfigAttr and
// replaces the element type with the given one.
//
// /param context The MLIR context.
// /param elementType The new element type.
// /return The new TensorConfigAttr with the given element type.
TensorConfigAttr TensorConfigAttr::withElementType(::mlir::MLIRContext *context,
                                                   Type elementType) {
  return TensorConfigAttr::get(
      context, getLinear(), getGrid(),
      buildMemRef<BufferType, BufferTypeAttr>(context, getShardShape(),
                                              elementType, getBufferType()),
      getMemLayout());
}

// Construct a new TensorConfigAttr
//
// This function creates a deep copy of the current TensorConfigAttr and
// replaces the memory space with the given one.
//
// /param context The MLIR context.
// /param memorySpace The new memory space.
// /return The new TensorConfigAttr with the given memory space.
TensorConfigAttr TensorConfigAttr::withBufferType(::mlir::MLIRContext *context,
                                                  BufferType memorySpace) {
  return TensorConfigAttr::get(
      context, getLinear(), getGrid(),
      buildMemRef<BufferType, BufferTypeAttr>(context, getShardShape(),
                                              getElementType(), memorySpace),
      getMemLayout());
}

// Construct a new TensorConfigAttr
//
// This function creates a deep copy of the current TensorConfigAttr and
// replaces the memory layout with the given one.
//
// /param context The MLIR context.
// /param memLayout The new memory layout.
// /return The new TensorConfigAttr with the given memory layout.
TensorConfigAttr
TensorConfigAttr::withMemoryLayout(::mlir::MLIRContext *context,
                                   TensorMemoryLayout memLayout) {
  return TensorConfigAttr::get(
      context, getLinear(), getGrid(),
      buildMemRef<BufferType, BufferTypeAttr>(
          context, getShardShape(), getElementType(), getBufferType()),
      memLayout);
}

TensorConfigAttr
TensorConfigAttr::withShardShape(::mlir::MLIRContext *context,
                                 llvm::SmallVector<int64_t> shardShape) {
  return TensorConfigAttr::get(
      context, getLinear(), getGrid(),
      buildMemRef<BufferType, BufferTypeAttr>(
          context, shardShape, getElementType(), getBufferType()),
      getMemLayout());
}

// Construct a new TensorConfigAttr
//
// This function constructs a new TensorConfigAttr with the given parameters.
//
// /param context The MLIR context.
// /param tensorShape The shape of the tensor (i.e 6x10x10)
// /param elementType The type of the element. Could be
// FloatType/IntegerType/TileType /param bufferType The type of the buffer.
// Could be SystemMemory/L1/DRAM/L1Small /param grid The grid where the tensor
// will be placed /param collapseIntervals The intervals to collapse (i.e. {{0,
// -1}}) /param memLayout The memory layout of the tensor
// (Interleaved/HeightSharded...) /return The constructed TensorConfigAttr
TensorConfigAttr TensorConfigAttr::get(
    ::mlir::MLIRContext *context, ArrayRef<int64_t> tensorShape,
    Type elementType, BufferType bufferType, GridAttr grid,
    TensorMemoryLayout memLayout,
    ArrayRef<std::pair<std::int64_t, std::int64_t>> collapseIntervals) {
  AffineMap linear = collapsedLinearAffineMap(
      context, tensorShape, grid.getShape(), collapseIntervals);
  mlir::SmallVector<int64_t, 4> shardShape =
      calculateLogicalShardShape(tensorShape, linear, grid);
  MemRefType memRefType = buildMemRef<BufferType, BufferTypeAttr>(
      context, shardShape, elementType, bufferType);
  return get(context, linear, grid, memRefType, memLayout);
}