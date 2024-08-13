// SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#ifndef TTMLIR_DIALECT_TT_IR_TTOPSTYPES_H
#define TTMLIR_DIALECT_TT_IR_TTOPSTYPES_H

#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/OpImplementation.h"

#include "ttmlir/Dialect/TT/IR/TTOpsEnums.h.inc"

namespace mlir::tt {
inline bool isSystemMemorySpace(MemorySpace memorySpace) {
  return memorySpace == MemorySpace::System ||
         memorySpace == MemorySpace::SystemMMIO;
}

inline bool isDeviceMemorySpace(MemorySpace memorySpace) {
  return memorySpace == MemorySpace::DeviceDRAM ||
         memorySpace == MemorySpace::DeviceL1;
}
inline void printDimensionList(::mlir::AsmPrinter &printer,
                               ::llvm::ArrayRef<int64_t> shape) {
  printer.printDimensionList(shape);
}

inline ::mlir::ParseResult
parseDimensionList(::mlir::AsmParser &odsParser,
                   ::llvm::SmallVector<int64_t> &dimensions) {
  return odsParser.parseDimensionList(dimensions, false, false);
}

// Custom formatter for printing Dim2D as int64_t arrays representing core
// coordinates as Y-X pairs.
inline void printCoreList(::mlir::AsmPrinter &printer,
                          ::llvm::ArrayRef<int64_t> core_coord) {
  assert(core_coord.size() % 2 == 0 &&
         "Core coord should contain pairs of values");

  printer << "["; // Start the list with a square bracket
  for (size_t i = 0; i < core_coord.size(); i += 2) {
    if (i != 0) {
      printer << ", "; // Separator between pairs
    }
    printer << core_coord[i] << "-"
            << core_coord[i + 1]; // Hyphen between y and x
  }
  printer << "]"; // End the list with a square bracket
}

inline ::mlir::ParseResult
parseCoreList(::mlir::AsmParser &parser,
              ::llvm::SmallVector<int64_t> &coreMappings) {
  // Parse the '[' token to start the list
  if (parser.parseLSquare())
    return ::mlir::failure();

  // Loop to parse multiple y-x pairs
  do {
    for (int i = 0; i < 2; ++i) {
      int64_t value;
      if (parser.parseInteger(value))
        return ::mlir::failure();
      coreMappings.push_back(value);

      // For the first value (y), parse the '-' separator
      if (i == 0 && parser.parseKeyword("-"))
        return ::mlir::failure();
    }
  } while (parser.parseOptionalComma()
               .succeeded()); // Continue parsing pairs separated by commas

  // Parse the ']' token to end the list
  if (parser.parseRSquare())
    return ::mlir::failure();

  return ::mlir::success();
}

} // namespace mlir::tt

#define GET_ATTRDEF_CLASSES
#include "ttmlir/Dialect/TT/IR/TTOpsAttrDefs.h.inc"

#define GET_TYPEDEF_CLASSES
#include "ttmlir/Dialect/TT/IR/TTOpsTypes.h.inc"

namespace mlir::tt {
SystemDescAttr getCurrentScopeSystemDesc(Operation *op);
DeviceAttr getCurrentScopeDevice(Operation *op);
} // namespace mlir::tt

#endif
