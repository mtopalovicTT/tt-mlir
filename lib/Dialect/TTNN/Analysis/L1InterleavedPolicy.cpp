// SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttmlir/Dialect/TTNN/Analysis/L1InterleavedPolicy.h"
#include "ttmlir/Dialect/TT/IR/TTOpsTypes.h"
#include "ttmlir/Scheduler/PrecedenceScheduler.h"
#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mlir/IR/Operation.h>

namespace mlir::tt::ttnn {

bool L1InterleavedPolicy::hasDRAMLayout(mlir::Operation *op) {
  return std::find_if(legalLayouts[op].begin(), legalLayouts[op].end(),
                      [](tt::LayoutAttr layout) {
                        return layout.getMemorySpace() ==
                               tt::MemorySpace::DeviceDRAM;
                      }) != legalLayouts[op].end();
}

tt::LayoutAttr L1InterleavedPolicy::getDRAMLayout(mlir::Operation *op) {
  assert(hasDRAMLayout(op));
  auto dramLayoutIter = std::find_if(
      legalLayouts[op].begin(), legalLayouts[op].end(),
      [](tt::LayoutAttr layout) {
        return layout.getMemorySpace() == tt::MemorySpace::DeviceDRAM;
      });
  return *dramLayoutIter;
}

bool L1InterleavedPolicy::hasL1InterleavedLayout(mlir::Operation *op) {
  return std::find_if(legalLayouts[op].begin(), legalLayouts[op].end(),
                      [](tt::LayoutAttr layout) {
                        return layout.hasInterleavedL1TensorMemoryLayout();
                      }) != legalLayouts[op].end();
}

tt::LayoutAttr
L1InterleavedPolicy::getL1InterleavedLayout(mlir::Operation *op) {
  assert(hasL1InterleavedLayout(op));
  auto l1InterleaveLayoutIter =
      std::find_if(legalLayouts[op].begin(), legalLayouts[op].end(),
                   [](tt::LayoutAttr layout) {
                     return layout.hasInterleavedL1TensorMemoryLayout();
                   });
  return *l1InterleaveLayoutIter;
}

void L1InterleavedPolicy::run() {
  rootOp->walk([&](func::FuncOp func) {
    // DeviceAttr deviceAttr = getCurrentScopeDevice(func);

    // Create mapping from MLIR op to OpNode
    //
    llvm::DenseMap<mlir::Operation *, OpNode> opNodeMap;
    func->walk([&](Operation *op) {
      OpNode opNode;

      // Fill in the OpNode
      opNode.op = op;

      // Insert the OpNode
      opNodeMap[op] = opNode;
    });

    // Start the policy.
    //
    mlir::tt::scheduler::PrecedenceScheduler scheduler(&func);
    llvm::SmallVector<mlir::Operation *> scheduleableOps;

    while (scheduler.hasUnscheduledOps()) {
      scheduleableOps = scheduler.getScheduleableOps();

      for (mlir::Operation *op : scheduleableOps) {
        // In the current implementation (V2) of the L1InterleavedPolicy, we do
        // not deal with fork ops.
        //
        if (op->hasOneUse()) {
          ;
        }
      }
    }
  });
}

uint64_t getOutputL1Usage(Operation *op, tt::LayoutAttr opLayout,
                          DeviceAttr &deviceAttr) {
  assert(opLayout.hasInterleavedL1TensorMemoryLayout());

  llvm::ArrayRef<int64_t> opOutputTensorShape =
      mlir::cast<RankedTensorType>(op->getResult(0).getType()).getShape();

  uint64_t opL1OutputUsage = deviceAttr.getLayoutSizeBytes(
      opOutputTensorShape, opLayout, opLayout.getMemorySpace());
  return opL1OutputUsage;
}

} // namespace mlir::tt::ttnn
