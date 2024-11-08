// SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttmlir/Dialect/TTNN/Analysis/L1InterleavedPolicy.h"
#include "ttmlir/Dialect/TT/IR/TTOpsTypes.h"
#include "ttmlir/Scheduler/PrecedenceScheduler.h"
#include <algorithm>
#include <cstdint>
#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mlir/IR/Operation.h>

namespace mlir::tt::ttnn {

bool L1InterleavedPolicy::hasDRAMLayout(Operation *op) {
  return std::find_if(legalLayouts[op].begin(), legalLayouts[op].end(),
                      [](tt::LayoutAttr layout) {
                        return layout.getMemorySpace() ==
                               tt::MemorySpace::DeviceDRAM;
                      }) != legalLayouts[op].end();
}

tt::LayoutAttr L1InterleavedPolicy::getDRAMLayout(Operation *op) {
  assert(hasDRAMLayout(op));
  auto dramLayoutIter = std::find_if(
      legalLayouts[op].begin(), legalLayouts[op].end(),
      [](tt::LayoutAttr layout) {
        return layout.getMemorySpace() == tt::MemorySpace::DeviceDRAM;
      });
  return *dramLayoutIter;
}

bool L1InterleavedPolicy::hasL1InterleavedLayout(Operation *op) {
  return std::find_if(legalLayouts[op].begin(), legalLayouts[op].end(),
                      [](tt::LayoutAttr layout) {
                        return layout.hasInterleavedL1TensorMemoryLayout();
                      }) != legalLayouts[op].end();
}

tt::LayoutAttr L1InterleavedPolicy::getL1InterleavedLayout(Operation *op) {
  assert(hasL1InterleavedLayout(op));
  auto l1InterleaveLayoutIter =
      std::find_if(legalLayouts[op].begin(), legalLayouts[op].end(),
                   [](tt::LayoutAttr layout) {
                     return layout.hasInterleavedL1TensorMemoryLayout();
                   });
  return *l1InterleaveLayoutIter;
}

uint64_t getOpOutputL1Usage(Operation *op, tt::LayoutAttr opLayout,
                            DeviceAttr &deviceAttr) {
  // In case the opLayout is not in L1 memory space, L1 memory usage is 0.
  //
  if (!opLayout.hasInterleavedL1TensorMemoryLayout()) {
    return 0;
  }

  llvm::ArrayRef<int64_t> opOutputTensorShape =
      mlir::cast<RankedTensorType>(op->getResult(0).getType()).getShape();

  uint64_t opL1OutputUsage = deviceAttr.getLayoutSizeBytes(
      opOutputTensorShape, opLayout, opLayout.getMemorySpace());
  return opL1OutputUsage;
}

bool L1InterleavedPolicy::isOpInputLegal(Operation *op,
                                         DeviceAttr &deviceAttr) {
  // TODO(fbajraktari): Placeholder for now
  //
  return true;
}

uint64_t L1InterleavedPolicy::getOpInputL1Usage(Operation *op,
                                                DeviceAttr &deviceAttr) {
  // Calculate the total L1 memory usage of all op's operands.
  //
  uint64_t inputL1Usage = 0;
  for (auto operand : op->getOperands()) {
    Operation *operandOp = operand.getDefiningOp();
    inputL1Usage +=
        getOpOutputL1Usage(operandOp, opNodeMap[operandOp].layout, deviceAttr);
  }

  return inputL1Usage;
}

uint64_t L1InterleavedPolicy::getOpMaxL1Usage(Operation *op,
                                              DeviceAttr &deviceAttr) {
  // Calculate the maximum L1 memory usage of all op's operands.
  //
  uint64_t maxL1Usage = 0;
  for (auto operand : op->getOperands()) {
    Operation *operandOp = operand.getDefiningOp();
    maxL1Usage = std::max(maxL1Usage, opNodeMap[operandOp].maxL1Usage);
  }

  // Calculate the total L1 memory usage of the op.
  //
  uint64_t totalOpL1Usage =
      getOpInputL1Usage(op, deviceAttr) +
      getOpOutputL1Usage(op, opNodeMap[op].layout, deviceAttr);

  return std::max(maxL1Usage, totalOpL1Usage);
}

void L1InterleavedPolicy::run() {
  rootOp->walk([&](func::FuncOp func) {
    DeviceAttr deviceAttr = getCurrentScopeDevice(func);

    // Start the policy.
    //
    mlir::tt::scheduler::PrecedenceScheduler scheduler(&func);
    llvm::SmallVector<Operation *> scheduleableOps;

    while (scheduler.hasUnscheduledOps()) {
      scheduleableOps = scheduler.getScheduleableOps();

      for (Operation *op : scheduleableOps) {
        // Schedule the op.
        //
        scheduler.scheduleOp(op);

        // Skip ops without legal layouts. Those ops don't have output tensors.
        //
        if (legalLayouts[op].size() > 0) {
          // Create the OpNode.
          //
          OpNode opNode;
          opNode.op = op;

          // Analyzing the op implies that all of its operands are scheduled and
          // their respective output layout candidates are chosen. Since those
          // layouts are chosen independently of each other, we have to check if
          // they produce a legal L1 memory state. In case they don't, we have
          // to relax the L1 memory by moving some tensors into DRAM. It is
          // guaranteed that if all operands are in DRAM it is a legal input.
          //
          while (!isOpInputLegal(op, deviceAttr)) {
            ;
          }

          // In the current implementation of the L1InterleavedPolicy (V2), we
          // do not deal with fork ops. Also, we assume that every op has a
          // legal DRAM layout.
          //
          opNode.layout = getDRAMLayout(op);
          if (op->hasOneUse() && hasL1InterleavedLayout(op)) {
            // Figure out this const based on exec data, but will be replaced
            // with API.
            //
            constexpr float tensorL1UsageCap = 0.8;
            uint64_t inputL1Usage = getOpInputL1Usage(op, deviceAttr);
            uint64_t outputL1Usage =
                getOpOutputL1Usage(op, getL1InterleavedLayout(op), deviceAttr);
            bool l1UsageValid = (inputL1Usage + outputL1Usage) <
                                tensorL1UsageCap * usableL1CacheSize;

            if (l1UsageValid) {
              opNodeMap[op].layout = getL1InterleavedLayout(op);
            }
          }

          // Update op's maxL1Usage.
          //
          opNode.maxL1Usage = getOpMaxL1Usage(op, deviceAttr);

          // Insert the OpNode into the map.
          //
          opNodeMap[op] = opNode;
        }
      }
    }
  });
}

} // namespace mlir::tt::ttnn
