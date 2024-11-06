// SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttmlir/Scheduler/QueueScheduler.h"

namespace mlir::tt::scheduler {

void QueueScheduler::scheduleOp(mlir::Operation *op) {
  scheduledOps.insert(op);
  unscheduledOps.erase(op);
  schedule.push_back(op);
}

llvm::SmallVector<mlir::Operation *> QueueScheduler::getScheduleableOps() {
  llvm::SmallVector<mlir::Operation *> scheduleableOps;
  for (auto &op : unscheduledOps) {
    if (canSchedule(op)) {
      scheduleableOps.push_back(op);
    }
  }

  return scheduleableOps;
}

llvm::SmallVector<mlir::Operation *> QueueScheduler::getSchedule() {
  return schedule;
}

std::unique_ptr<Scheduler> QueueScheduler::snapshot() {
  return std::make_unique<QueueScheduler>(*this);
}

bool QueueScheduler::canSchedule(mlir::Operation *op) {
  for (mlir::Operation *dep : dependencies[op]) {
    if (!scheduledOps.count(dep)) {
      return false;
    }
  }

  return true;
}

} // namespace mlir::tt::scheduler
