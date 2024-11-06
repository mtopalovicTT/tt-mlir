// SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#ifndef TTMLIR_SCHEDULER_QUEUESCHEDULER_H
#define TTMLIR_SCHEDULER_QUEUESCHEDULER_H

#include "ttmlir/Scheduler/Scheduler.h"

namespace mlir::tt::scheduler {

class QueueScheduler : public Scheduler {
public:
  // Constructor taking an MLIR Operation (or a module)
  QueueScheduler(func::FuncOp *root) : Scheduler(root) {};

  // Copy constructor
  QueueScheduler(const QueueScheduler &scheduler) : Scheduler(scheduler) {};

  // Method to schedule an operation
  void scheduleOp(mlir::Operation *op) final;

  // Method to get the next set of schedulable operations
  llvm::SmallVector<mlir::Operation *> getScheduleableOps() final;

  // Method to get the scheduled operations
  llvm::SmallVector<mlir::Operation *> getSchedule() final;

  // Method to take a snapshot of the scheduler
  std::unique_ptr<Scheduler> snapshot() final;

private:
  // Method to check if an operation can be scheduled
  bool canSchedule(mlir::Operation *op);
};

} // namespace mlir::tt::scheduler

#endif // TTMLIR_SCHEDULER_QUEUESCHEDULER_H
