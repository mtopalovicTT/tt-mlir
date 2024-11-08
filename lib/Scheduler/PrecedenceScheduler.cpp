// SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttmlir/Scheduler/PrecedenceScheduler.h"

namespace mlir::tt::scheduler {

PrecedenceScheduler::PrecedenceScheduler(func::FuncOp *root) : Scheduler(root) {
  root->walk([&](mlir::Operation *op) {
    if (op->hasTrait<mlir::OpTrait::ReturnLike>()) {
      outputOp = op;
    }
  });
}

void PrecedenceScheduler::scheduleOp(mlir::Operation *op) {
  unscheduledOps.erase(op);
  scheduleableOps.erase(op);
  scheduledOps.insert(op);

  OpResult result = op->getResult(0);
  for (mlir::Operation *use : result.getUsers()) {
    precedence[use].push_back(op);

    // Check the schedulability of the user op after scheduling the current op
    //
    if (canSchedule(use)) {
      scheduleableOps.insert(use);
    }
  }
}

llvm::SmallVector<mlir::Operation *> PrecedenceScheduler::getScheduleableOps() {
  return llvm::SmallVector<mlir::Operation *>(scheduleableOps.begin(),
                                              scheduleableOps.end());
}

llvm::SmallVector<mlir::Operation *> PrecedenceScheduler::getSchedule() {
  constructSchedule(outputOp);
  return schedule;
}

std::unique_ptr<Scheduler> PrecedenceScheduler::snapshot() {
  return std::make_unique<PrecedenceScheduler>(*this);
}

void PrecedenceScheduler::constructSchedule(mlir::Operation *op) {
  // Schedule all the precedents of the current operation
  //
  for (mlir::Operation *precedent : precedence[op]) {
    if (!visitedOps.count(precedent)) {
      constructSchedule(precedent);
    }
  }

  // Schedule the current operation
  //
  visitedOps.insert(op);
  schedule.push_back(op);
}

} // namespace mlir::tt::scheduler
