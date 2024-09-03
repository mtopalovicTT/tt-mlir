# Runtime Stitching

Runtime stitching adds the ability for the runtime to stitch together multiple,
indepently compiled programs together at runtime, ie. without compiler knowledge of
how the binary programs will be composed.

## Motivation

In order to flexibly support arbitrary training schedules / composing multiple
models together we want to have the ability for the runtime to stitch
graphs together.  To achieve this we need to define an ABI kind of interface
between the compiler and the runtime.

### Simple Example
```
mod_a = forge.compile(PyTorch_module_a)
mod_b = forge.compile(PyTorch_module_b)

for i in range(10):
    outs_a = mod_a(ins_a)
    outs_b = mod_b(outs_a)
```


`mod_a` and `mod_b` are 2 independent compile steps, during the compile step for
`mod_a` it should be completely unaware that `mod_b` will take place and vice-versa.
In order to achieve this we propose a new runtime concept called stitching:

- forge invokes compile step for `mod_a`, tt-mlir compiler determines where the
  inputs (`ins_a`) should live, host, device dram, device l1. tt-mlir returns
  metadata to forge describing where it wants the tensors to reside before invoking
  flatbuffer submission.
- forge invokes compile step for `mod_b`, same happens as bullet 1
- `mod_a` is invoked at runtime, forge runtime needs to inspect the compiler metadata
  to determine where the tensors should live.  Runtime manually invokes a new data
  copy command to get the tenors to the correct memory space / correct memory address.
- forge runtime invokes `mod_a` program submit
- `mod_b` is invoked at runtime, this time it might be that the compiler left
  the tensor outputs in L1, so no data copy is needed to start running `mod_b`
  since the inputs are already in the correct location.

> A more concrete usecase would be a training loop where there are often
> multiple graphs composed together.
> [#82](https://github.com/tenstorrent/tt-mlir/issues/82)
> Or when we eventually support torch 2.0, the torch runtime can arbitrarily
> break the graph anywhere.

## Proposed Changes

### Compiler Metadata

Compiler will encode the input tensor layout information directly into the
flatbuffer tensor desc.  The flatbuffer schema already exists to express this,
we just need to adopt populating it instead of assuming a canonical host layout.

Compiler will decide where the tensors should live, host, device dram, device l1.

### Runtime

- Runtime will inspect the tensor desc metadata to determine where the tensors
  need to end up / what layout they should be in before invoking the program.
- New runtime API `Tensor toLayout(Tensor tensor, ::tt::target::TensorDesc*
  tensorDesc);`
- Runtime will need to invoke `toLayout` on all input tensors before invoking
  the program.

## Test Plan

- Add a new test to the runtime gtest suite that verifies the runtime can
  correctly stitch together 2 independently compiled programs.

## Concerns

- Tensors pass through device memory spaces (dram, L1) will have a dynamic
  address, some arbitrary run order of flatbuffer could cause tensors to end up
  in non-ideal locations in memory.  Specifically, L1, a poorly placed tensor
  might not be able to be moved to a better location without a bounce through
  DRAM.
