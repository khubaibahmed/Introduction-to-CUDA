# An Even Easier Introduction to CUDA — Colab Notebook

This repository contains a Google Colab–friendly Jupyter notebook that accompanies NVIDIA’s free DLI course:

- **NVIDIA DLI (Free)**: *An Even Easier Introduction to CUDA*  
  https://learn.nvidia.com/courses/course-detail?course_id=course-v1:DLI+T-AC-01+V1

The course accompanies Mark Harris’s blog post **“An Even Easier Introduction to CUDA”** and walks through the first practical steps of CUDA C++ development: compiling with `nvcc`, launching kernels, scaling parallelism with threads/blocks, and observing performance changes.

## What you’ll learn

By working through the notebook you will be able to:

- Launch massively parallel CUDA kernels on an NVIDIA GPU
- Organise parallel thread execution for large datasets
- Manage memory between CPU and GPU using Unified Memory
- Measure and compare performance (kernel timing in-code; optional timeline profiling)

## What’s in the notebook

The notebook builds the same vector-add example in stages (1,048,576 floats):

1. **CPU baseline** (`add.cpp`) — simple C++ loop
2. **CUDA: 1 thread** (`add.cu`) — first kernel launch (`<<<1,1>>>`)
3. **CUDA: 256 threads (1 block)** (`add_block.cu`) — block-stride loop using `threadIdx.x`
4. **CUDA: many blocks × 256 threads** (`add_grid.cu`) — grid-stride loop using `blockIdx.x`, `gridDim.x`, plus Unified Memory prefetching and averaged kernel timing

## How to run (Google Colab)

1. Open the notebook in Colab.
2. Set **Runtime → Change runtime type → Hardware accelerator → GPU**.
3. Run the cells top-to-bottom.

### Important: `-arch` setting

The compile cells use `-arch=sm_75` (NVIDIA T4), which is common on Colab.  
If Colab assigns a different GPU, update the `-arch` flag:

- P100: `sm_60`
- V100: `sm_70`
- T4: `sm_75`
- A100: `sm_80`
- L4: `sm_89`

You can check the GPU model using:

```bash
nvidia-smi -L
```

## Profiling notes

The original DLI course outline mentions `nvprof`, but availability depends on your CUDA toolkit and environment. In this Colab-oriented notebook, the programmes print kernel timings using **CUDA events**, which is reliable and keeps the focus on CUDA fundamentals. An optional cell shows how to capture a CUDA trace with **Nsight Systems** (`nsys`) if it is available in your runtime.

## Files

- `An_Even_Easier_Introduction_to_CUDA_Colab.ipynb` — main notebook

## Acknowledgements

- NVIDIA Deep Learning Institute (DLI) course: *An Even Easier Introduction to CUDA*  
- Mark Harris’s companion blog post: *An Even Easier Introduction to CUDA*
