# CUDA Kernel Optimization 🔥

> From naive CUDA to Flash Attention — implementing and 
> benchmarking GPU kernels from scratch on NVIDIA L4 GPU

## Hardware
- **GPU:** NVIDIA L4 (23.7GB, 58 SMs, 7,424 CUDA cores)
- **CUDA Capability:** 8.9 (Ampere)
- **Peak FP32:** ~30.3 TFLOPS

## Results

### Vector Addition (1M elements)
| Kernel | Latency | vs PyTorch |
|--------|---------|-----------|
| PyTorch built-in | 19.99 μs | 1.00x |
| Naive CUDA v1 | 42.65 μs | 2.13x slower |
| Grid-stride v2 | 32.26 μs | 1.61x slower |
| Float4 vectorized v3 | 31.69 μs | 1.59x slower |

**Lesson:** Memory coalescing, vectorized float4 loads, 
grid-stride loops

---

### Matrix Multiply (1024×1024)
| Kernel | Latency | vs Naive |
|--------|---------|---------|
| PyTorch cuBLAS | 239.6 μs | 6.62x faster |
| Naive CUDA | 1585.4 μs | baseline |
| Tiled 16×16 (shared mem) | 1154.5 μs | 1.37x faster |
| **Triton 32×32** | **210.3 μs** | **7.54x faster** |

**Lesson:** Shared memory tiling reduces HBM reads 16x. 
Triton automatically leverages tensor cores — matching cuBLAS

---

### Flash Attention vs Standard Attention
| Seq Len | Standard | Flash | Speedup | Mem Saved |
|---------|----------|-------|---------|-----------|
| 256 | 100.1 μs | 79.7 μs | 1.26x | 0.13MB |
| 512 | 108.8 μs | 68.1 μs | **1.60x** | 0.52MB |
| 1024 | 93.1 μs | 84.2 μs | 1.11x | 2.10MB |
| 2048 | 101.6 μs | 131.6 μs | 0.77x* | 8.39MB |

*tile size needs tuning at 2048 — see autotuning notes

**Memory reduction: 5.3x (2.10MB → 0.39MB)**

---

## Key Concepts Implemented

### 1. CUDA Thread Hierarchy
- Threads → Warps (32) → Blocks → Grid
- `idx = blockIdx.x * blockDim.x + threadIdx.x`
- Warp divergence avoidance
- Memory coalescing

### 2. Memory Hierarchy
- Registers (~20 TB/s) → Shared Memory (~10 TB/s) 
  → L2 Cache → HBM (300 GB/s)
- **Core insight: GPU is memory-bound, not compute-bound**
- Optimization = keeping data close to compute units

### 3. Shared Memory Tiling
- Load 16×16 tile cooperatively into shared memory
- Reuse 16 times before eviction
- 16x reduction in HBM reads
- Foundation of Flash Attention

### 4. Triton Kernels
- Python-like syntax, CUDA-level performance
- Automatic tensor core utilization
- 7.54x faster than naive CUDA
- Matches cuBLAS without manual optimization

### 5. Flash Attention
- Online softmax — never write N×N matrix to HBM
- O(N²) → O(N) memory complexity
- Implemented from scratch in Triton
- Powers GPT-4, Llama, Gemini

## The Core Insight
```
GPU performance is almost never compute-bound.
It is almost always memory-bound.
Good kernel engineering = keeping data as close
to compute units as possible, for as long as possible.
```

## Tech Stack
```
CUDA C++ · Triton 3.6 · PyTorch 2.10 · CUDA 12.8
torch.profiler · load_inline · NVIDIA L4 GPU
```

## Part of ML Systems Optimization Suite
- ✅ Module 1 — Inference Optimization (ONNX + Quantization)
- ✅ Module 2 — CUDA Kernel Optimization (this repo)
- 🔜 Module 3 — Distributed Training (FSDP + NCCL)
- 🔜 Module 4 — TensorRT Optimization
- 🔜 Module 5 — Agentic AI Systems

## Author
**Piyush Kunjilwar**
MS Information Systems — Northeastern University (May 2026)
[LinkedIn](https://linkedin.com/in/piyush-kunjilwar) ·
[GitHub](https://github.com/piyush12kunjilwar) ·
[Portfolio](https://piyush12kunjilwar.github.io)
```

---

```
CUDA Kernel Optimization (Personal Project)
- Implemented matrix multiply in CUDA C++ — naive → 
  shared memory tiling → Triton, achieving 7.54x 
  speedup over naive implementation
- Built Flash Attention from scratch in Triton — 
  1.60x speedup at seq_len=512, 5.3x memory reduction
  (O(N²) → O(N)) on NVIDIA L4 GPU
- Profiled GPU memory hierarchy — proved memory-bound 
  bottleneck using arithmetic intensity analysis
- Tech: CUDA C++, Triton, torch.profiler, L4 GPU# CUDA-Kernel-Optimization
