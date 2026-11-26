#include <cstddef>
#include <cstdio>

#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include "cuda_tile.h"

namespace ct = cuda::tiles;

__tile_global__ void residual_add(
    const __nv_bfloat16* residual,
    const __nv_bfloat16* update,
    __nv_bfloat16* output,
    std::size_t count) {
    using namespace ct::literals;
    auto residual_view = ct::partition_view{
        ct::tensor_span{residual, ct::extents{count}}, ct::shape{128_ic},
    };
    auto update_view = ct::partition_view{
        ct::tensor_span{update, ct::extents{count}}, ct::shape{128_ic},
    };
    auto output_view = ct::partition_view{
        ct::tensor_span{output, ct::extents{count}}, ct::shape{128_ic},
    };
    auto index = ct::bid().x;
    auto left = ct::element_cast<float>(residual_view.load_masked(index));
    auto right = ct::element_cast<float>(update_view.load_masked(index));
    output_view.store_masked(ct::element_cast<__nv_bfloat16>(left + right), index);
}

// The runner composes the existing RMSNorm, rotary, attention, and MLP libraries.
// This module-specific kernel supplies Qwen3DecoderLayer's two residual additions.
extern "C" int qwen3_residual_add_cuda(
    const __nv_bfloat16* host_residual,
    const __nv_bfloat16* host_update,
    __nv_bfloat16* host_output,
    int count) {
    const std::size_t element_count = static_cast<std::size_t>(count);
    __nv_bfloat16* buffer = nullptr;
    cudaError_t status = cudaMalloc(
        reinterpret_cast<void**>(&buffer),
        3 * element_count * sizeof(__nv_bfloat16));
    if (status != cudaSuccess) return static_cast<int>(status);

    __nv_bfloat16* device_residual = buffer;
    __nv_bfloat16* device_update = device_residual + element_count;
    __nv_bfloat16* device_output = device_update + element_count;
    status = cudaMemcpy(device_residual, host_residual,
        element_count * sizeof(__nv_bfloat16), cudaMemcpyHostToDevice);
    if (status == cudaSuccess) {
        status = cudaMemcpy(device_update, host_update,
            element_count * sizeof(__nv_bfloat16), cudaMemcpyHostToDevice);
    }
    if (status == cudaSuccess) {
        residual_add<<<(element_count + 127) / 128, 1>>>(
            device_residual, device_update, device_output, element_count);
        status = cudaGetLastError();
    }
    if (status == cudaSuccess) status = cudaDeviceSynchronize();
    if (status == cudaSuccess) {
        status = cudaMemcpy(host_output, device_output,
            element_count * sizeof(__nv_bfloat16), cudaMemcpyDeviceToHost);
    }
    const cudaError_t free_status = cudaFree(buffer);
    if (status == cudaSuccess) status = free_status;
    if (status != cudaSuccess) {
        std::fprintf(stderr, "Qwen3DecoderLayer residual failure: %s\n",
            cudaGetErrorString(status));
    }
    return static_cast<int>(status);
}
