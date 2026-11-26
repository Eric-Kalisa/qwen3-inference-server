#include <cstddef>
#include <cstdio>

#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include "cuda_tile.h"

namespace ct = cuda::tiles;

static bool check_cuda(const char* operation, cudaError_t status) {
    if (status == cudaSuccess) {
        return true;
    }

    std::fprintf(
        stderr, "CUDA: %s failed: %s (%d)\n",
        operation, cudaGetErrorString(status), static_cast<int>(status));
    std::fflush(stderr);
    return false;
}

// PyTorch BF16 buffers cross the C ABI without conversion; linear accumulation is FP32.
__tile_global__ void linear(
    const __nv_bfloat16* input,
    const __nv_bfloat16* weight,
    float* accumulator,
    __nv_bfloat16* output,
    std::size_t rows,
    std::size_t input_size,
    std::size_t output_size) {
    using namespace ct::literals;

    auto input_view = ct::partition_view{
        ct::tensor_span{input, ct::extents{rows, input_size}},
        ct::shape{1_ic, 128_ic},
    };
    auto weight_view = ct::partition_view{
        ct::tensor_span{weight, ct::extents{output_size, input_size}},
        ct::shape{1_ic, 128_ic},
    };
    auto output_view = ct::partition_view{
        ct::tensor_span{output, ct::extents{rows, output_size}},
        ct::shape{1_ic, 1_ic},
    };
    auto accumulator_view = ct::partition_view{
        ct::tensor_span{accumulator, ct::extents{rows, output_size}},
        ct::shape{1_ic, 1_ic},
    };

    auto row = ct::bid().x;
    auto output_index = ct::bid().y;
    // A 128-wide tile bounds each reduction. ceil-dividing input_size includes
    // a final partial tile, which load_masked handles without reading past K.
    auto num_k_tiles = (input_size + 127) / 128;
    // The reduction is per 128-element chunk. Store the running FP32 sum in
    // scratch memory rather than a loop-carried Tile value: the latter hung
    // in testing, while this memory-backed form completed and matched cases.
    // Each CTA owns one (row, output) pair, so no other CTA updates its slot.
    for (auto k : ct::irange(std::size_t{0}, num_k_tiles)) {
        auto input_tile = ct::element_cast<float>(input_view.load_masked(row, k));
        auto weight_tile = ct::element_cast<float>(weight_view.load_masked(output_index, k));
        auto partial = ct::sum(input_tile * weight_tile, 1_ic);
        auto current = accumulator_view.load_masked(row, output_index);
        accumulator_view.store(current + partial, row, output_index);
    }

    auto result = accumulator_view.load_masked(row, output_index);
    // Round only at the projection boundary to match the BF16 PyTorch module.
    output_view.store(ct::element_cast<__nv_bfloat16>(result), row, output_index);
}

__tile_global__ void silu_multiply(
    const __nv_bfloat16* gate,
    const __nv_bfloat16* up,
    __nv_bfloat16* output,
    std::size_t count) {
    using namespace ct::literals;

    auto gate_view = ct::partition_view{
        ct::tensor_span{gate, ct::extents{count}},
        ct::shape{128_ic},
    };
    auto up_view = ct::partition_view{
        ct::tensor_span{up, ct::extents{count}},
        ct::shape{128_ic},
    };
    auto output_view = ct::partition_view{
        ct::tensor_span{output, ct::extents{count}},
        ct::shape{128_ic},
    };

    auto tile_index = ct::bid().x;
    auto gate_tile = ct::element_cast<float>(gate_view.load_masked(tile_index));
    auto up_tile = up_view.load_masked(tile_index);
    auto activated = gate_tile / (1.0f + ct::exp(-gate_tile));
    auto activated_bf16 = ct::element_cast<__nv_bfloat16>(activated);
    output_view.store_masked(activated_bf16 * up_tile, tile_index);
}

extern "C" int qwen3_mlp_cuda(
    const __nv_bfloat16* host_input,
    const __nv_bfloat16* host_gate_weight,
    const __nv_bfloat16* host_up_weight,
    const __nv_bfloat16* host_down_weight,
    __nv_bfloat16* host_output,
    int rows,
    int hidden_size,
    int intermediate_size) {
    const std::size_t input_count = static_cast<std::size_t>(rows) * hidden_size;
    const std::size_t projection_weight_count =
        static_cast<std::size_t>(intermediate_size) * hidden_size;
    const std::size_t down_weight_count =
        static_cast<std::size_t>(hidden_size) * intermediate_size;
    const std::size_t intermediate_count =
        static_cast<std::size_t>(rows) * intermediate_size;
    const std::size_t total_count =
        input_count + 2 * projection_weight_count + down_weight_count +
        3 * intermediate_count + input_count;
    const std::size_t accumulator_count = static_cast<std::size_t>(rows) *
        (intermediate_size > hidden_size ? intermediate_size : hidden_size);

    __nv_bfloat16* device_buffer = nullptr;
    float* device_accumulator = nullptr;
    cudaError_t status;
    {
        status = cudaMalloc(
            reinterpret_cast<void**>(&device_buffer),
            total_count * sizeof(__nv_bfloat16));
    }
    if (!check_cuda("cudaMalloc", status)) {
        return static_cast<int>(status);
    }
    {
        status = cudaMalloc(
            reinterpret_cast<void**>(&device_accumulator),
            accumulator_count * sizeof(float));
    }
    if (!check_cuda("accumulator cudaMalloc", status)) {
        cudaFree(device_buffer);
        return static_cast<int>(status);
    }

    __nv_bfloat16* device_input = device_buffer;
    __nv_bfloat16* device_gate_weight = device_input + input_count;
    __nv_bfloat16* device_up_weight = device_gate_weight + projection_weight_count;
    __nv_bfloat16* device_down_weight = device_up_weight + projection_weight_count;
    __nv_bfloat16* device_gate = device_down_weight + down_weight_count;
    __nv_bfloat16* device_up = device_gate + intermediate_count;
    __nv_bfloat16* device_product = device_up + intermediate_count;
    __nv_bfloat16* device_output = device_product + intermediate_count;

    {
        status = cudaMemcpy(
            device_input, host_input, input_count * sizeof(__nv_bfloat16),
            cudaMemcpyHostToDevice);
        if (status == cudaSuccess) {
            status = cudaMemcpy(
                device_gate_weight, host_gate_weight,
                projection_weight_count * sizeof(__nv_bfloat16),
                cudaMemcpyHostToDevice);
        }
        if (status == cudaSuccess) {
            status = cudaMemcpy(
                device_up_weight, host_up_weight,
                projection_weight_count * sizeof(__nv_bfloat16),
                cudaMemcpyHostToDevice);
        }
        if (status == cudaSuccess) {
            status = cudaMemcpy(
                device_down_weight, host_down_weight,
                down_weight_count * sizeof(__nv_bfloat16),
                cudaMemcpyHostToDevice);
        }
    }
    if (!check_cuda("copy inputs and weights", status)) goto cleanup;

    {
        // Reuse scratch across projections, but clear each active output range
        // so a projection never starts with the previous projection's sums.
        status = cudaMemset(
            device_accumulator, 0,
            static_cast<std::size_t>(rows) * intermediate_size * sizeof(float));
    }
    if (!check_cuda("clear gate accumulator", status)) goto cleanup;
    {
        linear<<<dim3(rows, intermediate_size), 1>>>(
            device_input, device_gate_weight, device_accumulator, device_gate,
            rows, hidden_size, intermediate_size);
        status = cudaGetLastError();
        if (status == cudaSuccess) {
            status = cudaDeviceSynchronize();
        }
    }
    if (!check_cuda("gate projection", status)) goto cleanup;

    {
        status = cudaMemset(
            device_accumulator, 0,
            static_cast<std::size_t>(rows) * intermediate_size * sizeof(float));
    }
    if (!check_cuda("clear up accumulator", status)) goto cleanup;
    {
        linear<<<dim3(rows, intermediate_size), 1>>>(
            device_input, device_up_weight, device_accumulator, device_up,
            rows, hidden_size, intermediate_size);
        status = cudaGetLastError();
        if (status == cudaSuccess) {
            status = cudaDeviceSynchronize();
        }
    }
    if (!check_cuda("up projection", status)) goto cleanup;

    {
        silu_multiply<<<(intermediate_count + 127) / 128, 1>>>(
            device_gate, device_up, device_product, intermediate_count);
        status = cudaGetLastError();
        if (status == cudaSuccess) {
            status = cudaDeviceSynchronize();
        }
    }
    if (!check_cuda("SiLU and elementwise multiply", status)) goto cleanup;

    {
        status = cudaMemset(
            device_accumulator, 0,
            static_cast<std::size_t>(rows) * hidden_size * sizeof(float));
    }
    if (!check_cuda("clear down accumulator", status)) goto cleanup;
    {
        linear<<<dim3(rows, hidden_size), 1>>>(
            device_product, device_down_weight, device_accumulator, device_output,
            rows, intermediate_size, hidden_size);
        status = cudaGetLastError();
        if (status == cudaSuccess) {
            status = cudaDeviceSynchronize();
        }
    }
    if (!check_cuda("down projection", status)) goto cleanup;

    {
        status = cudaMemcpy(
            host_output, device_output, input_count * sizeof(__nv_bfloat16),
            cudaMemcpyDeviceToHost);
    }
    if (!check_cuda("copy output", status)) goto cleanup;

cleanup:
    {
        const cudaError_t free_status = cudaFree(device_accumulator);
        if (!check_cuda("accumulator cudaFree", free_status) && status == cudaSuccess) {
            status = free_status;
        }
    }
    {
        const cudaError_t free_status = cudaFree(device_buffer);
        if (!check_cuda("cudaFree", free_status) && status == cudaSuccess) {
            status = free_status;
        }
    }
    return static_cast<int>(status);
}
