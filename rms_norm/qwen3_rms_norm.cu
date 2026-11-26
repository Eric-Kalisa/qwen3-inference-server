#include <cstddef>
#include <cstdio>

#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include "cuda_tile.h"

namespace ct = cuda::tiles;

__tile_global__ void rms_norm(
    const __nv_bfloat16* input,
    const __nv_bfloat16* weight,
    __nv_bfloat16* output,
    float* sum_squares,
    std::size_t rows,
    std::size_t hidden_size,
    float epsilon) {
    using namespace ct::literals;
    auto input_view = ct::partition_view{
        ct::tensor_span{input, ct::extents{rows, hidden_size}},
        ct::shape{1_ic, 128_ic},
    };
    auto weight_view = ct::partition_view{
        ct::tensor_span{weight, ct::extents{1, hidden_size}},
        ct::shape{1_ic, 128_ic},
    };
    auto output_view = ct::partition_view{
        ct::tensor_span{output, ct::extents{rows, hidden_size}},
        ct::shape{1_ic, 128_ic},
    };
    auto sum_view = ct::partition_view{
        ct::tensor_span{sum_squares, ct::extents{rows, 1}},
        ct::shape{1_ic, 1_ic},
    };

    auto row = ct::bid().x;
    const std::size_t num_tiles = (hidden_size + 127) / 128;
    // Qwen3-0.6B has a 1024-wide model hidden state. Accumulate the variance
    // through scratch memory so this remains correct beyond a single 128 tile.
    for (auto tile_index : ct::irange(std::size_t{0}, num_tiles)) {
        auto values = ct::element_cast<float>(
            input_view.load_masked(row, tile_index));
        auto partial = ct::sum(values * values, 1_ic);
        auto current = sum_view.load_masked(row, 0);
        sum_view.store(current + partial, row, 0);
    }
    auto mean_square = sum_view.load_masked(row, 0) /
        static_cast<float>(hidden_size);
    auto inverse_rms = 1.0f / ct::sqrt(mean_square + epsilon);
    for (auto tile_index : ct::irange(std::size_t{0}, num_tiles)) {
        auto values = ct::element_cast<float>(
            input_view.load_masked(row, tile_index));
        auto scale = weight_view.load_masked(0, tile_index);
        auto rounded = ct::element_cast<__nv_bfloat16>(values * inverse_rms);
        output_view.store_masked(rounded * scale, row, tile_index);
    }
}

extern "C" int qwen3_rms_norm_cuda(
    const __nv_bfloat16* host_input,
    const __nv_bfloat16* host_weight,
    __nv_bfloat16* host_output,
    int rows,
    int hidden_size,
    float epsilon) {
    const std::size_t input_count = static_cast<std::size_t>(rows) * hidden_size;
    __nv_bfloat16* buffer = nullptr;
    float* device_sum_squares = nullptr;
    cudaError_t status = cudaMalloc(
        reinterpret_cast<void**>(&buffer),
        (2 * input_count + hidden_size) * sizeof(__nv_bfloat16));
    if (status != cudaSuccess) return static_cast<int>(status);
    status = cudaMalloc(
        reinterpret_cast<void**>(&device_sum_squares),
        static_cast<std::size_t>(rows) * sizeof(float));
    if (status != cudaSuccess) {
        cudaFree(buffer);
        return static_cast<int>(status);
    }
    status = cudaMemset(device_sum_squares, 0,
        static_cast<std::size_t>(rows) * sizeof(float));

    __nv_bfloat16* device_input = buffer;
    __nv_bfloat16* device_weight = device_input + input_count;
    __nv_bfloat16* device_output = device_weight + hidden_size;
    if (status == cudaSuccess) {
        status = cudaMemcpy(device_input, host_input,
            input_count * sizeof(__nv_bfloat16), cudaMemcpyHostToDevice);
    }
    if (status == cudaSuccess) {
        status = cudaMemcpy(device_weight, host_weight,
            hidden_size * sizeof(__nv_bfloat16), cudaMemcpyHostToDevice);
    }
    if (status == cudaSuccess) {
        rms_norm<<<rows, 1>>>(
            device_input, device_weight, device_output, device_sum_squares,
            rows, hidden_size, epsilon);
        status = cudaGetLastError();
    }
    if (status == cudaSuccess) status = cudaDeviceSynchronize();
    if (status == cudaSuccess) {
        status = cudaMemcpy(host_output, device_output,
            input_count * sizeof(__nv_bfloat16), cudaMemcpyDeviceToHost);
    }
    const cudaError_t free_status = cudaFree(buffer);
    if (status == cudaSuccess) status = free_status;
    const cudaError_t sum_free_status = cudaFree(device_sum_squares);
    if (status == cudaSuccess) status = sum_free_status;
    if (status != cudaSuccess) {
        std::fprintf(stderr, "Qwen3RMSNorm CUDA failure: %s\n",
            cudaGetErrorString(status));
    }
    return static_cast<int>(status);
}
