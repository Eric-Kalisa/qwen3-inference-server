#include <cstddef>
#include <cstdint>
#include <cstdio>

#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include "cuda_tile.h"

namespace ct = cuda::tiles;

__tile_global__ void rotary_embedding(
    const std::int64_t* position_ids,
    __nv_bfloat16* cos_output,
    __nv_bfloat16* sin_output,
    std::size_t value_count,
    std::size_t head_dim,
    float rope_theta,
    float attention_scaling) {
    using namespace ct::literals;
    using scalar_tile = ct::tile<float, ct::shape<1>>;
    auto position_view = ct::partition_view{
        ct::tensor_span{position_ids, ct::extents{value_count / head_dim}},
        ct::shape{1_ic},
    };
    auto cos_view = ct::partition_view{
        ct::tensor_span{cos_output, ct::extents{value_count}},
        ct::shape{1_ic},
    };
    auto sin_view = ct::partition_view{
        ct::tensor_span{sin_output, ct::extents{value_count}},
        ct::shape{1_ic},
    };

    auto index = ct::bid().x;
    auto position_index = index / head_dim;
    auto dimension = index % head_dim;
    auto half_dim = head_dim / 2;
    auto frequency_index = dimension < half_dim ? dimension : dimension - half_dim;
    auto exponent = static_cast<float>(2 * frequency_index) /
        static_cast<float>(head_dim);
    auto exponent_tile = ct::full<scalar_tile>(exponent);
    auto theta_tile = ct::full<scalar_tile>(rope_theta);
    auto inverse_frequency = 1.0f / ct::pow(theta_tile, exponent_tile);
    auto position = ct::element_cast<float>(position_view.load_masked(position_index));
    auto angle = position * inverse_frequency;
    auto scale = ct::full<scalar_tile>(attention_scaling);
    cos_view.store(ct::element_cast<__nv_bfloat16>(ct::cos(angle) * scale), index);
    sin_view.store(ct::element_cast<__nv_bfloat16>(ct::sin(angle) * scale), index);
}

extern "C" int qwen3_rotary_embedding_cuda(
    const std::int64_t* host_position_ids,
    __nv_bfloat16* host_cos,
    __nv_bfloat16* host_sin,
    int rows,
    int head_dim,
    float rope_theta,
    float attention_scaling) {
    const std::size_t position_count = static_cast<std::size_t>(rows);
    const std::size_t value_count = position_count * head_dim;
    std::int64_t* device_positions = nullptr;
    __nv_bfloat16* device_values = nullptr;
    cudaError_t status = cudaMalloc(
        reinterpret_cast<void**>(&device_positions),
        position_count * sizeof(std::int64_t));
    if (status == cudaSuccess) {
        status = cudaMalloc(
            reinterpret_cast<void**>(&device_values),
            2 * value_count * sizeof(__nv_bfloat16));
    }
    if (status == cudaSuccess) {
        status = cudaMemcpy(device_positions, host_position_ids,
            position_count * sizeof(std::int64_t), cudaMemcpyHostToDevice);
    }
    if (status == cudaSuccess) {
        auto* device_cos = device_values;
        auto* device_sin = device_cos + value_count;
        rotary_embedding<<<value_count, 1>>>(
            device_positions, device_cos, device_sin, value_count,
            head_dim, rope_theta, attention_scaling);
        status = cudaGetLastError();
        if (status == cudaSuccess) status = cudaDeviceSynchronize();
        if (status == cudaSuccess) {
            status = cudaMemcpy(host_cos, device_cos,
                value_count * sizeof(__nv_bfloat16), cudaMemcpyDeviceToHost);
        }
        if (status == cudaSuccess) {
            status = cudaMemcpy(host_sin, device_sin,
                value_count * sizeof(__nv_bfloat16), cudaMemcpyDeviceToHost);
        }
    }
    if (device_values != nullptr) cudaFree(device_values);
    if (device_positions != nullptr) cudaFree(device_positions);
    if (status != cudaSuccess) {
        std::fprintf(stderr, "Qwen3RotaryEmbedding CUDA failure: %s\n",
            cudaGetErrorString(status));
    }
    return static_cast<int>(status);
}
