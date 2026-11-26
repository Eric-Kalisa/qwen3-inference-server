#include <cstddef>
#include <cstdint>
#include <cstdio>

#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include "cuda_tile.h"

namespace ct = cuda::tiles;

__tile_global__ void embedding_lookup(
    const std::int64_t* token_ids,
    const __nv_bfloat16* embedding_weight,
    __nv_bfloat16* output,
    std::size_t rows,
    std::size_t vocabulary_size,
    std::size_t hidden_size) {
    using namespace ct::literals;
    auto ids_view = ct::partition_view{
        ct::tensor_span{token_ids, ct::extents{rows}}, ct::shape{1_ic},
    };
    auto weight_view = ct::partition_view{
        ct::tensor_span{embedding_weight,
            ct::extents{vocabulary_size * hidden_size}},
        ct::shape{1_ic},
    };
    auto output_view = ct::partition_view{
        ct::tensor_span{output, ct::extents{rows * hidden_size}},
        ct::shape{1_ic},
    };

    auto flat_index = ct::bid().x;
    auto row = flat_index / hidden_size;
    auto dimension = flat_index % hidden_size;
    const auto token_id = static_cast<std::size_t>(ids_view.load(row));
    auto weight_index = token_id * hidden_size + dimension;
    output_view.store(weight_view.load_masked(weight_index), flat_index);
}

__tile_global__ void vocab_projection(
    const __nv_bfloat16* hidden_states,
    const __nv_bfloat16* weight,
    float* accumulator,
    __nv_bfloat16* logits,
    std::size_t hidden_size,
    std::size_t vocabulary_size) {
    using namespace ct::literals;
    auto input_view = ct::partition_view{
        ct::tensor_span{hidden_states, ct::extents{1, hidden_size}},
        ct::shape{1_ic, 128_ic},
    };
    auto weight_view = ct::partition_view{
        ct::tensor_span{weight, ct::extents{vocabulary_size, hidden_size}},
        ct::shape{1_ic, 128_ic},
    };
    auto accumulator_view = ct::partition_view{
        ct::tensor_span{accumulator, ct::extents{vocabulary_size, 1}},
        ct::shape{1_ic, 1_ic},
    };
    auto logits_view = ct::partition_view{
        ct::tensor_span{logits, ct::extents{vocabulary_size, 1}},
        ct::shape{1_ic, 1_ic},
    };

    auto token_id = ct::bid().x;
    auto num_k_tiles = (hidden_size + 127) / 128;
    for (auto k : ct::irange(std::size_t{0}, num_k_tiles)) {
        auto input_tile = ct::element_cast<float>(input_view.load_masked(0, k));
        auto weight_tile = ct::element_cast<float>(weight_view.load_masked(token_id, k));
        auto partial = ct::sum(input_tile * weight_tile, 1_ic);
        auto current = accumulator_view.load_masked(token_id, 0);
        accumulator_view.store(current + partial, token_id, 0);
    }
    auto result = accumulator_view.load_masked(token_id, 0);
    logits_view.store(
        ct::element_cast<__nv_bfloat16>(result), token_id, 0);
}

extern "C" int qwen3_embedding_cuda(
    const std::int64_t* host_token_ids,
    const __nv_bfloat16* host_embedding_weight,
    __nv_bfloat16* host_output,
    int rows,
    int vocabulary_size,
    int hidden_size) {
    const std::size_t id_count = static_cast<std::size_t>(rows);
    const std::size_t weight_count =
        static_cast<std::size_t>(vocabulary_size) * hidden_size;
    const std::size_t output_count = id_count * hidden_size;
    const std::size_t bfloat_count = weight_count + output_count;
    __nv_bfloat16* device_buffer = nullptr;
    std::int64_t* device_ids = nullptr;
    cudaError_t status = cudaMalloc(
        reinterpret_cast<void**>(&device_buffer),
        bfloat_count * sizeof(__nv_bfloat16));
    if (status == cudaSuccess) {
        status = cudaMalloc(
            reinterpret_cast<void**>(&device_ids),
            id_count * sizeof(std::int64_t));
    }
    if (status == cudaSuccess) {
        status = cudaMemcpy(device_ids, host_token_ids,
            id_count * sizeof(std::int64_t), cudaMemcpyHostToDevice);
    }
    if (status == cudaSuccess) {
        status = cudaMemcpy(device_buffer, host_embedding_weight,
            weight_count * sizeof(__nv_bfloat16), cudaMemcpyHostToDevice);
    }
    if (status == cudaSuccess) {
        auto* device_output = device_buffer + weight_count;
        embedding_lookup<<<output_count, 1>>>(
            device_ids, device_buffer, device_output,
            id_count, vocabulary_size, hidden_size);
        status = cudaGetLastError();
        if (status == cudaSuccess) status = cudaDeviceSynchronize();
        if (status == cudaSuccess) {
            status = cudaMemcpy(host_output, device_output,
                output_count * sizeof(__nv_bfloat16), cudaMemcpyDeviceToHost);
        }
    }
    if (device_ids != nullptr) cudaFree(device_ids);
    if (device_buffer != nullptr) cudaFree(device_buffer);
    if (status != cudaSuccess) {
        std::fprintf(stderr, "Qwen3 embedding CUDA failure: %s\n",
            cudaGetErrorString(status));
    }
    return static_cast<int>(status);
}

extern "C" int qwen3_vocab_projection_cuda(
    const __nv_bfloat16* host_hidden_states,
    const __nv_bfloat16* host_weight,
    __nv_bfloat16* host_logits,
    int hidden_size,
    int vocabulary_size) {
    const std::size_t input_count = static_cast<std::size_t>(hidden_size);
    const std::size_t weight_count =
        static_cast<std::size_t>(hidden_size) * vocabulary_size;
    __nv_bfloat16* device_buffer = nullptr;
    float* device_accumulator = nullptr;
    cudaError_t status = cudaMalloc(
        reinterpret_cast<void**>(&device_buffer),
        (input_count + weight_count + vocabulary_size) * sizeof(__nv_bfloat16));
    if (status == cudaSuccess) {
        status = cudaMalloc(
            reinterpret_cast<void**>(&device_accumulator),
            static_cast<std::size_t>(vocabulary_size) * sizeof(float));
    }
    if (status == cudaSuccess) {
        status = cudaMemcpy(device_buffer, host_hidden_states,
            input_count * sizeof(__nv_bfloat16), cudaMemcpyHostToDevice);
    }
    auto* device_weight = device_buffer == nullptr ? nullptr : device_buffer + input_count;
    if (status == cudaSuccess) {
        status = cudaMemcpy(device_weight, host_weight,
            weight_count * sizeof(__nv_bfloat16), cudaMemcpyHostToDevice);
    }
    if (status == cudaSuccess) {
        status = cudaMemset(device_accumulator, 0,
            static_cast<std::size_t>(vocabulary_size) * sizeof(float));
    }
    if (status == cudaSuccess) {
        auto* device_logits = device_weight + weight_count;
        vocab_projection<<<vocabulary_size, 1>>>(
            device_buffer, device_weight, device_accumulator, device_logits,
            hidden_size, vocabulary_size);
        status = cudaGetLastError();
        if (status == cudaSuccess) status = cudaDeviceSynchronize();
        if (status == cudaSuccess) {
            status = cudaMemcpy(host_logits, device_logits,
                static_cast<std::size_t>(vocabulary_size) * sizeof(__nv_bfloat16),
                cudaMemcpyDeviceToHost);
        }
    }
    if (device_accumulator != nullptr) cudaFree(device_accumulator);
    if (device_buffer != nullptr) cudaFree(device_buffer);
    if (status != cudaSuccess) {
        std::fprintf(stderr, "Qwen3 vocabulary projection failure: %s\n",
            cudaGetErrorString(status));
    }
    return static_cast<int>(status);
}
