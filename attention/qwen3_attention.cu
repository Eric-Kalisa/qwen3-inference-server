#include <cmath>
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

// Projection math mirrors Qwen3MLP. A scratch FP32 scalar per output avoids
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
    auto accumulator_view = ct::partition_view{
        ct::tensor_span{accumulator, ct::extents{rows, output_size}},
        ct::shape{1_ic, 1_ic},
    };
    auto output_view = ct::partition_view{
        ct::tensor_span{output, ct::extents{rows, output_size}},
        ct::shape{1_ic, 1_ic},
    };

    auto row = ct::bid().x;
    auto output_index = ct::bid().y;
    auto num_k_tiles = (input_size + 127) / 128;
    for (auto k : ct::irange(std::size_t{0}, num_k_tiles)) {
        auto input_tile = ct::element_cast<float>(input_view.load_masked(row, k));
        auto weight_tile = ct::element_cast<float>(weight_view.load_masked(output_index, k));
        auto partial = ct::sum(input_tile * weight_tile, 1_ic);
        auto current = accumulator_view.load_masked(row, output_index);
        accumulator_view.store(current + partial, row, output_index);
    }
    auto result = accumulator_view.load_masked(row, output_index);
    output_view.store(ct::element_cast<__nv_bfloat16>(result), row, output_index);
}

__tile_global__ void rms_norm(
    __nv_bfloat16* values,
    const __nv_bfloat16* norm_weight,
    std::size_t vector_count,
    std::size_t head_dim,
    float epsilon) {
    using namespace ct::literals;
    auto values_view = ct::partition_view{
        ct::tensor_span{values, ct::extents{vector_count, head_dim}},
        ct::shape{1_ic, 128_ic},
    };
    auto weight_view = ct::partition_view{
        ct::tensor_span{norm_weight, ct::extents{1, head_dim}},
        ct::shape{1_ic, 128_ic},
    };
    auto vector = ct::bid().x;
    auto input = ct::element_cast<float>(values_view.load_masked(vector, 0));
    auto weight = weight_view.load_masked(0, 0);
    auto mean_square = ct::sum(input * input, 1_ic) / static_cast<float>(head_dim);
    auto inverse_rms = 1.0f / ct::sqrt(mean_square + epsilon);
    auto normalized = ct::element_cast<__nv_bfloat16>(input * inverse_rms);
    values_view.store_masked(normalized * weight, vector, 0);
}

__tile_global__ void apply_rope(
    const __nv_bfloat16* input,
    const __nv_bfloat16* cos,
    const __nv_bfloat16* sin,
    __nv_bfloat16* output,
    std::size_t rows,
    std::size_t heads,
    std::size_t head_dim) {
    using namespace ct::literals;
    auto input_view = ct::partition_view{
        ct::tensor_span{input, ct::extents{rows * heads * head_dim}},
        ct::shape{1_ic},
    };
    auto cos_view = ct::partition_view{
        ct::tensor_span{cos, ct::extents{rows * head_dim}},
        ct::shape{1_ic},
    };
    auto sin_view = ct::partition_view{
        ct::tensor_span{sin, ct::extents{rows * head_dim}},
        ct::shape{1_ic},
    };
    auto output_view = ct::partition_view{
        ct::tensor_span{output, ct::extents{rows * heads * head_dim}},
        ct::shape{1_ic},
    };

    auto index = ct::bid().x;
    auto dimension = index % head_dim;
    auto row = index / (heads * head_dim);
    auto mate_dimension = dimension < head_dim / 2
        ? dimension + head_dim / 2
        : dimension - head_dim / 2;
    auto mate_index = index - dimension + mate_dimension;
    auto position_index = row * head_dim + dimension;
    auto value = ct::element_cast<float>(input_view.load_masked(index));
    auto mate = ct::element_cast<float>(input_view.load_masked(mate_index));
    auto cosine = ct::element_cast<float>(cos_view.load_masked(position_index));
    auto sine = ct::element_cast<float>(sin_view.load_masked(position_index));
    auto rotated = dimension < head_dim / 2 ? -mate : mate;
    auto direct_product = ct::element_cast<__nv_bfloat16>(value * cosine);
    auto rotated_product = ct::element_cast<__nv_bfloat16>(rotated * sine);
    auto result = direct_product + rotated_product;
    output_view.store(result, index);
}

__tile_global__ void transpose_values(
    const __nv_bfloat16* input,
    __nv_bfloat16* output,
    std::size_t batch,
    std::size_t sequence,
    std::size_t kv_heads,
    std::size_t head_dim) {
    using namespace ct::literals;
    auto input_count = batch * sequence * kv_heads * head_dim;
    auto input_view = ct::partition_view{
        ct::tensor_span{input, ct::extents{input_count}},
        ct::shape{1_ic},
    };
    auto output_view = ct::partition_view{
        ct::tensor_span{output, ct::extents{input_count}},
        ct::shape{1_ic},
    };
    auto index = ct::bid().x;
    auto dimension = index % head_dim;
    auto kv_head = (index / head_dim) % kv_heads;
    auto row = index / (kv_heads * head_dim);
    auto batch_index = row / sequence;
    auto position = row % sequence;
    auto output_index = ((batch_index * kv_heads + kv_head) * head_dim + dimension) *
        sequence + position;
    output_view.store(input_view.load_masked(index), output_index);
}

__tile_global__ void attention_scores(
    const __nv_bfloat16* query,
    const __nv_bfloat16* key,
    const __nv_bfloat16* mask,
    float* scores,
    std::size_t batch,
    std::size_t sequence,
    std::size_t heads,
    std::size_t kv_heads,
    std::size_t head_dim,
    float scale) {
    using namespace ct::literals;
    using scalar_tile = ct::tile<float, ct::shape<1, 1>>;
    auto query_view = ct::partition_view{
        ct::tensor_span{query, ct::extents{batch * sequence * heads, head_dim}},
        ct::shape{1_ic, 128_ic},
    };
    auto key_view = ct::partition_view{
        ct::tensor_span{key, ct::extents{batch * sequence * kv_heads, head_dim}},
        ct::shape{1_ic, 128_ic},
    };
    auto mask_view = ct::partition_view{
        ct::tensor_span{mask, ct::extents{batch * sequence * sequence, 1}},
        ct::shape{1_ic, 1_ic},
    };
    auto score_view = ct::partition_view{
        // One 128-key tile covers the generated sequence lengths; padded keys
        // are written as a large negative score and vanish in the softmax.
        ct::tensor_span{scores, ct::extents{batch * heads * sequence, 128}},
        ct::shape{1_ic, 1_ic},
    };

    auto query_slot = ct::bid().x;
    auto key_position = ct::bid().y;
    if (key_position >= sequence) {
        auto invalid = ct::full<scalar_tile>(-1.0e30f);
        score_view.store(invalid, query_slot, key_position);
        return;
    }

    auto batch_index = query_slot / (heads * sequence);
    auto head = (query_slot / sequence) % heads;
    auto query_position = query_slot % sequence;
    auto kv_head = head / (heads / kv_heads);
    auto query_row = (batch_index * sequence + query_position) * heads + head;
    auto key_row = (batch_index * sequence + key_position) * kv_heads + kv_head;
    auto query_tile = ct::element_cast<float>(query_view.load_masked(query_row, 0));
    auto key_tile = ct::element_cast<float>(key_view.load_masked(key_row, 0));
    auto dot = ct::sum(query_tile * key_tile, 1_ic);
    auto mask_index = (batch_index * sequence + query_position) * sequence + key_position;
    auto mask_value = ct::element_cast<float>(
        mask_view.load_masked(mask_index, 0));
    // Keep dot accumulation, scaling, and mask addition in FP32 through softmax.
    auto score = dot * scale + mask_value;
    score_view.store(score, query_slot, key_position);
}

__tile_global__ void attention_softmax(
    const float* scores,
    __nv_bfloat16* weights,
    std::size_t query_count,
    std::size_t sequence) {
    using namespace ct::literals;
    auto score_view = ct::partition_view{
        ct::tensor_span{scores, ct::extents{query_count, 128}},
        ct::shape{1_ic, 128_ic},
    };
    auto weight_view = ct::partition_view{
        ct::tensor_span{weights, ct::extents{query_count, sequence}},
        ct::shape{1_ic, 128_ic},
    };
    auto query_slot = ct::bid().x;
    auto score_tile = score_view.load_masked(query_slot, 0);
    auto exponentials = ct::exp(score_tile);
    auto denominator = ct::sum(exponentials, 1_ic);
    auto probabilities = ct::element_cast<__nv_bfloat16>(exponentials / denominator);
    weight_view.store_masked(probabilities, query_slot, 0);
}

__tile_global__ void attention_context(
    const __nv_bfloat16* weights,
    const __nv_bfloat16* values,
    __nv_bfloat16* context,
    std::size_t batch,
    std::size_t sequence,
    std::size_t heads,
    std::size_t kv_heads,
    std::size_t head_dim) {
    using namespace ct::literals;
    auto weights_view = ct::partition_view{
        ct::tensor_span{weights, ct::extents{batch * heads * sequence, sequence}},
        ct::shape{1_ic, 128_ic},
    };
    auto values_view = ct::partition_view{
        ct::tensor_span{values, ct::extents{batch * kv_heads * head_dim, sequence}},
        ct::shape{1_ic, 128_ic},
    };
    auto context_view = ct::partition_view{
        ct::tensor_span{context, ct::extents{batch * sequence, heads * head_dim}},
        ct::shape{1_ic, 1_ic},
    };

    auto row = ct::bid().x;
    auto component = ct::bid().y;
    auto head = component / head_dim;
    auto dimension = component % head_dim;
    auto batch_index = row / sequence;
    auto position = row % sequence;
    auto query_slot = (batch_index * heads + head) * sequence + position;
    auto kv_head = head / (heads / kv_heads);
    auto value_row = (batch_index * kv_heads + kv_head) * head_dim + dimension;
    auto weight_tile = ct::element_cast<float>(weights_view.load_masked(query_slot, 0));
    auto value_tile = ct::element_cast<float>(values_view.load_masked(value_row, 0));
    auto result = ct::sum(weight_tile * value_tile, 1_ic);
    context_view.store_masked(
        ct::element_cast<__nv_bfloat16>(result), row, component);
}

static cudaError_t run_linear(
    const char* name,
    const __nv_bfloat16* input,
    const __nv_bfloat16* weight,
    float* accumulator,
    __nv_bfloat16* output,
    std::size_t rows,
    std::size_t input_size,
    std::size_t output_size) {
    cudaError_t status = cudaMemset(
        accumulator, 0, rows * output_size * sizeof(float));
    if (status == cudaSuccess) {
        linear<<<dim3(rows, output_size), 1>>>(
            input, weight, accumulator, output, rows, input_size, output_size);
        status = cudaGetLastError();
    }
    if (status == cudaSuccess) {
        status = cudaDeviceSynchronize();
    }
    return status;
}

extern "C" int qwen3_attention_cuda(
    const __nv_bfloat16* host_hidden_states,
    const __nv_bfloat16* host_cos,
    const __nv_bfloat16* host_sin,
    const __nv_bfloat16* host_attention_mask,
    const __nv_bfloat16* host_q_weight,
    const __nv_bfloat16* host_k_weight,
    const __nv_bfloat16* host_v_weight,
    const __nv_bfloat16* host_o_weight,
    const __nv_bfloat16* host_q_norm,
    const __nv_bfloat16* host_k_norm,
    __nv_bfloat16* host_output,
    __nv_bfloat16* host_attention_weights,
    int batch,
    int sequence,
    int hidden_size,
    int num_heads,
    int num_kv_heads,
    int head_dim,
    float rms_norm_eps) {
    const std::size_t rows = static_cast<std::size_t>(batch) * sequence;
    const std::size_t query_width = static_cast<std::size_t>(num_heads) * head_dim;
    const std::size_t kv_width = static_cast<std::size_t>(num_kv_heads) * head_dim;
    const std::size_t input_count = rows * hidden_size;
    const std::size_t query_count = rows * query_width;
    const std::size_t kv_count = rows * kv_width;
    // Qwen3's context width can differ from hidden_size.
    const std::size_t context_count = query_count;
    const std::size_t output_count = input_count;
    const std::size_t o_weight_count = query_width * hidden_size;
    const std::size_t cos_count = rows * head_dim;
    const std::size_t mask_count = static_cast<std::size_t>(batch) * sequence * sequence;
    const std::size_t attention_count =
        static_cast<std::size_t>(batch) * num_heads * sequence * sequence;
    const std::size_t query_slots = static_cast<std::size_t>(batch) * num_heads * sequence;
    const std::size_t score_count = query_slots * 128;
    const std::size_t accumulator_count = rows *
        (query_width > hidden_size ? query_width : hidden_size);
    const std::size_t bfloat_count = input_count + 2 * cos_count + mask_count +
        query_width * hidden_size + 2 * kv_width * hidden_size +
        o_weight_count + 2 * head_dim + 2 * query_count +
        4 * kv_count + context_count + output_count + attention_count;

    __nv_bfloat16* device_values = nullptr;
    float* device_float_values = nullptr;
    cudaError_t status;
    {
        status = cudaMalloc(
            reinterpret_cast<void**>(&device_values),
            bfloat_count * sizeof(__nv_bfloat16));
        if (status == cudaSuccess) {
            status = cudaMalloc(
                reinterpret_cast<void**>(&device_float_values),
                (score_count + accumulator_count) * sizeof(float));
        }
    }
    if (!check_cuda("attention workspace allocation", status)) {
        if (device_values != nullptr) cudaFree(device_values);
        return static_cast<int>(status);
    }

    __nv_bfloat16* device_input = device_values;
    __nv_bfloat16* device_cos = device_input + input_count;
    __nv_bfloat16* device_sin = device_cos + cos_count;
    __nv_bfloat16* device_mask = device_sin + cos_count;
    __nv_bfloat16* device_q_weight = device_mask + mask_count;
    __nv_bfloat16* device_k_weight = device_q_weight + query_width * hidden_size;
    __nv_bfloat16* device_v_weight = device_k_weight + kv_width * hidden_size;
    __nv_bfloat16* device_o_weight = device_v_weight + kv_width * hidden_size;
    __nv_bfloat16* device_q_norm = device_o_weight + o_weight_count;
    __nv_bfloat16* device_k_norm = device_q_norm + head_dim;
    __nv_bfloat16* device_q = device_k_norm + head_dim;
    __nv_bfloat16* device_k = device_q + query_count;
    __nv_bfloat16* device_v = device_k + kv_count;
    __nv_bfloat16* device_q_rotated = device_v + kv_count;
    __nv_bfloat16* device_k_rotated = device_q_rotated + query_count;
    __nv_bfloat16* device_v_transposed = device_k_rotated + kv_count;
    __nv_bfloat16* device_context = device_v_transposed + kv_count;
    __nv_bfloat16* device_output = device_context + context_count;
    __nv_bfloat16* device_attention_weights = device_output + output_count;
    float* device_scores = device_float_values;
    float* device_accumulator = device_scores + score_count;

    {
        status = cudaMemcpy(device_input, host_hidden_states,
            input_count * sizeof(__nv_bfloat16), cudaMemcpyHostToDevice);
        if (status == cudaSuccess) status = cudaMemcpy(device_cos, host_cos,
            cos_count * sizeof(__nv_bfloat16), cudaMemcpyHostToDevice);
        if (status == cudaSuccess) status = cudaMemcpy(device_sin, host_sin,
            cos_count * sizeof(__nv_bfloat16), cudaMemcpyHostToDevice);
        if (status == cudaSuccess) status = cudaMemcpy(device_mask, host_attention_mask,
            mask_count * sizeof(__nv_bfloat16), cudaMemcpyHostToDevice);
        if (status == cudaSuccess) status = cudaMemcpy(device_q_weight, host_q_weight,
            query_width * hidden_size * sizeof(__nv_bfloat16), cudaMemcpyHostToDevice);
        if (status == cudaSuccess) status = cudaMemcpy(device_k_weight, host_k_weight,
            kv_width * hidden_size * sizeof(__nv_bfloat16), cudaMemcpyHostToDevice);
        if (status == cudaSuccess) status = cudaMemcpy(device_v_weight, host_v_weight,
            kv_width * hidden_size * sizeof(__nv_bfloat16), cudaMemcpyHostToDevice);
        if (status == cudaSuccess) status = cudaMemcpy(device_o_weight, host_o_weight,
            o_weight_count * sizeof(__nv_bfloat16), cudaMemcpyHostToDevice);
        if (status == cudaSuccess) status = cudaMemcpy(device_q_norm, host_q_norm,
            head_dim * sizeof(__nv_bfloat16), cudaMemcpyHostToDevice);
        if (status == cudaSuccess) status = cudaMemcpy(device_k_norm, host_k_norm,
            head_dim * sizeof(__nv_bfloat16), cudaMemcpyHostToDevice);
    }
    if (!check_cuda("copy attention inputs and weights", status)) goto cleanup;

    status = run_linear("query projection", device_input, device_q_weight,
        device_accumulator, device_q, rows, hidden_size, query_width);
    if (!check_cuda("query projection", status)) goto cleanup;
    status = run_linear("key projection", device_input, device_k_weight,
        device_accumulator, device_k, rows, hidden_size, kv_width);
    if (!check_cuda("key projection", status)) goto cleanup;
    status = run_linear("value projection", device_input, device_v_weight,
        device_accumulator, device_v, rows, hidden_size, kv_width);
    if (!check_cuda("value projection", status)) goto cleanup;

    {
        rms_norm<<<rows * num_heads, 1>>>(
            device_q, device_q_norm, rows * num_heads, head_dim, rms_norm_eps);
        status = cudaGetLastError();
        if (status == cudaSuccess) {
            rms_norm<<<rows * num_kv_heads, 1>>>(
                device_k, device_k_norm, rows * num_kv_heads, head_dim, rms_norm_eps);
            status = cudaGetLastError();
        }
        if (status == cudaSuccess) status = cudaDeviceSynchronize();
    }
    if (!check_cuda("query and key RMSNorm", status)) goto cleanup;

    {
        apply_rope<<<query_count, 1>>>(
            device_q, device_cos, device_sin, device_q_rotated,
            rows, num_heads, head_dim);
        status = cudaGetLastError();
        if (status == cudaSuccess) {
            apply_rope<<<kv_count, 1>>>(
                device_k, device_cos, device_sin, device_k_rotated,
                rows, num_kv_heads, head_dim);
            status = cudaGetLastError();
        }
        if (status == cudaSuccess) status = cudaDeviceSynchronize();
    }
    if (!check_cuda("query and key rotary embeddings", status)) goto cleanup;

    {
        transpose_values<<<kv_count, 1>>>(
            device_v, device_v_transposed, batch, sequence, num_kv_heads, head_dim);
        status = cudaGetLastError();
        if (status == cudaSuccess) status = cudaDeviceSynchronize();
    }
    if (!check_cuda("transpose values", status)) goto cleanup;

    {
        attention_scores<<<dim3(query_slots, 128), 1>>>(
            device_q_rotated, device_k_rotated, device_mask, device_scores,
            batch, sequence, num_heads, num_kv_heads, head_dim,
            1.0f / std::sqrt(static_cast<float>(head_dim)));
        status = cudaGetLastError();
        if (status == cudaSuccess) status = cudaDeviceSynchronize();
    }
    if (!check_cuda("attention score and mask", status)) goto cleanup;

    {
        attention_softmax<<<query_slots, 1>>>(
            device_scores, device_attention_weights, query_slots, sequence);
        status = cudaGetLastError();
        if (status == cudaSuccess) status = cudaDeviceSynchronize();
    }
    if (!check_cuda("attention softmax", status)) goto cleanup;

    {
        attention_context<<<dim3(rows, query_width), 1>>>(
            device_attention_weights, device_v_transposed, device_context,
            batch, sequence, num_heads, num_kv_heads, head_dim);
        status = cudaGetLastError();
        if (status == cudaSuccess) status = cudaDeviceSynchronize();
    }
    if (!check_cuda("attention value aggregation", status)) goto cleanup;

    status = run_linear("output projection", device_context, device_o_weight,
        device_accumulator, device_output, rows, query_width, hidden_size);
    if (!check_cuda("output projection", status)) goto cleanup;

    status = cudaMemcpy(host_output, device_output,
        output_count * sizeof(__nv_bfloat16), cudaMemcpyDeviceToHost);
    if (status == cudaSuccess) {
        status = cudaMemcpy(host_attention_weights, device_attention_weights,
            attention_count * sizeof(__nv_bfloat16), cudaMemcpyDeviceToHost);
    }
    if (!check_cuda("copy attention results", status)) goto cleanup;

cleanup:
    cudaFree(device_float_values);
    cudaFree(device_values);
    return static_cast<int>(status);
}
