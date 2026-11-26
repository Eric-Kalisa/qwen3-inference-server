#include "tokenizer.hpp"

#include <algorithm>
#include <memory>
#include <string>
#include <vector>

struct TokenizerHandle {
    std::shared_ptr<tokenizer::PreTrainedTokenizer> value;
};

extern "C" {

void* qwen3_tokenizer_load(const char* model_directory) {
    try {
        auto tokenizer = tokenizer::AutoTokenizer::from_pretrained(model_directory);
        if (!tokenizer) return nullptr;
        return new TokenizerHandle{tokenizer};
    } catch (...) {
        return nullptr;
    }
}

void qwen3_tokenizer_free(void* handle) {
    delete static_cast<TokenizerHandle*>(handle);
}

int qwen3_tokenizer_encode(
    void* handle, const char* text, int* output_ids, int capacity) {
    auto* tokenizer_handle = static_cast<TokenizerHandle*>(handle);
    const std::vector<int> ids = tokenizer_handle->value->encode(text);
    if (output_ids == nullptr || capacity < static_cast<int>(ids.size())) {
        return static_cast<int>(ids.size());
    }
    std::copy(ids.begin(), ids.end(), output_ids);
    return static_cast<int>(ids.size());
}

int qwen3_tokenizer_decode(
    void* handle, const int* ids, int count, char* output, int capacity) {
    auto* tokenizer_handle = static_cast<TokenizerHandle*>(handle);
    const std::vector<int> token_ids(ids, ids + count);
    const std::string text = tokenizer_handle->value->decode(token_ids);
    const int required = static_cast<int>(text.size());
    if (output == nullptr || capacity <= required) return required;
    std::copy(text.begin(), text.end(), output);
    output[required] = '\0';
    return required;
}

int qwen3_tokenizer_eos_id(void* handle) {
    auto* tokenizer_handle = static_cast<TokenizerHandle*>(handle);
    return tokenizer_handle->value->eos_token_id();
}

int qwen3_tokenizer_special_token_id(void* handle, const char* token) {
    auto* tokenizer_handle = static_cast<TokenizerHandle*>(handle);
    return tokenizer_handle->value->token_to_id(token);
}

}
