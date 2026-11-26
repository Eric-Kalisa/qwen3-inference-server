#include "tokenizer.hpp"
#include <algorithm>
#include <memory>
#include <string>
#include <vector>

struct Handle { std::shared_ptr<tokenizer::PreTrainedTokenizer> value; };
extern "C" void* qwen3_tokenizer_load(const char* directory) {
    try {
        auto value=tokenizer::AutoTokenizer::from_pretrained(directory);
        if(!value) return nullptr;
        // Cleanup can revise text already streamed to the client.
        value->set_clean_up_tokenization_spaces(false);
        return new Handle{value};
    } catch(...) { return nullptr; }
}
extern "C" void qwen3_tokenizer_free(void* handle) { delete static_cast<Handle*>(handle); }
extern "C" int qwen3_tokenizer_encode(void* handle,const char* text,int* output,int capacity) {
    try {
        if(!handle||!text||capacity<0) return -1;
        auto ids=static_cast<Handle*>(handle)->value->encode(text);
        if(output&&capacity>=(int)ids.size()) std::copy(ids.begin(),ids.end(),output);
        return (int)ids.size();
    } catch(...) { return -1; }
}
extern "C" int qwen3_tokenizer_decode_flags(void* handle,const int* ids,int count,char* output,int capacity,int skip_special) {
    try {
        if(!handle||count<0||capacity<0||(count&&!ids)) return -1;
        std::vector<int> tokens;
        if(count) tokens.assign(ids,ids+count);
        auto text=static_cast<Handle*>(handle)->value->decode(tokens,skip_special!=0);
        if(output&&capacity>(int)text.size()) {
            std::copy(text.begin(),text.end(),output); output[text.size()]='\0';
        }
        return (int)text.size();
    } catch(...) { return -1; }
}
extern "C" int qwen3_tokenizer_decode(void* handle,const int* ids,int count,char* output,int capacity) {
    return qwen3_tokenizer_decode_flags(handle,ids,count,output,capacity,1);
}
