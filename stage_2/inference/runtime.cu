// Minimal BF16 Qwen3 executor. One context, one owning host thread, one stream.
// All public mutations synchronize before returning: a block is never recycled
// while a kernel can still read it. GEMMs use cuBLAS; cache/attention are CUDA.
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cuda_tile.h>
#include <cublas_v2.h>
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <map>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

using B = __nv_bfloat16;
static thread_local std::string error;
static void ck(cudaError_t s) { if (s != cudaSuccess) throw std::runtime_error(cudaGetErrorString(s)); }
static void cb(cublasStatus_t s) { if (s != CUBLAS_STATUS_SUCCESS) throw std::runtime_error("cuBLAS error " + std::to_string(s)); }

namespace ct = cuda::tiles;
using namespace ct::literals;

constexpr int vector_width = 256;
constexpr int tokens_per_tile = 16;

template<int N>
using Indices = ct::tile<int, ct::shape<N>>;

template<class T>
__tile__ auto vector_tiles(T* data, int size) {
    return ct::partition_view{ct::tensor_span{data, ct::extents{size}}, ct::shape<vector_width>{}};
}

template<class T>
__tile__ auto fp32(T tile) {
    return ct::tile_with_element_t<T, float>{tile};
}

template<class T>
__tile__ auto bf16(T tile) {
    return ct::tile_with_element_t<T, B>{tile};
}

template<class Pointers, class Mask>
__tile__ auto load_bf16(Pointers pointers, Mask mask) {
    return fp32(ct::load_masked(pointers, mask, ct::zeros<ct::tile<B, ct::shape<>>>()));
}

static int tile_count(int size) { return (size - 1) / vector_width + 1; }

// Embedding lookup and logit row selection both copy indexed rows.
__tile_global__ void gather_rows(const B* input, B* output, const int* rows, int width, int column_tiles) {
    int block = ct::bid().x;
    int row = block / column_tiles;
    int source = rows[row];
    int column = block % column_tiles;
    auto in = vector_tiles(input + (size_t)source * width, width);
    auto out = vector_tiles(output + (size_t)row * width, width);
    out.store_masked(in.load_masked(column), column);
}

__tile_global__ void norm(const B* x, const B* w, B* y, int width, float eps) {
    size_t base = (size_t)ct::bid().x * width;
    auto in = vector_tiles(x + base, width);
    auto weights = vector_tiles(w, width);
    auto out = vector_tiles(y + base, width);
    using FloatVector = ct::tile<float, ct::shape<vector_width>>;
    auto squares = ct::zeros<FloatVector>();
    int chunks = (width - 1) / vector_width + 1;
    for (int i = 0; i < chunks; ++i) {
        auto values = fp32(in.load_masked(i));
        squares = squares + values * values;
    }
    auto scale = ct::rsqrt(ct::sum(squares, 0_ic) / (float)width + eps);
    for (int i = 0; i < chunks; ++i) {
        auto values = fp32(in.load_masked(i));
        out.store_masked(bf16(values * scale * fp32(weights.load_masked(i))), i);
    }
}

// Q/K normalization and split-half rotary stay in one kernel.
__tile_global__ void rope(B* x, const B* weight, const int* pos, int heads, int dim, float eps, float theta) {
    int head_row = ct::bid().x;
    B* first = x + (size_t)head_row * dim;
    int half = dim / 2;
    B* second = first + half;
    auto d = ct::iota<Indices<128>>();
    auto valid = d < half;
    auto a = load_bf16(first + d, valid);
    auto b = load_bf16(second + d, valid);
    auto scale = ct::rsqrt(ct::sum(a * a + b * b, 0_ic) / (float)dim + eps);
    // Round the normalized halves and rotary coefficients to BF16, as in Qwen3.
    a = fp32(bf16(a * scale * load_bf16(weight + d, valid)));
    b = fp32(bf16(b * scale * load_bf16(weight + half + d, valid)));
    auto exponent = -2.0f * fp32(d) / (float)dim;
    auto angle = (float)pos[head_row / heads] * ct::pow(theta, exponent);
    auto c = fp32(bf16(ct::cos(angle)));
    auto s = fp32(bf16(ct::sin(angle)));
    ct::store_masked(first + d, bf16(a * c - b * s), valid);
    ct::store_masked(second + d, bf16(b * c + a * s), valid);
}

__tile_global__ void cache_write(const B* k, const B* v, B* kc, B* vc, const int* slots,
                                int width, int column_tiles) {
    int block = ct::bid().x;
    int row = block / column_tiles;
    int column = block % column_tiles;
    size_t src = (size_t)row * width;
    size_t dst = (size_t)slots[row] * width;
    auto keys = vector_tiles(k + src, width);
    auto values = vector_tiles(v + src, width);
    auto key_cache = vector_tiles(kc + dst, width);
    auto value_cache = vector_tiles(vc + dst, width);
    key_cache.store_masked(keys.load_masked(column), column);
    value_cache.store_masked(values.load_masked(column), column);
}

// Each tile block owns one query row and head. Online softmax and the weighted
// sum stay local to that block, even when the sequence spans disjoint pages.
template<int HeadWidth>
__tile_global__ void attention(const B* q, const B* kc, const B* vc, B* out,
                               const int* tables, const int* seqs, const int* positions,
                               int maxblocks, int blocksize, int qheads, int kvheads, int dim) {
    int head_row = ct::bid().x;
    int row = head_row / qheads;
    int kv_head = (head_row % qheads) / (qheads / kvheads);
    const int* page_table = tables + (size_t)seqs[row] * maxblocks;
    int last = positions[row];
    int token_stride = kvheads * dim;
    size_t query_base = (size_t)head_row * dim;
    float score_scale = ct::rsqrt((float)dim);
    constexpr float lowest = -0x1.fffffep+127f;

    auto d = ct::iota<ct::tile<int, ct::shape<1, HeadWidth>>>();
    auto token = ct::iota<ct::tile<int, ct::shape<tokens_per_tile, 1>>>();
    auto query = load_bf16(q + query_base + d, d < dim);
    auto weighted = ct::zeros<ct::tile<float, ct::shape<1, HeadWidth>>>();
    auto maximum = ct::full<ct::tile<float, ct::shape<1, 1>>>(lowest);
    auto total = ct::zeros<ct::tile<float, ct::shape<1, 1>>>();

    for (int page = 0; page <= last / blocksize; ++page) {
        size_t base = ((size_t)page_table[page] * blocksize * kvheads + kv_head) * dim;
        int remaining = last - page * blocksize + 1;
        int count = remaining < blocksize ? remaining : blocksize;
        for (int tile = 0; tile <= (count - 1) / tokens_per_tile; ++tile) {
            int start = tile * tokens_per_tile;
            size_t tile_base = base + (size_t)start * token_stride;
            auto valid_token = token < count - start;
            auto mask = valid_token & (d < dim);
            auto offsets = token * token_stride + d;
            auto keys = load_bf16(kc + tile_base + offsets, mask);
            auto values = load_bf16(vc + tile_base + offsets, mask);
            auto scores = ct::sum(keys * query, 1_ic) * score_scale;
            scores = ct::select(valid_token, scores, ct::full<decltype(scores)>(lowest));
            auto next_max = ct::max(maximum, ct::reduce_max(scores, 0_ic));
            auto alpha = ct::exp(maximum - next_max);
            auto probs = ct::select(valid_token, ct::exp(scores - next_max), ct::zeros<decltype(scores)>());
            weighted = weighted * alpha + ct::sum(probs * values, 0_ic);
            total = total * alpha + ct::sum(probs, 0_ic);
            maximum = next_max;
        }
    }
    ct::store_masked(out + query_base + d, bf16(weighted / total), d < dim);
}

__tile_global__ void add(B* x, const B* y, int n) {
    int tile = ct::bid().x;
    auto dst = vector_tiles(x, n);
    auto src = vector_tiles(y, n);
    dst.store_masked(bf16(fp32(dst.load_masked(tile)) + fp32(src.load_masked(tile))), tile);
}

__tile_global__ void silu(B* gate, const B* up, int n) {
    int tile = ct::bid().x;
    auto gates = vector_tiles(gate, n);
    auto inputs = vector_tiles(up, n);
    auto g = fp32(gates.load_masked(tile));
    auto u = fp32(inputs.load_masked(tile));
    gates.store_masked(bf16(g / (1.0f + ct::exp(-g)) * u), tile);
}

__tile_global__ void to_float(const B* x, float* y, int n) {
    int tile = ct::bid().x;
    vector_tiles(y, n).store_masked(fp32(vector_tiles(x, n).load_masked(tile)), tile);
}

struct Engine {
    int h, mid, layers, qh, kh, dim, vocab, blocks, bs, context, rows, sequences, mb;
    float eps, theta;
    std::thread::id owner=std::this_thread::get_id();
    cudaStream_t stream=nullptr;
    cublasHandle_t blas=nullptr;
    std::map<std::string,B*> weights;
    std::vector<void*> allocations;
    B *kc,*vc,*x,*n,*q,*k,*v,*att,*tmp,*gate,*up,*selected,*logits;
    float* flogits;
    int *ids,*pos,*seq,*slots,*tables,*last;
    template<class T> T* alloc(size_t count) {
        T* p=nullptr; ck(cudaMalloc((void**)&p,count*sizeof(T)));
        allocations.push_back(p); ck(cudaMemsetAsync(p,0,count*sizeof(T),stream)); return p;
    }
    void check() { if(owner!=std::this_thread::get_id()) throw std::runtime_error("CUDA context used by non-owner thread"); }
    void sync() { ck(cudaGetLastError()); ck(cudaStreamSynchronize(stream)); }
    ~Engine() {
        if(stream) cudaStreamSynchronize(stream);
        for(void* p:allocations) cudaFree(p);
        if(blas) cublasDestroy(blas);
        if(stream) cudaStreamDestroy(stream);
    }
    void init() {
        ck(cudaStreamCreateWithFlags(&stream,cudaStreamNonBlocking));
        cb(cublasCreate(&blas)); cb(cublasSetStream(blas,stream));
        mb=(context+bs-1)/bs;
        size_t cache=(size_t)layers*blocks*bs*kh*dim;
        kc=alloc<B>(cache); vc=alloc<B>(cache);
        x=alloc<B>((size_t)rows*h); n=alloc<B>((size_t)rows*h);
        q=alloc<B>((size_t)rows*qh*dim); k=alloc<B>((size_t)rows*kh*dim); v=alloc<B>((size_t)rows*kh*dim);
        att=alloc<B>((size_t)rows*qh*dim); tmp=alloc<B>((size_t)rows*h);
        gate=alloc<B>((size_t)rows*mid); up=alloc<B>((size_t)rows*mid);
        selected=alloc<B>((size_t)sequences*h); logits=alloc<B>((size_t)sequences*vocab);
        flogits=alloc<float>((size_t)sequences*vocab);
        ids=alloc<int>(rows); pos=alloc<int>(rows); seq=alloc<int>(rows); slots=alloc<int>(rows);
        tables=alloc<int>((size_t)sequences*mb); last=alloc<int>(sequences); sync();
    }
    B* weight(const std::string& name) { return weights.at(name); }
    void linear(const B* input,B* output,const std::string& name,int r,int in,int out) {
        float a=1,b=0;
        // Row-major X W^T is column-major W X^T. W is stored [out,in].
        cb(cublasGemmEx(blas,CUBLAS_OP_T,CUBLAS_OP_N,out,r,in,&a,
                       weight(name),CUDA_R_16BF,in,input,CUDA_R_16BF,in,
                       &b,output,CUDA_R_16BF,out,CUBLAS_COMPUTE_32F,CUBLAS_GEMM_DEFAULT));
    }
    void forward(const int* hi,const int* hp,const int* hs,const int* ht,
                 const int* hl,int r,int ns,float* output) {
        check();
        if(r<1||r>rows||ns<1||ns>sequences) throw std::runtime_error("Batch capacity exceeded");
        std::vector<int> write(r), maxpos(ns,-1), reader(blocks,-1);
        std::vector<unsigned char> seen((size_t)blocks*bs,0);
        for(int i=0;i<r;i++) {
            if(hi[i]<0||hi[i]>=vocab||hp[i]<0||hp[i]>=context||hs[i]<0||hs[i]>=ns) throw std::runtime_error("Invalid row metadata");
            maxpos[hs[i]]=std::max(maxpos[hs[i]],hp[i]);
            int block=ht[hs[i]*mb+hp[i]/bs];
            if(block<0||block>=blocks) throw std::runtime_error("Invalid block mapping");
            write[i]=ht[hs[i]*mb+hp[i]/bs]*bs+hp[i]%bs;
            if(seen[write[i]]++) throw std::runtime_error("Duplicate cache write slot");
        }
        // A write to a block used by another sequence would bypass COW and race
        // with its attention reads, even if those writes have different offsets.
        for(int s=0;s<ns;s++) {
            if(maxpos[s]<0) throw std::runtime_error("Sequence has no query rows");
            std::vector<unsigned char> mapped(blocks,0);
            for(int b=0;b<=maxpos[s]/bs;b++) {
                int id=ht[s*mb+b];
                if(id<0||id>=blocks||mapped[id]++) throw std::runtime_error("Invalid/aliased block mapping");
                reader[id]=reader[id]==-1 ? s : -2;
            }
        }
        for(int i=0;i<r;i++) if(reader[write[i]/bs]!=hs[i])
            throw std::runtime_error("Shared block is writable");
        for(int s=0;s<ns;s++) if(hl[s]<0||hl[s]>=r||hs[hl[s]]!=s) throw std::runtime_error("Invalid selected row");
        ck(cudaMemcpyAsync(ids,hi,r*sizeof(int),cudaMemcpyHostToDevice,stream));
        ck(cudaMemcpyAsync(pos,hp,r*sizeof(int),cudaMemcpyHostToDevice,stream));
        ck(cudaMemcpyAsync(seq,hs,r*sizeof(int),cudaMemcpyHostToDevice,stream));
        ck(cudaMemcpyAsync(slots,write.data(),r*sizeof(int),cudaMemcpyHostToDevice,stream));
        ck(cudaMemcpyAsync(tables,ht,(size_t)ns*mb*sizeof(int),cudaMemcpyHostToDevice,stream));
        ck(cudaMemcpyAsync(last,hl,ns*sizeof(int),cudaMemcpyHostToDevice,stream));
        int hidden_tiles = tile_count(h);
        int kv_tiles = tile_count(kh * dim);
        gather_rows<<<r * hidden_tiles, 1, 0, stream>>>(
            weight("model.embed_tokens.weight"), x, ids, h, hidden_tiles);
        size_t stride=(size_t)blocks*bs*kh*dim;
        for(int l=0;l<layers;l++) {
            std::string p="model.layers."+std::to_string(l)+".";
            norm<<<r, 1, 0, stream>>>(x,weight(p+"input_layernorm.weight"),n,h,eps);
            linear(n,q,p+"self_attn.q_proj.weight",r,h,qh*dim);
            linear(n,k,p+"self_attn.k_proj.weight",r,h,kh*dim);
            linear(n,v,p+"self_attn.v_proj.weight",r,h,kh*dim);
            rope<<<r * qh, 1, 0, stream>>>(q,weight(p+"self_attn.q_norm.weight"),pos,qh,dim,eps,theta);
            rope<<<r * kh, 1, 0, stream>>>(k,weight(p+"self_attn.k_norm.weight"),pos,kh,dim,eps,theta);
            B* keys = kc + l * stride;
            B* values = vc + l * stride;
            cache_write<<<r * kv_tiles, 1, 0, stream>>>(k,v,keys,values,slots,kh*dim,kv_tiles);
            if (dim <= 128)
                attention<128><<<r * qh, 1, 0, stream>>>(q,keys,values,att,tables,seq,pos,mb,bs,qh,kh,dim);
            else
                attention<256><<<r * qh, 1, 0, stream>>>(q,keys,values,att,tables,seq,pos,mb,bs,qh,kh,dim);
            linear(att,tmp,p+"self_attn.o_proj.weight",r,qh*dim,h);
            add<<<tile_count(r * h), 1, 0, stream>>>(x,tmp,r*h);
            norm<<<r, 1, 0, stream>>>(x,weight(p+"post_attention_layernorm.weight"),n,h,eps);
            linear(n,gate,p+"mlp.gate_proj.weight",r,h,mid);
            linear(n,up,p+"mlp.up_proj.weight",r,h,mid);
            silu<<<tile_count(r * mid), 1, 0, stream>>>(gate,up,r*mid);
            linear(gate,tmp,p+"mlp.down_proj.weight",r,mid,h);
            add<<<tile_count(r * h), 1, 0, stream>>>(x,tmp,r*h);
        }
        gather_rows<<<ns * hidden_tiles, 1, 0, stream>>>(x,selected,last,h,hidden_tiles);
        norm<<<ns, 1, 0, stream>>>(selected,weight("model.norm.weight"),n,h,eps);
        linear(n,logits,"lm_head.weight",ns,h,vocab);
        to_float<<<tile_count(ns * vocab), 1, 0, stream>>>(logits,flogits,ns*vocab);
        ck(cudaMemcpyAsync(output,flogits,(size_t)ns*vocab*sizeof(float),cudaMemcpyDeviceToHost,stream));
        sync();
    }
};

#define API_BEGIN try { error.clear();
#define API_END } catch(const std::exception& e) { error=e.what(); return -1; } return 0;
extern "C" const char* q_error() { return error.c_str(); }
extern "C" void* q_create(int h,int mid,int layers,int qh,int kh,int dim,int vocab,
                          int blocks,int bs,int context,int rows,int sequences,float eps,float theta) {
    Engine* e=nullptr;
    try {
        if(h<1||mid<1||layers<1||kh<1||qh<kh||qh%kh||dim<2||dim>256||dim%2||vocab<1||blocks<1||bs<1||context<1||rows<1||sequences<1||sequences>rows||eps<=0||theta<=0)
            throw std::runtime_error("Unsupported model/runtime dimensions");
        e=new Engine; e->h=h;e->mid=mid;e->layers=layers;e->qh=qh;e->kh=kh;e->dim=dim;e->vocab=vocab;
        e->blocks=blocks;e->bs=bs;e->context=context;e->rows=rows;e->sequences=sequences;e->eps=eps;e->theta=theta;
        e->init(); return e;
    } catch(const std::exception& ex) { error=ex.what(); delete e; return nullptr; }
}
extern "C" int q_weight(void* handle,const char* name,const void* data,size_t count) {
    API_BEGIN
    auto* e=(Engine*)handle; e->check();
    if(e->weights.count(name)) throw std::runtime_error("Duplicate weight");
    B* p=e->alloc<B>(count); ck(cudaMemcpyAsync(p,data,count*sizeof(B),cudaMemcpyHostToDevice,e->stream));
    e->sync(); e->weights[name]=p;
    API_END
}
extern "C" int q_forward(void* handle,const int* ids,const int* pos,const int* seq,const int* table,const int* last,int rows,int ns,float* logits) {
    API_BEGIN ((Engine*)handle)->forward(ids,pos,seq,table,last,rows,ns,logits); API_END
}
extern "C" int q_copy(void* handle,int src,int dst) {
    API_BEGIN
    auto* e=(Engine*)handle; e->check();
    if(src<0||dst<0||src>=e->blocks||dst>=e->blocks||src==dst) throw std::runtime_error("Invalid block copy");
    size_t size=(size_t)e->bs*e->kh*e->dim, stride=(size_t)e->blocks*size;
    for(int l=0;l<e->layers;l++) for(B* cache:{e->kc,e->vc})
        ck(cudaMemcpyAsync(cache+l*stride+dst*size,cache+l*stride+src*size,size*sizeof(B),cudaMemcpyDeviceToDevice,e->stream));
    e->sync(); API_END
}
extern "C" int q_destroy(void* handle) {
    API_BEGIN auto* e=(Engine*)handle; e->check(); delete e; API_END
}
