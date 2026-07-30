// .nnb container loader: mmap the file, expose tensors by name. Q1 tensors
// are consumed IN PLACE (18-byte units, see model/Q1_0_LAYOUT.md) — the only
// float weight data in the process are the f32 norm vectors. No copies.
#pragma once
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

#include <cstdint>
#include <cstring>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

#include "json_mini.h"

struct Tensor {
    enum Kind { Q1, F32 } kind;
    std::vector<int64_t> dims;   // [ne0(=input, contiguous), ne1(=output)]
    const uint8_t* data = nullptr;
    size_t nbytes = 0;
    int64_t ne(int i) const { return i < (int)dims.size() ? dims[i] : 1; }
};

struct ModelMeta {
    std::string arch;            // "qwen3" (dense) | "qwen35" (hybrid GDN)
    int hidden, layers, heads, kv_heads, head_dim, ffn, vocab;
    float rms_eps, rope_theta, yarn_factor;
    int yarn_orig_ctx, eos, pad;
    std::string chat_template;
    // ---- qwen35 (Qwen3.5/3.6 hybrid) only; zero/empty on qwen3 ----
    // layer_types is one char per block: 'L' = Gated-DeltaNet linear
    // attention, 'F' = gated full attention. Empty means "uniform dense".
    std::string layer_types;
    int full_attn_interval = 0;
    int rope_dim_count = 0;      // partial RoPE: rotate only the first N dims
    int ssm_conv_kernel = 0, ssm_state_size = 0, ssm_group_count = 0;
    int ssm_time_step_rank = 0, ssm_inner_size = 0;
};

class Nnb {
  public:
    explicit Nnb(const std::string& path) {
        fd_ = open(path.c_str(), O_RDONLY);
        if (fd_ < 0) throw std::runtime_error("nnb: cannot open " + path);
        struct stat st{};
        fstat(fd_, &st);
        size_ = (size_t)st.st_size;
        base_ = (const uint8_t*)mmap(nullptr, size_, PROT_READ, MAP_PRIVATE, fd_, 0);
        if (base_ == MAP_FAILED) throw std::runtime_error("nnb: mmap failed");
        if (memcmp(base_, "NNB1BIT\x00", 8) != 0) throw std::runtime_error("nnb: bad magic");
        uint64_t hlen;
        memcpy(&hlen, base_ + 8, 8);
        std::string hdr((const char*)base_ + 16, hlen);
        auto root = jmini::parse(hdr);
        const uint8_t* blob = base_ + 16 + hlen;

        const auto& m = root->at("meta");
        meta.arch = m.has("arch") ? m.at("arch").s() : "qwen3";
        meta.hidden = (int)m.at("hidden").i();
        meta.layers = (int)m.at("layers").i();
        meta.heads = (int)m.at("heads").i();
        meta.kv_heads = (int)m.at("kv_heads").i();
        meta.head_dim = (int)m.at("head_dim").i();
        meta.ffn = (int)m.at("ffn").i();
        meta.vocab = (int)m.at("vocab").i();
        meta.rms_eps = (float)m.at("rms_eps").d();
        meta.rope_theta = (float)m.at("rope_theta").d();
        meta.yarn_factor = (float)m.at("yarn_factor").d();
        meta.yarn_orig_ctx = (int)m.at("yarn_orig_ctx").i();
        meta.eos = (int)m.at("eos").i();
        meta.pad = (int)m.at("pad").i();
        meta.chat_template = m.at("chat_template").s();
        if (m.has("layer_types")) meta.layer_types = m.at("layer_types").s();
        auto opt = [&](const char* k, int& dst) {
            if (m.has(k)) dst = (int)m.at(k).i();
        };
        opt("full_attn_interval", meta.full_attn_interval);
        opt("rope_dim_count", meta.rope_dim_count);
        opt("ssm_conv_kernel", meta.ssm_conv_kernel);
        opt("ssm_state_size", meta.ssm_state_size);
        opt("ssm_group_count", meta.ssm_group_count);
        opt("ssm_time_step_rank", meta.ssm_time_step_rank);
        opt("ssm_inner_size", meta.ssm_inner_size);

        for (const auto& [name, tv] : root->at("tensors").obj) {
            Tensor t;
            t.kind = tv->at("kind").s() == "q1" ? Tensor::Q1 : Tensor::F32;
            for (const auto& d : tv->at("dims").arr) t.dims.push_back(d->i());
            t.data = blob + (size_t)tv->at("offset").i();
            t.nbytes = (size_t)tv->at("nbytes").i();
            tensors_[name] = std::move(t);
        }
    }
    ~Nnb() {
        if (base_ && base_ != MAP_FAILED) munmap((void*)base_, size_);
        if (fd_ >= 0) close(fd_);
    }
    const Tensor& get(const std::string& name) const {
        auto it = tensors_.find(name);
        if (it == tensors_.end())
            throw std::runtime_error("nnb: missing tensor " + name);
        return it->second;
    }
    bool has(const std::string& name) const { return tensors_.count(name) != 0; }

    ModelMeta meta;

  private:
    int fd_ = -1;
    size_t size_ = 0;
    const uint8_t* base_ = nullptr;
    std::unordered_map<std::string, Tensor> tensors_;
};
