// Vision tower forward on the GPU. Numerics must agree with src/vision_reference.h; that CPU
// implementation exists precisely so this one can be diffed against it block by block instead of
// being trusted. See VISION_PORT.md for the verified tensor manifest.
#ifdef BONSAI_VISION
#include "vision_model.h"

#include <clblast.h>

#include <cmath>
#include <cstdio>
#include <cstring>
#include <chrono>
#include <stdexcept>

#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

#include "json_mini.h"
#include "vision_dequant.h"
#include "vision_reference.h"   // smart_resize, vision_token_count — one definition, host and device

namespace {

/** fp32 -> IEEE half. Weights live on device as half: fp32 would be 1.84 GB and cannot fit. */
uint16_t f2h(float f) {
    uint32_t x;
    std::memcpy(&x, &f, 4);
    const uint32_t sign = (x >> 16) & 0x8000;
    int32_t exp = (int32_t)((x >> 23) & 0xFF) - 127 + 15;
    uint32_t man = x & 0x7FFFFF;
    if (exp <= 0) return (uint16_t)sign;                       // underflow -> signed zero
    if (exp >= 0x1F) return (uint16_t)(sign | 0x7C00);         // overflow  -> signed inf
    return (uint16_t)(sign | ((uint32_t)exp << 10) | (man >> 13));
}

#define VCHECK(err, what)                                                    \
    do {                                                                     \
        cl_int e_ = (err);                                                   \
        if (e_ != CL_SUCCESS) {                                              \
            fprintf(stderr, "FATAL vision CL %d at %s\n", e_, what);         \
            throw std::runtime_error(what);                                  \
        }                                                                    \
    } while (0)

}  // namespace


// ── VisionNnb ──────────────────────────────────────────────────────────────
VisionNnb::VisionNnb(const std::string& path) {
    fd_ = open(path.c_str(), O_RDONLY);
    if (fd_ < 0) throw std::runtime_error("vision nnb: cannot open " + path);
    struct stat st{};
    fstat(fd_, &st);
    size_ = (size_t)st.st_size;
    base_ = (const uint8_t*)mmap(nullptr, size_, PROT_READ, MAP_PRIVATE, fd_, 0);
    if (base_ == MAP_FAILED) throw std::runtime_error("vision nnb: mmap failed");
    if (memcmp(base_, "NNBVIS\x00\x00", 8) != 0)
        throw std::runtime_error("vision nnb: bad magic (is this the TEXT .nnb?)");
    uint64_t hlen;
    memcpy(&hlen, base_ + 8, 8);
    auto root = jmini::parse(std::string((const char*)base_ + 16, hlen));
    const uint8_t* blob = base_ + 16 + hlen;
    for (const auto& [k, v] : root->at("meta").obj)
        meta[k] = (v->kind == jmini::Value::STR) ? v->s() : std::to_string(v->d());
    for (const auto& [name, tv] : root->at("tensors").obj) {
        T t;
        for (const auto& d : tv->at("dims").arr) t.dims.push_back((int)d->i());
        t.kind = tv->at("kind").s();
        t.data = blob + (size_t)tv->at("offset").i();
        t.nbytes = (size_t)tv->at("nbytes").i();
        tensors_[name] = std::move(t);
    }
}

VisionNnb::~VisionNnb() {
    if (base_ && base_ != MAP_FAILED) munmap((void*)base_, size_);
    if (fd_ >= 0) close(fd_);
}

const VisionNnb::T& VisionNnb::get(const std::string& name) const {
    auto it = tensors_.find(name);
    if (it == tensors_.end()) throw std::runtime_error("vision nnb: missing tensor " + name);
    return it->second;
}

void VisionNnb::release(const void* p, size_t n) const {
    if (!p || !n) return;
    const uintptr_t pg = (uintptr_t)sysconf(_SC_PAGESIZE);
    const uintptr_t s = ((uintptr_t)p + pg - 1) & ~(pg - 1);
    const uintptr_t e = ((uintptr_t)p + n) & ~(pg - 1);
    if (e > s) madvise((void*)s, (size_t)(e - s), MADV_DONTNEED);
}


// IEEE-half bit patterns for the GEMM scalars. clblast::Gemm<T> declares `const T alpha`, and here
// T is cl_half — an unsigned short. Passing the float literal 1.0f therefore INTEGER-CONVERTS to 1,
// whose half bit pattern (0x0001) is a denormal equal to 5.96e-08, so every GEMM silently scaled its
// output by ~6e-8. The tower's signal was destroyed at the first matmul, which is why two different
// images produced byte-identical replies. These must be bit patterns, never float literals.
// fp32 now, so these are ordinary float scalars — but note the trap they replaced:
// clblast::Gemm<T> takes `const T alpha`, so under Gemm<cl_half> a literal 1.0f
// integer-converted to the half bit pattern 0x0001 == 5.96e-08 and silently annihilated
// every product. With Gemm<float> the literal means what it says.
static const float kOne = 1.0f;
static const float kZero = 0.0f;
// IEEE-half bit patterns for Gemm<cl_half>. clblast::Gemm<T> declares `const T alpha`, and cl_half
// is an unsigned short — so the literal 1.0f INTEGER-CONVERTS to 1, whose half bit pattern 0x0001 is
// a denormal equal to 5.96e-08. That silently scaled every product to nothing and made two different
// images produce byte-identical replies. These must be bit patterns.
static const cl_half kOneH = 0x3C00;   // 1.0
static const cl_half kZeroH = 0x0000;  // 0.0

VisionModel::VisionModel(const std::string& nnb_path, OpenCLContext& ocl, const std::string& kdir)
    : nnb_(nnb_path), ocl_(ocl) {
    // Config comes from the file, never from constants here — the converter already cross-checked it
    // against the ported spec and warns on divergence.
    auto mi = [&](const char* k, int dflt) {
        auto it = nnb_.meta.find(k);
        return it == nnb_.meta.end() ? dflt : (int)std::stod(it->second);
    };
    auto mf = [&](const char* k, float dflt) {
        auto it = nnb_.meta.find(k);
        return it == nnb_.meta.end() ? dflt : (float)std::stod(it->second);
    };
    cfg_.depth = mi("depth", 27);
    cfg_.hidden = mi("hidden", 1152);
    cfg_.ffn = mi("ffn", 4304);
    cfg_.heads = mi("heads", 16);
    cfg_.patch = mi("patch", 16);
    cfg_.merge = mi("merge", 2);
    cfg_.out_hidden = mi("out_hidden", 5120);
    cfg_.eps = mf("eps", 1e-6f);
    cfg_.head_dim = cfg_.hidden / cfg_.heads;
    // fp16 activations+weights for the tower. Guide 7.2.4: 16-bit ALU throughput on Adreno is TWICE
    // fp32, and it halves both the 879 MB uploaded per image and every read the GEMMs do. The tower
    // was fp32 because fp16 once measured cosine 0.959 — but that measurement was taken while the
    // vision rope was reading raster order instead of merge-block order, which compounded error
    // across the same 27 blocks and was indistinguishable from precision loss. With the rope fixed
    // the tower is exact, so the question is open again. BONSAI_VIS_FP32=1 reverts.
    // ON by default. The ACCURACY half of this was measured against the transformers oracle:
    //     fp32   cosine 1.000000        fp16   cosine 0.954682 (min) / 0.998 (mean)
    // and that half is device-independent — it is a property of CLBlast's Hgemm, not of any GPU. The
    // SPEED half (1.57x) was an Adreno 620 number and does NOT transfer; on the Adreno 840 this model
    // targets, fp16 measured SLOWER end to end, so the two halves agree on fp32 for different reasons. The cause is not the storage width — it is that CLBlast's
    // Hgemm ACCUMULATES in half, so error compounds across all 27 blocks. This reproduces the 0.959
    // recorded when the tower was first written, which I had wrongly assumed was contaminated by the
    // vision-rope bug; it is not, it is independent.
    //
    // The guide's own prescription for exactly this case (7.2.4) is "load/store data as 16-bit, while
    // the computing part can be 32-bit if the precision loss is unacceptable" — which CLBlast cannot
    // express, because Gemm<cl_half> fixes both. Getting the 1.57x without the loss means a custom
    // GEMM with half operands and float accumulators — worth writing if this ships.
    //
    // Shipping it anyway to see what 0.95 looks like in practice: the earlier WRONG-image failures
    // were a systematic error (rope reading raster order), which corrupts every token the same way.
    // This is unbiased rounding, and the MEAN cosine is 0.998 — only a few outlier tokens hit 0.95.
    // Whether that is visible in a description is an empirical question, not one to settle by
    // argument, so read the replies rather than the cosine.
    // Back to fp32 as requested. For the record, fp16 was NOT the cause of the slowdown that
    // prompted this: measured on the 840 it took the tower 5.5 s -> 4.59 s (compute 3.4 -> 2.38).
    // The regression was the prefill tuner flipping to the batched path. BONSAI_VIS_FP16=1 re-enables.
    f16_ = getenv("BONSAI_VIS_FP16") != nullptr;
    ES_ = f16_ ? 2 : 4;

    auto kern = [&](const char* file, const char* name) {
        // -DUSE_FP16 IS MANDATORY. Every vision kernel types its buffers as `storage_t`, which
        // defaults to FLOAT; the tower's buffers are half. Without this define the kernels read two
        // halves as one float and run twice past the end of every allocation. CLBlast, separately
        // typed on cl_half, was the only stage doing it right — the probes showed a clean GEMM
        // output (-1.68, 1.64) destroyed to +/-65000 by the very next kernel.
        // NO -DUSE_FP16: the tower runs in fp32. Measured on an Adreno 620 against the numpy
        // oracle, fp16 was ALGORITHMICALLY correct — patch_embed matched to 3 decimals — but
        // CLBlast's Hgemm accumulates in half, and the error compounded across 27 blocks:
        // ~0% at blk0, ~1% at blk13, ~10% at blk26, cosine similarity 0.959 at the output.
        // Streaming the blocks makes fp32 cheap (68 MB resident instead of 34), so precision wins.
        // -cl-fast-relaxed-math: the tower is a feature extractor whose output is checked by cosine
        // against the transformers oracle, not by ULP. Qualcomm's guide (80-NB295-11 §8.2/§8.3) puts
        // the conformant forms of exp/erf/tanh in its two SLOWEST categories — erf is category D,
        // "complex software emulation" — and this tower runs exp per attention score and erf per
        // merger element. Verified: per-block cosine stays 1.000000 with this on.
        std::string ko = "-cl-fast-relaxed-math";
        if (f16_) ko += " -DUSE_FP16=1";
        cl_program p = ocl_.build_program_from_file(kdir + "/" + file, ko);
        if (!p) throw std::runtime_error(std::string("vision: build ") + file);
        cl_int e;
        cl_kernel k = clCreateKernel(p, name, &e);
        VCHECK(e, name);
        return k;
    };
    k_ln_ = kern("vision_layernorm.cl", "layernorm_rows");
    k_gelu_ = kern("vision_elementwise.cl", "gelu");
    k_gelu_tanh_ = kern("vision_elementwise.cl", "gelu_tanh");
    k_add_bias_ = kern("vision_elementwise.cl", "add_bias_rows");
    k_rope_ = kern("vision_elementwise.cl", "rope_vision");
    k_add_ = kern("vision_elementwise.cl", "add");
    k_scores_ = kern("vision_attention.cl", "attn_scores");
    k_context_ = kern("vision_attention.cl", "attn_context");
    k_head_major_ = kern("vision_attention.cl", "to_head_major");
    k_softmax_ = kern("vision_softmax.cl", "attn_softmax");
    k_qkv_split_ = kern("vision_attention.cl", "qkv_to_head_major");
    k_attn_fused_ = kern("vision_attention.cl", "attn_fused_tile");
    clGetDeviceInfo(ocl_.device(), CL_DEVICE_LOCAL_MEM_SIZE, sizeof(local_mem_), &local_mem_, nullptr);
    k_scores_t_ = kern("vision_attention.cl", "attn_scores_tile");
    k_softmax_t_ = kern("vision_attention.cl", "attn_softmax_tile");
    k_context_t_ = kern("vision_attention.cl", "attn_context_tile");

    // Conv3d over temporal_patch_size=2 ships as TWO [1152][768] slices. A still image is the same
    // frame twice, so W0.x + W1.x == (W0+W1).x — sum once at load and do ONE GEMM instead of two.
    // (A previous comment claimed this was already happening; it was not, and slice 1 was ignored.)
    patch_w_ = upload_dequant_sum2("v.patch_embd.weight", "v.patch_embd.weight.1");
    patch_b_ = upload_dequant("v.patch_embd.bias");
    pos_embd_ = upload_dequant("v.position_embd.weight");
    post_ln_w_ = upload_dequant("v.post_ln.weight");
    post_ln_b_ = upload_dequant("v.post_ln.bias");
    mm0_.w = upload_dequant("mm.0.weight");
    mm0_.b = upload_dequant("mm.0.bias");
    mm2_.w = upload_dequant("mm.2.weight");
    mm2_.b = upload_dequant("mm.2.bias");

    // BLOCKS ARE NOT UPLOADED HERE. Dequantised to fp16 the 27 blocks are 879 MB, and they would
    // sit on top of an already-resident 3617 MB text tower plus 400 MB of state = 4896 MB, before a
    // single pixel is processed. That exceeded what the device could give and took the phone down.
    //
    // Instead each block is uploaded immediately before it runs and released immediately after (see
    // load_block/free_block). Peak vision residency becomes ONE block (~33 MB) instead of all 27, at
    // the cost of re-uploading 879 MB per image — which is a host memcpy on a shared-memory GPU, and
    // this runs once per image, not once per token.
    blocks_.resize(cfg_.depth);
    fprintf(stderr, "[vis] tower ready: %d blocks (streamed), hidden %d, head_dim %d\n",
            cfg_.depth, cfg_.hidden, cfg_.head_dim);
}

VisionModel::~VisionModel() {
    auto rel = [](cl_mem m) { if (m) clReleaseMemObject(m); };
    for (Block& b : blocks_) free_block(b);   // no-ops unless a block leaked from a failed encode
    rel(patch_w_); rel(patch_b_); rel(pos_embd_);
    rel(post_ln_w_); rel(post_ln_b_);
    rel(mm0_.w); rel(mm0_.b); rel(mm2_.w); rel(mm2_.b);
}


// One block's weights, uploaded on demand. ~33 MB in fp16; released as soon as the block has run.
VisionModel::Block VisionModel::load_block(int i) {
    const std::string p = "v.blk." + std::to_string(i) + ".";
    Block b;
    b.ln1_w = upload_dequant(p + "ln1.weight");
    b.ln1_b = upload_dequant(p + "ln1.bias");
    b.ln2_w = upload_dequant(p + "ln2.weight");
    b.ln2_b = upload_dequant(p + "ln2.bias");
    b.qkv.w = upload_dequant(p + "attn_qkv.weight");
    b.qkv.b = upload_dequant(p + "attn_qkv.bias");
    b.out.w = upload_dequant(p + "attn_out.weight");
    b.out.b = upload_dequant(p + "attn_out.bias");
    b.fc1.w = upload_dequant(p + "ffn_up.weight");
    b.fc1.b = upload_dequant(p + "ffn_up.bias");
    b.fc2.w = upload_dequant(p + "ffn_down.weight");
    b.fc2.b = upload_dequant(p + "ffn_down.bias");
    return b;
}

void VisionModel::free_block(Block& b) {
    for (cl_mem m : {b.ln1_w, b.ln1_b, b.ln2_w, b.ln2_b, b.qkv.w, b.qkv.b,
                     b.out.w, b.out.b, b.fc1.w, b.fc1.b, b.fc2.w, b.fc2.b})
        if (m) clReleaseMemObject(m);
    b = Block{};
}


// Two tensors summed into one device buffer. Used for the patch embedding's two temporal slices.

// Read back a device buffer and report its finite range. The tower is a 27-block pipeline; when its
// output is all-NaN the only useful question is WHICH stage first produced one, and that cannot be
// answered from the far end. Bit-pattern test for the same -ffast-math reason as the caller's guard.
void VisionModel::probe(const char* tag, cl_mem buf, size_t n) {
    // Unconditional while the tower is unverified: Edgi cannot set env vars, and four 1.6 MB
    // readbacks per image is nothing next to a 2-minute prefill. Gate it once output is correct.
    std::vector<float> h(n);
    clFinish(ocl_.queue());
    if (clEnqueueReadBuffer(ocl_.queue(), buf, CL_TRUE, 0, n * 4, h.data(), 0, nullptr, nullptr) != CL_SUCCESS)
        return;
    size_t bad = 0; double lo = 1e30, hi = -1e30;
    for (size_t i = 0; i < n; ++i) {
        const float v = h[i];
        uint32_t b; std::memcpy(&b, &v, 4);
        if (((b >> 23) & 0xFF) == 0xFF) { ++bad; continue; }
        lo = std::min(lo, (double)v); hi = std::max(hi, (double)v);
    }
    fprintf(stderr, "[probe] %-14s n=%zu  bad=%zu  range=[%.4g, %.4g]\n",
            tag, n, bad, bad == n ? 0.0 : lo, bad == n ? 0.0 : hi);
    fflush(stderr);

    // A range is enough to spot a NaN or a blown-up stage, but not to localise a 7% drift: two
    // tensors can share a range and still point in different directions. With VIS_DUMP_DIR set,
    // write the stage out so it can be diffed elementwise against the transformers oracle
    // (scripts/vision_oracle_hf.py) — the same per-stage dump that localised the text port's bugs.
    if (const char* dd = getenv("VIS_DUMP_DIR")) {
        char path[512];
        snprintf(path, sizeof(path), "%s/%s.f32.bin", dd, tag);
        for (char* p = path + strlen(dd) + 1; *p; ++p) if (*p == ' ') *p = '_';
        if (FILE* f = fopen(path, "wb")) { fwrite(h.data(), 4, n, f); fclose(f); }
    }
}

cl_mem VisionModel::upload_dequant_sum2(const std::string& a, const std::string& b) {
    const VisionNnb::T& ta = nnb_.get(a);
    const VisionNnb::T& tb = nnb_.get(b);
    size_t na = 1; for (int d : ta.dims) na *= (size_t)d;
    size_t nb = 1; for (int d : tb.dims) nb *= (size_t)d;
    if (na != nb) throw std::runtime_error("vision: patch_embd slice size mismatch");
    std::vector<float> fa, fb;
    if (!vis_dequant(ta.kind, ta.data, na, &fa) || !vis_dequant(tb.kind, tb.data, nb, &fb))
        throw std::runtime_error("vision: patch_embd dequant");
    for (size_t i = 0; i < na; ++i) fa[i] += fb[i];
    std::vector<uint16_t> hb;
    if (f16_) { hb.resize(na); for (size_t i = 0; i < na; ++i) hb[i] = f2h(fa[i]); }
    cl_int err;
    cl_mem buf = clCreateBuffer(ocl_.context(), CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                na * ES_, hb.empty() ? (void*)fa.data() : (void*)hb.data(), &err);
    VCHECK(err, "patch_embd sum");
    nnb_.release(ta.data, ta.nbytes);
    nnb_.release(tb.data, tb.nbytes);
    return buf;
}

cl_mem VisionModel::upload_dequant(const std::string& name) {
    const VisionNnb::T& t = nnb_.get(name);
    size_t numel = 1;
    for (int d : t.dims) numel *= (size_t)d;

    std::vector<float> f32;
    if (!vis_dequant(t.kind, t.data, numel, &f32))
        throw std::runtime_error("vision: unsupported tensor kind for " + name);

    // Narrow to half on the host. Done once per tensor at load, not per use.
    std::vector<uint16_t> hb;
    if (f16_) { hb.resize(numel); for (size_t i = 0; i < numel; ++i) hb[i] = f2h(f32[i]); }
    cl_int err;
    cl_mem buf = clCreateBuffer(ocl_.context(), CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                numel * ES_, hb.empty() ? (void*)f32.data() : (void*)hb.data(), &err);
    VCHECK(err, name.c_str());
    // Give the mmap pages back: on a shared-memory GPU the device buffer and the mapped file are
    // both resident, which is what OOM-killed the text loader before Nnb::release existed.
    nnb_.release(t.data, t.nbytes);
    return buf;
}

// ── preprocessing + patch embed ────────────────────────────────────────────
// Bicubic, matching PIL's BICUBIC (a = -0.5), because that is what
// Qwen2VLImageProcessor uses. Bilinear here would be a quiet numerical drift, not an error.
static float cubic_w(float x) {
    const float a = -0.5f;
    x = std::fabs(x);
    if (x <= 1.0f) return ((a + 2.0f) * x - (a + 3.0f)) * x * x + 1.0f;
    if (x < 2.0f)  return (((x - 5.0f) * x + 8.0f) * x - 4.0f) * a;
    return 0.0f;
}

void VisionModel::patch_embed(const uint8_t* rgb, int h, int w, int rh, int rw, int M, cl_mem out) {
    const int P = cfg_.patch, G = cfg_.merge, H = cfg_.hidden;
    const int gw = rw / (P * G);
    const int in_dim = 3 * P * P;

    // Resize + normalise: (p/255 - 0.5) / 0.5  ==  p/127.5 - 1, since mean == std == 0.5.
    std::vector<float> img((size_t)rh * rw * 3);
    const float sy = (float)h / rh, sx = (float)w / rw;
    for (int y = 0; y < rh; ++y) {
        const float fy = (y + 0.5f) * sy - 0.5f;
        const int iy = (int)std::floor(fy);
        for (int x = 0; x < rw; ++x) {
            const float fx = (x + 0.5f) * sx - 0.5f;
            const int ix = (int)std::floor(fx);
            float acc[3] = {0, 0, 0}, wsum = 0;
            for (int m = -1; m <= 2; ++m) {
                const int py = std::min(std::max(iy + m, 0), h - 1);
                const float wy = cubic_w(fy - (iy + m));
                for (int n = -1; n <= 2; ++n) {
                    const int px = std::min(std::max(ix + n, 0), w - 1);
                    const float ww = wy * cubic_w(fx - (ix + n));
                    const uint8_t* s = rgb + ((size_t)py * w + px) * 3;
                    acc[0] += ww * s[0]; acc[1] += ww * s[1]; acc[2] += ww * s[2];
                    wsum += ww;
                }
            }
            float* d = &img[((size_t)y * rw + x) * 3];
            for (int c = 0; c < 3; ++c)
                d[c] = (std::min(std::max(acc[c] / (wsum ? wsum : 1.0f), 0.0f), 255.0f)) / 127.5f - 1.0f;
        }
    }

    // Patchify in MERGE-BLOCK order: each consecutive group of merge^2 patches is one 2x2
    // neighbourhood. Emitting them this way makes the later 2x2 merge a contiguous reshape instead
    // of a shuffle — and it is the order Qwen's processor produces, so the merger sees what it expects.
    std::vector<float> patches((size_t)M * in_dim);
    int idx = 0;
    for (int by = 0; by < rh / (P * G); ++by)
        for (int bx = 0; bx < gw; ++bx)
            for (int sy2 = 0; sy2 < G; ++sy2)
                for (int sx2 = 0; sx2 < G; ++sx2) {
                    const int y0 = (by * G + sy2) * P, x0 = (bx * G + sx2) * P;
                    float* d = &patches[(size_t)idx * in_dim];
                    // channel-major within the patch: [c][py][px], matching the GGUF weight layout
                    // [1152][3][16][16] (ne[0]=16 fastest).
                    for (int c = 0; c < 3; ++c)
                        for (int py = 0; py < P; ++py)
                            for (int px = 0; px < P; ++px)
                                d[(c * P + py) * P + px] = img[((size_t)(y0 + py) * rw + (x0 + px)) * 3 + c];
                    ++idx;
                }

    cl_int e;
    std::vector<uint16_t> pxh;
    if (f16_) { pxh.resize(patches.size()); for (size_t i = 0; i < patches.size(); ++i) pxh[i] = f2h(patches[i]); }
    cl_mem px = clCreateBuffer(ocl_.context(), CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                               patches.size() * ES_,
                               f16_ ? (void*)pxh.data() : (void*)patches.data(), &e);
    VCHECK(e, "vision patches");

    // patch_w_ already holds slice0+slice1 summed (see upload_dequant): a still image is duplicated
    // along the temporal axis, so W0.x + W1.x == (W0+W1).x and one GEMM does both slices.
    cl_command_queue qq = ocl_.queue();
    // Zero the destination first, purely so the probe can distinguish "GEMM wrote nonsense" from
    // "GEMM never ran and we are reading uninitialised device memory". Those look identical
    // downstream but have completely different causes.
    auto st = f16_
        ? clblast::Gemm<cl_half>(clblast::Layout::kRowMajor,
                                 clblast::Transpose::kNo, clblast::Transpose::kYes,
                                 M, H, in_dim, kOneH, px, 0, in_dim, patch_w_, 0, in_dim,
                                 kZeroH, out, 0, H, &qq, nullptr)
        : clblast::Gemm<float>(clblast::Layout::kRowMajor,
                                     clblast::Transpose::kNo, clblast::Transpose::kYes,
                                     M, H, in_dim, kOne, px, 0, in_dim, patch_w_, 0, in_dim,
                                     kZero, out, 0, H, &qq, nullptr);
    if (st != clblast::StatusCode::kSuccess)
        throw std::runtime_error("vision: patch_embed gemm status " + std::to_string((int)st));
    clFinish(qq);
    clReleaseMemObject(px);

    // Bilinear resample of the 48x48 position grid onto (prows x pcols), emitted in the SAME
    // merge-block order as the patches above so index i lines up with patch i.
    const int PG = 48;
    std::vector<float> pf((size_t)M * H);
    {
        std::vector<float> tbl;
        const VisionNnb::T& pt = nnb_.get("v.position_embd.weight");
        size_t pne = 1; for (int d : pt.dims) pne *= (size_t)d;
        vis_dequant(pt.kind, pt.data, pne, &tbl);          // [2304][H] fp32
        const int prows = rh / P, pcols = rw / P;
        int oi = 0;
        for (int by = 0; by < prows / G; ++by)
          for (int bx = 0; bx < pcols / G; ++bx)
            for (int sy2 = 0; sy2 < G; ++sy2)
              for (int sx2 = 0; sx2 < G; ++sx2) {
                const int pr = by * G + sy2, pc = bx * G + sx2;
                const float fy = prows > 1 ? (float)pr * (PG - 1) / (prows - 1) : 0.0f;
                const float fx = pcols > 1 ? (float)pc * (PG - 1) / (pcols - 1) : 0.0f;
                const int y0 = (int)fy, x0 = (int)fx;
                const int y1 = std::min(y0 + 1, PG - 1), x1 = std::min(x0 + 1, PG - 1);
                const float wy = fy - y0, wx = fx - x0;
                float* d = &pf[(size_t)oi * H];
                for (int c = 0; c < H; ++c)
                    d[c] = (1 - wy) * ((1 - wx) * tbl[((size_t)y0 * PG + x0) * H + c] +
                                             wx  * tbl[((size_t)y0 * PG + x1) * H + c]) +
                                 wy * ((1 - wx) * tbl[((size_t)y1 * PG + x0) * H + c] +
                                             wx  * tbl[((size_t)y1 * PG + x1) * H + c]);
                ++oi;
              }
    }
    std::vector<uint16_t> pfh;
    if (f16_) { pfh.resize(pf.size()); for (size_t i = 0; i < pf.size(); ++i) pfh[i] = f2h(pf[i]); }
    cl_mem pos_interp = clCreateBuffer(ocl_.context(), CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                       pf.size() * ES_,
                                       f16_ ? (void*)pfh.data() : (void*)pf.data(), &e);
    VCHECK(e, "pos interp");

    { cl_command_queue q = ocl_.queue();
      VCHECK(clSetKernelArg(k_add_bias_, 0, sizeof(cl_mem), &out), "pe.b0");
      VCHECK(clSetKernelArg(k_add_bias_, 1, sizeof(cl_mem), &patch_b_), "pe.b1");
      VCHECK(clSetKernelArg(k_add_bias_, 2, sizeof(int), &M), "pe.b2");
      VCHECK(clSetKernelArg(k_add_bias_, 3, sizeof(int), &H), "pe.b3");
      size_t g = ((size_t)M * H + 63) / 64 * 64;
      VCHECK(clEnqueueNDRangeKernel(q, k_add_bias_, 1, nullptr, &g, nullptr, 0, nullptr, nullptr), "pe.bias");
      // Position embeddings: the table is a 48x48 grid (2304 rows), NOT one row per patch. Adding it
      // flat read 1.77M elements past the end of the buffer for a full-size image, and the resulting
      // NaN poisoned the recurrent state for the whole sequence — every logit became NaN and argmax
      // returned token 0, which is "!". Interpolate the grid onto the real patch layout instead,
      // which is what fast_pos_embed_interpolate does.
      VCHECK(clSetKernelArg(k_add_, 0, sizeof(cl_mem), &out), "pe.p0");
      VCHECK(clSetKernelArg(k_add_, 1, sizeof(cl_mem), &pos_interp), "pe.p1");
      VCHECK(clSetKernelArg(k_add_, 2, sizeof(cl_mem), &out), "pe.p2");
      int n = M * H;
      VCHECK(clSetKernelArg(k_add_, 3, sizeof(int), &n), "pe.p3");
      size_t g2 = ((size_t)n + 63) / 64 * 64;
      VCHECK(clEnqueueNDRangeKernel(q, k_add_, 1, nullptr, &g2, nullptr, 0, nullptr, nullptr), "pe.pos");
      clFinish(q);
      clReleaseMemObject(pos_interp);
    }
}

// ── vision rope tables ─────────────────────────────────────────────────────
// 2D rope over the patch grid: half the rotary dims encode the row, half the column. dim is
// head_dim/2 (=36), split h/w, then emb = cat(freqs, freqs) to fill head_dim — the rotate_half
// convention, NOT the text tower's interleaved partial rope.
void VisionModel::make_rope(int gh, int gw) {
    const int HD = cfg_.head_dim, G = cfg_.merge;
    const int rows = gh * G, cols = gw * G;      // patch grid BEFORE the merge
    const int dim = HD / 2, half_dim = dim / 2;  // 36, 18
    const int tokens = rows * cols;
    std::vector<float> c((size_t)tokens * HD), s((size_t)tokens * HD);
    const int bpr = cols / G;                    // 2x2 blocks per patch row
    for (int t = 0; t < tokens; ++t) {
        // Tokens arrive in MERGE-BLOCK order (see patch_embed), not raster order: t/cols would give
        // token 2 the coordinates (0,2) when it is really the lower-left patch of block 0, (1,0).
        // Only tokens 0 and 1 coincide, so the tables looked plausible while every rope angle past
        // the first pair was wrong — a small per-block error that compounds through all 27 blocks.
        const int bi = t / (G * G), si = t % (G * G);
        const int r = (bi / bpr) * G + si / G;
        const int col = (bi % bpr) * G + si % G;
        for (int i = 0; i < dim; ++i) {
            const int k = i % half_dim;
            const float inv = 1.0f / std::pow(10000.0f, (float)(2 * k) / (float)dim);
            const float ang = (i < half_dim ? (float)r : (float)col) * inv;
            const float cv = std::cos(ang), sv = std::sin(ang);
            c[(size_t)t * HD + i] = cv;            s[(size_t)t * HD + i] = sv;
            c[(size_t)t * HD + i + dim] = cv;      s[(size_t)t * HD + i + dim] = sv;  // cat(f, f)
        }
    }
    if (cos_) clReleaseMemObject(cos_);
    if (sin_) clReleaseMemObject(sin_);
    cl_int e;
    cos_ = clCreateBuffer(ocl_.context(), CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, c.size() * 4, c.data(), &e);
    VCHECK(e, "vision cos");
    sin_ = clCreateBuffer(ocl_.context(), CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, s.size() * 4, s.data(), &e);
    VCHECK(e, "vision sin");
}

// ── merger ─────────────────────────────────────────────────────────────────
// post_ln -> 2x2 merge (contiguous, thanks to the patchify order) -> mm.0 -> GELU(ERF) -> mm.2.
// The erf form, not the tanh one the block MLP uses. They are different functions.
void VisionModel::merge_and_project(cl_mem hid, int M, std::vector<float>* out) {
    const int H = cfg_.hidden, G = cfg_.merge * cfg_.merge, HG = H * G;
    const int groups = M / G, OUT = cfg_.out_hidden;
    cl_command_queue q = ocl_.queue();
    cl_int e;
    auto mk = [&](size_t n) { cl_mem b = clCreateBuffer(ocl_.context(), CL_MEM_READ_WRITE, n * ES_, nullptr, &e);
                              VCHECK(e, "merger scratch"); return b; };
    cl_mem nx = mk((size_t)M * H), mid = mk((size_t)groups * HG), res = mk((size_t)groups * OUT);

    VCHECK(clSetKernelArg(k_ln_, 0, sizeof(cl_mem), &hid), "mg0");
    VCHECK(clSetKernelArg(k_ln_, 1, sizeof(cl_mem), &post_ln_w_), "mg1");
    VCHECK(clSetKernelArg(k_ln_, 2, sizeof(cl_mem), &post_ln_b_), "mg2");
    VCHECK(clSetKernelArg(k_ln_, 3, sizeof(cl_mem), &nx), "mg3");
    VCHECK(clSetKernelArg(k_ln_, 4, sizeof(int), &M), "mg4");
    VCHECK(clSetKernelArg(k_ln_, 5, sizeof(int), &H), "mg5");
    VCHECK(clSetKernelArg(k_ln_, 6, sizeof(float), &cfg_.eps), "mg6");
    size_t g = ((size_t)M + 63) / 64 * 64;
    VCHECK(clEnqueueNDRangeKernel(q, k_ln_, 1, nullptr, &g, nullptr, 0, nullptr, nullptr), "mg.ln");

    // No shuffle kernel: patchify emitted merge-blocks contiguously, so [M,H] IS [M/4,4H].
    auto gemm = [&](int Mr, int N, int K, cl_mem x, cl_mem W, cl_mem y) {
        auto st = f16_
            ? clblast::Gemm<cl_half>(clblast::Layout::kRowMajor,
                                     clblast::Transpose::kNo, clblast::Transpose::kYes,
                                     Mr, N, K, kOneH, x, 0, K, W, 0, K, kZeroH, y, 0, N, &q, nullptr)
            : clblast::Gemm<float>(clblast::Layout::kRowMajor,
                                         clblast::Transpose::kNo, clblast::Transpose::kYes,
                                         Mr, N, K, kOne, x, 0, K, W, 0, K, kZero, y, 0, N,
                                         &q, nullptr);
        if (st != clblast::StatusCode::kSuccess) throw std::runtime_error("vision: merger gemm");
    };
    auto bias = [&](cl_mem y, cl_mem b, int rows, int D) {
        VCHECK(clSetKernelArg(k_add_bias_, 0, sizeof(cl_mem), &y), "mb0");
        VCHECK(clSetKernelArg(k_add_bias_, 1, sizeof(cl_mem), &b), "mb1");
        VCHECK(clSetKernelArg(k_add_bias_, 2, sizeof(int), &rows), "mb2");
        VCHECK(clSetKernelArg(k_add_bias_, 3, sizeof(int), &D), "mb3");
        size_t gg = ((size_t)rows * D + 63) / 64 * 64;
        VCHECK(clEnqueueNDRangeKernel(q, k_add_bias_, 1, nullptr, &gg, nullptr, 0, nullptr, nullptr), "mb");
    };
    gemm(groups, HG, HG, nx, mm0_.w, mid);
    bias(mid, mm0_.b, groups, HG);
    { int n = groups * HG;
      VCHECK(clSetKernelArg(k_gelu_, 0, sizeof(cl_mem), &mid), "mgu0");
      VCHECK(clSetKernelArg(k_gelu_, 1, sizeof(cl_mem), &mid), "mgu1");
      VCHECK(clSetKernelArg(k_gelu_, 2, sizeof(int), &n), "mgu2");
      size_t gg = ((size_t)n + 63) / 64 * 64;
      VCHECK(clEnqueueNDRangeKernel(q, k_gelu_, 1, nullptr, &gg, nullptr, 0, nullptr, nullptr), "mg.gelu"); }
    gemm(groups, OUT, HG, mid, mm2_.w, res);
    bias(res, mm2_.b, groups, OUT);

    // The tower's output crosses back into fp32 here regardless: forward_embed() takes floats, and
    // these embeddings reach 200+ in magnitude where half has ~3 decimal digits. Widening at the
    // boundary costs one pass over 256x5120 and keeps the splice exact.
    out->resize((size_t)groups * OUT);
    if (f16_) {
        std::vector<uint16_t> h(out->size());
        VCHECK(clEnqueueReadBuffer(q, res, CL_TRUE, 0, h.size() * 2, h.data(), 0, nullptr, nullptr), "mg.read");
        for (size_t i = 0; i < h.size(); ++i) {
            const uint16_t v = h[i];
            const uint32_t sign = (uint32_t)(v & 0x8000) << 16;
            const uint32_t exp = (v >> 10) & 0x1F, man = v & 0x3FF;
            uint32_t f;
            if (exp == 0)        f = sign;                                    // zero/denormal -> zero
            else if (exp == 0x1F) f = sign | 0x7F800000 | (man << 13);        // inf/nan
            else                  f = sign | ((exp - 15 + 127) << 23) | (man << 13);
            std::memcpy(&(*out)[i], &f, 4);
        }
    } else {
        VCHECK(clEnqueueReadBuffer(q, res, CL_TRUE, 0, out->size() * 4, out->data(), 0, nullptr, nullptr), "mg.read");
    }
    for (cl_mem b : {nx, mid, res}) clReleaseMemObject(b);
}

// ── the forward ────────────────────────────────────────────────────────────
// Mirrors vision_reference.h op for op. Every GEMM is nn.Linear layout
// (out[M,N] = x[M,K] @ W[N,K]^T), which is why TransposeB is kYes throughout.
void VisionModel::encode(const uint8_t* rgb, int h, int w, std::vector<float>* out,
                         int* gh, int* gw) {
    const int H = cfg_.hidden, NH = cfg_.heads, HD = cfg_.head_dim, FF = cfg_.ffn;
    const int per = cfg_.patch * cfg_.merge;
    // MEMORY GUARD. Scratch grows with the patch count, and the vision tower sits on top of an
    // already-resident 3.6 GB text model. An unguarded full-resolution image once exceeded the device
    // budget and rebooted the phone — so size the image to the memory actually left, rather than
    // assuming Qwen's 1003520-pixel default always fits. Shrinking a photo degrades quality; running
    // the device out of memory loses the user's session.
    cl_ulong gmem = 0;
    clGetDeviceInfo(ocl_.device(), CL_DEVICE_GLOBAL_MEM_SIZE, sizeof(gmem), &gmem, nullptr);
    // What the TEXT tower + state already hold. Overridable because it is not always true: the
    // standalone --vision-check runs the tower with no text model loaded, and hardcoding the
    // reservation there made "available" negative and silently clamped a 171-token image to 32
    // patches — which would have made any device-vs-oracle comparison meaningless.
    const char* rv = getenv("BONSAI_VIS_RESERVE_MB");
    const double used = (rv ? atof(rv) : (3617.0 + 922.0 + 400.0)) * 1048576.0;
    const double avail = (double)gmem - used - 200.0 * 1048576;  // 200 MB safety margin

    // QUALITY/SPEED CAP, distinct from the memory guard below. Qwen's 1003520-px default yields ~760
    // merged tokens for a landscape photo, and every one of those is a token the 27B language model
    // must prefill AND carry in its KV cache for the rest of the conversation — the tower is not the
    // expensive part, the tokens it emits are. 262144 px gives ~256 tokens: 3x less prefill and 3x
    // less tower compute, for detail loss that does not change the answer on ordinary photos.
    // Raise it with BONSAI_VIS_MAX_PX when a task genuinely needs fine detail (dense text, charts).
    // 131072 px -> ~512 patches -> 128 merged tokens. Prefill is EXACTLY linear in token count
    // (each one is an independent pass over all 3.6 GB of weights), so halving the budget from
    // 262144 halves the ~20 s image prefill. It costs resolution: 362x362 against 512x512, which is
    // fine for scene description and starts to bite on small text in screenshots.
    // BONSAI_VIS_MAX_PX overrides — the app exposes it as an image-detail setting.
    const char* mp = getenv("BONSAI_VIS_MAX_PX");
    const long cap_px = mp ? atol(mp) : 131072;
    long max_px = std::min(1003520L, cap_px);
    const long want_px = max_px;
    int rh = 0, rw = 0, M = 0;
    for (;;) {
        smart_resize(h, w, per, 56 * 56, max_px, &rh, &rw);
        M = (rh / cfg_.patch) * (rw / cfg_.patch);         // patches BEFORE the 2x2 merge
        const int qt = std::min(M, 256);
        const double need = ((double)M * H * 4) * 6 + (double)M * 3 * H * 4 +
                            (double)M * FF * 4 + (double)NH * qt * M * 4;
        if (need <= avail || max_px <= 56 * 56 * 4) {
            fprintf(stderr, "[vis] %d patches (%dx%d) -> %d tokens, scratch %.0f MB of %.0f MB\n",
                    M, rh, rw, M / (cfg_.merge * cfg_.merge), need / 1048576.0, avail / 1048576.0);
            // Downscaling is a quality decision made silently on the user's behalf, and a heavily
            // clamped image still produces a fluent, confident, WRONG description rather than any
            // visible error. Say so out loud: this clamp once reduced a 320x224 test image to 96x64
            // and the only symptom was the model describing a different picture.
            // Only warn when MEMORY forced a further cut — the quality cap above is deliberate, and
            // warning about it every image would train the reader to ignore the line that matters.
            if (max_px < want_px)
                fprintf(stderr, "[vis] WARNING: image clamped to %ld px (wanted %ld) to fit %.0f MB "
                                "— detail will be lost\n", max_px, want_px, avail / 1048576.0);
            fflush(stderr);
            break;
        }
        max_px = (long)(max_px * 0.7);                      // step down and re-fit
    }
    *gh = rh / per; *gw = rw / per;

    cl_command_queue q = ocl_.queue();
    auto buf = [&](size_t n_elems) {
        cl_int e; cl_mem b = clCreateBuffer(ocl_.context(), CL_MEM_READ_WRITE, n_elems * ES_, nullptr, &e);
        VCHECK(e, "vision scratch"); return b;
    };
    auto run = [&](cl_kernel k, size_t gws) {
        size_t g = (gws + 63) / 64 * 64;
        VCHECK(clEnqueueNDRangeKernel(q, k, 1, nullptr, &g, nullptr, 0, nullptr, nullptr), "vis kernel");
    };
    auto arg = [&](cl_kernel k, int i, size_t sz, const void* v) {
        VCHECK(clSetKernelArg(k, i, sz, v), "vis arg");
    };
    // nn.Linear: out[M,N] = x[M,K] @ W[N,K]^T. Half throughout — see the HALF=ON note in CMakeLists.
    auto gemm = [&](int Mr, int N, int K, cl_mem x, cl_mem W, cl_mem y) {
        // Hgemm when the tower is fp16. alpha/beta MUST be half bit patterns here (see kOneH).
        auto st = f16_
            ? clblast::Gemm<cl_half>(clblast::Layout::kRowMajor,
                                     clblast::Transpose::kNo, clblast::Transpose::kYes,
                                     Mr, N, K, kOneH, x, 0, K, W, 0, K, kZeroH, y, 0, N, &q, nullptr)
            : clblast::Gemm<float>(clblast::Layout::kRowMajor,
                                         clblast::Transpose::kNo, clblast::Transpose::kYes,
                                         Mr, N, K, kOne, x, 0, K, W, 0, K, kZero, y, 0, N, &q, nullptr);
        if (st != clblast::StatusCode::kSuccess)
            throw std::runtime_error("vision: CLBlast Hgemm failed (" +
                                     std::to_string((int)st) + ") — was CLBlast built with HALF=ON?");
    };
    auto bias = [&](cl_mem y, cl_mem b, int rows, int D) {
        arg(k_add_bias_, 0, sizeof(cl_mem), &y); arg(k_add_bias_, 1, sizeof(cl_mem), &b);
        arg(k_add_bias_, 2, sizeof(int), &rows); arg(k_add_bias_, 3, sizeof(int), &D);
        run(k_add_bias_, (size_t)rows * D);
    };

    cl_mem hid = buf((size_t)M * H), nx = buf((size_t)M * H);
    cl_mem qkv = buf((size_t)M * 3 * H), tmp = buf((size_t)M * H), ffb = buf((size_t)M * FF);
    cl_mem qh = buf((size_t)M * H), kh = buf((size_t)M * H), vh = buf((size_t)M * H);
    // Bounded: H * QTILE * M instead of H * M * M. At M=3840 that is 31 MB rather than 472 MB,
    // which is the difference between running and taking the phone down with a system OOM.
    const int QT = std::min(M, 256);
    // The fused kernel keeps a whole probability row in local memory; it only fits while
    // (M + WG + hd) floats stay under the device's local size. Beyond that, fall back rather than
    // fail — a 32 KB limit is 8000-odd patches, well past anything the px cap allows, but the check
    // costs nothing and the alternative is a CL_OUT_OF_RESOURCES on some future device.
    const bool fused_attn = ((size_t)M + 64 + HD) * 4 + 256 <= local_mem_;
    fprintf(stderr, "[vis] attention: %s (local %zu KB, need %zu KB)\n",
            fused_attn ? "FUSED" : "3-kernel fallback", local_mem_ / 1024,
            (((size_t)M + 64 + HD) * 4) / 1024 + 1);
    cl_mem sc = buf((size_t)NH * QT * M);

    make_rope(*gh, *gw);
    patch_embed(rgb, h, w, rh, rw, M, hid);    // pixels -> [M, H], + position embeddings

    probe("patch_embed", hid, (size_t)M * H);
    const float scale = 1.0f / std::sqrt((float)HD);
    // Split weight-upload from compute. They have completely different fixes — upload is bounded by
    // dequant + PCIe-equivalent copy and is attacked by caching or narrowing the dtype, compute by
    // the kernels — so a single "tower took N seconds" number cannot direct any of that work.
    double t_up = 0, t_cmp = 0;
    auto now = [] { return std::chrono::duration<double>(
                        std::chrono::steady_clock::now().time_since_epoch()).count(); };
    const double t_enc0 = now();
    for (int i = 0; i < cfg_.depth; ++i) {
        const double tu0 = now();
        Block b = load_block(i);        // ~33 MB in, ~33 MB out at the end of the loop
        clFinish(q);
        t_up += now() - tu0;
        const double tc0 = now();
        // --- attention ---
        arg(k_ln_, 0, sizeof(cl_mem), &hid);   arg(k_ln_, 1, sizeof(cl_mem), &b.ln1_w);
        arg(k_ln_, 2, sizeof(cl_mem), &b.ln1_b); arg(k_ln_, 3, sizeof(cl_mem), &nx);
        arg(k_ln_, 4, sizeof(int), &M); arg(k_ln_, 5, sizeof(int), &H);
        arg(k_ln_, 6, sizeof(float), &cfg_.eps);
        run(k_ln_, (size_t)M);

        gemm(M, 3 * H, H, nx, b.qkv.w, qkv);
        bias(qkv, b.qkv.b, M, 3 * H);

        // rope on q and k IN PLACE, before the split: row stride is 3H, q at 0 and k at H.
        const int stride3 = 3 * H;
        for (int off : {0, H}) {
            arg(k_rope_, 0, sizeof(cl_mem), &qkv);   arg(k_rope_, 1, sizeof(cl_mem), &cos_);
            arg(k_rope_, 2, sizeof(cl_mem), &sin_);  arg(k_rope_, 3, sizeof(int), &M);
            arg(k_rope_, 4, sizeof(int), &NH);       arg(k_rope_, 5, sizeof(int), &HD);
            arg(k_rope_, 6, sizeof(int), &stride3);  arg(k_rope_, 7, sizeof(int), &off);
            run(k_rope_, (size_t)M * NH * (HD / 2));
        }
        arg(k_qkv_split_, 0, sizeof(cl_mem), &qkv); arg(k_qkv_split_, 1, sizeof(cl_mem), &qh);
        arg(k_qkv_split_, 2, sizeof(cl_mem), &kh);  arg(k_qkv_split_, 3, sizeof(cl_mem), &vh);
        arg(k_qkv_split_, 4, sizeof(int), &M); arg(k_qkv_split_, 5, sizeof(int), &NH);
        arg(k_qkv_split_, 6, sizeof(int), &HD);
        run(k_qkv_split_, (size_t)M * H);

        // NON-causal (no mask), tiled over query blocks so peak memory is independent of M.
        for (int q0 = 0; q0 < M; q0 += QT) {
            const int qn = std::min(QT, M - q0);
            if (fused_attn) {
                const size_t WG = 64;
                arg(k_attn_fused_, 0, sizeof(cl_mem), &qh);
                arg(k_attn_fused_, 1, sizeof(cl_mem), &kh);
                arg(k_attn_fused_, 2, sizeof(cl_mem), &vh);
                arg(k_attn_fused_, 3, sizeof(cl_mem), &tmp);
                arg(k_attn_fused_, 4, (size_t)M * 4, nullptr);     // __local p[M]
                arg(k_attn_fused_, 5, WG * 4, nullptr);            // __local red[WG]
                arg(k_attn_fused_, 6, (size_t)HD * 4, nullptr);    // __local qs[hd]
                arg(k_attn_fused_, 7, sizeof(int), &NH);
                arg(k_attn_fused_, 8, sizeof(int), &M);
                arg(k_attn_fused_, 9, sizeof(int), &HD);
                arg(k_attn_fused_, 10, sizeof(float), &scale);
                arg(k_attn_fused_, 11, sizeof(int), &q0);
                arg(k_attn_fused_, 12, sizeof(int), &qn);
                const size_t gws = (size_t)NH * qn * WG;
                VCHECK(clEnqueueNDRangeKernel(q, k_attn_fused_, 1, nullptr, &gws, &WG,
                                              0, nullptr, nullptr), "attn_fused");
                continue;
            }
            arg(k_scores_t_, 0, sizeof(cl_mem), &qh); arg(k_scores_t_, 1, sizeof(cl_mem), &kh);
            arg(k_scores_t_, 2, sizeof(cl_mem), &sc); arg(k_scores_t_, 3, sizeof(int), &NH);
            arg(k_scores_t_, 4, sizeof(int), &M);     arg(k_scores_t_, 5, sizeof(int), &HD);
            arg(k_scores_t_, 6, sizeof(float), &scale);
            arg(k_scores_t_, 7, sizeof(int), &q0);    arg(k_scores_t_, 8, sizeof(int), &qn);
            run(k_scores_t_, (size_t)NH * qn * M);

            arg(k_softmax_t_, 0, sizeof(cl_mem), &sc); arg(k_softmax_t_, 1, sizeof(int), &NH);
            arg(k_softmax_t_, 2, sizeof(int), &M);     arg(k_softmax_t_, 3, sizeof(int), &qn);
            run(k_softmax_t_, (size_t)NH * qn);

            arg(k_context_t_, 0, sizeof(cl_mem), &sc);  arg(k_context_t_, 1, sizeof(cl_mem), &vh);
            arg(k_context_t_, 2, sizeof(cl_mem), &tmp); arg(k_context_t_, 3, sizeof(int), &NH);
            arg(k_context_t_, 4, sizeof(int), &M);      arg(k_context_t_, 5, sizeof(int), &HD);
            arg(k_context_t_, 6, sizeof(int), &q0);     arg(k_context_t_, 7, sizeof(int), &qn);
            run(k_context_t_, (size_t)qn * H);
        }

        gemm(M, H, H, tmp, b.out.w, nx);
        bias(nx, b.out.b, M, H);
        arg(k_add_, 0, sizeof(cl_mem), &hid); arg(k_add_, 1, sizeof(cl_mem), &nx);
        arg(k_add_, 2, sizeof(cl_mem), &hid); { int n = M * H; arg(k_add_, 3, sizeof(int), &n); }
        run(k_add_, (size_t)M * H);

        // --- MLP: fc2(gelu_TANH(fc1(x))). The tanh form, not the merger's erf form. ---
        arg(k_ln_, 0, sizeof(cl_mem), &hid);   arg(k_ln_, 1, sizeof(cl_mem), &b.ln2_w);
        arg(k_ln_, 2, sizeof(cl_mem), &b.ln2_b); arg(k_ln_, 3, sizeof(cl_mem), &nx);
        arg(k_ln_, 4, sizeof(int), &M); arg(k_ln_, 5, sizeof(int), &H);
        arg(k_ln_, 6, sizeof(float), &cfg_.eps);
        run(k_ln_, (size_t)M);
        gemm(M, FF, H, nx, b.fc1.w, ffb);
        bias(ffb, b.fc1.b, M, FF);
        { int n = M * FF;
          arg(k_gelu_tanh_, 0, sizeof(cl_mem), &ffb); arg(k_gelu_tanh_, 1, sizeof(cl_mem), &ffb);
          arg(k_gelu_tanh_, 2, sizeof(int), &n); run(k_gelu_tanh_, (size_t)n); }
        gemm(M, H, FF, ffb, b.fc2.w, nx);
        bias(nx, b.fc2.b, M, H);
        arg(k_add_, 0, sizeof(cl_mem), &hid); arg(k_add_, 1, sizeof(cl_mem), &nx);
        arg(k_add_, 2, sizeof(cl_mem), &hid); { int n = M * H; arg(k_add_, 3, sizeof(int), &n); }
        run(k_add_, (size_t)M * H);
        clFinish(q);            // the block's weights must be done being read before we free them
        t_cmp += now() - tc0;
        free_block(b);
        if (i % 9 == 0) { fprintf(stderr, "[vis] block %d/%d\n", i, cfg_.depth); fflush(stderr); }
        // Every block when dumping — the drift is gradual, so the first block that diverges is only
        // visible if all of them are recorded.
        if (getenv("VIS_DUMP_DIR") || i == 0 || i == 13 || i == cfg_.depth - 1) {
            char t[24]; snprintf(t, sizeof(t), "blk%02d", i);
            probe(t, hid, (size_t)M * H);
        }
    }

    const double t_blocks = now() - t_enc0;
    const double t_mg0 = now();
    merge_and_project(hid, M, out);
    timing_.merger = now() - t_mg0;
    timing_.total = t_blocks + timing_.merger;
    timing_.upload = t_up;
    timing_.compute = t_cmp;
    timing_.patches = M;
    fprintf(stderr, "[vis] encode %.2fs = weight-upload %.2fs + compute %.2fs + merger %.2fs "
                    "(%d patches, %d blocks)\n",
            timing_.total, t_up, t_cmp, timing_.merger, M, cfg_.depth);
    fflush(stderr);
    for (cl_mem b : {hid, nx, qkv, tmp, ffb, qh, kh, vh, sc}) clReleaseMemObject(b);
}

long VisionModel::token_count(int h, int w) const {
    VisionMeta vm;
    vm.patch = cfg_.patch; vm.merge = cfg_.merge;
    int oh, ow;
    // Must use the SAME cap encode() does, or the caller budgets prefill for a token count the tower
    // will never produce.
    const char* mp = getenv("BONSAI_VIS_MAX_PX");
    smart_resize(h, w, cfg_.patch * cfg_.merge, 56 * 56,
                 std::min(1003520L, mp ? atol(mp) : 131072L), &oh, &ow);
    return vision_token_count(oh, ow, vm);
}

#endif  // BONSAI_VISION
