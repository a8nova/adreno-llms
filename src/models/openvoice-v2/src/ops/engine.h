// OpenVoice V2 ToneColorConverter — shared OpenCL engine (manual fp16 port).
// Primitives + the WaveNet (shared by enc_q and flow) live here. The three
// model components are split across enc_q.cpp / flow.cpp / dec.cpp; each is a
// (declared-here, defined-there) Engine method so it can call the primitives
// directly. Built + run ONLY on the Android device (see scripts/).
#pragma once
#include "opencl_context.h"
#include "weights.h"
#include <CL/cl.h>
#include <clblast_c.h>      // #5 im2col → GEMM
#include <cstdio>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <ctime>
#include <string>
#include <vector>
#include <unordered_map>

// ── fp16 host helpers ───────────────────────────────────────────────────────
static inline uint16_t f2h(float f){
    uint32_t x; __builtin_memcpy(&x,&f,4);
    uint32_t sign=(x>>16)&0x8000u; int32_t exp=((x>>23)&0xff)-127+15; uint32_t man=x&0x7fffffu;
    if(exp<=0){ if(exp<-10) return (uint16_t)sign; man|=0x800000u; uint32_t sh=(uint32_t)(14-exp);
        uint32_t h=man>>sh; if((man>>(sh-1))&1) h++; return (uint16_t)(sign|h);}
    if(exp>=31) return (uint16_t)(sign|0x7c00u);
    uint16_t h=(uint16_t)(sign|(exp<<10)|(man>>13)); if((man>>12)&1) h++; return h;
}
static inline float h2f(uint16_t h){
    uint32_t sign=(h&0x8000u)<<16; uint32_t exp=(h>>10)&0x1f; uint32_t man=h&0x3ffu; uint32_t out;
    if(exp==0){ if(man==0){out=sign;} else { exp=127-15+1; while(!(man&0x400u)){man<<=1;exp--;} man&=0x3ffu; out=sign|(exp<<23)|(man<<13);} }
    else if(exp==31){ out=sign|0x7f800000u|(man<<13);} else { out=sign|((exp-15+127)<<23)|(man<<13);}
    float f; __builtin_memcpy(&f,&out,4); return f;
}
static inline double now_ms(){ struct timespec ts; clock_gettime(CLOCK_MONOTONIC,&ts); return ts.tv_sec*1000.0 + ts.tv_nsec/1e6; }

// ── storage dtype — fp16 by default; -DSTORE_FP32 stores activations as float ─
#ifdef STORE_FP32
  typedef float store_t;
  static inline store_t st_enc(float f){ return f; }
  static inline float   st_dec(store_t h){ return h; }
  static const char* STORE_DTYPE = "float32";
  static const char* CL_STORE_OPT = " -DSTORE_FP32";
#else
  typedef uint16_t store_t;
  static inline store_t st_enc(float f){ return f2h(f); }
  static inline float   st_dec(store_t h){ return h2f(h); }
  static const char* STORE_DTYPE = "float16";
  static const char* CL_STORE_OPT = "";
#endif

// On-device buffer with shape [C, T] -----------------------------------------
struct Buf { cl_mem mem=nullptr; int C=0, T=0; size_t n() const { return (size_t)C*T; } };

struct Engine {
    OpenCLContext& cl; Weights& W; cl_program prog;
    std::unordered_map<std::string,cl_mem> wcache;       // folded weights cache (persistent)
    std::unordered_map<cl_mem,size_t> live_;             // buffer→bytes for peak accounting
    size_t live_bytes=0, peak_bytes=0;

    Engine(OpenCLContext& c, Weights& w): cl(c), W(w) {
        // Concatenate the per-functional-unit kernel files — _preamble.cl FIRST so the
        // storage macros + TILE_T/NC are defined before any kernel uses them — then build
        // as one program (mirrors adreno-llms GpuOps::init).
        const char* files[]={ "_preamble.cl","elementwise.cl","wavenet.cl","conv_1d.cl",
            "conv_1d_image.cl","conv_transpose_1d.cl","gemm.cl","hwprobe.cl" };
        std::string src;
        for(const char* fn : files){
            std::string path=std::string("kernels/")+fn;
            FILE* f=fopen(path.c_str(),"rb");
            if(!f){ fprintf(stderr,"FATAL: %s not found\n", path.c_str()); exit(2); }
            fseek(f,0,SEEK_END); long sz=ftell(f); fseek(f,0,SEEK_SET);
            std::string s(sz,0); fread(&s[0],1,sz,f); fclose(f);
            src += "\n// ==== "; src += fn; src += " ====\n"; src += s;
        }
        // conv1d_2d NC (output-channels/thread = input-reuse factor) is compile-time in the
        // kernel (sizes float8 a[NC]) AND drives the C++ dispatch grid — they MUST agree.
        // NNOPT_NC re-tunes the bandwidth-bound convs per-GPU without editing source: a higher
        // NC reuses each half8 input load across more output channels → less input traffic.
        // Default NC=4: Adreno 620 (Razr) sweep winner (1.69× on conv1d_2d vs NC=2 in-pipeline,
        // cos 1.0000). Adreno 619 (tablet) prefers NC=2 — set NNOPT_NC=2 there. GPU model is not
        // exposed via CL_DEVICE_NAME (both report "QUALCOMM Adreno(TM)") so this can't auto-detect.
        int nc_override = 4;
        if(const char* e=getenv("NNOPT_NC")){ int v=atoi(e); if(v>=1 && v<=8) nc_override=v; }
        std::string flags = std::string("-cl-mad-enable")+CL_STORE_OPT+" -DNC="+std::to_string(nc_override);
        prog = cl.build_program(src, flags);
        if(!prog){ fprintf(stderr,"FATAL: converter.cl failed to build (see CL build log above)\n"); exit(3); }
        NC = nc_override;
        if(getenv("NNOPT_NC")) printf("conv1d_2d NC override: %d channels/thread\n", NC);
        // conv_transpose1d_opt work-group (driver-default until tuned). NNOPT_TLX/TLY.
        if(const char* e=getenv("NNOPT_TLX")) tlx=(size_t)atoi(e);
        if(const char* e=getenv("NNOPT_TLY")) tly=(size_t)atoi(e);
        if(getenv("NNOPT_TLX")||getenv("NNOPT_TLY"))
            printf("conv_transpose1d_opt work-group override: {%zu,%zu}\n", tlx, tly);
        cl_ulong maxAlloc=0, globalMem=0;
        clGetDeviceInfo(cl.device(), CL_DEVICE_MAX_MEM_ALLOC_SIZE, sizeof(maxAlloc), &maxAlloc, 0);
        clGetDeviceInfo(cl.device(), CL_DEVICE_GLOBAL_MEM_SIZE, sizeof(globalMem), &globalMem, 0);
        printf("storage dtype: %s | device max_alloc=%lluMB global_mem=%lluMB\n",
            STORE_DTYPE, (unsigned long long)(maxAlloc>>20), (unsigned long long)(globalMem>>20));
        // Per-GPU conv1d_2d work-group tuning: {4,32} won the Adreno 619 sweep, but the
        // optimal local shape is wave-geometry-specific. Override at runtime to re-tune for
        // a new GPU (e.g. Adreno 620) without a rebuild: NNOPT_LX / NNOPT_LY.
        if(const char* e=getenv("NNOPT_LX")) lx=(size_t)atoi(e);
        if(const char* e=getenv("NNOPT_LY")) ly=(size_t)atoi(e);
        if(getenv("NNOPT_LX")||getenv("NNOPT_LY"))
            printf("conv1d_2d work-group override: {%zu,%zu} = %zu work-items\n", lx, ly, lx*ly);
    }

    // ── memory accounting + buffer arena (#16) ──────────────────────────────
    // Activation buffers are pooled by byte-size and recycled on rel() instead of
    // clCreateBuffer/clReleaseMemObject per op (kills host alloc/free overhead).
    // Safe with the in-order queue: a recycled buffer's new kernel is enqueued after
    // the old kernel that used it, so it can't run before the old one completes.
    bool arena = true;
    std::unordered_map<size_t,std::vector<cl_mem>> pool;   // bytes → free buffers
    void track(cl_mem m, size_t bytes){ if(!m) return; live_[m]=bytes; live_bytes+=bytes; if(live_bytes>peak_bytes) peak_bytes=live_bytes; }
    void rel(cl_mem m){ if(!m) return; auto it=live_.find(m); if(it!=live_.end()){ size_t b=it->second; live_bytes-=b; live_.erase(it);
        if(arena){ pool[b].push_back(m); return; } } clReleaseMemObject(m); }
    cl_mem pooled(size_t bytes){ if(!arena) return nullptr; auto it=pool.find(bytes); if(it!=pool.end() && !it->second.empty()){ cl_mem m=it->second.back(); it->second.pop_back(); return m; } return nullptr; }
    double peak_mb() const { return peak_bytes/(double)(1u<<20); }

    // ── GPU-timer profiling (Qualcomm OpenCL guide §4.5.2): the accurate kernel
    // execution time comes from clGetEventProfilingInfo START→END on the device's
    // own counter, NOT host wall-clock. Queue is created with CL_QUEUE_PROFILING_ENABLE.
    double gpu_ns=0; std::unordered_map<std::string,double> gpu_ns_by;
    // Explicit local work-group size for conv1d_opt (#2). {4,32}=128 work-items won the
    // microbench sweep (1.64× over driver default). 0 ⇒ let the driver choose.
    // {2,32}: Adreno 620 sweep winner (paired with NC=4). Adreno 619 prefers {4,32} — override
    // with NNOPT_LX=4. With NC=4 the channel grid (Cout/NC) is small so lx=2 fits it tighter.
    size_t lx=2, ly=32;
    // #8 batch mode: per-kernel event-wait ONLY when profiling; else enqueue back-to-back
    // into the in-order queue and sync once via finish(). NNOPT_PROFILE=1 ⇒ GPU timer on.
    bool profile = (getenv("NNOPT_PROFILE")!=nullptr);
    void finish(){ clFinish(cl.queue()); }
    // #10 persistent kernels: create each cl_kernel once and reuse (args are reset per call).
    std::unordered_map<std::string,cl_kernel> kcache;
    cl_kernel kern(const char* n){ auto it=kcache.find(n); if(it!=kcache.end()) return it->second;
        cl_kernel k=clCreateKernel(prog,n,nullptr); kcache[n]=k; return k; }
    // #17 warm-up (§4.5.4): no root to pin clocks (device idles at 266MHz, max 840MHz),
    // so ramp the DVFS governor with ~0.6s of sustained conv before timing.
    void warmup(){
        std::vector<float> in((size_t)128*16384, 0.1f);
        Buf x{alloc(in.size()),128,16384}; upload(x.mem,in);
        bool sv=profile; profile=false;
        double t0=now_ms();
        while(now_ms()-t0 < 600.0){ Buf o=conv1d_vec(x,"dec.resblocks.5.convs1.0",128,3,1,1,1); finish(); rel(o.mem); }
        profile=sv; rel(x.mem);
    }
    void run_kernel(cl_kernel k, cl_uint dim, const size_t* g, const char* name){
        run_kernel(k, dim, g, nullptr, name);
    }
    void run_kernel(cl_kernel k, cl_uint dim, const size_t* g, const size_t* local, const char* name){
        if(!profile){ clEnqueueNDRangeKernel(cl.queue(), k, dim, nullptr, g, local, 0, nullptr, nullptr); return; }
        cl_event ev=nullptr;
        clEnqueueNDRangeKernel(cl.queue(), k, dim, nullptr, g, local, 0, nullptr, &ev);
        clWaitForEvents(1, &ev);                          // GPU timer needs the cmd complete
        cl_ulong s=0,e=0;
        clGetEventProfilingInfo(ev, CL_PROFILING_COMMAND_START, sizeof(s), &s, nullptr);
        clGetEventProfilingInfo(ev, CL_PROFILING_COMMAND_END,   sizeof(e), &e, nullptr);
        double ns=(double)(e-s); gpu_ns+=ns; if(name) gpu_ns_by[name]+=ns;
        clReleaseEvent(ev);
    }
    double gpu_ms() const { return gpu_ns/1e6; }
    void reset_gpu_timer(){ gpu_ns=0; gpu_ns_by.clear(); }
    void report_gpu(){
        printf("  GPU-timer (kernel exec, device counter): %.1f ms total\n", gpu_ms());
        for(auto& kv : gpu_ns_by) printf("    %-22s %.1f ms\n", kv.first.c_str(), kv.second/1e6);
    }

    cl_mem alloc(size_t n){
        size_t bytes=n*sizeof(store_t);
        cl_mem m=pooled(bytes);
        if(!m){ cl_int err=CL_SUCCESS; m=clCreateBuffer(cl.context(), CL_MEM_READ_WRITE, bytes, nullptr, &err);
            if(err!=CL_SUCCESS || !m){ fprintf(stderr,"ALLOC FAILED: %zu elems (%zu MB) store=%s err=%d\n", n, bytes/(1u<<20), STORE_DTYPE, (int)err); return nullptr; } }
        track(m, bytes); return m;
    }
    void upload(cl_mem m, const std::vector<float>& v){
        std::vector<store_t> h(v.size()); for(size_t i=0;i<v.size();++i) h[i]=st_enc(v[i]);
        clEnqueueWriteBuffer(cl.queue(), m, CL_TRUE, 0, h.size()*sizeof(store_t), h.data(), 0,0,0);
    }
    // Weights are ALWAYS fp16 (mixed precision). Separate alloc/upload at 2 bytes.
    cl_mem alloc_half(size_t n){
        size_t bytes=n*2; cl_mem m=pooled(bytes);
        if(!m){ cl_int err=CL_SUCCESS; m=clCreateBuffer(cl.context(), CL_MEM_READ_WRITE, bytes, nullptr, &err);
            if(err!=CL_SUCCESS || !m){ fprintf(stderr,"WEIGHT ALLOC FAILED: %zu elems err=%d\n", n, (int)err); return nullptr; } }
        track(m, bytes); return m;
    }
    void upload_half(cl_mem m, const std::vector<float>& v){
        std::vector<uint16_t> h(v.size()); for(size_t i=0;i<v.size();++i) h[i]=f2h(v[i]);
        clEnqueueWriteBuffer(cl.queue(), m, CL_TRUE, 0, h.size()*2, h.data(), 0,0,0);
    }
    std::vector<float> download(cl_mem m, size_t n){
        std::vector<store_t> h(n); clEnqueueReadBuffer(cl.queue(), m, CL_TRUE, 0, n*sizeof(store_t), h.data(),0,0,0);
        std::vector<float> v(n); for(size_t i=0;i<n;++i) v[i]=st_dec(h[i]); return v;
    }
    // Weight-norm fold (G7/G9): w = g * v/||v|| with norm over each dim-0 slice
    // (PyTorch weight_norm default dim=0). Works for BOTH Conv1d weight
    // [Cout,Cin,K] (d0=Cout) and ConvTranspose1d weight [Cin,Cout,K] (d0=Cin).
    cl_mem weight(const std::string& base, int /*Cout*/, int /*Cin*/, int /*K*/){
        auto it=wcache.find(base); if(it!=wcache.end()) return it->second;
        std::vector<float> w;
        if (W.has_tensor(base+".weight")) { w = W.get_host_vec(base+".weight"); }
        else {
            std::vector<float> g=W.get_host_vec(base+".weight_g");   // [d0,1,1]
            std::vector<float> v=W.get_host_vec(base+".weight_v");   // [d0, ...]
            std::vector<int> shp=W.get_shape(base+".weight_v");
            int d0 = shp.empty() ? (int)g.size() : shp[0];
            int rest = d0>0 ? (int)(v.size()/d0) : 0;
            w.resize(v.size());
            for(int s=0; s<d0; ++s){
                double nrm=0; for(int i=0;i<rest;++i){ double x=v[(size_t)s*rest+i]; nrm+=x*x; }
                nrm=std::sqrt(nrm)+1e-12; float sc=(float)(g[s]/nrm);
                for(int i=0;i<rest;++i) w[(size_t)s*rest+i]=v[(size_t)s*rest+i]*sc;
            }
        }
        cl_mem m=alloc_half(w.size()); upload_half(m,w); wcache[base]=m; return m;   // weights fp16
    }
    cl_mem bias(const std::string& base){
        std::string k=base+".bias"; if(!W.has_tensor(k)) return nullptr;
        auto v=W.get_host_vec(k); cl_mem m=alloc_half(v.size()); upload_half(m,v); return m;  // bias fp16
    }
    // Naive conv1d (1 output/thread) — kept for the microbenchmark baseline only.
    Buf conv1d_naive(const Buf& in, const std::string& base, int Cout, int K, int stride, int dil, int pad){
        cl_mem w=weight(base, Cout, in.C, K); cl_mem b=bias(base);
        int Tout=(in.T + 2*pad - dil*(K-1) - 1)/stride + 1;
        Buf out{alloc((size_t)Cout*Tout), Cout, Tout};
        cl_kernel k=kern("conv1d"); int has_b=b?1:0; cl_mem nullb=b?b:in.mem;
        clSetKernelArg(k,0,sizeof(cl_mem),&in.mem); clSetKernelArg(k,1,sizeof(cl_mem),&w);
        clSetKernelArg(k,2,sizeof(cl_mem),&nullb); clSetKernelArg(k,3,sizeof(cl_mem),&out.mem);
        clSetKernelArg(k,4,4,&in.C); clSetKernelArg(k,5,4,&in.T); clSetKernelArg(k,6,4,&Cout);
        clSetKernelArg(k,7,4,&Tout); clSetKernelArg(k,8,4,&K); clSetKernelArg(k,9,4,&stride);
        clSetKernelArg(k,10,4,&dil); clSetKernelArg(k,11,4,&pad); clSetKernelArg(k,12,4,&has_b);
        size_t g[2]={(size_t)Cout,(size_t)Tout}; run_kernel(k,2,g,"conv1d");
        if(b) rel(b); return out;
    }
    static const int TILE_T = 8;
    // Model-facing conv1d: dispatch to the __local-weight kernel when the work-group's
    // weights fit on-chip (#4/#6), else the register-tiled+half8 kernel.
    int NC = 4;   // conv1d_2d channels/thread (#3 2D); set in ctor from NNOPT_NC (default 4, Adreno 620)
    Buf conv1d(const Buf& in, const std::string& base, int Cout, int K, int stride, int dil, int pad){
        // #5: CLBlast Hgemm measured NET-NEGATIVE in the full pipeline — it edges out vec2d
        // only on large-channel small-T convs in isolation, but the per-call im2col buffer
        // alloc + integration overhead makes dec CPU e2e WORSE (13.8→14.5s), and it loses
        // 2-3× on the bandwidth-bound low-channel convs. conv1d_gemm kept (documented) but unused.
        if (NC>1 && (Cout%NC)==0) return conv1d_2d(in,base,Cout,K,stride,dil,pad);
        return conv1d_vec(in,base,Cout,K,stride,dil,pad);
    }
    // fp16-accumulate variant (2× ALU roof). Same dispatch as conv1d_2d.
    Buf conv1d_2dh(const Buf& in, const std::string& base, int Cout, int K, int stride, int dil, int pad){
        cl_mem w=weight(base, Cout, in.C, K); cl_mem b=bias(base);
        int Tout=(in.T + 2*pad - dil*(K-1) - 1)/stride + 1;
        Buf out{alloc((size_t)Cout*Tout), Cout, Tout};
        cl_kernel k=kern("conv1d_2dh"); int has_b=b?1:0; cl_mem nullb=b?b:in.mem;
        clSetKernelArg(k,0,sizeof(cl_mem),&in.mem); clSetKernelArg(k,1,sizeof(cl_mem),&w);
        clSetKernelArg(k,2,sizeof(cl_mem),&nullb); clSetKernelArg(k,3,sizeof(cl_mem),&out.mem);
        clSetKernelArg(k,4,4,&in.C); clSetKernelArg(k,5,4,&in.T); clSetKernelArg(k,6,4,&Cout);
        clSetKernelArg(k,7,4,&Tout); clSetKernelArg(k,8,4,&K); clSetKernelArg(k,9,4,&stride);
        clSetKernelArg(k,10,4,&dil); clSetKernelArg(k,11,4,&pad); clSetKernelArg(k,12,4,&has_b);
        size_t g[2]={(size_t)((Cout+NC-1)/NC),(size_t)((Tout+TILE_T-1)/TILE_T)};
        if(lx&&ly){ size_t loc[2]={lx,ly}; g[0]=((g[0]+lx-1)/lx)*lx; g[1]=((g[1]+ly-1)/ly)*ly; run_kernel(k,2,g,loc,"conv1d_2dh"); }
        else run_kernel(k,2,g,"conv1d_2dh");
        if(b) rel(b); return out;
    }
    Buf conv1d_2d(const Buf& in, const std::string& base, int Cout, int K, int stride, int dil, int pad){
        cl_mem w=weight(base, Cout, in.C, K); cl_mem b=bias(base);
        int Tout=(in.T + 2*pad - dil*(K-1) - 1)/stride + 1;
        Buf out{alloc((size_t)Cout*Tout), Cout, Tout};
        cl_kernel k=kern("conv1d_2d"); int has_b=b?1:0; cl_mem nullb=b?b:in.mem;
        clSetKernelArg(k,0,sizeof(cl_mem),&in.mem); clSetKernelArg(k,1,sizeof(cl_mem),&w);
        clSetKernelArg(k,2,sizeof(cl_mem),&nullb); clSetKernelArg(k,3,sizeof(cl_mem),&out.mem);
        clSetKernelArg(k,4,4,&in.C); clSetKernelArg(k,5,4,&in.T); clSetKernelArg(k,6,4,&Cout);
        clSetKernelArg(k,7,4,&Tout); clSetKernelArg(k,8,4,&K); clSetKernelArg(k,9,4,&stride);
        clSetKernelArg(k,10,4,&dil); clSetKernelArg(k,11,4,&pad); clSetKernelArg(k,12,4,&has_b);
        size_t g[2]={(size_t)((Cout+NC-1)/NC),(size_t)((Tout+TILE_T-1)/TILE_T)};
        if(lx&&ly){ size_t loc[2]={lx,ly}; g[0]=((g[0]+lx-1)/lx)*lx; g[1]=((g[1]+ly-1)/ly)*ly; run_kernel(k,2,g,loc,"conv1d_2d"); }
        else run_kernel(k,2,g,"conv1d_2d");
        if(b) rel(b); return out;
    }
    // conv1d — register-tiled + half8-vectorized.
    Buf conv1d_vec(const Buf& in, const std::string& base, int Cout, int K, int stride, int dil, int pad){
        cl_mem w=weight(base, Cout, in.C, K); cl_mem b=bias(base);
        int Tout=(in.T + 2*pad - dil*(K-1) - 1)/stride + 1;
        Buf out{alloc((size_t)Cout*Tout), Cout, Tout};
        cl_kernel k=kern("conv1d_opt"); int has_b=b?1:0; cl_mem nullb=b?b:in.mem;
        clSetKernelArg(k,0,sizeof(cl_mem),&in.mem); clSetKernelArg(k,1,sizeof(cl_mem),&w);
        clSetKernelArg(k,2,sizeof(cl_mem),&nullb); clSetKernelArg(k,3,sizeof(cl_mem),&out.mem);
        clSetKernelArg(k,4,4,&in.C); clSetKernelArg(k,5,4,&in.T); clSetKernelArg(k,6,4,&Cout);
        clSetKernelArg(k,7,4,&Tout); clSetKernelArg(k,8,4,&K); clSetKernelArg(k,9,4,&stride);
        clSetKernelArg(k,10,4,&dil); clSetKernelArg(k,11,4,&pad); clSetKernelArg(k,12,4,&has_b);
        size_t g[2]={(size_t)Cout,(size_t)((Tout+TILE_T-1)/TILE_T)};
        if(lx&&ly){
            size_t loc[2]={lx,ly};
            g[0]=((g[0]+lx-1)/lx)*lx; g[1]=((g[1]+ly-1)/ly)*ly;   // pad to local multiple (kernel guards OOB)
            run_kernel(k,2,g,loc,"conv1d_opt");
        } else run_kernel(k,2,g,"conv1d_opt");
        if(b) rel(b); return out;
    }
    // conv1d via im2col + CLBlast Hgemm (#5/#7). out[Cout,Tout] = W[Cout,Cin*K] · col[Cin*K,Tout].
    // Weights (fp16, row-major [Cout,Cin,K]) ARE the [Cout,Cin*K] matrix — no repack.
    Buf conv1d_gemm(const Buf& in, const std::string& base, int Cout, int K, int stride, int dil, int pad){
        cl_mem w=weight(base, Cout, in.C, K);
        int Tout=(in.T + 2*pad - dil*(K-1) - 1)/stride + 1;
        int Kd = in.C*K;
        Buf col{alloc((size_t)Kd*Tout), Kd, Tout};
        cl_kernel ik=kern("im2col");
        clSetKernelArg(ik,0,sizeof(cl_mem),&in.mem); clSetKernelArg(ik,1,sizeof(cl_mem),&col.mem);
        clSetKernelArg(ik,2,4,&in.C); clSetKernelArg(ik,3,4,&in.T); clSetKernelArg(ik,4,4,&Tout);
        clSetKernelArg(ik,5,4,&K); clSetKernelArg(ik,6,4,&stride); clSetKernelArg(ik,7,4,&dil); clSetKernelArg(ik,8,4,&pad);
        size_t ig[2]={(size_t)Kd,(size_t)Tout}; run_kernel(ik,2,ig,"im2col");
        Buf out{alloc((size_t)Cout*Tout), Cout, Tout};
        cl_half one=f2h(1.0f), zero=0; cl_command_queue q=cl.queue();
        CLBlastStatusCode st=CLBlastHgemm(CLBlastLayoutRowMajor, CLBlastTransposeNo, CLBlastTransposeNo,
            (size_t)Cout,(size_t)Tout,(size_t)Kd, one, w,0,(size_t)Kd, col.mem,0,(size_t)Tout,
            zero, out.mem,0,(size_t)Tout, &q, nullptr);
        if(st!=CLBlastSuccess) fprintf(stderr,"CLBlast Hgemm err=%d (%s)\n",(int)st, base.c_str());
        cl_mem b=bias(base);
        if(b){ cl_kernel bk=kern("add_bias"); clSetKernelArg(bk,0,sizeof(cl_mem),&out.mem); clSetKernelArg(bk,1,sizeof(cl_mem),&b);
               clSetKernelArg(bk,2,4,&Cout); clSetKernelArg(bk,3,4,&Tout);
               size_t bg[2]={(size_t)Cout,(size_t)Tout}; run_kernel(bk,2,bg,"add_bias"); rel(b); }
        rel(col.mem); return out;
    }
    // conv1d reading input via texture engine / L1 cache (image2d). Adreno-recommended
    // for memory-bound reuse. Images NOT pooled (different object type) — direct release.
    Buf conv1d_img(const Buf& in, const std::string& base, int Cout, int K, int stride, int dil, int pad){
        cl_mem w=weight(base, Cout, in.C, K);
        int Tout=(in.T + 2*pad - dil*(K-1) - 1)/stride + 1;
        cl_image_format fmt; fmt.image_channel_order=CL_R; fmt.image_channel_data_type=CL_HALF_FLOAT;
        cl_image_desc desc; memset(&desc,0,sizeof(desc));
        desc.image_type=CL_MEM_OBJECT_IMAGE2D; desc.image_width=in.T; desc.image_height=in.C;
        cl_int err=CL_SUCCESS;
        cl_mem img=clCreateImage(cl.context(), CL_MEM_READ_ONLY, &fmt, &desc, nullptr, &err);
        if(err!=CL_SUCCESS||!img){ fprintf(stderr,"clCreateImage err=%d\n",(int)err); return Buf{}; }
        size_t origin[3]={0,0,0}, region[3]={(size_t)in.T,(size_t)in.C,1};
        clEnqueueCopyBufferToImage(cl.queue(), in.mem, img, 0, origin, region, 0,0,0);
        Buf out{alloc((size_t)Cout*Tout), Cout, Tout};
        cl_mem b=bias(base); int has_b=b?1:0; cl_mem nullb=b?b:w;
        cl_kernel k=kern("conv1d_img");
        clSetKernelArg(k,0,sizeof(cl_mem),&img); clSetKernelArg(k,1,sizeof(cl_mem),&w);
        clSetKernelArg(k,2,sizeof(cl_mem),&nullb); clSetKernelArg(k,3,sizeof(cl_mem),&out.mem);
        clSetKernelArg(k,4,4,&in.C); clSetKernelArg(k,5,4,&in.T); clSetKernelArg(k,6,4,&Cout);
        clSetKernelArg(k,7,4,&Tout); clSetKernelArg(k,8,4,&K); clSetKernelArg(k,9,4,&stride);
        clSetKernelArg(k,10,4,&dil); clSetKernelArg(k,11,4,&pad); clSetKernelArg(k,12,4,&has_b);
        size_t g[2]={(size_t)((Cout+NC-1)/NC),(size_t)((Tout+TILE_T-1)/TILE_T)};
        if(lx&&ly){ size_t loc[2]={lx,ly}; g[0]=((g[0]+lx-1)/lx)*lx; g[1]=((g[1]+ly-1)/ly)*ly; run_kernel(k,2,g,loc,"conv1d_img"); }
        else run_kernel(k,2,g,"conv1d_img");
        if(b) rel(b); clReleaseMemObject(img); return out;
    }
    // conv1d via im2col + 2-level-tiled GEMM microkernel (register × local blocking).
    Buf conv1d_tiled(const Buf& in, const std::string& base, int Cout, int K, int stride, int dil, int pad){
        cl_mem w=weight(base, Cout, in.C, K);
        int Tout=(in.T + 2*pad - dil*(K-1) - 1)/stride + 1;
        int Kd=in.C*K;
        Buf col{alloc((size_t)Kd*Tout), Kd, Tout};
        cl_kernel ik=kern("im2col");
        clSetKernelArg(ik,0,sizeof(cl_mem),&in.mem); clSetKernelArg(ik,1,sizeof(cl_mem),&col.mem);
        clSetKernelArg(ik,2,4,&in.C); clSetKernelArg(ik,3,4,&in.T); clSetKernelArg(ik,4,4,&Tout);
        clSetKernelArg(ik,5,4,&K); clSetKernelArg(ik,6,4,&stride); clSetKernelArg(ik,7,4,&dil); clSetKernelArg(ik,8,4,&pad);
        size_t ig[2]={(size_t)Kd,(size_t)Tout}; run_kernel(ik,2,ig,"im2col");
        Buf out{alloc((size_t)Cout*Tout), Cout, Tout};
        cl_kernel k=kern("gemm_tiled");
        clSetKernelArg(k,0,sizeof(cl_mem),&w); clSetKernelArg(k,1,sizeof(cl_mem),&col.mem); clSetKernelArg(k,2,sizeof(cl_mem),&out.mem);
        clSetKernelArg(k,3,4,&Cout); clSetKernelArg(k,4,4,&Tout); clSetKernelArg(k,5,4,&Kd);
        size_t loc[2]={8,8};                              // BM/MR=8, BN/NR=8
        size_t g[2]={(size_t)(((Cout+31)/32)*8), (size_t)(((Tout+63)/64)*8)};
        run_kernel(k,2,g,loc,"gemm_tiled");
        cl_mem b=bias(base);
        if(b){ cl_kernel bk=kern("add_bias"); clSetKernelArg(bk,0,sizeof(cl_mem),&out.mem); clSetKernelArg(bk,1,sizeof(cl_mem),&b);
               clSetKernelArg(bk,2,4,&Cout); clSetKernelArg(bk,3,4,&Tout);
               size_t bg[2]={(size_t)Cout,(size_t)Tout}; run_kernel(bk,2,bg,"add_bias"); rel(b); }
        rel(col.mem); return out;
    }
    // conv1d with weights staged in __local (#4/#6). Requires local {lx,ly}.
    Buf conv1d_local(const Buf& in, const std::string& base, int Cout, int K, int stride, int dil, int pad){
        cl_mem w=weight(base, Cout, in.C, K); cl_mem b=bias(base);
        int Tout=(in.T + 2*pad - dil*(K-1) - 1)/stride + 1;
        Buf out{alloc((size_t)Cout*Tout), Cout, Tout};
        cl_kernel k=kern("conv1d_local"); int has_b=b?1:0; cl_mem nullb=b?b:in.mem;
        clSetKernelArg(k,0,sizeof(cl_mem),&in.mem); clSetKernelArg(k,1,sizeof(cl_mem),&w);
        clSetKernelArg(k,2,sizeof(cl_mem),&nullb); clSetKernelArg(k,3,sizeof(cl_mem),&out.mem);
        clSetKernelArg(k,4,(size_t)lx*in.C*K*2,nullptr);                  // __local wl (LCH*Cin*K halfs)
        clSetKernelArg(k,5,4,&in.C); clSetKernelArg(k,6,4,&in.T); clSetKernelArg(k,7,4,&Cout);
        clSetKernelArg(k,8,4,&Tout); clSetKernelArg(k,9,4,&K); clSetKernelArg(k,10,4,&stride);
        clSetKernelArg(k,11,4,&dil); clSetKernelArg(k,12,4,&pad); clSetKernelArg(k,13,4,&has_b);
        size_t loc[2]={lx,ly};
        size_t g[2]={(size_t)Cout,(size_t)((Tout+TILE_T-1)/TILE_T)};
        g[0]=((g[0]+lx-1)/lx)*lx; g[1]=((g[1]+ly-1)/ly)*ly;
        run_kernel(k,2,g,loc,"conv1d_local");
        if(b) rel(b); return out;
    }
    // ConvTranspose1d (naive, 1 output/thread) — microbench baseline only.
    Buf convT_naive(const Buf& in, const std::string& base, int Cout, int K, int stride, int pad){
        cl_mem w=weight(base, in.C, Cout, K); cl_mem b=bias(base);
        int Tout=(in.T-1)*stride - 2*pad + K;
        Buf out{alloc((size_t)Cout*Tout), Cout, Tout};
        cl_kernel k=kern("conv_transpose1d"); int has_b=b?1:0; cl_mem nullb=b?b:in.mem;
        clSetKernelArg(k,0,sizeof(cl_mem),&in.mem); clSetKernelArg(k,1,sizeof(cl_mem),&w);
        clSetKernelArg(k,2,sizeof(cl_mem),&nullb); clSetKernelArg(k,3,sizeof(cl_mem),&out.mem);
        clSetKernelArg(k,4,4,&in.C); clSetKernelArg(k,5,4,&in.T); clSetKernelArg(k,6,4,&Cout);
        clSetKernelArg(k,7,4,&Tout); clSetKernelArg(k,8,4,&K); clSetKernelArg(k,9,4,&stride);
        clSetKernelArg(k,10,4,&pad); clSetKernelArg(k,11,4,&has_b);
        size_t g[2]={(size_t)Cout,(size_t)Tout}; run_kernel(k,2,g,"conv_transpose1d");
        if(b) rel(b); return out;
    }
    // Model-facing convT: dispatch by kernel size. Tiled+phase wins for heavy taps
    // (K=16 ups); the original wins for light K=4 ups (tiling overhead not amortized).
    Buf convT(const Buf& in, const std::string& base, int Cout, int K, int stride, int pad){
        return (K >= 8) ? convT_tiled(in,base,Cout,K,stride,pad)
                        : convT_naive(in,base,Cout,K,stride,pad);
    }
    // ConvTranspose1d — register-tiled over output channels + phase-stepping (#15).
    static const int TILE_CO = 8;
    // conv_transpose1d_opt work-group. {2,64}: Adreno 620 sweep winner (1.53× microbench /
    // ~1.74× on the tiled path vs driver-default, cos 1.0000). Override via NNOPT_TLX/TLY.
    size_t tlx=2, tly=64;   // work-group for convT (0,0 = driver default)
    Buf convT_tiled(const Buf& in, const std::string& base, int Cout, int K, int stride, int pad){
        cl_mem w=weight(base, in.C, Cout, K); cl_mem b=bias(base);
        int Tout=(in.T-1)*stride - 2*pad + K;
        Buf out{alloc((size_t)Cout*Tout), Cout, Tout};
        cl_kernel k=kern("conv_transpose1d_opt"); int has_b=b?1:0; cl_mem nullb=b?b:in.mem;
        clSetKernelArg(k,0,sizeof(cl_mem),&in.mem); clSetKernelArg(k,1,sizeof(cl_mem),&w);
        clSetKernelArg(k,2,sizeof(cl_mem),&nullb); clSetKernelArg(k,3,sizeof(cl_mem),&out.mem);
        clSetKernelArg(k,4,4,&in.C); clSetKernelArg(k,5,4,&in.T); clSetKernelArg(k,6,4,&Cout);
        clSetKernelArg(k,7,4,&Tout); clSetKernelArg(k,8,4,&K); clSetKernelArg(k,9,4,&stride);
        clSetKernelArg(k,10,4,&pad); clSetKernelArg(k,11,4,&has_b);
        size_t g[2]={(size_t)((Cout+TILE_CO-1)/TILE_CO),(size_t)Tout};
        if(tlx&&tly){ size_t loc[2]={tlx,tly}; g[0]=((g[0]+tlx-1)/tlx)*tlx; g[1]=((g[1]+tly-1)/tly)*tly; run_kernel(k,2,g,loc,"conv_transpose1d_opt"); }
        else run_kernel(k,2,g,"conv_transpose1d_opt");
        if(b) rel(b); return out;
    }
    void dump(const Buf& b, const std::string& name){
        auto v=download(b.mem, b.n());
        std::string p="layer_dumps/"+name+"__pass0.bin"; FILE* f=fopen(p.c_str(),"wb");
        std::vector<store_t> h(v.size()); for(size_t i=0;i<v.size();++i) h[i]=st_enc(v[i]);
        fwrite(h.data(),sizeof(store_t),h.size(),f); fclose(f);
        std::string mp=p+".meta.json"; FILE* mf=fopen(mp.c_str(),"wb");
        fprintf(mf,"{\"dtype\":\"%s\"}", STORE_DTYPE); fclose(mf);
    }

    // fused_add_tanh_sigmoid_multiply with cond sliced from gcond [2H*n_layers,1]
    // at channel goff, broadcast over T. x_in: [2H,T] → out: [H,T].
    Buf fused(const Buf& x_in, const Buf& gcond, int goff, int H, int T){
        Buf out{alloc((size_t)H*T), H, T};
        cl_kernel k=kern("fused_tanh_sigmoid_bcast");
        clSetKernelArg(k,0,sizeof(cl_mem),&x_in.mem); clSetKernelArg(k,1,sizeof(cl_mem),&gcond.mem);
        clSetKernelArg(k,2,sizeof(cl_mem),&out.mem); clSetKernelArg(k,3,4,&H); clSetKernelArg(k,4,4,&T);
        clSetKernelArg(k,5,4,&goff);
        size_t g[2]={(size_t)H,(size_t)T}; run_kernel(k,2,g,"fused");
        return out;
    }
    // dst[dst_choff + c, :] += src[src_choff + c, :]   for c in 0..H-1
    void add_slice(Buf& dst, const Buf& src, int dst_choff, int src_choff, int H, int T){
        cl_kernel k=kern("add_channel_slice");
        clSetKernelArg(k,0,sizeof(cl_mem),&dst.mem); clSetKernelArg(k,1,sizeof(cl_mem),&src.mem);
        clSetKernelArg(k,2,4,&dst_choff); clSetKernelArg(k,3,4,&src_choff);
        clSetKernelArg(k,4,4,&H); clSetKernelArg(k,5,4,&T);
        size_t g[2]={(size_t)H,(size_t)T}; run_kernel(k,2,g,"add_slice");
       
    }
    // #14: no clFinish on copies — the in-order queue preserves ordering vs dependent
    // kernels; only blocking reads (download/dump) need to sync. Removes per-copy stalls.
    Buf clone(const Buf& a){ Buf o{alloc(a.n()),a.C,a.T}; clEnqueueCopyBuffer(cl.queue(),a.mem,o.mem,0,0,a.n()*sizeof(store_t),0,0,0); return o; }
    Buf slice_time(const Buf& a, int N){    // first N frames along T (all channels) → [C,N]
        Buf o{alloc((size_t)a.C*N), a.C, N};
        for(int c=0;c<a.C;++c)
            clEnqueueCopyBuffer(cl.queue(), a.mem, o.mem,
                (size_t)c*a.T*sizeof(store_t), (size_t)c*N*sizeof(store_t),
                (size_t)N*sizeof(store_t), 0,0,0);
        return o;
    }
    Buf slice(const Buf& a, int c0, int nC){ // channels [c0,c0+nC) → [nC,T]
        Buf o{alloc((size_t)nC*a.T), nC, a.T};
        clEnqueueCopyBuffer(cl.queue(),a.mem,o.mem,(size_t)c0*a.T*sizeof(store_t),0,(size_t)nC*a.T*sizeof(store_t),0,0,0);
        return o;
    }
    Buf concat(const Buf& a, const Buf& b){  // concat along channels → [a.C+b.C,T]
        Buf o{alloc((size_t)(a.C+b.C)*a.T), a.C+b.C, a.T};
        clEnqueueCopyBuffer(cl.queue(),a.mem,o.mem,0,0,a.n()*sizeof(store_t),0,0,0);
        clEnqueueCopyBuffer(cl.queue(),b.mem,o.mem,0,a.n()*sizeof(store_t),b.n()*sizeof(store_t),0,0,0);
        return o;
    }
    Buf flip(const Buf& a){
        Buf o{alloc(a.n()), a.C, a.T};
        cl_kernel k=kern("flip_channels");
        clSetKernelArg(k,0,sizeof(cl_mem),&a.mem); clSetKernelArg(k,1,sizeof(cl_mem),&o.mem);
        clSetKernelArg(k,2,4,&a.C); clSetKernelArg(k,3,4,&a.T);
        size_t g[2]={(size_t)a.C,(size_t)a.T}; run_kernel(k,2,g,"flip");
        return o;
    }
    void lrelu(Buf& a, float slope){
        int N=(int)a.n(); cl_kernel k=kern("leaky_relu");
        clSetKernelArg(k,0,sizeof(cl_mem),&a.mem); clSetKernelArg(k,1,4,&N); clSetKernelArg(k,2,4,&slope);
        size_t g=(size_t)N; run_kernel(k,1,&g,"leaky_relu");
    }
    void tanh_(Buf& a){
        int N=(int)a.n(); cl_kernel k=kern("tanh_inplace");
        clSetKernelArg(k,0,sizeof(cl_mem),&a.mem); clSetKernelArg(k,1,4,&N);
        size_t g=(size_t)N; run_kernel(k,1,&g,"tanh");
    }
    Buf addbuf(const Buf& a, const Buf& b){  // out = a + b (same shape) → new buffer
        Buf o{alloc(a.n()), a.C, a.T}; int N=(int)a.n();
        cl_kernel k=kern("add");
        clSetKernelArg(k,0,sizeof(cl_mem),&a.mem); clSetKernelArg(k,1,sizeof(cl_mem),&b.mem);
        clSetKernelArg(k,2,sizeof(cl_mem),&o.mem); clSetKernelArg(k,3,4,&N);
        size_t g=(size_t)N; run_kernel(k,1,&g,"add");
        return o;
    }
    void scale_(Buf& a, float s){
        int N=(int)a.n(); cl_kernel k=kern("scale");
        clSetKernelArg(k,0,sizeof(cl_mem),&a.mem); clSetKernelArg(k,1,4,&N); clSetKernelArg(k,2,4,&s);
        size_t g=(size_t)N; run_kernel(k,1,&g,"scale");
    }
    // x[c,t] += cond[c,0] (broadcast time-invariant cond over T)
    Buf add_cond_bcast(Buf& x, const Buf& cond){
        cl_kernel k=kern("add_cond_broadcast");
        clSetKernelArg(k,0,sizeof(cl_mem),&x.mem); clSetKernelArg(k,1,sizeof(cl_mem),&cond.mem);
        clSetKernelArg(k,2,4,&x.C); clSetKernelArg(k,3,4,&x.T);
        size_t g[2]={(size_t)x.C,(size_t)x.T}; run_kernel(k,2,g,"add_cond");
        return x;
    }

    // WaveNet (modules.py::WN): x [H,T], g [gin,1] → out [H,T]. Shared by enc_q & flow.
    //   wkey  = weight-key prefix with DOTS    (e.g. "enc_q.enc")
    //   dname = dump-name  prefix with UNDERSCORES (e.g. "enc_q_enc")
    // dilation_rate is 1 for OpenVoice (verified from weight shapes), so dilation=1.
    Buf wn(const Buf& x_in, cl_mem g_src, int H, int gin, int n_layers, int K,
           const std::string& wkey, const std::string& dname, bool dump_layers){
        Buf gbuf{g_src, gin, 1};
        Buf gcond = conv1d(gbuf, wkey+".cond_layer", 2*H*n_layers, 1, 1, 1, 0);   // [2H*n_layers,1]
        Buf x{alloc(x_in.n()), x_in.C, x_in.T};
        clEnqueueCopyBuffer(cl.queue(),x_in.mem,x.mem,0,0,x_in.n()*sizeof(store_t),0,0,0);   // #14: no clFinish
        Buf output{alloc((size_t)H*x.T), H, x.T};
        { std::vector<float> z((size_t)H*x.T,0.f); upload(output.mem,z); }
        int pad=(K-1)/2;   // dilation=1
        for(int i=0;i<n_layers;++i){
            Buf x_inl = conv1d(x, wkey+".in_layers."+std::to_string(i), 2*H, K, 1, 1, pad); // [2H,T]
            if(dump_layers) dump(x_inl, dname+"_in_layers_"+std::to_string(i));
            Buf acts = fused(x_inl, gcond, i*2*H, H, x.T);                                   // [H,T]
            int rsC = (i<n_layers-1) ? 2*H : H;
            Buf rs = conv1d(acts, wkey+".res_skip_layers."+std::to_string(i), rsC, 1, 1, 1, 0);
            if(dump_layers) dump(rs, dname+"_res_skip_layers_"+std::to_string(i));
            if(i<n_layers-1){
                add_slice(x,      rs, 0, 0, H, x.T);   // x[:H]      += rs[:H]
                add_slice(output, rs, 0, H, H, x.T);   // output[:H] += rs[H:2H]
            } else {
                add_slice(output, rs, 0, 0, H, x.T);   // output[:H] += rs[:H]
            }
            rel(x_inl.mem); rel(acts.mem); rel(rs.mem);
        }
        rel(x.mem); rel(gcond.mem);
        return output;
    }

    // ── components: declared here, DEFINED in flow.cpp / dec.cpp ─────────────
    Buf coupling(const Buf& x, cl_mem g, int gin, const std::string& wkey, bool reverse);  // flow.cpp
    Buf flow(const Buf& x_in, cl_mem g, int gin, bool reverse);                             // flow.cpp
    Buf resblock(const Buf& x_in, const std::string& wkey, int K);                          // dec.cpp
    Buf dec(const Buf& z, cl_mem g, int gin, bool dump_stages);                             // dec.cpp
};

// ── shared harness helpers ──────────────────────────────────────────────────
inline Buf load_ref(Engine& E, const char* path, int C){   // fp32 ref dump → [C, len/C]
    FILE* f=fopen(path,"rb"); if(!f){ printf("missing %s\n", path); return Buf{}; }
    fseek(f,0,SEEK_END); long sz=ftell(f); fseek(f,0,SEEK_SET);
    int n=sz/4; std::vector<float> v(n); fread(v.data(),4,n,f); fclose(f);
    Buf b{E.alloc(n), C, n/C}; E.upload(b.mem, v); return b;
}
inline cl_mem load_g(Engine& E, const char* path){
    FILE* f=fopen(path,"rb"); if(!f){ printf("missing %s\n", path); return nullptr; }
    std::vector<float> g(256); fread(g.data(),4,256,f); fclose(f);
    cl_mem m=E.alloc(256); E.upload(m,g); return m;
}
// zero_g=True (converter config): enc_q & dec receive g=ZEROS (only flow uses the embedding).
inline cl_mem zero_g(Engine& E){
    std::vector<float> z(256, 0.0f); cl_mem m=E.alloc(256); E.upload(m,z); return m;
}
inline int boot(OpenCLContext& cl, Weights& W){
    if(!cl.initialize(0,0)){ printf("no cl\n"); return 1; }
    if(!W.load("weights/model.fp16.bin","weights/model.fp16.meta.json", cl.context())){ printf("no weights\n"); return 1; }
    return 0;
}
