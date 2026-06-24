// conv microbenchmark — naive vs optimized for conv1d AND conv_transpose1d, on the
// hot decoder shapes (5s-scale). Synthetic deterministic input + real folded weights.
// Reports cos(naive,opt) (must be 1.0000 — no regression) and GPU-timer ms + speedup.
#include "engine.h"
#include <algorithm>

static void fill_rand(std::vector<float>& v, uint32_t seed){
    uint32_t s=seed; for(size_t i=0;i<v.size();++i){ s=s*1664525u+1013904223u; v[i]=((s>>9)&0x7fffffu)/4194304.0f-1.0f; }
}
static double cosv(const std::vector<float>& a, const std::vector<float>& b){
    size_t n=std::min(a.size(),b.size()); double dot=0,na=0,nb=0;
    for(size_t i=0;i<n;++i){ dot+=a[i]*b[i]; na+=a[i]*a[i]; nb+=b[i]*b[i]; }
    return dot/(std::sqrt(na)*std::sqrt(nb)+1e-12);
}

int run_hwprobe(){
    OpenCLContext cl; Weights W; if(boot(cl,W)) return 1; Engine E(cl,W); E.profile=true;
    char name[256]={0}; cl_uint cu=0; cl_uint mhz=0;
    clGetDeviceInfo(cl.device(), CL_DEVICE_NAME, sizeof(name), name, 0);
    clGetDeviceInfo(cl.device(), CL_DEVICE_MAX_COMPUTE_UNITS, sizeof(cu), &cu, 0);
    clGetDeviceInfo(cl.device(), CL_DEVICE_MAX_CLOCK_FREQUENCY, sizeof(mhz), &mhz, 0);
    printf("device: %s | compute_units=%u | max_clock=%u MHz\n", name, cu, mhz);
    cl_bool imgsup=0; size_t iw=0,ih=0; cl_ulong lmem=0;
    clGetDeviceInfo(cl.device(), CL_DEVICE_IMAGE_SUPPORT, sizeof(imgsup), &imgsup, 0);
    clGetDeviceInfo(cl.device(), CL_DEVICE_IMAGE2D_MAX_WIDTH, sizeof(iw), &iw, 0);
    clGetDeviceInfo(cl.device(), CL_DEVICE_IMAGE2D_MAX_HEIGHT, sizeof(ih), &ih, 0);
    clGetDeviceInfo(cl.device(), CL_DEVICE_LOCAL_MEM_SIZE, sizeof(lmem), &lmem, 0);
    printf("image_support=%d  image2d_max=%zux%zu  local_mem=%lluKB\n", (int)imgsup, iw, ih, (unsigned long long)(lmem>>10));
    { std::vector<char> ext(8192,0); clGetDeviceInfo(cl.device(), CL_DEVICE_EXTENSIONS, ext.size(), ext.data(), 0);
      std::string e(ext.data());
      printf("ALL EXT: %s\n", e.c_str()); const char* want[]={"dot_product","integer_dot","int8","dp4a","cl_qcom_dot"};
      printf("extensions of interest:");
      for(auto w: want){ if(e.find(w)!=std::string::npos) printf(" [%s]", w); }
      printf("\n"); }

    // ---- peak compute (FMA chain, no memory) ----
    {
        size_t G = (size_t)cu*4096; int iters=20000;
        cl_mem o=E.alloc(64); cl_kernel k=E.kern("compute_peak");
        clSetKernelArg(k,0,sizeof(cl_mem),&o); clSetKernelArg(k,1,4,&iters);
        E.reset_gpu_timer(); E.run_kernel(k,1,&G,"compute_peak"); double ms=E.gpu_ms();
        double flop=(double)G*iters*128.0;                // 8 float8-FMA/iter × 8 lanes × 2 FLOP
        printf("peak compute : %.1f GFLOP/s  (%.1f ms for %.1f GFLOP)\n", flop/ms/1e6, ms, flop/1e9);
        E.rel(o);
    }
    // ---- peak bandwidth (read 2 + write 1, fp16) ----
    {
        int N=32*1024*1024; std::vector<float> z(N,0.5f);
        cl_mem a=E.alloc(N), b=E.alloc(N), o=E.alloc(N); E.upload(a,z); E.upload(b,z);
        cl_kernel k=E.kern("bw_copy");
        clSetKernelArg(k,0,sizeof(cl_mem),&a); clSetKernelArg(k,1,sizeof(cl_mem),&b);
        clSetKernelArg(k,2,sizeof(cl_mem),&o); clSetKernelArg(k,3,4,&N);
        size_t G=(size_t)N;
        E.reset_gpu_timer(); E.run_kernel(k,1,&G,"bw_copy"); double ms=E.gpu_ms();
        double bytes=(double)N*6.0;                        // 2 read + 1 write, 2 B each
        printf("peak bandwidth: %.1f GB/s  (%.1f ms for %.0f MB)\n", bytes/ms/1e6, ms, bytes/1e6);
        E.rel(a); E.rel(b); E.rel(o);
    }
    return 0;
}

int run_bench(){
    OpenCLContext cl; Weights W; if(boot(cl,W)) return 1; Engine E(cl,W);
    E.profile=true;                       // bench always uses the GPU timer

    // conv1d resblock shapes (square C→C), one per decoder level.
    struct C1{ const char* key; int C,K,T,dil; };
    C1 c1[]={ {"dec.resblocks.2.convs1.2",256,11,3440,5},
              {"dec.resblocks.5.convs1.0",128, 3,27520,1},
              {"dec.resblocks.8.convs1.1", 64, 7,55040,1},
              {"dec.resblocks.11.convs1.2",32,11,110080,5} };
    printf("== conv1d ==  %-26s %8s %9s %9s %9s %8s  %s\n","layer","T","naive","vec2d","image","image↑","cos v/i");
    double tn=0,tv=0,ti=0;
    for(auto& s: c1){
        std::vector<float> in((size_t)s.C*s.T); fill_rand(in,12345u+s.K+s.T);
        Buf x{E.alloc(in.size()),s.C,s.T}; E.upload(x.mem,in); int pad=s.dil*(s.K-1)/2;
        E.reset_gpu_timer(); Buf a=E.conv1d_naive(x,s.key,s.C,s.K,1,s.dil,pad); E.finish(); double a_ms=E.gpu_ms();
        E.reset_gpu_timer(); Buf b=E.conv1d_2d   (x,s.key,s.C,s.K,1,s.dil,pad); E.finish(); double v_ms=E.gpu_ms();
        E.reset_gpu_timer(); Buf c=E.conv1d_img  (x,s.key,s.C,s.K,1,s.dil,pad); E.finish(); double i_ms=E.gpu_ms();
        auto va=E.download(a.mem,a.n()), vb=E.download(b.mem,b.n()), vc=E.download(c.mem,c.n());
        printf("              %-26s %8d %9.1f %9.1f %9.1f %7.2fx  %.4f/%.4f\n",
               s.key,s.T,a_ms,v_ms,i_ms,v_ms/i_ms,cosv(va,vb),cosv(va,vc));
        tn+=a_ms; tv+=v_ms; ti+=i_ms; E.rel(x.mem); E.rel(a.mem); E.rel(b.mem); E.rel(c.mem);
    }
    printf("              %-26s %8s %9.1f %9.1f %9.1f %7.2fx\n","TOTAL","",tn,tv,ti,tv/ti);

    // conv_transpose1d ups shapes.
    struct CT{ const char* key; int Cin,Cout,K,Tin,stride,pad; };
    CT ct[]={ {"dec.ups.0",512,256,16,  430,8,4},
              {"dec.ups.1",256,128,16, 3440,8,4},
              {"dec.ups.2",128, 64, 4,27520,2,1},
              {"dec.ups.3", 64, 32, 4,55040,2,1} };
    printf("== convT  ==  %-26s %8s %10s %10s %8s %8s\n","layer","Tin","naive ms","opt ms","speedup","cos");
    double tnt=0,tot=0;
    for(auto& s: ct){
        std::vector<float> in((size_t)s.Cin*s.Tin); fill_rand(in,777u+s.K+s.Tin);
        Buf x{E.alloc(in.size()),s.Cin,s.Tin}; E.upload(x.mem,in);
        E.reset_gpu_timer(); Buf a=E.convT_naive(x,s.key,s.Cout,s.K,s.stride,s.pad); double a_ms=E.gpu_ms();
        E.reset_gpu_timer(); Buf b=E.convT      (x,s.key,s.Cout,s.K,s.stride,s.pad); double b_ms=E.gpu_ms();
        printf("              %-26s %8d %10.1f %10.1f %7.2fx %8.4f\n", s.key,s.Tin,a_ms,b_ms,a_ms/b_ms,
               cosv(E.download(a.mem,a.n()),E.download(b.mem,b.n())));
        tnt+=a_ms; tot+=b_ms; E.rel(x.mem); E.rel(a.mem); E.rel(b.mem);
    }
    printf("              %-26s %8s %10.1f %10.1f %7.2fx\n","TOTAL","",tnt,tot,tnt/tot);
    return 0;
}
