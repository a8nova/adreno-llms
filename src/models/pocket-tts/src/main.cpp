// main.cpp — pocket-tts hand port driver.
//
// Phase 1 (current): per-op validation harness. Runs ONE named op on a
// reference input tensor and writes the output, so scripts/cos_check.py can
// compare against reference/layers/<node>_output.bin (predecessor-output-as-
// input isolation, see .nnport/PORT_SPEC.md §6). Grows into the full
// text→waveform pipeline as ops land.
//
// Usage:
//   pocket_tts_inference_fp16 <weights.bin> <weights.meta.json> <op> <in.bin> <out.bin> [int args...]
//
// All .bin tensors are little-endian float32 (matching the reference dumps and
// cos_check.py). The harness uploads f32→fp16, runs on device, downloads fp16→f32.
#include "opencl_context.h"
#include "weights.h"
#include "gpu_ops.h"
#include "utils.h"
#include "pipeline.h"
#include "tokenizer.h"
#include "debug_utils.h"
#include "profiler.h"
#include <iostream>
#include <cmath>
#include <cstdint>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

static std::vector<float> read_f32(const std::string& path) {
    FILE* f = fopen(path.c_str(), "rb");
    if (!f) { fprintf(stderr, "ERROR: cannot open %s\n", path.c_str()); return {}; }
    fseek(f, 0, SEEK_END); long sz = ftell(f); fseek(f, 0, SEEK_SET);
    std::vector<float> v(sz / sizeof(float));
    size_t rd = fread(v.data(), sizeof(float), v.size(), f);
    (void)rd; fclose(f);
    return v;
}
static void write_f32(const std::string& path, const std::vector<float>& v) {
    FILE* f = fopen(path.c_str(), "wb");
    if (!f) { fprintf(stderr, "ERROR: cannot write %s\n", path.c_str()); return; }
    fwrite(v.data(), sizeof(float), v.size(), f);
    fclose(f);
    fprintf(stderr, "WROTE %s (%zu floats)\n", path.c_str(), v.size());
}

int main(int argc, char** argv) {
    // App entrypoint (the host process spawns `lib<key>.so --serve` from a workdir that
    // holds weights/ + kernels/). Default the paths to cwd so no args are needed.
    std::string wbin, wmeta, op, inpath, outpath;
    if (argc >= 2 && std::string(argv[1]) == "--serve") {
        wbin = "weights/model.fp16.bin"; wmeta = "weights/model.fp16.meta.json";
        op = "serve"; inpath = "weights/tokenizer_vocab.bin"; outpath = "";
    } else {
        if (argc < 6) {
            fprintf(stderr, "usage: %s <weights.bin> <weights.meta.json> <op> <in.bin> <out.bin> [args...]\n", argv[0]);
            fprintf(stderr, "   or: %s --serve   (app mode: reads weights/ + kernels/ from cwd)\n", argv[0]);
            return 2;
        }
        wbin = argv[1]; wmeta = argv[2]; op = argv[3]; inpath = argv[4]; outpath = argv[5];
    }
    std::vector<int> a;
    for (int i = 6; i < argc; ++i) a.push_back(atoi(argv[i]));

    using _clk = std::chrono::steady_clock;
    auto _ms_since = [](std::chrono::steady_clock::time_point t) {
        return std::chrono::duration<double, std::milli>(_clk::now() - t).count();
    };

    auto _t_init = _clk::now();
    OpenCLContext ctx;
    if (!ctx.initialize()) { fprintf(stderr, "ERROR: OpenCL init failed\n"); return 1; }
    fprintf(stderr, "PHASE opencl_init_ms: %.1f\n", _ms_since(_t_init));
    fprintf(stderr, "OpenCL device: %s\n", ctx.device_name().c_str());
    {   // image-path feasibility (for texture-weight GEMV)
        cl_device_id dev = ctx.device();
        cl_bool img = CL_FALSE; size_t maxbuf = 0, w2d = 0, h2d = 0;
        clGetDeviceInfo(dev, CL_DEVICE_IMAGE_SUPPORT, sizeof(img), &img, nullptr);
        clGetDeviceInfo(dev, CL_DEVICE_IMAGE_MAX_BUFFER_SIZE, sizeof(maxbuf), &maxbuf, nullptr);
        clGetDeviceInfo(dev, CL_DEVICE_IMAGE2D_MAX_WIDTH, sizeof(w2d), &w2d, nullptr);
        clGetDeviceInfo(dev, CL_DEVICE_IMAGE2D_MAX_HEIGHT, sizeof(h2d), &h2d, nullptr);
        fprintf(stderr, "IMGINFO support=%d image1d_buffer_max=%zu image2d_max=%zux%zu\n",
                (int)img, maxbuf, w2d, h2d);
    }

    auto _t_load = _clk::now();
    Weights weights;
    if (!weights.load(wbin, wmeta, ctx.context())) { fprintf(stderr, "ERROR: weights load failed\n"); return 1; }
    fprintf(stderr, "PHASE weight_load_ms: %.1f\n", _ms_since(_t_load));
    fprintf(stderr, "weights loaded.\n");

    auto _t_build = _clk::now();
    GpuOps g(ctx, weights);
    if (!g.init()) { fprintf(stderr, "ERROR: gpu_ops init failed\n"); return 1; }
    fprintf(stderr, "PHASE program_build_ms: %.1f\n", _ms_since(_t_build));

    // ── Recordable-queue probe: record K cheap dispatches, replay N×, compare to
    //    live enqueue. Confirms cl_qcom_recordable_queues works + the per-dispatch
    //    cost ratio on THIS binary before we restructure decode to record/replay.
    if (op == "recordprobe") {
        if (!ctx.has_recordable_queues()) { fprintf(stderr, "RECORDPROBE: recordable queues NOT available\n"); return 1; }
        cl_command_queue recq = ctx.create_recordable_queue();
        if (!recq) { fprintf(stderr, "RECORDPROBE: create_recordable_queue failed\n"); return 1; }
        const int K = 64, N = 200;
        cl_mem a = g.alloc(4096), b = g.alloc(4096);
        cl_kernel k = g.kernel("copy_buf");
        int n4096 = 4096; size_t gws = 4096;
        clSetKernelArg(k, 0, sizeof(cl_mem), &a);
        clSetKernelArg(k, 1, sizeof(cl_mem), &b);
        clSetKernelArg(k, 2, sizeof(int), &n4096);
        // CORRECTNESS: dependent CHAIN a→t1→t2→b with FRESH kernels per dispatch
        // (mirrors the backbone: many dispatches, data deps, distinct kernel objects).
        {
            int nn=4096; size_t g1=4096;
            std::vector<float> av(4096); for (int i=0;i<4096;++i) av[i]=(float)(i%97)*0.013f;
            cl_mem ca=g.upload(av), t1=g.alloc(4096), t2=g.alloc(4096), bb=g.alloc(4096);
            cl_int ke;
            cl_kernel k1=clCreateKernel(g.prog(),"copy_buf",&ke);
            cl_kernel k2=clCreateKernel(g.prog(),"copy_buf",&ke);
            cl_kernel k3=clCreateKernel(g.prog(),"copy_buf",&ke);
            clSetKernelArg(k1,0,sizeof(cl_mem),&ca); clSetKernelArg(k1,1,sizeof(cl_mem),&t1); clSetKernelArg(k1,2,sizeof(int),&nn);
            clSetKernelArg(k2,0,sizeof(cl_mem),&t1); clSetKernelArg(k2,1,sizeof(cl_mem),&t2); clSetKernelArg(k2,2,sizeof(int),&nn);
            clSetKernelArg(k3,0,sizeof(cl_mem),&t2); clSetKernelArg(k3,1,sizeof(cl_mem),&bb); clSetKernelArg(k3,2,sizeof(int),&nn);
            cl_recording_qcom rc = ctx.new_recording(recq);
            clEnqueueNDRangeKernel(recq,k1,1,nullptr,&g1,nullptr,0,nullptr,nullptr);
            clEnqueueNDRangeKernel(recq,k2,1,nullptr,&g1,nullptr,0,nullptr,nullptr);
            clEnqueueNDRangeKernel(recq,k3,1,nullptr,&g1,nullptr,0,nullptr,nullptr);
            ctx.end_recording(rc);
            ctx.enqueue_recording(ctx.queue(), rc, 0, nullptr);
            clFinish(ctx.queue());
            std::vector<float> bv = g.download(bb, 4096);
            double err=0; for (int i=0;i<4096;++i) err += std::abs(bv[i]-av[i]);
            fprintf(stderr, "RECORDPROBE chain a->t1->t2->b (3 fresh kernels): total_abs_err=%.5f (0=ok) b[10]=%.4f want=%.4f\n",
                    err, bv[10], av[10]);
            ctx.release_recording(rc); g.release(ca);
            clReleaseKernel(k1); clReleaseKernel(k2); clReleaseKernel(k3);
        }
        // CLBlast-recordability: route a GEMM onto the recordable queue, replay, verify
        // vs a live GEMM. If correct, the whole mimi (CLBlast) can be recorded — no
        // custom GEMMs needed. (If CLBlast uses non-NDRange enqueues, this breaks.)
        {
            int M=16, N=128, Kd=64;
            std::vector<float> Av((size_t)M*Kd), Wv((size_t)N*Kd);
            for (size_t i=0;i<Av.size();++i) Av[i]=std::sin(0.01f*i);
            for (size_t i=0;i<Wv.size();++i) Wv[i]=std::cos(0.013f*i);
            cl_mem A=g.upload(Av), W=g.upload(Wv), Cl=g.alloc((size_t)M*N), Cr=g.alloc((size_t)M*N);
            pytorch_linear(ctx.queue(), M, N, Kd, A, W, Cl); clFinish(ctx.queue());   // LIVE ref
            std::vector<float> live = g.download(Cl, (size_t)M*N);
            cl_recording_qcom rg = ctx.new_recording(recq);
            bool ok = pytorch_linear(recq, M, N, Kd, A, W, Cr);                        // route to recq
            cl_int er = ctx.end_recording(rg);
            ctx.enqueue_recording(ctx.queue(), rg, 0, nullptr); clFinish(ctx.queue());
            std::vector<float> rep = g.download(Cr, (size_t)M*N);
            double err=0; for (size_t i=0;i<rep.size();++i) err += std::abs(rep[i]-live[i]);
            fprintf(stderr, "RECORDPROBE CLBlast-record: pytorch_linear ok=%d end_rec=%d | replay-vs-live abs_err=%.4f C[5] rep=%.3f live=%.3f\n",
                    (int)ok, (int)er, err, rep[5], live[5]);
            ctx.release_recording(rg);
        }
        cl_recording_qcom rec = ctx.new_recording(recq);
        for (int i = 0; i < K; ++i)
            clEnqueueNDRangeKernel(recq, k, 1, nullptr, &gws, nullptr, 0, nullptr, nullptr);
        if (ctx.end_recording(rec) != CL_SUCCESS) { fprintf(stderr, "RECORDPROBE: end_recording failed\n"); return 1; }
        clFinish(ctx.queue());
        auto r0 = _clk::now();
        for (int i = 0; i < N; ++i) ctx.enqueue_recording(ctx.queue(), rec, 0, nullptr);
        clFinish(ctx.queue());
        double replay_ms = _ms_since(r0);
        clFinish(ctx.queue());
        auto l0 = _clk::now();
        for (int i = 0; i < N; ++i) for (int j = 0; j < K; ++j)
            clEnqueueNDRangeKernel(ctx.queue(), k, 1, nullptr, &gws, nullptr, 0, nullptr, nullptr);
        clFinish(ctx.queue());
        double live_ms = _ms_since(l0);
        fprintf(stderr, "RECORDPROBE: %dx%d dispatches | live=%.1fms (%.2fus/disp) replay=%.1fms (%.2fus/disp) | %.2fx cheaper\n",
                K, N, live_ms, live_ms*1000.0/(K*N), replay_ms, replay_ms*1000.0/(K*N), live_ms/replay_ms);
        ctx.release_recording(rec);
        return 0;
    }

    // ── End-to-end generation: text→waveform, all on device. inpath unused;
    //    outpath gets the fp32 waveform. argv[6]=n_frames (default 23). PORT_SPEC §1.
    if (op == "generate") {
        int n_frames = (argc > 6) ? atoi(argv[6]) : 23;
        float noise_std = (argc > 7) ? (float)atof(argv[7]) : 0.0f;
        // argv[8] = comma-separated SentencePiece token ids (e.g. "364,1143,..").
        // empty / "-1" ⇒ voice-only.
        std::vector<int> text_ids;
        if (argc > 8) {
            std::string ts = argv[8]; size_t p = 0;
            while (p < ts.size()) {
                size_t c = ts.find(',', p);
                std::string tok = ts.substr(p, c == std::string::npos ? std::string::npos : c - p);
                int v = atoi(tok.c_str());
                if (v >= 0) text_ids.push_back(v);
                if (c == std::string::npos) break; p = c + 1;
            }
        }
        // optional: argv[9]=voice file, argv[10]="kv" (v3 KV) else v1 audio_prompt.
        std::string vp = (argc > 9) ? argv[9] : "";
        bool vkv = (argc > 10) && std::string(argv[10]) == "kv";
        std::vector<float> wave = tts_generate(g, n_frames, noise_std, text_ids,
                                               nullptr, nullptr, nullptr, false, 0.0f, vp, vkv);
        write_f32(outpath, wave);
        clFinish(ctx.queue());
        KernelProfiler::dump_summary();   // per-kernel GPU breakdown when NNOPT_PROFILE=1
        fprintf(stderr, "OK: generated %zu samples (%.2f s @ 24kHz)\n", wave.size(), wave.size()/24000.0);
        return 0;
    }

    // ── tokenize: text (argv[6..], joined) → SentencePiece ids (validation). inpath = vocab.bin.
    if (op == "tokenize") {
        Tokenizer tok;
        if (!tok.load(inpath)) { fprintf(stderr, "ERROR: tokenizer load '%s'\n", inpath.c_str()); return 1; }
        std::string text;
        for (int i = 6; i < argc; ++i) { if (i > 6) text += " "; text += argv[i]; }
        std::vector<int> ids = tok.encode(text);
        printf("ids:");
        for (int id : ids) printf(" %d", id);
        printf("\n");
        return 0;
    }

    // ── serve: warm REPL. stdin = one line of text per utterance; emit per-utterance
    //    `POCKET_PCM_BEGIN <n> <sr>` (stderr) + n*2 bytes int16 LE PCM (stdout) + `POCKET_UTT_END`.
    //    inpath = tokenizer_vocab.bin. The voice (audio_prompt) is baked into the weights.
    if (op == "serve") {
        Tokenizer tok;
        if (!tok.load(inpath)) { fprintf(stderr, "ERROR: tokenizer load '%s'\n", inpath.c_str()); return 1; }
        const int   kSampleRate = 24000;
        const int   kMaxFrames  = 180;      // ~14.4 s cap (KV CAP=512 headroom)
        const float kNoiseStd   = 0.6f;
        const float kEosThresh  = 0.0f;
        std::string cur_voice = "";      // "" = baked voice; else a path
        bool cur_voice_kv = false;       // true ⇒ v3 KV file, else v1 audio_prompt
        fprintf(stderr, "ready.\n"); fflush(stderr);
        std::string line;
        while (std::getline(std::cin, line)) {
            // trim
            size_t b = line.find_first_not_of(" \t\r\n");
            if (b == std::string::npos) continue;
            size_t e = line.find_last_not_of(" \t\r\n");
            std::string text = line.substr(b, e - b + 1);
            // Voice select: "@voice <path> [kv]" — sets the voice for subsequent utterances.
            if (text.rfind("@voice", 0) == 0) {
                std::string rest = text.size() > 6 ? text.substr(7) : "";
                size_t sp = rest.find(' ');
                cur_voice = rest.substr(0, sp);
                cur_voice_kv = (sp != std::string::npos) && rest.substr(sp + 1).find("kv") != std::string::npos;
                fprintf(stderr, "voice set: '%s' kv=%d\n", cur_voice.c_str(), (int)cur_voice_kv); fflush(stderr);
                continue;
            }
            std::vector<int> ids = tok.encode(text);
            std::vector<float> wave = tts_generate(g, kMaxFrames, kNoiseStd, ids,
                                                   nullptr, nullptr, nullptr, /*stop_on_eos=*/true, kEosThresh,
                                                   cur_voice, cur_voice_kv);
            // fp32 [-1,1] → int16 LE PCM
            std::vector<int16_t> pcm(wave.size());
            for (size_t i = 0; i < wave.size(); ++i) {
                float v = wave[i]; if (v > 1.0f) v = 1.0f; else if (v < -1.0f) v = -1.0f;
                pcm[i] = (int16_t)lrintf(v * 32767.0f);
            }
            fprintf(stderr, "POCKET_PCM_BEGIN %zu %d\n", pcm.size(), kSampleRate); fflush(stderr);
            fwrite(pcm.data(), sizeof(int16_t), pcm.size(), stdout); fflush(stdout);
            fprintf(stderr, "POCKET_UTT_END\n"); fflush(stderr);
        }
        return 0;
    }

    std::vector<float> in = read_f32(inpath);
    if (in.empty()) return 1;
    size_t n = in.size();
    cl_mem x = g.upload(in);
    cl_mem y = nullptr;

    // ── genref: run generation with a DUMPED reference noise sequence (inpath =
    //    n_frames*32 f32), text tokens in argv[8]. Writes my per-frame c
    //    (my_c.bin), latent (my_latent.bin), and waveform (outpath) for a
    //    frame-by-frame cosine diff vs the real-model dump.
    if (op == "genref") {
        std::vector<float> noise_seq = read_f32(inpath);
        int n_frames = (int)(noise_seq.size() / 32);
        std::vector<int> tids;
        if (argc > 6) { std::string ts = argv[6]; size_t p = 0;
            while (p < ts.size()) { size_t cc = ts.find(',', p);
                tids.push_back(atoi(ts.substr(p, cc==std::string::npos?std::string::npos:cc-p).c_str()));
                if (cc==std::string::npos) break; p = cc+1; } }
        std::vector<float> c_out, lat_out;
        std::vector<float> wave = tts_generate(g, n_frames, 0.0f, tids, &noise_seq, &c_out, &lat_out);
        write_f32("my_c.bin", c_out);
        write_f32("my_latent.bin", lat_out);
        write_f32(outpath, wave);
        fprintf(stderr, "OK: genref %d frames\n", n_frames);
        return 0;
    }

    // ── mimicmp: streaming vs non-streaming decode of a VARYING latent seq.
    //    They must be identical; any diff = streaming-state bug. No reference needed.
    if (op == "mimicmp") {
        int N = (argc > 6) ? atoi(argv[6]) : 8;
        std::vector<float> lats(N * 32);
        for (int f = 0; f < N; ++f) for (int j = 0; j < 32; ++j)
            lats[f*32+j] = 0.8f * std::sin(0.17f*f + 0.31f*j + 0.5f);   // varying, non-DC
        std::vector<float> ws = mimi_decode_seq_vec(g, lats, N);
        std::vector<float> wn = mimi_decode_nonstream_pub(g, lats, N);
        size_t n = std::min(ws.size(), wn.size());
        double dot=0, na=0, nb=0, maxd=0;
        for (size_t i=0;i<n;++i){ dot+=ws[i]*wn[i]; na+=ws[i]*ws[i]; nb+=wn[i]*wn[i]; double d=std::abs(ws[i]-wn[i]); if(d>maxd)maxd=d; }
        fprintf(stderr, "MIMICMP streaming-vs-nonstreaming: n=%zu cos=%.6f maxdiff=%.5f\n", n, dot/(std::sqrt(na)*std::sqrt(nb)+1e-20), maxd);
        // per-frame cosine to localize WHERE they diverge
        for (int f=0; f<N; ++f){ double d2=0,a2=0,b2=0; for(int i=0;i<1920;++i){ float xa=ws[f*1920+i], xb=wn[f*1920+i]; d2+=xa*xb; a2+=xa*xa; b2+=xb*xb;} fprintf(stderr,"  frame %d cos=%.5f\n", f, d2/(std::sqrt(a2)*std::sqrt(b2)+1e-20)); }
        return 0;
    }

    // ── mimi decode the SAME latent N times (streaming) — knock test.
    if (op == "mimiseq") {
        std::vector<float> lat(in.begin(), in.begin() + 32);
        int nf = (argc > 6) ? atoi(argv[6]) : 10;
        std::vector<float> w = mimi_decode_seq(g, lat, nf);
        write_f32(outpath, w);
        fprintf(stderr, "OK: mimiseq %d frames → %zu samples\n", nf, w.size());
        return 0;
    }

    // ── mimi decode one latent: inpath = latent[32] → 1920-sample waveform.
    if (op == "mimiframe") {
        std::vector<float> lat(in.begin(), in.begin() + 32);
        std::vector<float> w = mimi_decode_one(g, lat);
        write_f32(outpath, w);
        fprintf(stderr, "OK: mimiframe → %zu samples\n", w.size());
        return 0;
    }

    // ── flow_net device test: inpath = c[1024] (++ optional noise[32]). If the
    //    input is 1056 floats, the trailing 32 are used as noise; else noise=0.
    //    Lets us validate against the real model with the recovered noise.
    if (op == "flownet") {
        std::vector<float> cvec(in.begin(), in.begin() + 1024);
        std::vector<float> nz(32, 0.0f);
        if (in.size() >= 1056) for (int j = 0; j < 32; ++j) nz[j] = in[1024 + j];
        cl_mem cbuf = g.upload(cvec);
        cl_mem noise = g.upload(nz);
        cl_mem out = g.flow_net(cbuf, 0.0f, 1.0f, noise);
        clFinish(ctx.queue());
        write_f32(outpath, g.download(out, 32));
        fprintf(stderr, "OK: flownet (noise %s)\n", in.size() >= 1056 ? "supplied" : "zero");
        return 0;
    }

    // ── Single transformer-layer device test: input [T,512] → mimi layer 0
    //    (d512,h8,dff2048,ctx250,layer_scale,offset=0,fresh KV) → dump. For
    //    device-vs-host validation of the shared transformer_layer.
    if (op == "tlayer") {
        int T = a.empty() ? 3 : a[0];
        cl_mem kc = g.alloc((size_t)T * 512), vc = g.alloc((size_t)T * 512);
        cl_mem out = g.transformer_layer(x, T, 512, 8, 2048, 0, 250,
                        "mimi.decoder_transformer.transformer.layers.0", true, kc, vc, 10000.0f);
        clFinish(ctx.queue());
        write_f32(outpath, g.download(out, (size_t)T * 512));
        fprintf(stderr, "OK: tlayer T=%d\n", T);
        return 0;
    }

    // ── Isolated SEANet validation: feed each op its REFERENCE predecessor
    //    output (not the chained output) → run the single op → dump. Removes
    //    error accumulation so a buggy op shows up directly. inpath = device dir
    //    holding the reference *_output.bin files.
    if (op == "isoseanet") {
        std::string D = inpath;   // dir prefix, e.g. "refs"
        auto load = [&](const char* nm) -> cl_mem {
            std::vector<float> v = read_f32(D + "/" + nm + "_output.bin");
            return v.empty() ? nullptr : g.upload(v);
        };
        auto dump = [&](const char* nm, cl_mem b, size_t cnt) {
            write_f32(std::string("layer_dumps/iso_") + nm + ".bin", g.download(b, cnt));
        };
        const std::string P = "mimi.decoder.model";
        cl_mem i0 = load("mimi_decoder_transformer");
        cl_mem o0 = g.conv1d_causal(i0, 512, 512, 16, 7, 1, P+".0.conv.weight", P+".0.conv.bias");
        dump("mimi_decoder_model_0", o0, 512*16);
        cl_mem o2 = g.convtranspose1d(load("mimi_decoder_model_1"), 512, 256, 16, 12, 6, P+".2.convtr.weight", P+".2.convtr.bias");
        dump("mimi_decoder_model_2", o2, 256*96);
        cl_mem o3 = g.seanet_resnet_block(load("mimi_decoder_model_2"), 256, 96, P+".3");
        dump("mimi_decoder_model_3", o3, 256*96);
        cl_mem o5 = g.convtranspose1d(load("mimi_decoder_model_4"), 256, 128, 96, 10, 5, P+".5.convtr.weight", P+".5.convtr.bias");
        dump("mimi_decoder_model_5", o5, 128*480);
        cl_mem o6 = g.seanet_resnet_block(load("mimi_decoder_model_5"), 128, 480, P+".6");
        dump("mimi_decoder_model_6", o6, 128*480);
        cl_mem o11 = g.conv1d_causal(load("mimi_decoder_model_10"), 64, 1, 1920, 3, 1, P+".11.conv.weight", P+".11.conv.bias");
        dump("mimi_decoder_model_11", o11, 1920);
        clFinish(ctx.queue());
        fprintf(stderr, "OK: isoseanet done\n");
        return 0;
    }

    // ── Flow-net stateless validation: cond_embed(c) + time_embed_{0,1}(s,t).
    //    inpath = out_norm output [1024] (the backbone context c). PORT_SPEC §2c.
    if (op == "flowval") {
        auto dump = [&](const char* name, cl_mem b, size_t cnt) {
            std::vector<float> v = g.download(b, cnt);
            write_f32(std::string("layer_dumps/") + name + ".bin", v);
        };
        cl_mem ce = g.linear(x, 1, 512, 1024, "flow_lm.flow_net.cond_embed.weight", "flow_lm.flow_net.cond_embed.bias");
        dump("flow_lm_flow_net_cond_embed", ce, 512);
        cl_mem te0 = g.timestep_embedder(0.0f, "flow_lm.flow_net.time_embed.0");
        dump("flow_lm_flow_net_time_embed_0", te0, 512);
        cl_mem te1 = g.timestep_embedder(1.0f, "flow_lm.flow_net.time_embed.1");
        dump("flow_lm_flow_net_time_embed_1", te1, 512);
        clFinish(ctx.queue());
        fprintf(stderr, "OK: flowval done\n");
        return 0;
    }

    // ── SEANet decoder chain (nodes 29-41), single frame. Dumps every node to
    //    layer_dumps/<name>.bin for per-node cosine. Input = decoder_transformer
    //    output [512,16]. See PORT_SPEC §3.19.
    if (op == "seanet") {
        auto dump = [&](const char* name, cl_mem b, size_t cnt) {
            std::vector<float> v = g.download(b, cnt);
            write_f32(std::string("layer_dumps/") + name + ".bin", v);
        };
        const std::string P = "mimi.decoder.model";
        cl_mem c0 = g.conv1d_causal(x, 512, 512, 16, 7, 1, P+".0.conv.weight", P+".0.conv.bias");
        dump("mimi_decoder_model_0", c0, 512*16);
        cl_mem c1 = g.elu(c0, 512*16);                              dump("mimi_decoder_model_1", c1, 512*16);
        cl_mem c2 = g.convtranspose1d(c1, 512, 256, 16, 12, 6, P+".2.convtr.weight", P+".2.convtr.bias");
        dump("mimi_decoder_model_2", c2, 256*96);
        cl_mem c3 = g.seanet_resnet_block(c2, 256, 96, P+".3");     dump("mimi_decoder_model_3", c3, 256*96);
        cl_mem c4 = g.elu(c3, 256*96);                             dump("mimi_decoder_model_4", c4, 256*96);
        cl_mem c5 = g.convtranspose1d(c4, 256, 128, 96, 10, 5, P+".5.convtr.weight", P+".5.convtr.bias");
        dump("mimi_decoder_model_5", c5, 128*480);
        cl_mem c6 = g.seanet_resnet_block(c5, 128, 480, P+".6");    dump("mimi_decoder_model_6", c6, 128*480);
        cl_mem c7 = g.elu(c6, 128*480);                            dump("mimi_decoder_model_7", c7, 128*480);
        cl_mem c8 = g.convtranspose1d(c7, 128, 64, 480, 8, 4, P+".8.convtr.weight", P+".8.convtr.bias");
        dump("mimi_decoder_model_8", c8, 64*1920);
        cl_mem c9 = g.seanet_resnet_block(c8, 64, 1920, P+".9");    dump("mimi_decoder_model_9", c9, 64*1920);
        cl_mem c10 = g.elu(c9, 64*1920);                           dump("mimi_decoder_model_10", c10, 64*1920);
        cl_mem c11 = g.conv1d_causal(c10, 64, 1, 1920, 3, 1, P+".11.conv.weight", P+".11.conv.bias");
        dump("mimi_decoder_model_11", c11, 1920);
        clFinish(ctx.queue());
        fprintf(stderr, "OK: seanet chain done (12 nodes dumped)\n");
        return 0;
    }

    // Dispatch. Ops needing dims/keys read them from `a` / argv tail.
    if      (op == "elu")  y = g.elu(x, n);
    else if (op == "silu") y = g.silu(x, n);
    else if (op == "gelu") y = g.gelu(x, n);
    else if (op == "copy") y = g.copy(x, n);
    else if (op == "layernorm") {
        // args: rows C eps_x1e6  [w_key b_key]  (eps passed as micro to stay int)
        int rows = a[0], C = a[1]; float eps = (a.size()>2? a[2]:10) * 1e-6f;
        std::string wk = (argc>9? argv[9]:""), bk = (argc>10? argv[10]:"");
        y = g.layernorm(x, rows, C, eps, wk, bk);
    }
    else { fprintf(stderr, "ERROR: unknown op '%s'\n", op.c_str()); return 2; }

    if (!y) { fprintf(stderr, "ERROR: op '%s' returned null\n", op.c_str()); return 1; }
    clFinish(ctx.queue());
    std::vector<float> out = g.download(y, n);  // most ops here are shape-preserving
    write_f32(outpath, out);
    fprintf(stderr, "OK: ran '%s' on %zu floats\n", op.c_str(), n);
    return 0;
}
