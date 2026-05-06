// Auto-generated inference entry point for Qwen/Qwen2.5-0.5B
// Model type: qwen2

#include "model.h"
#include "tokenizer.h"
#include "sampler.h"
#include "opencl_context.h"
#include "weights.h"
#include "utils.h"
#include "debug_utils.h"
#include "benchmark.h"
#include <iostream>
#include <string>
#include <cstring>
#include <cstdio>
#include <cstdint>
#include <vector>
#include <chrono>
#include <fstream>
#include <dlfcn.h>
#include <CL/cl.h>

// STREAM-style microbenchmark — measures the practical streaming-read
// ceiling on this device for both buffer-cache and texture-cache reads.
// Prints achieved GB/s. Triggered via NNOPT_BW_PROBE=1, runs and exits
// before any LLM work begins.
static int run_bw_probe(OpenCLContext& cl_ctx) {
    using namespace std::chrono;
    cl_context  ctx   = cl_ctx.context();
    cl_command_queue q = cl_ctx.queue();
    cl_device_id dev  = cl_ctx.device();
    cl_int err = CL_SUCCESS;

    // 256 MB fp16 buffer — much larger than the 64 KB L2 cache so every
    // read goes to DRAM (no cache amplification).
    const size_t TOTAL_BYTES   = 256ull * 1024 * 1024;
    const size_t TOTAL_HALVES  = TOTAL_BYTES / 2;
    const size_t TOTAL_VEC4    = TOTAL_HALVES / 4;       // # fp16x4 elements
    const size_t WG            = 64;
    const size_t WG_COUNT      = 256;
    const size_t TOTAL_THREADS = WG * WG_COUNT;          // 16384
    const size_t VEC4_PER_THREAD = TOTAL_VEC4 / TOTAL_THREADS;
    const size_t HALVES_PER_THREAD = VEC4_PER_THREAD * 4;

    cl_mem src = clCreateBuffer(ctx, CL_MEM_READ_ONLY,  TOTAL_BYTES,            nullptr, &err);
    if (err != CL_SUCCESS) { std::cerr << "src alloc fail " << err << "\n"; return 1; }
    cl_mem dce = clCreateBuffer(ctx, CL_MEM_READ_WRITE, TOTAL_THREADS * sizeof(float), nullptr, &err);
    if (err != CL_SUCCESS) { std::cerr << "dce alloc fail " << err << "\n"; return 1; }
    {
        const cl_uchar pat[2] = {0x00, 0x3c};  // fp16 1.0 little-endian (0x3C00)
        clEnqueueFillBuffer(q, src, pat, 2, 0, TOTAL_BYTES, 0, nullptr, nullptr);
        clFinish(q);
    }

    // Build the bw kernels via the same path as our gemv_m1 program.
    std::ifstream f("kernels/gemv_m1.cl", std::ios::binary);
    if (!f.is_open()) { std::cerr << "open gemv_m1.cl fail\n"; return 1; }
    std::string src_text((std::istreambuf_iterator<char>(f)),
                          std::istreambuf_iterator<char>());
    cl_program prog = OpenCLContext::build_cached_program_from_queue(q, src_text, "");
    if (!prog) { std::cerr << "build kernels fail\n"; return 1; }

    cl_kernel k_buf = clCreateKernel(prog, "gemv_stream_buf", &err);
    if (err != CL_SUCCESS) { std::cerr << "createKernel buf " << err << "\n"; return 1; }

    // ── Buffer-cache STREAM (coalesced wave-stride pattern) ──
    // iters_per_thread = total_vec4 / total_threads
    int iters_per_thread = (int)(TOTAL_VEC4 / TOTAL_THREADS);
    clSetKernelArg(k_buf, 0, sizeof(cl_mem), &src);
    clSetKernelArg(k_buf, 1, sizeof(cl_mem), &dce);
    clSetKernelArg(k_buf, 2, sizeof(int),    &iters_per_thread);

    double best_buf_gbs = 0.0;
    for (int trial = 0; trial < 5; ++trial) {
        size_t gws = TOTAL_THREADS, lws = WG;
        // Warm up the cache and any frequency ramp.
        clEnqueueNDRangeKernel(q, k_buf, 1, nullptr, &gws, &lws, 0, nullptr, nullptr);
        clFinish(q);

        auto t0 = high_resolution_clock::now();
        clEnqueueNDRangeKernel(q, k_buf, 1, nullptr, &gws, &lws, 0, nullptr, nullptr);
        clFinish(q);
        auto t1 = high_resolution_clock::now();

        double secs = duration<double>(t1 - t0).count();
        double gbs  = (double)TOTAL_BYTES / secs / 1e9;
        if (gbs > best_buf_gbs) best_buf_gbs = gbs;
    }
    std::cerr << "STREAM[buf]: " << TOTAL_BYTES/(1024.0*1024.0) << " MB read "
              << "→ " << best_buf_gbs << " GB/s (best of 5)\n";

    // ── Texture-cache STREAM (image2d_t over the same buffer) ──
    cl_kernel k_img = clCreateKernel(prog, "gemv_stream_img", &err);
    if (err != CL_SUCCESS) { std::cerr << "createKernel img " << err << "\n"; return 1; }

    // image2d view dimensions: pix-width = 4096 (must fit in 16384 max),
    // height = TOTAL_VEC4 / pix-width. CL_RGBA half = 4 fp16 per pixel.
    const int img_w = 4096;
    const int img_h = (int)(TOTAL_VEC4 / img_w);
    cl_image_format fmt = { CL_RGBA, CL_HALF_FLOAT };
    cl_image_desc desc = {};
    desc.image_type        = CL_MEM_OBJECT_IMAGE2D;
    desc.image_width       = img_w;
    desc.image_height      = img_h;
    desc.buffer            = src;
    cl_mem img = clCreateImage(ctx, CL_MEM_READ_ONLY, &fmt, &desc, nullptr, &err);
    if (err != CL_SUCCESS) { std::cerr << "createImage " << err << " (h=" << img_h << ")\n"; return 1; }

    // Each WG handles `rows_per_wg` rows (wave-stride pattern within row).
    // Pick rows_per_wg so total WGs = 256 (same as buf path) for fair compare.
    int rows_per_wg = img_h / 256;
    if (rows_per_wg < 1) rows_per_wg = 1;
    int row_pixels = img_w;
    clSetKernelArg(k_img, 0, sizeof(cl_mem), &img);
    clSetKernelArg(k_img, 1, sizeof(cl_mem), &dce);
    clSetKernelArg(k_img, 2, sizeof(int),    &rows_per_wg);
    clSetKernelArg(k_img, 3, sizeof(int),    &row_pixels);

    double best_img_gbs = 0.0;
    for (int trial = 0; trial < 5; ++trial) {
        size_t wg_count = (size_t)(img_h / rows_per_wg);
        size_t lws = WG, gws = wg_count * lws;
        clEnqueueNDRangeKernel(q, k_img, 1, nullptr, &gws, &lws, 0, nullptr, nullptr);
        clFinish(q);

        auto t0 = high_resolution_clock::now();
        clEnqueueNDRangeKernel(q, k_img, 1, nullptr, &gws, &lws, 0, nullptr, nullptr);
        clFinish(q);
        auto t1 = high_resolution_clock::now();

        double secs = duration<double>(t1 - t0).count();
        double bytes = (double)img_h * img_w * 8.0;  // 4 fp16 per pixel = 8 bytes
        double gbs   = bytes / secs / 1e9;
        if (gbs > best_img_gbs) best_img_gbs = gbs;
    }
    std::cerr << "STREAM[img]: " << ((double)img_h*img_w*8.0)/(1024.0*1024.0) << " MB read via image2d "
              << "→ " << best_img_gbs << " GB/s (best of 5)\n";

    // Theoretical roofline summary using these MEASURED ceilings.
    const double weight_mb = 942.0;
    std::cerr << "\n=== Practical roofline for Qwen2.5-0.5B (942 MB/token) ===\n"
              << "  Buffer-cache ceiling: " << best_buf_gbs << " GB/s → max "
              <<  (best_buf_gbs * 1000.0 / weight_mb) << " tok/s\n"
              << "  Texture-cache ceiling: " << best_img_gbs << " GB/s → max "
              <<  (best_img_gbs * 1000.0 / weight_mb) << " tok/s\n";

    clReleaseMemObject(img);
    clReleaseKernel(k_img);
    clReleaseKernel(k_buf);
    clReleaseProgram(prog);
    clReleaseMemObject(dce);
    clReleaseMemObject(src);
    return 0;
}

// cl_qcom_recordable_queues probe — INVESTIGATIVE / NOT WORKING.
//
// Goal: prove cl_qcom_recordable_queues lets us record the 480-launch
// decode sequence once and replay it ~zero-overhead, eliminating the
// ~24 ms/token of CPU-side launch bookkeeping.
//
// Status (2026-05-03): symbols exist (clNewRecordingQCOM,
// clEndRecordingQCOM, clEnqueueRecordingQCOM, clReleaseRecordingQCOM)
// and resolve via dlsym(RTLD_DEFAULT). dlsym, perf-hint, and BW probes
// all work via this path.
//
// BUT clNewRecordingQCOM returns CL_INVALID_COMMAND_QUEUE (-35) on every
// queue we construct, including:
//   - Default in-order with profiling
//   - clCreateCommandQueueWithProperties with 8 different bit candidates
//     in the CL_QUEUE_PROPERTIES bitfield (1<<4 .. 1<<31)
//   - clCreateCommandQueueWithProperties with 8 standalone property keys
//     in the 0x40Cx-0x40Dx Qualcomm extension keyspace
//   - Out-of-order queues
//   - Default-properties queues (nullptr)
// And alternate first-arg signatures (context-only, context+device,
// out-arg variants) all confirm the function takes a cl_command_queue
// + cl_int* errcode_ret — so the queue itself is what's "invalid for
// recording", not the call shape.
//
// Without Qualcomm's SDK headers we can't discover the property bit
// that flips a queue from "regular" to "recordable". The vendor
// libOpenCL.so on this device is a thin ICD shim (1041 strings); the
// real driver implementation lives in a closed-source layer we can't
// inspect.
//
// Re-test on a device shipping the Adreno 7xx-class driver and the
// modern cl_khr_command_buffer extension (Khronos-standard equivalent),
// where this class of optimization is properly documented.
static int run_record_probe(OpenCLContext& cl_ctx) {
    using namespace std::chrono;
    cl_context  ctx = cl_ctx.context();
    cl_command_queue q = cl_ctx.queue();
    cl_int err = CL_SUCCESS;

    // Best-guess signatures — opaque handle for recording.
    typedef void* cl_recording_qcom;
    typedef cl_recording_qcom (CL_API_CALL *clNewRecordingQCOM_fn)(
        cl_command_queue, cl_int*);
    typedef cl_int (CL_API_CALL *clEndRecordingQCOM_fn)(cl_recording_qcom);
    typedef cl_int (CL_API_CALL *clReleaseRecordingQCOM_fn)(cl_recording_qcom);
    // Per Qualcomm convention, clEnqueueRecordingQCOM takes 4 (size, ptr)
    // pairs of per-replay overrides (args, global_offsets, gws, lws) before
    // the standard event-list trio. Pass all overrides as 0/NULL for the
    // simplest no-override replay.
    struct cl_array_arg_qcom {
        cl_kernel    kernel;
        cl_uint      arg_indx;
        size_t       arg_size;
        const void*  arg_value;
    };
    struct cl_array_kernel_exec_info_qcom {
        cl_kernel       kernel;
        cl_uint         indx;
        size_t          param_value_size;
        const void*     param_value;
    };
    typedef cl_int (CL_API_CALL *clEnqueueRecordingQCOM_fn)(
        cl_command_queue queue,
        cl_recording_qcom recording,
        size_t num_args,
        const cl_array_arg_qcom* args,
        size_t num_global_offsets,
        const cl_array_kernel_exec_info_qcom* global_offsets,
        size_t num_global_work_sizes,
        const cl_array_kernel_exec_info_qcom* global_work_sizes,
        size_t num_local_work_sizes,
        const cl_array_kernel_exec_info_qcom* local_work_sizes,
        cl_uint num_events_in_wait_list,
        const cl_event* event_wait_list,
        cl_event* event);

    auto fnNew     = (clNewRecordingQCOM_fn)    dlsym(RTLD_DEFAULT, "clNewRecordingQCOM");
    auto fnEnd     = (clEndRecordingQCOM_fn)    dlsym(RTLD_DEFAULT, "clEndRecordingQCOM");
    auto fnRelease = (clReleaseRecordingQCOM_fn)dlsym(RTLD_DEFAULT, "clReleaseRecordingQCOM");
    auto fnEnqueue = (clEnqueueRecordingQCOM_fn)dlsym(RTLD_DEFAULT, "clEnqueueRecordingQCOM");
    if (!fnNew || !fnEnd || !fnRelease || !fnEnqueue) {
        std::cerr << "Record: missing one or more entry points\n";
        return 1;
    }

    // We compile against CL_TARGET_OPENCL_VERSION=120, but the device is 2.0.
    // Declare the 2.0 queue-properties type and entry point manually and
    // dlsym the function — same trick we used for clSetPerfHintQCOM.
    typedef cl_ulong cl_queue_properties_qcom;
    typedef cl_command_queue (CL_API_CALL *clCreateCommandQueueWithProperties_fn)(
        cl_context ctx, cl_device_id dev,
        const cl_queue_properties_qcom* properties, cl_int* errcode_ret);
    auto fnCreateQ = (clCreateCommandQueueWithProperties_fn)
        dlsym(RTLD_DEFAULT, "clCreateCommandQueueWithProperties");
    if (!fnCreateQ) {
        std::cerr << "Record: clCreateCommandQueueWithProperties not exposed; aborting probe\n";
        return 1;
    }
    constexpr cl_queue_properties_qcom CL_QUEUE_PROPERTIES_QCOM = 0x1093;  // standard

    // Probe: search for the right CL_QUEUE_RECORDABLE_QCOM property value.
    cl_device_id dev = cl_ctx.device();
    cl_command_queue probe_q = nullptr;
    cl_recording_qcom winning_rec = nullptr;
    int win_attempt = -1;
    cl_command_queue live_q = q;  // original in-order PROFILE queue for replay

    auto try_recording = [&](cl_command_queue qq, int attempt_id) -> bool {
        cl_int e = 0;
        cl_recording_qcom h = fnNew(qq, &e);
        std::cerr << "  attempt " << attempt_id << ": clNewRecordingQCOM err="
                  << e << " handle=" << h << "\n";
        if (e == CL_SUCCESS && h) {
            probe_q = qq;
            // Tear down the probe recording immediately — we're just testing
            // whether recording starts at all. The actual recording for the
            // perf measurement is created later, after the baseline.
            fnEnd(h);
            fnRelease(h);
            win_attempt = attempt_id;
            winning_rec = (cl_recording_qcom)1;  // sentinel: probe succeeded
            return true;
        }
        return false;
    };

    // Attempt 0: the existing in-order PROFILING queue (confirm -35).
    if (!winning_rec) try_recording(q, 0);

    // Diagnostic: ask the device what queue properties it claims to support.
    {
        cl_command_queue_properties supp = 0;
        clGetDeviceInfo(dev, CL_DEVICE_QUEUE_PROPERTIES, sizeof(supp), &supp, nullptr);
        std::cerr << "  CL_DEVICE_QUEUE_PROPERTIES = 0x" << std::hex << supp << std::dec
                  << " (bit 0=OOO, bit 1=PROFILING; bit 30=CL_QUEUE_RECORDABLE_QCOM)\n";
    }

    // Direct candidates for the recordable property — test exactly what the
    // device claims (0x40000003 — OOO + PROFILING + RECORDABLE) and partial
    // subsets. Snapdragon Programming Guide says use clCreateCommandQueue.
    constexpr cl_command_queue_properties RECORD_BIT = (cl_command_queue_properties)1 << 30;
    cl_command_queue_properties direct_candidates[] = {
        RECORD_BIT,                                                        // alone
        RECORD_BIT | CL_QUEUE_PROFILING_ENABLE,                            // + profile
        RECORD_BIT | CL_QUEUE_OUT_OF_ORDER_EXEC_MODE_ENABLE,               // + ooo
        RECORD_BIT | CL_QUEUE_PROFILING_ENABLE | CL_QUEUE_OUT_OF_ORDER_EXEC_MODE_ENABLE,
    };
    for (size_t i = 0; i < sizeof(direct_candidates)/sizeof(direct_candidates[0]); ++i) {
        if (winning_rec) break;
        cl_int qerr = 0;
        cl_command_queue qq = clCreateCommandQueue(ctx, dev, direct_candidates[i], &qerr);
        std::cerr << "  direct: props=0x" << std::hex << direct_candidates[i] << std::dec
                  << " qerr=" << qerr << " qq=" << qq << "\n";
        if (qerr == CL_SUCCESS && qq) {
            if (!try_recording(qq, 400 + (int)i)) clReleaseCommandQueue(qq);
        }
    }

    // The Snapdragon OpenCL Programming Guide (80-NB295-11 Rev. C, §9.1.3)
    // specifies clCreateCommandQueue (the OpenCL 1.2 API) — NOT
    // clCreateCommandQueueWithProperties — for the recordable queue.
    // The 1.2 path may accept the QCOM bit where the 2.0 path rejects it.
    // Iterate through candidate bits in the cl_command_queue_properties
    // bitfield (bits 0,1 are reserved for OOO and profiling; try bits 2..31).
    for (int bit = 2; bit < 32; ++bit) {
        if (winning_rec) break;
        cl_command_queue_properties props =
            (cl_command_queue_properties)(((cl_ulong)1 << bit) | CL_QUEUE_PROFILING_ENABLE);
        cl_int qerr = 0;
        cl_command_queue qq = clCreateCommandQueue(ctx, dev, props, &qerr);
        std::cerr << "  v12 bit " << bit
                  << " (props=0x" << std::hex << (cl_ulong)props << std::dec << ")"
                  << ": qerr=" << qerr << " qq=" << qq << "\n";
        if (qerr == CL_SUCCESS && qq) {
            // Verify what properties came back.
            cl_command_queue_properties got = 0;
            clGetCommandQueueInfo(qq, CL_QUEUE_PROPERTIES, sizeof(got), &got, nullptr);
            std::cerr << "    got_props=0x" << std::hex << got << std::dec
                      << " (expected the high bit to remain)\n";
            if (!try_recording(qq, 300 + bit)) clReleaseCommandQueue(qq);
        }
    }

    // Sanity check 0a: does fnCreateQ even work with a vanilla properties array?
    {
        cl_queue_properties_qcom props[] = {
            CL_QUEUE_PROPERTIES_QCOM, CL_QUEUE_PROFILING_ENABLE,
            0
        };
        cl_int qe = 0;
        cl_command_queue qq = fnCreateQ(ctx, dev, props, &qe);
        std::cerr << "  sanity: fnCreateQ(profile-only) err=" << qe << " qq=" << qq << "\n";
        if (qq) {
            try_recording(qq, 100);  // attempt id 100 = with-vanilla-props
            if (!winning_rec) clReleaseCommandQueue(qq);
        }
    }

    // Sanity check 0b: does OOO queue allow recording?
    if (!winning_rec) {
        cl_queue_properties_qcom props[] = {
            CL_QUEUE_PROPERTIES_QCOM,
            CL_QUEUE_PROFILING_ENABLE | CL_QUEUE_OUT_OF_ORDER_EXEC_MODE_ENABLE,
            0
        };
        cl_int qe = 0;
        cl_command_queue qq = fnCreateQ(ctx, dev, props, &qe);
        std::cerr << "  sanity: fnCreateQ(ooo+profile) err=" << qe << " qq=" << qq << "\n";
        if (qq) {
            try_recording(qq, 101);
            if (!winning_rec) clReleaseCommandQueue(qq);
        }
    }

    // Sanity check 0c: nullptr properties (default queue).
    if (!winning_rec) {
        cl_int qe = 0;
        cl_command_queue qq = fnCreateQ(ctx, dev, nullptr, &qe);
        std::cerr << "  sanity: fnCreateQ(nullptr props) err=" << qe << " qq=" << qq << "\n";
        if (qq) {
            try_recording(qq, 102);
            if (!winning_rec) clReleaseCommandQueue(qq);
        }
    }

    // Attempts 1..8: try various queue-property bits in CL_QUEUE_PROPERTIES bitfield.
    cl_ulong candidate_props[] = {
        (cl_ulong)1 << 4,  (cl_ulong)1 << 5,  (cl_ulong)1 << 6,  (cl_ulong)1 << 7,
        (cl_ulong)1 << 8,  (cl_ulong)1 << 9,  (cl_ulong)1 << 30, (cl_ulong)1 << 31,
    };
    for (size_t i = 0; i < sizeof(candidate_props)/sizeof(candidate_props[0]); ++i) {
        if (winning_rec) break;
        cl_queue_properties_qcom props[] = {
            CL_QUEUE_PROPERTIES_QCOM, candidate_props[i] | CL_QUEUE_PROFILING_ENABLE,
            0
        };
        cl_int qerr = 0;
        cl_command_queue qq = fnCreateQ(ctx, dev, props, &qerr);
        if (qerr != CL_SUCCESS || !qq) {
            std::cerr << "  attempt " << (i+1) << ": queue err=" << qerr << "\n";
            continue;
        }
        if (!try_recording(qq, (int)(i + 1))) clReleaseCommandQueue(qq);
    }

    // Attempts 9..16: try standalone property KEYS in 0x40Cx-0x40Dx range.
    cl_uint candidate_keys[] = {
        0x40C0, 0x40C1, 0x40C6, 0x40C7, 0x40C8,
        0x40D0, 0x40D1, 0x40D2,
    };
    for (size_t i = 0; i < sizeof(candidate_keys)/sizeof(candidate_keys[0]); ++i) {
        if (winning_rec) break;
        cl_queue_properties_qcom props[] = {
            CL_QUEUE_PROPERTIES_QCOM, CL_QUEUE_PROFILING_ENABLE,
            (cl_queue_properties_qcom)candidate_keys[i], 1,
            0
        };
        cl_int qerr = 0;
        cl_command_queue qq = fnCreateQ(ctx, dev, props, &qerr);
        if (qerr != CL_SUCCESS || !qq) {
            std::cerr << "  attempt " << (i+9) << " (key 0x" << std::hex
                      << candidate_keys[i] << std::dec << "): queue err=" << qerr << "\n";
            continue;
        }
        if (!try_recording(qq, (int)(i + 9))) clReleaseCommandQueue(qq);
    }

    // Try alternative function signatures — maybe the first arg isn't
    // cl_command_queue. Some Qualcomm extensions create resource-like
    // objects from a context or device instead of a queue.
    if (!winning_rec) {
        std::cerr << "  altsig: trying clNewRecordingQCOM(context, &err)\n";
        typedef cl_recording_qcom (CL_API_CALL *fn_ctx_t)(cl_context, cl_int*);
        auto f = (fn_ctx_t)dlsym(RTLD_DEFAULT, "clNewRecordingQCOM");
        cl_int e = 0;
        cl_recording_qcom h = f(ctx, &e);
        std::cerr << "    err=" << e << " handle=" << h << "\n";
        if (e == CL_SUCCESS && h) { winning_rec = h; probe_q = q; win_attempt = 200; }
    }
    if (!winning_rec) {
        std::cerr << "  altsig: trying clNewRecordingQCOM(context, device, &err)\n";
        typedef cl_recording_qcom (CL_API_CALL *fn_cd_t)(cl_context, cl_device_id, cl_int*);
        auto f = (fn_cd_t)dlsym(RTLD_DEFAULT, "clNewRecordingQCOM");
        cl_int e = 0;
        cl_recording_qcom h = f(ctx, dev, &e);
        std::cerr << "    err=" << e << " handle=" << h << "\n";
        if (e == CL_SUCCESS && h) { winning_rec = h; probe_q = q; win_attempt = 201; }
    }
    if (!winning_rec) {
        std::cerr << "  altsig: trying clNewRecordingQCOM(queue, num_props=0, props=null, &err)\n";
        typedef cl_recording_qcom (CL_API_CALL *fn_qp_t)(cl_command_queue, cl_uint, const void*, cl_int*);
        auto f = (fn_qp_t)dlsym(RTLD_DEFAULT, "clNewRecordingQCOM");
        cl_int e = 0;
        cl_recording_qcom h = f(q, 0, nullptr, &e);
        std::cerr << "    err=" << e << " handle=" << h << "\n";
        if (e == CL_SUCCESS && h) { winning_rec = h; probe_q = q; win_attempt = 202; }
    }
    // Crucial finding: altsig (context, &err) got err=-36 (CL_INVALID_CONTEXT),
    // a different error than queue-arg gave (-35 CL_INVALID_COMMAND_QUEUE).
    // That's strong evidence the function takes a CONTEXT (not queue) and
    // that our context lacks a required property bit/key. Without the QCOM
    // SDK headers we don't know which property — would need binary RE.

    // Try a few last-ditch alternative signatures with output-arg style:
    if (!winning_rec) {
        std::cerr << "  altsig: trying clNewRecordingQCOM(queue, out_handle) status return\n";
        typedef cl_int (CL_API_CALL *fn_qout_t)(cl_command_queue, cl_recording_qcom*);
        auto f = (fn_qout_t)dlsym(RTLD_DEFAULT, "clNewRecordingQCOM");
        cl_recording_qcom h = nullptr;
        cl_int e = f(q, &h);
        std::cerr << "    status=" << e << " handle=" << h << "\n";
        if (e == CL_SUCCESS && h) { winning_rec = h; probe_q = q; win_attempt = 203; }
    }
    if (!winning_rec) {
        std::cerr << "  altsig: trying clNewRecordingQCOM(ctx, dev, out_handle) status\n";
        typedef cl_int (CL_API_CALL *fn_cdout_t)(cl_context, cl_device_id, cl_recording_qcom*);
        auto f = (fn_cdout_t)dlsym(RTLD_DEFAULT, "clNewRecordingQCOM");
        cl_recording_qcom h = nullptr;
        cl_int e = f(ctx, dev, &h);
        std::cerr << "    status=" << e << " handle=" << h << "\n";
        if (e == CL_SUCCESS && h) { winning_rec = h; probe_q = q; win_attempt = 204; }
    }

    if (!winning_rec) {
        std::cerr << "Record: no candidate signature/property combination worked. Aborting.\n";
        return 1;
    }
    std::cerr << "Record: WIN attempt " << win_attempt
              << " (probe_q=" << probe_q << " rec=" << winning_rec << ")\n";

    // Keep q as the LIVE in-order queue for baseline + replay.
    // probe_q is the RECORDABLE queue used only to capture the recording.
    // Per Snapdragon Programming Guide: "The live command queue for
    // [clEnqueueRecordingQCOM] is different from the one for recording".

    // Build the probe kernel.
    std::ifstream f("kernels/gemv_m1.cl", std::ios::binary);
    std::string src_text((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
    cl_program prog = OpenCLContext::build_cached_program_from_queue(q, src_text, "");
    if (!prog) { std::cerr << "build kernels fail\n"; return 1; }
    cl_kernel k = clCreateKernel(prog, "probe_noop", &err);
    if (err != CL_SUCCESS) { std::cerr << "createKernel fail " << err << "\n"; return 1; }

    cl_mem counter = clCreateBuffer(ctx, CL_MEM_READ_WRITE, sizeof(cl_int), nullptr, &err);
    if (err != CL_SUCCESS) { std::cerr << "alloc counter fail\n"; return 1; }
    int incr = 1;
    clSetKernelArg(k, 0, sizeof(cl_mem), &counter);
    clSetKernelArg(k, 1, sizeof(int),    &incr);

    constexpr int N_DISPATCH_PER_RECORDING = 100;
    constexpr int N_REPLAYS                = 100;
    constexpr int N_BASELINE_DISPATCHES    = N_DISPATCH_PER_RECORDING * N_REPLAYS;

    auto reset_counter = [&]() {
        cl_int zero = 0;
        clEnqueueWriteBuffer(live_q, counter, CL_TRUE, 0, sizeof(int), &zero, 0, nullptr, nullptr);
    };
    auto read_counter = [&]() {
        cl_int v = 0;
        clEnqueueReadBuffer(live_q, counter, CL_TRUE, 0, sizeof(int), &v, 0, nullptr, nullptr);
        return v;
    };

    // ── Baseline: N_BASELINE_DISPATCHES sequential clEnqueueNDRangeKernel
    //           on the LIVE in-order queue (NOT the recordable queue, which
    //           wouldn't actually execute anything) ──
    reset_counter();
    size_t gws = 1, lws = 1;
    auto t0 = high_resolution_clock::now();
    for (int i = 0; i < N_BASELINE_DISPATCHES; ++i) {
        clEnqueueNDRangeKernel(live_q, k, 1, nullptr, &gws, &lws, 0, nullptr, nullptr);
    }
    clFinish(live_q);
    auto t1 = high_resolution_clock::now();
    double base_ms = duration<double, std::milli>(t1 - t0).count();
    int base_val = read_counter();
    std::cerr << "Baseline: " << N_BASELINE_DISPATCHES << " sequential dispatches → "
              << base_ms << " ms, counter=" << base_val
              << " (expected " << N_BASELINE_DISPATCHES << ")\n";

    // ── Recording: create a fresh recording on probe_q for the perf test ──
    reset_counter();
    cl_int new_err = 0;
    cl_recording_qcom rec = fnNew(probe_q, &new_err);
    if (new_err != CL_SUCCESS || !rec) {
        std::cerr << "Record: post-baseline clNewRecordingQCOM failed err=" << new_err << "\n";
        return 1;
    }
    std::cerr << "Record: started fresh recording on probe_q (rec=" << rec << ")\n";

    // After clNewRecordingQCOM, subsequent enqueues on probe_q are
    // recorded (not executed). Record N_DISPATCH_PER_RECORDING dispatches.
    for (int i = 0; i < N_DISPATCH_PER_RECORDING; ++i) {
        cl_int e = clEnqueueNDRangeKernel(probe_q, k, 1, nullptr, &gws, &lws, 0, nullptr, nullptr);
        if (e != CL_SUCCESS) {
            std::cerr << "Record: enqueue during record failed at i=" << i << " err=" << e << "\n";
            fnRelease(rec);
            return 1;
        }
    }

    cl_int end_err = fnEnd(rec);
    std::cerr << "clEndRecordingQCOM → err=" << end_err << "\n";
    if (end_err != CL_SUCCESS) {
        std::cerr << "Record: end failed; aborting\n";
        fnRelease(rec);
        return 1;
    }

    int post_record_counter = read_counter();
    std::cerr << "After record (no replay): counter=" << post_record_counter
              << " (should be 0 — recording shouldn't execute)\n";

    // ── Replay N_REPLAYS times — on the LIVE in-order queue, not probe_q ──
    reset_counter();
    auto t2 = high_resolution_clock::now();
    for (int r = 0; r < N_REPLAYS; ++r) {
        cl_int e = fnEnqueue(live_q, rec,
            0, nullptr,    // num_args / args
            0, nullptr,    // num_global_offsets / global_offsets
            0, nullptr,    // num_global_work_sizes / gws
            0, nullptr,    // num_local_work_sizes  / lws
            0, nullptr, nullptr);  // events
        if (e != CL_SUCCESS) {
            std::cerr << "Record: replay failed at r=" << r << " err=" << e << "\n";
            fnRelease(rec);
            return 1;
        }
    }
    clFinish(live_q);
    auto t3 = high_resolution_clock::now();
    double replay_ms = duration<double, std::milli>(t3 - t2).count();
    int replay_val = read_counter();
    std::cerr << "Replay: " << N_REPLAYS << " × " << N_DISPATCH_PER_RECORDING
              << " = " << N_BASELINE_DISPATCHES << " dispatches → "
              << replay_ms << " ms, counter=" << replay_val << "\n";

    if (replay_val != base_val) {
        std::cerr << "Record: COUNTER MISMATCH (math is wrong; replay isn't equivalent)\n";
    } else {
        double speedup = base_ms / replay_ms;
        double per_dispatch_baseline = base_ms / N_BASELINE_DISPATCHES * 1000.0;  // µs
        double per_dispatch_replay   = replay_ms / N_BASELINE_DISPATCHES * 1000.0;
        std::cerr << "Record: speedup = " << speedup << "× ("
                  << per_dispatch_baseline << " µs/dispatch baseline → "
                  << per_dispatch_replay   << " µs/dispatch replay)\n";
    }

    fnRelease(rec);
    clReleaseMemObject(counter);
    clReleaseKernel(k);
    clReleaseProgram(prog);
    return 0;
}

static std::vector<int> load_token_ids_from_file(const std::string& path) {
    std::vector<int> ids;
    FILE* f = fopen(path.c_str(), "rb");
    if (!f) return ids;
    fseek(f, 0, SEEK_END);
    long sz = ftell(f);
    fseek(f, 0, SEEK_SET);
    int count = sz / sizeof(int32_t);
    ids.resize(count);
    fread(ids.data(), sizeof(int32_t), count, f);
    fclose(f);
    return ids;
}

static void print_usage(const char* prog) {
    std::cerr << "Usage: " << prog
              << " <prompt> [max_tokens]"
              << " [--temperature T] [--top-k K] [--top-p P]"
              << " [--repetition-penalty R] [--seed S]"
              << " [--token-ids <file>]"
              << std::endl;
}

int main(int argc, char* argv[]) {
    // ── Streaming UX: disable stdio buffering so each per-token
    // std::cout::flush() actually leaves the binary right away. When
    // stdout is a pipe (e.g. adb shell), the default _IOFBF would hold
    // up to 4 KB before flushing — the entire 32-token sentence would
    // appear at once. _IONBF + ostream::unitbuf makes every print emit
    // one syscall. The cost is trivial (one syscall per token, ~32
    // total per generation).
    setvbuf(stdout, nullptr, _IONBF, 0);
    setvbuf(stderr, nullptr, _IOLBF, 0);
    std::cout.setf(std::ios::unitbuf);

    nnopt_install_crash_handler();
    NNOPT_CHECKPOINT("main() started");

    if (argc < 2) {
        print_usage(argv[0]);
        return 1;
    }

    std::string prompt = argv[1];
    int max_tokens = 64;
    SamplerConfig sampler_config;
    std::string token_ids_file;

    // Parse positional and optional args
    int arg_idx = 2;
    if (arg_idx < argc && argv[arg_idx][0] != '-') {
        max_tokens = std::stoi(argv[arg_idx++]);
    }
    while (arg_idx < argc) {
        std::string flag = argv[arg_idx++];
        if (arg_idx >= argc) { print_usage(argv[0]); return 1; }
        if (flag == "--temperature")       sampler_config.temperature = std::stof(argv[arg_idx++]);
        else if (flag == "--top-k")        sampler_config.top_k = std::stoi(argv[arg_idx++]);
        else if (flag == "--top-p")        sampler_config.top_p = std::stof(argv[arg_idx++]);
        else if (flag == "--repetition-penalty") sampler_config.repetition_penalty = std::stof(argv[arg_idx++]);
        else if (flag == "--seed")         sampler_config.seed = static_cast<uint32_t>(std::stoul(argv[arg_idx++]));
        else if (flag == "--token-ids")    token_ids_file = argv[arg_idx++];
        else { std::cerr << "Unknown flag: " << flag << std::endl; print_usage(argv[0]); return 1; }
    }

    // Initialize OpenCL
    NNOPT_CHECKPOINT("initializing OpenCL");
    OpenCLContext cl_ctx;
    if (!cl_ctx.initialize()) {
        std::cerr << "Failed to initialize OpenCL" << std::endl;
        return 1;
    }
    std::cerr << "Device: " << cl_ctx.device_name() << std::endl;

    // Optional verbose device dump (NNOPT_DEVINFO=1) — extensions, vector
    // widths, image limits, local-mem size. Used to spot under-utilized
    // hardware capabilities (subgroups, half2 vec, etc.).
    if (const char* di = std::getenv("NNOPT_DEVINFO"); di && di[0] == '1') {
        cl_device_id dev = cl_ctx.device();
        auto pull = [&](cl_device_info q, const char* label) {
            char buf[4096] = {0};
            size_t got = 0;
            clGetDeviceInfo(dev, q, sizeof(buf), buf, &got);
            std::cerr << label << ": " << buf << "\n";
        };
        auto pulln = [&](cl_device_info q, const char* label) {
            cl_ulong v = 0;
            clGetDeviceInfo(dev, q, sizeof(v), &v, nullptr);
            std::cerr << label << ": " << v << "\n";
        };
        auto pullsz = [&](cl_device_info q, const char* label) {
            size_t v = 0;
            clGetDeviceInfo(dev, q, sizeof(v), &v, nullptr);
            std::cerr << label << ": " << v << "\n";
        };
        std::cerr << "==DEVINFO==\n";
        pull(CL_DEVICE_VERSION,           "CL_DEVICE_VERSION");
        pull(CL_DEVICE_OPENCL_C_VERSION,  "CL_DEVICE_OPENCL_C_VERSION");
        pull(CL_DRIVER_VERSION,           "CL_DRIVER_VERSION");
        pull(CL_DEVICE_EXTENSIONS,        "CL_DEVICE_EXTENSIONS");
        pulln(CL_DEVICE_MAX_COMPUTE_UNITS,             "MAX_COMPUTE_UNITS");
        pullsz(CL_DEVICE_MAX_WORK_GROUP_SIZE,           "MAX_WORK_GROUP_SIZE");
        pulln(CL_DEVICE_MAX_CLOCK_FREQUENCY,           "MAX_CLOCK_FREQUENCY_MHZ");
        pulln(CL_DEVICE_MAX_MEM_ALLOC_SIZE,            "MAX_MEM_ALLOC_SIZE_BYTES");
        pulln(CL_DEVICE_GLOBAL_MEM_SIZE,               "GLOBAL_MEM_SIZE_BYTES");
        pulln(CL_DEVICE_GLOBAL_MEM_CACHE_SIZE,         "GLOBAL_MEM_CACHE_SIZE_BYTES");
        pulln(CL_DEVICE_GLOBAL_MEM_CACHELINE_SIZE,     "GLOBAL_MEM_CACHELINE_SIZE_BYTES");
        pulln(CL_DEVICE_LOCAL_MEM_SIZE,                "LOCAL_MEM_SIZE_BYTES");
        pulln(CL_DEVICE_PREFERRED_VECTOR_WIDTH_HALF,   "PREF_VEC_WIDTH_HALF");
        pulln(CL_DEVICE_PREFERRED_VECTOR_WIDTH_FLOAT,  "PREF_VEC_WIDTH_FLOAT");
        pulln(CL_DEVICE_NATIVE_VECTOR_WIDTH_HALF,      "NATIVE_VEC_WIDTH_HALF");
        pulln(CL_DEVICE_NATIVE_VECTOR_WIDTH_FLOAT,     "NATIVE_VEC_WIDTH_FLOAT");
        pullsz(CL_DEVICE_IMAGE2D_MAX_WIDTH,             "IMAGE2D_MAX_WIDTH");
        pullsz(CL_DEVICE_IMAGE2D_MAX_HEIGHT,            "IMAGE2D_MAX_HEIGHT");
        pulln(CL_DEVICE_MEM_BASE_ADDR_ALIGN,           "MEM_BASE_ADDR_ALIGN_BITS");
        pulln(CL_DEVICE_HOST_UNIFIED_MEMORY,           "HOST_UNIFIED_MEMORY");
        std::cerr << "==/DEVINFO==\n";
    }
    NNOPT_CHECKPOINT("OpenCL initialized");

    // STREAM bandwidth probe — measures the practical streaming-read
    // ceiling on this device, then exits.
    if (const char* bw = std::getenv("NNOPT_BW_PROBE"); bw && bw[0] == '1') {
        return run_bw_probe(cl_ctx);
    }
    if (const char* rp = std::getenv("NNOPT_RECORD_PROBE"); rp && rp[0] == '1') {
        return run_record_probe(cl_ctx);
    }

    // Load tokenizer (still needed for decoding output)
    NNOPT_CHECKPOINT("loading tokenizer");
    Tokenizer tokenizer;
    bool tokenizer_ok = tokenizer.load("weights/tokenizer_vocab.bin");
    if (!tokenizer_ok && token_ids_file.empty()) {
        std::cerr << "Failed to load tokenizer (use --token-ids to bypass)" << std::endl;
        return 1;
    }
    if (tokenizer_ok) {
        sampler_config.eos_token_id = tokenizer.eos_token_id();
    }
    NNOPT_CHECKPOINT("tokenizer loaded");

    // Load weights — fp16 build pulls weights/model.fp16.bin (smaller, half-precision
    // storage); fp32 build uses weights/model.bin. Both written side-by-side by
    // ConvertWeights so they coexist on device and the right one loads by binary suffix.
    NNOPT_CHECKPOINT("loading weights");
    Weights weights;
#ifdef NNOPT_USE_FP16
    const char* nnopt_weights_bin  = "weights/model.fp16.bin";
    const char* nnopt_weights_meta = "weights/model.fp16.meta.json";
#else
    const char* nnopt_weights_bin  = "weights/model.bin";
    const char* nnopt_weights_meta = "weights/model.meta.json";
#endif
    if (!weights.load(nnopt_weights_bin, nnopt_weights_meta, cl_ctx.context())) {
        std::cerr << "Failed to load weights from " << nnopt_weights_bin << std::endl;
        return 1;
    }
    NNOPT_CHECKPOINT("weights loaded");

    // Create model (initializes layers, compiles kernels, uploads weight buffers)
    NNOPT_CHECKPOINT("creating model (layer init)");
    Model model(cl_ctx, weights);
    NNOPT_CHECKPOINT("model created");

    // Baseline benchmark instrumentation (emits BENCHMARK <key>: <value>
    // lines on stderr; parsed by runUtils.ts parseInferenceMetrics).
    //   inference_start = BEFORE tokenize  — TTFT numerator (matches vLLM / MLPerf / llama-bench:
    //                     TTFT is inclusive of tokenization, prefill, and first-token sampling;
    //                     exclusive of model load / OpenCL init, which are amortized setup).
    //   prefill_start   = right before model.generate() — bounds the forward-pass on the prompt
    //                     (llama-bench 'pp' convention: pure compute throughput, excludes tokenize).
    //   first_token     = stamped by NNOPT_BENCH_FIRST_TOKEN() inside generate(), immediately
    //                     after the first sampled token is appended — do NOT remove the macro
    //                     call from generate() or prefill/decode split collapses to -1.
    BenchmarkTimer& bench = BenchmarkTimer::instance();
    bench.mark_inference_start();

    // Get input token IDs: from file (bypass tokenizer) or by encoding prompt
    std::vector<int> input_ids;
    if (!token_ids_file.empty()) {
        input_ids = load_token_ids_from_file(token_ids_file);
        if (input_ids.empty()) {
            std::cerr << "Failed to load token IDs from " << token_ids_file << std::endl;
            return 1;
        }
        std::cerr << "Loaded " << input_ids.size() << " token IDs from file (tokenizer bypass)" << std::endl;
    } else {
        input_ids = tokenizer.encode(prompt);
        // Dump encode result so FinalizePort can diff C++ tokenizer output
        // against Python's reference_tokens.json.input_ids. Piggybacks on the
        // layer_dumps/ pull plumbing — no extra infrastructure. Written only
        // on the encode path (never with --token-ids).
        {
#ifdef _WIN32
            (void)system("mkdir layer_dumps 2> NUL");
#else
            (void)system("mkdir -p layer_dumps");
#endif
            FILE* tfe = fopen("layer_dumps/tokenizer_encode.json", "w");
            if (tfe) {
                fputs("{\n  \"ids\": [", tfe);
                for (size_t i = 0; i < input_ids.size(); i++) {
                    if (i > 0) fputs(", ", tfe);
                    fprintf(tfe, "%d", input_ids[i]);
                }
                fputs("],\n  \"prompt\": \"", tfe);
                for (char c : prompt) {
                    switch (c) {
                        case '"':  fputs("\\\"", tfe); break;
                        case '\\': fputs("\\\\", tfe); break;
                        case '\n': fputs("\\n", tfe);  break;
                        case '\r': fputs("\\r", tfe);  break;
                        case '\t': fputs("\\t", tfe);  break;
                        default:
                            if ((unsigned char)c < 0x20) fprintf(tfe, "\\u%04x", c);
                            else fputc(c, tfe);
                    }
                }
                fputs("\"\n}\n", tfe);
                fclose(tfe);
            }
        }
    }
    std::cerr << "Input tokens: " << input_ids.size() << " [";
    for (size_t i = 0; i < std::min(input_ids.size(), (size_t)15); i++) {
        if (i > 0) std::cerr << ", ";
        std::cerr << input_ids[i];
    }
    if (input_ids.size() > 15) std::cerr << ", ...";
    std::cerr << "]" << std::endl;
    std::cerr << "Sampling: temp=" << sampler_config.temperature
              << " top_k=" << sampler_config.top_k
              << " top_p=" << sampler_config.top_p
              << " rep_penalty=" << sampler_config.repetition_penalty
              << " eos=" << sampler_config.eos_token_id << std::endl;

#ifdef NNOPT_DEBUG
    // EMBEDDING_VERIFY: host-side wte readback for input_ids[0]. Bypasses the
    // embedding kernel and the LAYER_CHECK readback path, so the printed values
    // come strictly from the weight upload. Compare to
    // reference/layers/embedding_wte_output.bin first 8 floats:
    //   - match  -> bug is in the embedding kernel or LAYER_CHECK dump path
    //   - differ -> bug is in weight selection/upload (wrong meta, wrong dtype,
    //               wrong row stride). Check meta.dtype, model.fp16.bin vs model.bin.
    if (!input_ids.empty()) {
        const char* embed_key_candidates[] = {
            "backbone.embeddings.weight",
            "transformer.wte.weight",
            "model.embed_tokens.weight",
            "embed_tokens.weight",
            "tok_embeddings.weight",
        };
        cl_mem wte_buf = nullptr;
        const char* wte_key = nullptr;
        for (const char* k : embed_key_candidates) {
            wte_buf = weights.get_buffer(k, true);
            if (wte_buf) { wte_key = k; break; }
        }
        if (wte_buf) {
            const int tok = input_ids[0];
            const int n = 8;
            const size_t hsz = (size_t)MODEL_CONFIG::HIDDEN_SIZE;
#ifdef NNOPT_USE_FP16
            const size_t row_off = (size_t)tok * hsz * 2;
            std::vector<uint16_t> raw(n);
            cl_int e = clEnqueueReadBuffer(cl_ctx.queue(), wte_buf, CL_TRUE,
                row_off, (size_t)n * 2, raw.data(), 0, nullptr, nullptr);
            if (e == CL_SUCCESS) {
                fprintf(stderr, "EMBEDDING_VERIFY key=%s token=%d first%d:", wte_key, tok, n);
                for (int i = 0; i < n; i++)
                    fprintf(stderr, " %.4f", _nnopt_dbg_f16_to_f32(raw[i]));
                fprintf(stderr, "\n");
            } else {
                fprintf(stderr, "EMBEDDING_VERIFY: clEnqueueReadBuffer failed err=%d\n", e);
            }
#else
            const size_t row_off = (size_t)tok * hsz * 4;
            std::vector<float> raw(n);
            cl_int e = clEnqueueReadBuffer(cl_ctx.queue(), wte_buf, CL_TRUE,
                row_off, (size_t)n * 4, raw.data(), 0, nullptr, nullptr);
            if (e == CL_SUCCESS) {
                fprintf(stderr, "EMBEDDING_VERIFY key=%s token=%d first%d:", wte_key, tok, n);
                for (int i = 0; i < n; i++) fprintf(stderr, " %.4f", raw[i]);
                fprintf(stderr, "\n");
            } else {
                fprintf(stderr, "EMBEDDING_VERIFY: clEnqueueReadBuffer failed err=%d\n", e);
            }
#endif
            fflush(stderr);
        } else {
            fprintf(stderr, "EMBEDDING_VERIFY: no embedding weight key matched any of the common candidates\n");
            fflush(stderr);
        }
    }
#endif

    // Generate
    NNOPT_CHECKPOINT("starting generation");
    Timer timer;
    timer.start();

    // ── Streaming UX setup ──────────────────────────────────────────
    // Print the prompt prefix (decoded back through the tokenizer for
    // round-trip parity), then stream each new token's text as it's
    // produced by Model::generate via the on_token callback. We compute
    // the delta against the running emitted-byte count so multi-byte
    // (UTF-8 / CJK) tokens always print as complete characters — if a
    // partial UTF-8 sequence sits at the tail, decode().size() doesn't
    // grow until the next token completes the codepoint, and the
    // `text.size() > emitted_len` guard keeps us from printing garbage.
    std::string emitted_text;
    if (tokenizer_ok) {
        emitted_text = tokenizer.decode(input_ids);
        std::cout << emitted_text << std::flush;
    }
    const bool dbg_token_ids =
        []() { const char* d = std::getenv("NNOPT_DEBUG_LAYERS"); return d && d[0] != '0'; }();

    auto on_token = [&](int32_t new_tok, const std::vector<int32_t>& all_ids) {
        if (dbg_token_ids) {
            fprintf(stderr, "Generated token: %d\n", new_tok);
        }
        if (!tokenizer_ok) return;
        std::string full = tokenizer.decode(all_ids);
        if (full.size() > emitted_text.size()) {
            std::cout << full.substr(emitted_text.size()) << std::flush;
            emitted_text = std::move(full);
        }
    };

    // Prefill clock starts now — excludes tokenize (see comment block above the
    // mark_inference_start() call for the full metric contract).
    bench.mark_prefill_start();
    auto output_ids = model.generate(input_ids, max_tokens, sampler_config, on_token);
    bench.mark_end();
    double elapsed = timer.elapsed_ms();
    int gen_tokens = output_ids.size() - input_ids.size();
    NNOPT_CHECKPOINT("generation complete");

    // End the streaming line. If tokenizer wasn't loaded we never streamed
    // anything; print the raw IDs so callers (e.g. pipeline validators)
    // still get something deterministic on stdout.
    if (tokenizer_ok) {
        std::cout << std::endl;
    } else {
        std::cout << "Token IDs: ";
        for (size_t i = 0; i < output_ids.size(); ++i) {
            if (i > 0) std::cout << " ";
            std::cout << output_ids[i];
        }
        std::cout << std::endl;
    }

    // Stats (human-readable summary — kept for backward compat with old parsers)
    std::cerr << "Generated " << gen_tokens << " tokens in "
              << elapsed << " ms ("
              << (gen_tokens * 1000.0 / elapsed) << " tokens/sec)" << std::endl;

    // Structured baseline metrics for FinalizePort / README generation.
    bench.print_summary((int)input_ids.size(), gen_tokens);

    return 0;
}
