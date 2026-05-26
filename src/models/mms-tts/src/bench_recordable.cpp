// Microbenchmark: does cl_qcom_recordable_queues actually reduce the
// QUEUED→START dispatch gap on Adreno 620?
//
// Strategy: dispatch a trivial kernel N times, two ways:
//   (a) plain clEnqueueNDRangeKernel ×N
//   (b) record once, then clEnqueueRecordingQCOM ×(N-1)
// Measure wall time + GPU profiling events. If (b) is materially faster
// than (a), the full vocoder integration is justified. If not, the
// extension doesn't help this driver and we pivot.
//
// Build: linked as a separate executable; pushed alongside the main binary.

#include <CL/cl.h>
#include <dlfcn.h>
#include <cstdio>
#include <cstring>
#include <chrono>
#include <vector>

// Same inline declarations as opencl_context.h — keep this file self-contained.
typedef void* cl_recording_qcom;
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
typedef cl_recording_qcom (CL_API_CALL *PFN_New)(cl_command_queue, cl_int*);
typedef cl_int (CL_API_CALL *PFN_End)(cl_recording_qcom);
typedef cl_int (CL_API_CALL *PFN_Rel)(cl_recording_qcom);
typedef cl_int (CL_API_CALL *PFN_Enq)(
    cl_command_queue, cl_recording_qcom,
    size_t, const cl_array_arg_qcom*,
    size_t, const cl_array_kernel_exec_info_qcom*,
    size_t, const cl_array_kernel_exec_info_qcom*,
    size_t, const cl_array_kernel_exec_info_qcom*,
    cl_uint, const cl_event*, cl_event*);

static const char* kSrc =
    "__kernel void noop(__global int* x, const int v) {\n"
    "  int gid = get_global_id(0);\n"
    "  if (gid < 1) x[0] = v;\n"
    "}\n";

static double now_ms() {
    using namespace std::chrono;
    return duration<double, std::milli>(steady_clock::now().time_since_epoch()).count();
}

int main() {
    cl_uint np = 0; clGetPlatformIDs(0, nullptr, &np);
    if (!np) { fprintf(stderr, "no platforms\n"); return 1; }
    std::vector<cl_platform_id> plats(np); clGetPlatformIDs(np, plats.data(), nullptr);
    cl_platform_id plat = plats[0];

    cl_uint nd = 0; clGetDeviceIDs(plat, CL_DEVICE_TYPE_GPU, 0, nullptr, &nd);
    if (!nd) { fprintf(stderr, "no GPU\n"); return 1; }
    std::vector<cl_device_id> devs(nd); clGetDeviceIDs(plat, CL_DEVICE_TYPE_GPU, nd, devs.data(), nullptr);
    cl_device_id dev = devs[0];

    cl_int err = CL_SUCCESS;
    cl_context ctx = clCreateContext(nullptr, 1, &dev, nullptr, nullptr, &err);
    if (err != CL_SUCCESS) { fprintf(stderr, "ctx err=%d\n", err); return 1; }

    // Recording bit per LFM2 (validated on same A6xx driver family): bit 30.
    constexpr cl_command_queue_properties RECORD_BIT = (cl_command_queue_properties)1 << 30;
    cl_command_queue qlive = clCreateCommandQueue(ctx, dev, CL_QUEUE_PROFILING_ENABLE, &err);
    if (err != CL_SUCCESS) { fprintf(stderr, "qlive err=%d\n", err); return 1; }
    cl_command_queue qrec = clCreateCommandQueue(ctx, dev, RECORD_BIT, &err);
    if (err != CL_SUCCESS) { fprintf(stderr, "qrec err=%d\n", err); return 1; }

    // Build noop kernel
    cl_program prog = clCreateProgramWithSource(ctx, 1, &kSrc, nullptr, &err);
    if (err != CL_SUCCESS) { fprintf(stderr, "createProgram err=%d\n", err); return 1; }
    err = clBuildProgram(prog, 1, &dev, "", nullptr, nullptr);
    if (err != CL_SUCCESS) {
        size_t lsz = 0; clGetProgramBuildInfo(prog, dev, CL_PROGRAM_BUILD_LOG, 0, nullptr, &lsz);
        std::vector<char> log(lsz); clGetProgramBuildInfo(prog, dev, CL_PROGRAM_BUILD_LOG, lsz, log.data(), nullptr);
        fprintf(stderr, "buildProgram err=%d log=%s\n", err, log.data());
        return 1;
    }
    cl_kernel k = clCreateKernel(prog, "noop", &err);
    if (err != CL_SUCCESS) { fprintf(stderr, "createKernel err=%d\n", err); return 1; }

    cl_mem buf = clCreateBuffer(ctx, CL_MEM_READ_WRITE, 64, nullptr, &err);
    if (err != CL_SUCCESS) { fprintf(stderr, "createBuffer err=%d\n", err); return 1; }
    int v0 = 0;
    clSetKernelArg(k, 0, sizeof(cl_mem), &buf);
    clSetKernelArg(k, 1, sizeof(int), &v0);

    constexpr int N = 100;
    const size_t gws = 1;

    // Warmup
    for (int i = 0; i < 5; ++i) {
        clEnqueueNDRangeKernel(qlive, k, 1, nullptr, &gws, nullptr, 0, nullptr, nullptr);
    }
    clFinish(qlive);

    // ── (a) Plain enqueues ──────────────────────────────────────────
    double a0 = now_ms();
    for (int i = 0; i < N; ++i) {
        int v = i;
        clSetKernelArg(k, 1, sizeof(int), &v);
        clEnqueueNDRangeKernel(qlive, k, 1, nullptr, &gws, nullptr, 0, nullptr, nullptr);
    }
    clFinish(qlive);
    double a1 = now_ms();
    printf("(a) plain enqueues   N=%d wall=%.2f ms  (%.2f ms/call)\n",
           N, a1 - a0, (a1 - a0) / N);

    // Resolve extension function pointers
    auto fnNew = (PFN_New)dlsym(RTLD_DEFAULT, "clNewRecordingQCOM");
    auto fnEnd = (PFN_End)dlsym(RTLD_DEFAULT, "clEndRecordingQCOM");
    auto fnRel = (PFN_Rel)dlsym(RTLD_DEFAULT, "clReleaseRecordingQCOM");
    auto fnEnq = (PFN_Enq)dlsym(RTLD_DEFAULT, "clEnqueueRecordingQCOM");
    if (!fnNew || !fnEnd || !fnRel || !fnEnq) {
        fprintf(stderr, "function pointer load failed\n"); return 1;
    }

    // ── Build recording on qrec ─────────────────────────────────────
    cl_int rec_err = CL_SUCCESS;
    cl_recording_qcom rec = fnNew(qrec, &rec_err);
    if (rec_err != CL_SUCCESS || !rec) {
        fprintf(stderr, "clNewRecordingQCOM err=%d rec=%p\n", rec_err, rec); return 1;
    }
    // Record ONE kernel dispatch. Replays will execute that one dispatch.
    int v_rec = 7;
    clSetKernelArg(k, 1, sizeof(int), &v_rec);
    err = clEnqueueNDRangeKernel(qrec, k, 1, nullptr, &gws, nullptr, 0, nullptr, nullptr);
    if (err != CL_SUCCESS) { fprintf(stderr, "record enqueue err=%d\n", err); return 1; }
    err = fnEnd(rec);
    if (err != CL_SUCCESS) { fprintf(stderr, "endRecording err=%d\n", err); return 1; }

    // Try replay first with NO updates (simplest case).
    {
        cl_int e = fnEnq(qlive, rec, 0, nullptr, 0, nullptr, 0, nullptr, 0, nullptr, 0, nullptr, nullptr);
        printf("replay-no-updates err=%d\n", e);
        if (e == CL_SUCCESS) {
            clFinish(qlive);
            printf("replay-no-updates: SUCCESS\n");
        }
    }
    // Warmup replays (no arg updates — just exercise the replay path)
    for (int i = 0; i < 5; ++i) {
        fnEnq(qlive, rec, 0, nullptr, 0, nullptr, 0, nullptr, 0, nullptr, 0, nullptr, nullptr);
    }
    clFinish(qlive);

    // ── (b) Recording replays (no arg updates — pure dispatch-cost A/B) ──
    double b0 = now_ms();
    for (int i = 0; i < N; ++i) {
        cl_int e = fnEnq(qlive, rec, 0, nullptr, 0, nullptr, 0, nullptr, 0, nullptr, 0, nullptr, nullptr);
        if (e != CL_SUCCESS) { fprintf(stderr, "replay enqueue err=%d (i=%d)\n", e, i); return 1; }
    }
    clFinish(qlive);
    double b1 = now_ms();
    printf("(b) record + replay  N=%d wall=%.2f ms  (%.2f ms/call)\n",
           N, b1 - b0, (b1 - b0) / N);

    const double ratio = (a1 - a0) > 0 ? (a1 - a0) / (b1 - b0) : 0.0;
    printf("speedup (a/b) = %.2fx\n", ratio);

    fnRel(rec);
    clReleaseMemObject(buf);
    clReleaseKernel(k);
    clReleaseProgram(prog);
    clReleaseCommandQueue(qrec);
    clReleaseCommandQueue(qlive);
    clReleaseContext(ctx);

    if (ratio > 1.5) {
        puts("VERDICT: recordable queues materially reduce dispatch latency. Full vocoder integration justified.");
        return 0;
    } else if (ratio > 1.1) {
        puts("VERDICT: marginal speedup. Vocoder integration may help but risk/reward is unclear.");
        return 0;
    } else {
        puts("VERDICT: no meaningful speedup. Extension doesn't help on this driver. Abort this path.");
        return 0;
    }
}
