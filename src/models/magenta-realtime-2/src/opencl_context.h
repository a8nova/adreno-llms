#pragma once
#include <string>
#include <vector>

// Define OpenCL version BEFORE including headers to avoid version conflicts
#define CL_TARGET_OPENCL_VERSION 120

// Always use portable Khronos headers (works for cross-compilation to Android)
#include <CL/cl.h>

// ── cl_qcom_recordable_queues (not in stock cl_ext.h) ───────────────────────────────────────────
// A recordable queue CAPTURES dispatches instead of executing them; the capture is replayed on a
// live queue with per-replay argument overrides. The AR issues ~455 dispatches per frame with an
// identical structure every frame — only a position scalar changes — which is precisely the case
// guide 80-NB295-11 Rev C 9.1.3 describes. Recipe below is probe-validated in this repo by
// src/models/pocket-tts (Adreno 620): CL_QUEUE_RECORDABLE_QCOM is bit 30 ALONE, never combined with
// PROFILING_ENABLE, and pocket-tts measured replay at ~4x cheaper per dispatch than live enqueue.
//
// CLBlast cannot be recorded, so this covers the AR (our own kernels) and not the codec.
typedef void* cl_recording_qcom;
struct cl_array_arg_qcom {
    cl_kernel   kernel;
    cl_uint     arg_indx;
    size_t      arg_size;
    const void* arg_value;
};
struct cl_array_kernel_exec_info_qcom {
    cl_kernel   kernel;
    cl_uint     indx;
    size_t      param_value_size;
    const void* param_value;
};
typedef cl_recording_qcom (CL_API_CALL *clNewRecordingQCOM_fn)(cl_command_queue, cl_int*);
typedef cl_int (CL_API_CALL *clEndRecordingQCOM_fn)(cl_recording_qcom);
typedef cl_int (CL_API_CALL *clReleaseRecordingQCOM_fn)(cl_recording_qcom);
typedef cl_int (CL_API_CALL *clEnqueueRecordingQCOM_fn)(
    cl_command_queue, cl_recording_qcom,
    size_t, const cl_array_arg_qcom*,
    size_t, const cl_array_kernel_exec_info_qcom*,
    size_t, const cl_array_kernel_exec_info_qcom*,
    size_t, const cl_array_kernel_exec_info_qcom*,
    cl_uint, const cl_event*, cl_event*);

class OpenCLContext {
public:
    OpenCLContext();
    ~OpenCLContext();

    bool initialize(int platform_idx = 0, int device_idx = 0);
    cl_program build_program(const std::string& source, const std::string& options = "");
    cl_program build_program_from_file(const std::string& path, const std::string& options = "");

    cl_context context() const { return context_; }
    cl_command_queue queue() const { return queue_; }
    cl_device_id device() const { return device_; }

    // ── Multi-queue overlap (AR ∥ codec) ──────────────────────────────────
    // profEnqueue() submits to active_queue_ (defaults to queue_). The pipelined
    // driver flips this so AR kernels land on queue 1 and codec kernels on a 2nd
    // queue, letting the GPU-bound codec fill the host-dispatch-bound AR's GPU idle.
    cl_command_queue activeQueue() const;
    void setActiveQueue(cl_command_queue q) { active_queue_ = q ? q : queue_; }
    // Per-THREAD override, consulted before active_queue_. The codec runs on its own host thread so
    // it can issue while the AR keeps generating; without this the two threads would fight over one
    // shared "current queue" member and each would steal the other's dispatches.
    // Host-side dispatch cost — see the note in the .cpp. The AR is host-bound, so this is the
    // profile that ranks the work; the GPU one shows an empty timeline.
    static void hostProfReset();
    static void hostProfReport(double frame_ms, int frames);
    static void hostProfTotals(int frames, long* disp_per_frame, double* us_per_dispatch);
    // GPU kernel-execution ms for the ONE profiled frame, or 0 when not profiling.
    static double arProfFrameMs();
    static void setThreadQueue(cl_command_queue q);
    static cl_command_queue threadQueue();

    // ── Qualcomm-guide cl_event GPU profiler ─────────────────────────────
    // Enqueues a kernel; when NNOPT_PROFILE is set, captures the command's
    // CL_PROFILING_COMMAND_START/END and accumulates per-kernel GPU time.
    // Non-blocking during the run (events read at profReport()).
    cl_int profEnqueue(cl_kernel k, cl_uint dim, const size_t* gws, const size_t* lws, const char* name);
    // Result of the one-time workgroup sweep for a (kernel, global-size). use==false ⇒ driver default.
    struct LwsChoice { bool use = false; size_t l[3] = {0,0,0}; };
    LwsChoice tune_lws(cl_kernel k, cl_uint dim, const size_t* gws, const char* name);
    // The workgroup sweep TIMES A KERNEL BY RE-RUNNING IT with its live arguments, so it is only
    // safe where the result is thrown away: several kernels here accumulate in place (`ola` is an
    // overlap-add, `add`/`bias` write back into their input) and extra executions change the data.
    // Opened for the warm-up render, closed before SERVE_READY; afterwards untuned shapes simply
    // keep the driver default.
    static void setLwsTuningWindow(bool open);
    static bool lwsTuningWindowOpen();

    // ── recordable queues ──
    bool has_recordable_queues() const { return record_fns_loaded_; }
    cl_command_queue create_recordable_queue();
    cl_recording_qcom new_recording(cl_command_queue q) const;
    cl_int end_recording(cl_recording_qcom rec) const;
    cl_int release_recording(cl_recording_qcom rec) const;
    cl_int enqueue_recording(cl_command_queue live_q, cl_recording_qcom rec,
                             size_t num_args, const cl_array_arg_qcom* args) const;
    // Profile ONE live AR frame's GPU time, always on, no flag. Answers "where does the frame go"
    // now that host issue is no longer the bottleneck.
    void   arProfBegin(const char* tag = "AR_FRAME_GPU");
    void   arProfEnd(cl_command_queue q);

    bool   record_probe();   // verify capture+replay reproduces a live dispatch on THIS device
    // Does enqueueing to a recordable queue ALSO execute the work, or only record it? The AR has to
    // know: if capture does not execute, the captured frame's work never happened and its slot in
    // the KV cache and token grid is a hole. Determined once on the device by a one-dispatch probe
    // (never guessed, never a flag), cached. 1 = executes, 0 = records only, -1 = undetermined.
    int    record_capture_executes();
    static bool profEnabled();
    // Re-read NNOPT_PROFILE. Without this the flag latches at first use, so a serve process that
    // warmed up with profiling off could never be asked for a profile later — which is exactly the
    // case that matters, because the only way onto a device-farm phone is a request to a warm app.
    static void profRefresh();
    // Ask for an event slot for a kernel WE do not enqueue (CLBlast): returns nullptr when
    // profiling is off, so the callee is asked for no event and pays nothing. Pass the filled event
    // straight back via profEventAdd — do NOT retain the pointer, it is typically a stack local.
    // Without this the codec's GEMMs, its single biggest cost, are invisible to every profile.
    static bool profWantEvent();
    static void profEventAdd(const char* label, cl_event ev);   // takes ownership of ev
    static void profReport();   // prints sorted per-kernel GPU-time breakdown, then clears

    // Component (pipeline-stage) timing: compTic() starts; each compAccum(name) does a clFinish and
    // adds the elapsed wall-time since the previous tic/accum into that component. Profile-gated.
    void compTic();
    void compAccum(const char* name);
    static void compReport();   // prints sorted per-component wall-time breakdown, then clears

    // Device info
    std::string device_name() const;
    size_t max_work_group_size() const;
    size_t local_mem_size() const;

private:
    friend cl_mem pool_alloc(cl_context, cl_mem_flags, size_t, void*, cl_int*);
    cl_platform_id platform_ = nullptr;
    cl_device_id device_ = nullptr;
    cl_context context_ = nullptr;
    cl_command_queue queue_ = nullptr;
    cl_command_queue active_queue_ = nullptr;  // profEnqueue target; nullptr ⇒ queue_
    bool record_fns_loaded_ = false;
    clNewRecordingQCOM_fn     fn_new_recording_     = nullptr;
    clEndRecordingQCOM_fn     fn_end_recording_     = nullptr;
    clReleaseRecordingQCOM_fn fn_release_recording_ = nullptr;
    clEnqueueRecordingQCOM_fn fn_enqueue_recording_ = nullptr;
    void load_record_fns();
};

// ── Buffer pool (OPT #2) ──────────────────────────────────────────────────────
// Drop-in replacements for clCreateBuffer / clReleaseMemObject on the hot path.
// Only plain CL_MEM_READ_WRITE, no-host-ptr buffers are recycled (size-bucketed
// free lists); every other flag combo is forwarded to clCreateBuffer untouched.
// pool_free is tolerant: a buffer it never issued is just clReleaseMemObject'd.
// This kills the ~360 alloc/free pairs per AR frame. NNOPT_POOL=0 → passthrough.
cl_mem pool_alloc(cl_context ctx, cl_mem_flags flags, size_t bytes, void* host_ptr, cl_int* err);
// Hold every buffer freed while pinning is on, so a recorded frame's handles cannot be recycled and
// handed to the codec running concurrently on the other queue.
void pool_set_pin(bool on);
// Hand back the buffers retained while the pin was on. Call only AFTER the queues are drained:
// replays reference these buffers, and freeing them with work in flight is a GPU-side
// use-after-free, which is a device reset rather than an error code.
void pool_release_pinned();
void   pool_free(cl_mem m);
