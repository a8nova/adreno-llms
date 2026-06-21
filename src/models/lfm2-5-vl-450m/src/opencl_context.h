#pragma once
#include <string>
#include <vector>

// Define OpenCL version BEFORE including headers to avoid version conflicts
#define CL_TARGET_OPENCL_VERSION 120

// Always use portable Khronos headers (works for cross-compilation to Android)
#include <CL/cl.h>

// ── cl_qcom_recordable_queues vendor extension typedefs ──
// Struct layouts from MNN's vendored cl_ext_qcom.h (which has Qualcomm's
// official header). dispatch_index is an index into the recorded sequence,
// NOT a kernel pointer (this was the bug in the prior Qwen attempt).
// Adreno hides these entry points from clGetExtensionFunctionAddressForPlatform
// — must dlsym from RTLD_DEFAULT.
namespace nnopt_qcom_rec {
    typedef struct _cl_recording_qcom_opaque* cl_recording_qcom;

    typedef struct {
        cl_uint dispatch_index;
        cl_uint arg_index;
        size_t  arg_size;
        const void* arg_value;
    } cl_array_arg_qcom;

    typedef struct {
        cl_uint dispatch_index;
        const size_t* workgroup_size;
    } cl_workgroup_qcom;

    typedef struct {
        cl_uint dispatch_index;
        size_t offsets[3];
    } cl_offset_qcom;

    typedef cl_recording_qcom (CL_API_CALL *fn_new_t)(cl_command_queue, cl_int*);
    typedef cl_int (CL_API_CALL *fn_end_t)(cl_recording_qcom);
    typedef cl_int (CL_API_CALL *fn_release_t)(cl_recording_qcom);
    typedef cl_int (CL_API_CALL *fn_retain_t)(cl_recording_qcom);
    typedef cl_int (CL_API_CALL *fn_enqueue_t)(
        cl_command_queue, cl_recording_qcom,
        size_t, const cl_array_arg_qcom*,
        size_t, const cl_offset_qcom*,
        size_t, const cl_workgroup_qcom*,
        size_t, const cl_workgroup_qcom*,
        cl_uint, const cl_event*, cl_event*);

    // CL_QUEUE_RECORDABLE_QCOM = (1u << 30u) per Qualcomm header.
    constexpr cl_command_queue_properties RECORDABLE_BIT =
        (cl_command_queue_properties)1 << 30;
}

class OpenCLContext {
public:
    OpenCLContext();
    ~OpenCLContext();

    bool initialize(int platform_idx = 0, int device_idx = 0);
    cl_program build_program(const std::string& source, const std::string& options = "");
    cl_program build_program_from_file(const std::string& path, const std::string& options = "");

    cl_context context() const { return context_; }
    cl_command_queue queue() const { return queue_; }
    cl_command_queue recordable_queue() const { return recordable_queue_; }
    cl_device_id device() const { return device_; }

    // Vendor recordable-queues API (nullptr if extension unavailable).
    nnopt_qcom_rec::fn_new_t     rec_fn_new()     const { return rec_fn_new_; }
    nnopt_qcom_rec::fn_end_t     rec_fn_end()     const { return rec_fn_end_; }
    nnopt_qcom_rec::fn_release_t rec_fn_release() const { return rec_fn_release_; }
    nnopt_qcom_rec::fn_enqueue_t rec_fn_enqueue() const { return rec_fn_enqueue_; }
    bool recordable_queues_supported() const { return rec_fn_new_ != nullptr && recordable_queue_ != nullptr; }

    // Drain the GPU queue and briefly sleep so the foreground compositor can
    // grab the Adreno's single CU between bursts of compute. Without this, long
    // GPU passes (the SigLIP vision tower + the LM prefill of ~1300 image
    // tokens) saturate the GPU for ~1-2 min and the app's UI thread starves →
    // ANR ("isn't responding"). Gated by NNOPT_GPU_YIELD env (unset in bench
    // builds for max throughput). Cost: blocks the caller for `sleep_ms`; call
    // once per layer, not per kernel. Ported from smolvlm-256m-instruct.
    void yield_for_compositor(int sleep_ms = 12);

    // Device info
    std::string device_name() const;
    size_t max_work_group_size() const;
    size_t local_mem_size() const;

private:
    cl_platform_id platform_ = nullptr;
    cl_device_id device_ = nullptr;
    cl_context context_ = nullptr;
    cl_command_queue queue_ = nullptr;
    cl_command_queue recordable_queue_ = nullptr;

    nnopt_qcom_rec::fn_new_t     rec_fn_new_     = nullptr;
    nnopt_qcom_rec::fn_end_t     rec_fn_end_     = nullptr;
    nnopt_qcom_rec::fn_release_t rec_fn_release_ = nullptr;
    nnopt_qcom_rec::fn_enqueue_t rec_fn_enqueue_ = nullptr;
};
