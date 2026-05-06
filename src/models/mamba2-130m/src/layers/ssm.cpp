// Reference: model_info/transformers_src/modeling_mamba2.py:1-520 (Mamba2Mixer.torch_forward, segment_sum helpers)

#include "layers/ssm.h"

#include "debug_utils.h"
#include "kernel_profiler.h"
#include "opencl_context.h"
#include "utils.h"
#include "weights.h"

#include <clblast.h>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

Ssm::Ssm(OpenCLContext& cl_ctx, Weights& weights, const std::string& prefix, int layer_idx)
    : cl_ctx_(cl_ctx), weights_(weights), prefix_(prefix), layer_idx_(layer_idx) {}

Ssm::~Ssm() {
    if (conv_kernel_) clReleaseKernel(conv_kernel_);
    if (conv_silu_kernel_) clReleaseKernel(conv_silu_kernel_);
    if (update_conv_state_kernel_) clReleaseKernel(update_conv_state_kernel_);
    if (silu_inplace_kernel_) clReleaseKernel(silu_inplace_kernel_);
    if (conv_program_) clReleaseProgram(conv_program_);

    if (scan_kernel_) clReleaseKernel(scan_kernel_);
    if (softplus_kernel_) clReleaseKernel(softplus_kernel_);
    if (silu_mul_kernel_) clReleaseKernel(silu_mul_kernel_);
    if (ssd_scan_kernel_) clReleaseKernel(ssd_scan_kernel_);
    if (ssd_scan_coop_kernel_) clReleaseKernel(ssd_scan_coop_kernel_);
    if (ssd_scan_t_kernel_) clReleaseKernel(ssd_scan_t_kernel_);
    if (scan_program_) clReleaseProgram(scan_program_);

    if (norm_kernel_) clReleaseKernel(norm_kernel_);
    if (norm_program_) clReleaseProgram(norm_program_);

    if (ssm_state_) clReleaseMemObject(ssm_state_);
    if (conv_state_) clReleaseMemObject(conv_state_);

    release_activation_buffers_();
}

void Ssm::release_activation_buffers_() {
    if (buf_projected_states_) { clReleaseMemObject(buf_projected_states_); buf_projected_states_ = nullptr; }
    if (buf_gate_)              { clReleaseMemObject(buf_gate_);              buf_gate_ = nullptr; }
    if (buf_rest_after_gate_)   { clReleaseMemObject(buf_rest_after_gate_);   buf_rest_after_gate_ = nullptr; }
    if (buf_hidden_states_B_C_) { clReleaseMemObject(buf_hidden_states_B_C_); buf_hidden_states_B_C_ = nullptr; }
    if (buf_dt_raw_)            { clReleaseMemObject(buf_dt_raw_);            buf_dt_raw_ = nullptr; }
    if (buf_hbc_conv_)          { clReleaseMemObject(buf_hbc_conv_);          buf_hbc_conv_ = nullptr; }
    if (buf_x_inner_)           { clReleaseMemObject(buf_x_inner_);           buf_x_inner_ = nullptr; }
    if (buf_bc_)                { clReleaseMemObject(buf_bc_);                buf_bc_ = nullptr; }
    if (buf_B_)                 { clReleaseMemObject(buf_B_);                 buf_B_ = nullptr; }
    if (buf_C_)                 { clReleaseMemObject(buf_C_);                 buf_C_ = nullptr; }
    if (buf_y_scan_)            { clReleaseMemObject(buf_y_scan_);            buf_y_scan_ = nullptr; }
    if (buf_normed_)            { clReleaseMemObject(buf_normed_);            buf_normed_ = nullptr; }
    if (buf_out_)               { clReleaseMemObject(buf_out_);               buf_out_ = nullptr; }
    if (buf_gate_dec_)          { clReleaseMemObject(buf_gate_dec_);          buf_gate_dec_ = nullptr; }
    if (buf_hbc_dec_)           { clReleaseMemObject(buf_hbc_dec_);           buf_hbc_dec_ = nullptr; }
    if (buf_dt_dec_)            { clReleaseMemObject(buf_dt_dec_);            buf_dt_dec_ = nullptr; }
    if (buf_x_inner_dec_)       { clReleaseMemObject(buf_x_inner_dec_);       buf_x_inner_dec_ = nullptr; }
    if (buf_bc_dec_)            { clReleaseMemObject(buf_bc_dec_);            buf_bc_dec_ = nullptr; }
    if (buf_B_dec_)             { clReleaseMemObject(buf_B_dec_);             buf_B_dec_ = nullptr; }
    if (buf_C_dec_)             { clReleaseMemObject(buf_C_dec_);             buf_C_dec_ = nullptr; }
    buf_decode_subbufs_ready_ = false;
    buf_capacity_rows_ = 0;
}

bool Ssm::ensure_activation_buffers_(int rows) {
    if (rows <= buf_capacity_rows_ && buf_out_ != nullptr) {
        // Early return: capacity-fit. But Lever F's split-elimination only
        // applies when the buffers were created at rows==1 layout. If we
        // grew first for prefill (rows>1) and now at decode (rows=1), the
        // sub-buffer views were never created, so splits must run.
        // use_subbuffer_splits_ stays at whatever ensure_ first set it to.
        return true;
    }
    // Grow: drop existing and reallocate at the new (larger) row count.
    release_activation_buffers_();

    cl_int err = CL_SUCCESS;
    auto make = [&](cl_mem* dst, size_t cols, const char* name) -> bool {
        if (cols == 0) {
            NNOPT_ERROR_FMT("Ssm[%d]: ensure_activation_buffers_(%s) cols==0", layer_idx_, name);
            return false;
        }
        cl_int e = CL_SUCCESS;
        *dst = clCreateBuffer(cl_ctx_.context(), CL_MEM_READ_WRITE,
                              (size_t)rows * cols * sizeof(nnopt_storage_t), nullptr, &e);
        if (e != CL_SUCCESS || !*dst) {
            NNOPT_ERROR_FMT("Ssm[%d]: ensure_activation_buffers_(%s) failed (%d)", layer_idx_, name, (int)e);
            err = e;
            return false;
        }
        return true;
    };

    auto make_subbuf = [&](cl_mem* dst, cl_mem parent, size_t off_bytes, size_t size_bytes,
                           const char* name) -> bool {
        if (size_bytes == 0) return false;
        cl_int e = CL_SUCCESS;
        cl_buffer_region region = { off_bytes, size_bytes };
        *dst = clCreateSubBuffer(parent, CL_MEM_READ_WRITE,
                                 CL_BUFFER_CREATE_TYPE_REGION, &region, &e);
        if (e != CL_SUCCESS || !*dst) {
            NNOPT_ERROR_FMT("Ssm[%d]: clCreateSubBuffer(%s) failed (%d) — Lever F decode-skip disabled",
                            layer_idx_, name, (int)e);
            *dst = nullptr;
            return false;
        }
        return true;
    };

    const int projection_size = d_inner_ + conv_ch_ + n_v_heads_;
    const int rest_cols = projection_size - d_inner_;
    const int bc_cols = conv_ch_ - d_inner_;
    const int gc_cols = bc_cols / 2;

    if (!make(&buf_projected_states_,    (size_t)projection_size, "projected_states")) return false;
    if (!make(&buf_gate_,                (size_t)d_inner_,        "gate"))             return false;
    if (!make(&buf_rest_after_gate_,     (size_t)rest_cols,       "rest_after_gate"))  return false;
    if (!make(&buf_hidden_states_B_C_,   (size_t)conv_ch_,        "hidden_states_B_C")) return false;
    if (!make(&buf_dt_raw_,              (size_t)n_v_heads_,      "dt_raw"))           return false;
    if (!make(&buf_hbc_conv_,            (size_t)conv_ch_,        "hbc_conv"))         return false;
    if (!make(&buf_x_inner_,             (size_t)d_inner_,        "x_inner"))          return false;
    if (!make(&buf_bc_,                  (size_t)bc_cols,         "bc"))               return false;
    if (!make(&buf_B_,                   (size_t)gc_cols,         "B"))                return false;
    if (!make(&buf_C_,                   (size_t)gc_cols,         "C"))                return false;
    if (!make(&buf_y_scan_,              (size_t)d_inner_,        "y_scan"))           return false;
    if (!make(&buf_normed_,              (size_t)d_inner_,        "normed"))           return false;
    if (!make(&buf_out_,                 (size_t)d_model_,        "out"))              return false;

    buf_capacity_rows_ = rows;

    // Lever F: also create decode-only subbuffer aliases. These are zero-copy
    // single-row views into the parents (buf_projected_states_, buf_hbc_conv_).
    // At decode (seq_len==1) the dispatch sites use these and skip the splits.
    // Adreno's mem-base-addr-align is typically 1024 bits = 128 bytes — all our
    // offsets (multiples of 256 bytes for fp16) satisfy this.
    const size_t off_gate_b      = 0;
    const size_t off_hbc_b       = (size_t)d_inner_ * sizeof(nnopt_storage_t);
    const size_t off_dt_b        = (size_t)(d_inner_ + conv_ch_) * sizeof(nnopt_storage_t);
    const size_t sz_gate_b       = (size_t)d_inner_   * sizeof(nnopt_storage_t);
    const size_t sz_hbc_b        = (size_t)conv_ch_   * sizeof(nnopt_storage_t);
    const size_t sz_dt_b         = (size_t)n_v_heads_ * sizeof(nnopt_storage_t);

    const size_t off_x_inner_b   = 0;
    const size_t off_bc_b        = (size_t)d_inner_ * sizeof(nnopt_storage_t);
    const size_t off_B_b         = off_bc_b;
    const size_t off_C_b         = off_bc_b + (size_t)gc_cols * sizeof(nnopt_storage_t);
    const size_t sz_x_inner_b    = (size_t)d_inner_ * sizeof(nnopt_storage_t);
    const size_t sz_bc_b         = (size_t)bc_cols  * sizeof(nnopt_storage_t);
    const size_t sz_BorC_b       = (size_t)gc_cols  * sizeof(nnopt_storage_t);

    bool ok = true;
    ok &= make_subbuf(&buf_gate_dec_,    buf_projected_states_, off_gate_b,    sz_gate_b,    "gate_dec");
    ok &= make_subbuf(&buf_hbc_dec_,     buf_projected_states_, off_hbc_b,     sz_hbc_b,     "hbc_dec");
    ok &= make_subbuf(&buf_dt_dec_,      buf_projected_states_, off_dt_b,      sz_dt_b,      "dt_dec");
    ok &= make_subbuf(&buf_x_inner_dec_, buf_hbc_conv_,         off_x_inner_b, sz_x_inner_b, "x_inner_dec");
    ok &= make_subbuf(&buf_bc_dec_,      buf_hbc_conv_,         off_bc_b,      sz_bc_b,      "bc_dec");
    ok &= make_subbuf(&buf_B_dec_,       buf_hbc_conv_,         off_B_b,       sz_BorC_b,    "B_dec");
    ok &= make_subbuf(&buf_C_dec_,       buf_hbc_conv_,         off_C_b,       sz_BorC_b,    "C_dec");
    buf_decode_subbufs_ready_ = ok;

    return true;
}

bool Ssm::initialize() {
    // Load weights (all required for Mamba2 mixer)
    // Contract: (model metadata)
    in_proj_w_ = weights_.get_buffer(prefix_ + ".in_proj.weight");
    conv_w_ = weights_.get_buffer(prefix_ + ".conv1d.weight");
    conv_b_ = weights_.get_buffer(prefix_ + ".conv1d.bias");

    dt_bias_ = weights_.get_buffer(prefix_ + ".dt_bias");
    a_log_ = weights_.get_buffer(prefix_ + ".A_log");
    d_skip_ = weights_.get_buffer(prefix_ + ".D");

    norm_w_ = weights_.get_buffer(prefix_ + ".norm.weight");
    out_proj_w_ = weights_.get_buffer(prefix_ + ".out_proj.weight");

    if (!in_proj_w_ || !conv_w_ || !conv_b_ || !dt_bias_ || !a_log_ || !d_skip_ || !norm_w_ ||
        !out_proj_w_) {
        NNOPT_ERROR_FMT("Ssm[%d]: missing weight(s) for prefix=%s", layer_idx_, prefix_.c_str());
        return false;
    }

    if (!derive_dims_from_weights()) {
        NNOPT_ERROR_FMT("Ssm[%d]: derive_dims_from_weights() failed", layer_idx_);
        return false;
    }

    if (!build_kernels()) {
        NNOPT_ERROR_FMT("Ssm[%d]: build_kernels() failed", layer_idx_);
        return false;
    }

    cl_int err = CL_SUCCESS;

    // Build mixer RMSNormGated kernel.
    // Reference: model_info/transformers_src/modeling_mamba2.py:74-120 (MambaRMSNormGated.forward)
    norm_program_ = cl_ctx_.build_program_from_file("kernels/mamba_rms_norm_gated.cl");
    if (!norm_program_) {
        NNOPT_ERROR_FMT("Ssm[%d]: build_program_from_file(kernels/mamba_rms_norm_gated.cl) failed", layer_idx_);
        return false;
    }
    norm_kernel_ = clCreateKernel(norm_program_, "mamba_rms_norm_gated", &err);
    if (err != CL_SUCCESS || !norm_kernel_) {
        NNOPT_ERROR_FMT("Ssm[%d]: clCreateKernel(mamba_rms_norm_gated) failed (%d)", layer_idx_, (int)err);
        return false;
    }

    // Allocate persistent recurrent state (fp32)
    const int conv_state_elems = conv_ch_ * (d_conv_ - 1);
    conv_state_ = clCreateBuffer(cl_ctx_.context(), CL_MEM_READ_WRITE,
                                 (size_t)conv_state_elems * sizeof(float), nullptr, &err);
    if (err != CL_SUCCESS || !conv_state_) {
        NNOPT_ERROR_FMT("Ssm[%d]: clCreateBuffer(conv_state) failed (%d)", layer_idx_, err);
        return false;
    }

    const int ssm_state_elems = d_inner_ * d_state_;
    const size_t hstate_elt_size = use_fp16_hstate_ ? sizeof(nnopt_storage_t) : sizeof(float);
    ssm_state_ = clCreateBuffer(cl_ctx_.context(), CL_MEM_READ_WRITE,
                                (size_t)ssm_state_elems * hstate_elt_size, nullptr, &err);
    if (err != CL_SUCCESS || !ssm_state_) {
        NNOPT_ERROR_FMT("Ssm[%d]: clCreateBuffer(ssm_state) failed (%d)", layer_idx_, err);
        return false;
    }

    // Zero init state
    reset_state(cl_ctx_.queue());

    // causal_conv1d is depthwise and expects d_inner == channel count.
    // Here we run it on hidden_states_B_C, so pass conv_ch_.
    conv_d_inner_ = conv_ch_;

    return true;
}

cl_mem Ssm::forward(cl_command_queue queue, cl_mem input, int seq_len, int start_pos,
                    cl_mem hidden_residual) {
    // Full Mamba2 mixer forward: in_proj → split (gate, xBC, dt) →
    //   conv1d(xBC) → silu → split(x, B, C) → SSD scan(x, dt, A, B, C, D) →
    //   RMSNormGated(y, gate) → out_proj.
    // Reference: model_info/transformers_src/modeling_mamba2.py:Mamba2Mixer.torch_forward
    //
    // Step 5 (persistent activation buffers): all 13 intermediate buffers live
    // on `this` and are sized once for max(seq_len). Per-call clCreateBuffer/
    // clReleaseMemObject pairs are gone — saves ~50 µs/buffer × 13 × 24 layers
    // × N tokens of pure driver overhead. The output buffer is borrowed; the
    // caller must NOT release it (see ssm.h).
    (void)start_pos;  // recurrent state is held in conv_state_/ssm_state_, not in start_pos.

    if (!queue || !input || seq_len <= 0 || d_model_ <= 0) {
        NNOPT_ERROR_FMT("Ssm[%d]::forward invalid args (seq_len=%d d_model=%d)", layer_idx_, seq_len, d_model_);
        return nullptr;
    }

    if (layer_idx_ == 0) {
        NNOPT_LAYER_CHECK_INPUT("block0_sub_ssm_input", queue, input,
                                (size_t)seq_len * (size_t)d_model_);
    }

    cl_int err = CL_SUCCESS;
    const int rows = seq_len;
    const int projection_size = d_inner_ + conv_ch_ + n_v_heads_;
    const int bc_cols = conv_ch_ - d_inner_;          // 2 * n_groups * state_size
    const int gc_cols = bc_cols / 2;                   // n_groups * state_size
    if (gc_cols != n_groups_ * d_state_) {
        NNOPT_ERROR_FMT("Ssm[%d]::forward derived gc_cols=%d != n_groups*d_state=%d",
                        layer_idx_, gc_cols, n_groups_ * d_state_);
        return nullptr;
    }

    if (!ensure_activation_buffers_(rows)) {
        return nullptr;
    }

    // ── 1) projected_states = in_proj(input) ──
    if (!pytorch_linear(queue, rows, projection_size, d_model_, input, in_proj_w_, buf_projected_states_)) {
        return nullptr;
    }

    if (layer_idx_ == 0) {
        NNOPT_LAYER_CHECK("block0_sub_ssm_projected_states", queue, buf_projected_states_,
                          (size_t)rows * (size_t)projection_size);
        NNOPT_LAYER_CHECK("block0_sub_ssm_in_proj_output", queue, buf_projected_states_,
                          (size_t)rows * (size_t)projection_size);
    }

    // ── 2) Split projected_states → gate | hbc | dt
    // Lever F: at decode (rows==1), use the persistent subbuffer aliases
    // and skip the split kernels entirely. At prefill, run the splits.
    const bool use_dec_aliases = (rows == 1) && buf_decode_subbufs_ready_;
    cl_mem& sel_gate = use_dec_aliases ? buf_gate_dec_              : buf_gate_;
    cl_mem& sel_hbc  = use_dec_aliases ? buf_hbc_dec_               : buf_hidden_states_B_C_;
    cl_mem& sel_dt   = use_dec_aliases ? buf_dt_dec_                : buf_dt_raw_;
    const size_t gate_elems = (size_t)rows * (size_t)d_inner_;
    const size_t hbc_elems = (size_t)rows * (size_t)conv_ch_;
    const size_t dt_elems = (size_t)rows * (size_t)n_v_heads_;
    if (!use_dec_aliases) {
        if (!split_last_dim_unequal(queue, utils_program_, buf_projected_states_, buf_gate_, buf_rest_after_gate_,
                                    rows, projection_size, /*first_cols=*/d_inner_)) {
            return nullptr;
        }
        if (!split_last_dim_unequal(queue, utils_program_, buf_rest_after_gate_, buf_hidden_states_B_C_, buf_dt_raw_,
                                    rows, /*src_cols=*/(projection_size - d_inner_), /*first_cols=*/conv_ch_)) {
            return nullptr;
        }
    }

    if (layer_idx_ == 0) {
        NNOPT_LAYER_CHECK("block0_sub_ssm_gate", queue, buf_gate_, gate_elems);
        NNOPT_LAYER_CHECK("block0_sub_mamba2mixer_gate", queue, buf_gate_, gate_elems);
        NNOPT_LAYER_CHECK("block0_sub_ssm_conv1d_input", queue, buf_hidden_states_B_C_, hbc_elems);
        NNOPT_LAYER_CHECK("block0_sub_ssm_input_states", queue, buf_hidden_states_B_C_, hbc_elems);
        NNOPT_LAYER_CHECK("block0_sub_ssm_dt", queue, buf_dt_raw_, dt_elems);
    }

    // ── 4) causal_conv1d(hidden_states_B_C) [+ Lever C: fused silu] ──
    {
        cl_kernel k = conv_silu_kernel_ ? conv_silu_kernel_ : conv_kernel_;
        const char* prof_label = conv_silu_kernel_ ? "causal_conv1d_silu" : "causal_conv1d";
        int arg = 0;
        err  = clSetKernelArg(k, arg++, sizeof(cl_mem), &sel_hbc);  // Lever F: dec alias or full buf
        err |= clSetKernelArg(k, arg++, sizeof(cl_mem), &conv_w_);
        err |= clSetKernelArg(k, arg++, sizeof(cl_mem), &conv_b_);
        err |= clSetKernelArg(k, arg++, sizeof(cl_mem), &conv_state_);
        err |= clSetKernelArg(k, arg++, sizeof(cl_mem), &buf_hbc_conv_);
        err |= clSetKernelArg(k, arg++, sizeof(int), &seq_len);
        err |= clSetKernelArg(k, arg++, sizeof(int), &conv_d_inner_);
        err |= clSetKernelArg(k, arg++, sizeof(int), &d_conv_);
        if (err != CL_SUCCESS) {
            NNOPT_ERROR_FMT("Ssm[%d]::forward clSetKernelArg(%s) failed (%d)", layer_idx_, prof_label, (int)err);
            return nullptr;
        }
        const size_t gws = hbc_elems;
        cl_event* evt = KernelProfiler::event_for(prof_label);
        err = clEnqueueNDRangeKernel(queue, k, 1, nullptr, &gws, nullptr, 0, nullptr, evt);
        if (err != CL_SUCCESS) {
            NNOPT_ERROR_FMT("Ssm[%d]::forward dispatch %s failed (%d)", layer_idx_, prof_label, (int)err);
            return nullptr;
        }
    }

    if (layer_idx_ == 0) {
        NNOPT_LAYER_CHECK("block0_sub_ssm_conv1d_output", queue, buf_hbc_conv_, hbc_elems);
    }

    // Update conv_state with last (d_conv-1) tokens of the pre-conv input.
    {
        int arg = 0;
        err  = clSetKernelArg(update_conv_state_kernel_, arg++, sizeof(cl_mem), &sel_hbc);
        err |= clSetKernelArg(update_conv_state_kernel_, arg++, sizeof(cl_mem), &conv_state_);
        err |= clSetKernelArg(update_conv_state_kernel_, arg++, sizeof(int), &seq_len);
        err |= clSetKernelArg(update_conv_state_kernel_, arg++, sizeof(int), &conv_d_inner_);
        err |= clSetKernelArg(update_conv_state_kernel_, arg++, sizeof(int), &d_conv_);
        if (err == CL_SUCCESS) {
            const size_t gws = (size_t)conv_d_inner_;
            cl_event* evt = KernelProfiler::event_for("update_conv_state");
            err = clEnqueueNDRangeKernel(queue, update_conv_state_kernel_, 1, nullptr, &gws, nullptr,
                                         0, nullptr, evt);
        }
        if (err != CL_SUCCESS) {
            NNOPT_ERROR_FMT("Ssm[%d]::forward update_conv_state failed (%d)", layer_idx_, (int)err);
            return nullptr;
        }
    }

    // ── 5) silu(hbc_conv) in place ──
    // Lever C: when conv_silu_kernel_ is in use, the silu was already applied
    // by causal_conv1d_silu above. Skip the separate launch.
    if (!conv_silu_kernel_) {
        const int n_elems = (int)hbc_elems;
        int arg = 0;
        err  = clSetKernelArg(silu_inplace_kernel_, arg++, sizeof(cl_mem), &buf_hbc_conv_);
        err |= clSetKernelArg(silu_inplace_kernel_, arg++, sizeof(int), &n_elems);
        if (err == CL_SUCCESS) {
            const size_t gws = (size_t)n_elems;
            cl_event* evt = KernelProfiler::event_for("silu_inplace");
            err = clEnqueueNDRangeKernel(queue, silu_inplace_kernel_, 1, nullptr, &gws, nullptr,
                                         0, nullptr, evt);
        }
        if (err != CL_SUCCESS) {
            NNOPT_ERROR_FMT("Ssm[%d]::forward silu_inplace failed (%d)", layer_idx_, (int)err);
            return nullptr;
        }
    }

    // ── 6) Split hbc_conv → x_inner | bc, then bc → B | C
    // Lever F: at decode use the subbuffer aliases and skip the splits.
    cl_mem& sel_x_inner = use_dec_aliases ? buf_x_inner_dec_ : buf_x_inner_;
    cl_mem& sel_B       = use_dec_aliases ? buf_B_dec_       : buf_B_;
    cl_mem& sel_C       = use_dec_aliases ? buf_C_dec_       : buf_C_;
    const size_t hs_elems = (size_t)rows * (size_t)d_inner_;
    if (!use_dec_aliases) {
        if (!split_last_dim_unequal(queue, utils_program_, buf_hbc_conv_, buf_x_inner_, buf_bc_,
                                    rows, /*src_cols=*/conv_ch_, /*first_cols=*/d_inner_)) {
            return nullptr;
        }
        if (!split_last_dim_2(queue, utils_program_, buf_bc_, buf_B_, buf_C_, rows, gc_cols)) {
            return nullptr;
        }
    }

    if (layer_idx_ == 0) {
        NNOPT_LAYER_CHECK("block0_sub_ssm_hidden_states", queue, buf_x_inner_, hs_elems);
        NNOPT_LAYER_CHECK("block0_sub_mamba2mixer_hidden_states", queue, buf_x_inner_, hs_elems);
    }

    // ── 7) y = mamba2_ssd_scan(x_inner, dt_raw, A_log, dt_bias, B, C, D, ssm_state) ──
    // Dispatch the cooperative WG=128 variant when d_state==128 (mamba2-130m); fall
    // back to the scalar+vec4 variant otherwise. Same args, same outputs (modulo a
    // different fp32 reduction order in the coop variant).
    {
        const float dt_min = 0.001f;
        const float dt_max = 100.0f;
        // Priority: coop (opt-in env var) > transposed (Lever E, default-on) > original.
        cl_kernel k;
        const char* prof_label;
        if (ssd_scan_use_coop_) {
            k = ssd_scan_coop_kernel_;
            prof_label = "mamba2_ssd_scan_coop";
        } else if (ssd_scan_use_t_) {
            k = ssd_scan_t_kernel_;
            prof_label = "mamba2_ssd_scan_t";
        } else {
            k = ssd_scan_kernel_;
            prof_label = "mamba2_ssd_scan";
        }
        int arg = 0;
        err  = clSetKernelArg(k, arg++, sizeof(cl_mem), &sel_x_inner);
        err |= clSetKernelArg(k, arg++, sizeof(cl_mem), &sel_dt);
        err |= clSetKernelArg(k, arg++, sizeof(cl_mem), &a_log_);
        err |= clSetKernelArg(k, arg++, sizeof(cl_mem), &dt_bias_);
        err |= clSetKernelArg(k, arg++, sizeof(cl_mem), &sel_B);
        err |= clSetKernelArg(k, arg++, sizeof(cl_mem), &sel_C);
        err |= clSetKernelArg(k, arg++, sizeof(cl_mem), &d_skip_);
        err |= clSetKernelArg(k, arg++, sizeof(cl_mem), &ssm_state_);
        err |= clSetKernelArg(k, arg++, sizeof(cl_mem), &buf_y_scan_);
        err |= clSetKernelArg(k, arg++, sizeof(int), &seq_len);
        err |= clSetKernelArg(k, arg++, sizeof(int), &n_v_heads_);
        err |= clSetKernelArg(k, arg++, sizeof(int), &head_dim_);
        err |= clSetKernelArg(k, arg++, sizeof(int), &n_groups_);
        err |= clSetKernelArg(k, arg++, sizeof(int), &d_state_);
        err |= clSetKernelArg(k, arg++, sizeof(float), &dt_min);
        err |= clSetKernelArg(k, arg++, sizeof(float), &dt_max);
        if (err != CL_SUCCESS) {
            NNOPT_ERROR_FMT("Ssm[%d]::forward clSetKernelArg(%s) failed (%d)", layer_idx_, prof_label, (int)err);
            return nullptr;
        }
        cl_event* evt = KernelProfiler::event_for(prof_label);
        if (ssd_scan_use_coop_) {
            const size_t WG = 32;  // single Adreno A6xx wave; intra-wave barriers only
            const size_t gws = (size_t)d_inner_ * WG;  // 1 WG per (head, chan_in_head)
            const size_t lws = WG;
            err = clEnqueueNDRangeKernel(queue, k, 1, nullptr, &gws, &lws, 0, nullptr, evt);
        } else {
            const size_t gws = (size_t)d_inner_;  // one work-item per (head, chan_in_head)
            err = clEnqueueNDRangeKernel(queue, k, 1, nullptr, &gws, nullptr, 0, nullptr, evt);
        }
        if (err != CL_SUCCESS) {
            NNOPT_ERROR_FMT("Ssm[%d]::forward dispatch %s failed (%d)", layer_idx_, prof_label, (int)err);
            return nullptr;
        }
    }

    // ── 8) normed = mamba_rms_norm_gated(y_scan, gate, norm_w) ──
    {
        const int rows_i = seq_len;
        const int cols_i = d_inner_;
        const float eps = 1e-6f;
        int arg = 0;
        err  = clSetKernelArg(norm_kernel_, arg++, sizeof(cl_mem), &buf_y_scan_);
        err |= clSetKernelArg(norm_kernel_, arg++, sizeof(cl_mem), &sel_gate);
        err |= clSetKernelArg(norm_kernel_, arg++, sizeof(cl_mem), &norm_w_);
        err |= clSetKernelArg(norm_kernel_, arg++, sizeof(cl_mem), &buf_normed_);
        err |= clSetKernelArg(norm_kernel_, arg++, sizeof(int), &rows_i);
        err |= clSetKernelArg(norm_kernel_, arg++, sizeof(int), &cols_i);
        err |= clSetKernelArg(norm_kernel_, arg++, sizeof(float), &eps);
        if (err == CL_SUCCESS) {
            const size_t WG = 64;
            const size_t gws = (size_t)rows_i * WG;
            const size_t lws = WG;
            cl_event* evt = KernelProfiler::event_for("mamba_rms_norm_gated");
            err = clEnqueueNDRangeKernel(queue, norm_kernel_, 1, nullptr, &gws, &lws, 0, nullptr, evt);
        }
        if (err != CL_SUCCESS) {
            NNOPT_ERROR_FMT("Ssm[%d]::forward dispatch mamba_rms_norm_gated failed (%d)", layer_idx_, (int)err);
            return nullptr;
        }
    }

    if (layer_idx_ == 0) {
        NNOPT_LAYER_CHECK("block0_sub_ssm_norm_output", queue, buf_normed_, hs_elems);
    }

    // ── 9) out = out_proj(normed) ──
    // Lever 2 fast path: when caller provides hidden_residual at decode (M=1)
    // and the radd kernel is available, fuse the residual add into out_proj.
    // hidden_residual[:] += W^T @ normed — single launch, no separate
    // element_add. Falls back to plain pytorch_linear into buf_out_ otherwise.
    bool used_radd = false;
    if (hidden_residual != nullptr && rows == 1) {
        if (pytorch_linear_radd_fast(queue, /*M=*/rows, /*N=*/d_model_, /*K=*/d_inner_,
                                     buf_normed_, out_proj_w_, hidden_residual)) {
            used_radd = true;
        }
    }
    if (!used_radd) {
        if (!pytorch_linear(queue, rows, d_model_, d_inner_, buf_normed_, out_proj_w_, buf_out_)) {
            return nullptr;
        }
    }

    if (layer_idx_ == 0) {
        cl_mem dump_target = used_radd ? hidden_residual : buf_out_;
        NNOPT_LAYER_CHECK("block0_sub_ssm_contextualized_states", queue, dump_target,
                          (size_t)rows * (size_t)d_model_);
    }

    return used_radd ? hidden_residual : buf_out_;
}

void Ssm::reset_state(cl_command_queue queue) {
    if (!queue || !conv_state_ || !ssm_state_) return;

    const int conv_state_elems = conv_ch_ * (d_conv_ - 1);
    std::vector<float> zero_conv((size_t)conv_state_elems, 0.0f);
    cl_int err = clEnqueueWriteBuffer(queue, conv_state_, CL_TRUE, 0,
                                      (size_t)conv_state_elems * sizeof(float), zero_conv.data(), 0, nullptr, nullptr);
    if (err != CL_SUCCESS) {
        NNOPT_ERROR_FMT("Ssm[%d]: reset_state conv_state write failed (%d)", layer_idx_, err);
        return;
    }

    const int ssm_state_elems = d_inner_ * d_state_;
    if (use_fp16_hstate_) {
        // Lever I: fp16 ssm_state — half-byte zeros.
        std::vector<nnopt_storage_t> zero_ssm((size_t)ssm_state_elems, (nnopt_storage_t)0);
        err = clEnqueueWriteBuffer(queue, ssm_state_, CL_TRUE, 0,
                                   (size_t)ssm_state_elems * sizeof(nnopt_storage_t),
                                   zero_ssm.data(), 0, nullptr, nullptr);
    } else {
        std::vector<float> zero_ssm((size_t)ssm_state_elems, 0.0f);
        err = clEnqueueWriteBuffer(queue, ssm_state_, CL_TRUE, 0,
                                   (size_t)ssm_state_elems * sizeof(float),
                                   zero_ssm.data(), 0, nullptr, nullptr);
    }
    if (err != CL_SUCCESS) {
        NNOPT_ERROR_FMT("Ssm[%d]: reset_state ssm_state write failed (%d)", layer_idx_, err);
        return;
    }
}

bool Ssm::derive_dims_from_weights() {
    // Derive all key dims from weight shapes (single source of truth).
    // Reference: model_info/transformers_src/modeling_mamba2.py (Mamba2Mixer.__init__)
    // out_proj.weight: [hidden_size, intermediate_size]
    // conv1d.weight:   [conv_dim, 1, conv_kernel]
    // in_proj.weight:  [projection_size, hidden_size]

    auto out_shape = weights_.get_shape(prefix_ + ".out_proj.weight");
    auto conv_shape = weights_.get_shape(prefix_ + ".conv1d.weight");
    auto in_proj_shape = weights_.get_shape(prefix_ + ".in_proj.weight");

    if (out_shape.size() != 2 || conv_shape.size() != 3 || in_proj_shape.size() != 2) {
        NNOPT_ERROR_FMT("Ssm[%d]: unexpected weight ranks out=%zu conv=%zu in_proj=%zu", layer_idx_, out_shape.size(),
                        conv_shape.size(), in_proj_shape.size());
        return false;
    }

    d_model_ = out_shape[0];
    d_inner_ = out_shape[1];

    conv_ch_ = conv_shape[0];
    d_conv_ = conv_shape[2];

    const int proj_out = in_proj_shape[0];
    const int proj_in = in_proj_shape[1];
    if (proj_in != d_model_) {
        NNOPT_ERROR_FMT("Ssm[%d]: in_proj in_features mismatch: %d vs d_model=%d", layer_idx_, proj_in, d_model_);
        return false;
    }

    const int n_v = (int)weights_.get_num_elements(prefix_ + ".dt_bias");
    if (n_v <= 0) {
        NNOPT_ERROR_FMT("Ssm[%d]: dt_bias numel invalid", layer_idx_);
        return false;
    }
    n_v_heads_ = n_v;

    // Derive d_state and n_qk_heads using invariants from config/weights.
    // conv_dim = intermediate_size + 2 * n_groups * state_size
    // Here: conv_ch_ == conv_dim, d_inner_ == intermediate_size.
    // Let bc = conv_dim - intermediate_size = 2*n_groups*state_size.
    const int bc = conv_ch_ - d_inner_;
    if (bc <= 0 || (bc % 2) != 0) {
        NNOPT_ERROR_FMT("Ssm[%d]: conv_dim - intermediate_size invalid: conv_dim=%d inter=%d", layer_idx_, conv_ch_,
                        d_inner_);
        return false;
    }

    // This model uses n_groups=1; with dt_bias size (num_heads) == 24 and head_dim=64.
    // state_size inferred from bc/2/(n_groups). For this checkpoint: bc=256 => state_size=128.
    n_groups_ = 1;
    d_state_ = bc / (2 * n_groups_);
    n_qk_heads_ = n_groups_;
    if (n_v_heads_ <= 0 || (d_inner_ % n_v_heads_) != 0) {
        NNOPT_ERROR_FMT("Ssm[%d]: d_inner=%d not divisible by n_v_heads=%d", layer_idx_, d_inner_, n_v_heads_);
        return false;
    }
    head_dim_ = d_inner_ / n_v_heads_;

    // Sanity checks against known checkpoint shapes:
    // conv_dim = 1792, intermediate=1536 => bc=256 => d_state=128.
    if (d_state_ <= 0) {
        NNOPT_ERROR_FMT("Ssm[%d]: derived d_state invalid", layer_idx_);
        return false;
    }

    // in_proj projection_size should match derived
    const int projection_size = d_inner_ + conv_ch_ + n_v_heads_;
    if (proj_out != projection_size) {
        NNOPT_ERROR_FMT("Ssm[%d]: projection_size mismatch: in_proj_out=%d expected=%d", layer_idx_, proj_out,
                        projection_size);
        return false;
    }

    return true;
}

bool Ssm::build_kernels() {
    // Build scaffolded programs and kernels.
    // kernels/causal_conv1d.cl: causal_conv1d, update_conv_state, silu_inplace
    // kernels/selective_scan.cl: selective_scan, softplus, silu_mul
    // kernels/utils.cl: split_last_dim_2

    cl_int err = CL_SUCCESS;

    conv_program_ = cl_ctx_.build_program_from_file("kernels/causal_conv1d.cl");
    if (!conv_program_) return false;

    conv_kernel_ = clCreateKernel(conv_program_, "causal_conv1d", &err);
    if (err != CL_SUCCESS || !conv_kernel_) {
        NNOPT_ERROR_FMT("Ssm[%d]: clCreateKernel(causal_conv1d) failed (%d)", layer_idx_, err);
        return false;
    }

    conv_silu_kernel_ = clCreateKernel(conv_program_, "causal_conv1d_silu", &err);
    if (err != CL_SUCCESS || !conv_silu_kernel_) {
        NNOPT_ERROR_FMT("Ssm[%d]: clCreateKernel(causal_conv1d_silu) failed (%d) — Lever C disabled", layer_idx_, err);
        conv_silu_kernel_ = nullptr;
        err = CL_SUCCESS;  // non-fatal
    }

    update_conv_state_kernel_ = clCreateKernel(conv_program_, "update_conv_state", &err);
    if (err != CL_SUCCESS || !update_conv_state_kernel_) {
        NNOPT_ERROR_FMT("Ssm[%d]: clCreateKernel(update_conv_state) failed (%d)", layer_idx_, err);
        return false;
    }

    silu_inplace_kernel_ = clCreateKernel(conv_program_, "silu_inplace", &err);
    if (err != CL_SUCCESS || !silu_inplace_kernel_) {
        NNOPT_ERROR_FMT("Ssm[%d]: clCreateKernel(silu_inplace) failed (%d)", layer_idx_, err);
        return false;
    }

    // Lever I: opt-in fp16 ssm_state via env var. Halves state R/W traffic
    // for ssd_scan_t (the dominant non-GEMV kernel).
    const char* env_fp16_h = std::getenv("NNOPT_FP16_HSTATE");
    use_fp16_hstate_ = env_fp16_h && env_fp16_h[0] != '0';
    std::string scan_opts = use_fp16_hstate_ ? "-D USE_FP16_HSTATE=1" : "";
    scan_program_ = cl_ctx_.build_program_from_file("kernels/selective_scan.cl", scan_opts);
    if (!scan_program_) return false;

    scan_kernel_ = clCreateKernel(scan_program_, "selective_scan", &err);
    if (err != CL_SUCCESS || !scan_kernel_) {
        NNOPT_ERROR_FMT("Ssm[%d]: clCreateKernel(selective_scan) failed (%d)", layer_idx_, err);
        return false;
    }

    softplus_kernel_ = clCreateKernel(scan_program_, "softplus", &err);
    if (err != CL_SUCCESS || !softplus_kernel_) {
        NNOPT_ERROR_FMT("Ssm[%d]: clCreateKernel(softplus) failed (%d)", layer_idx_, err);
        return false;
    }

    silu_mul_kernel_ = clCreateKernel(scan_program_, "silu_mul", &err);
    if (err != CL_SUCCESS || !silu_mul_kernel_) {
        NNOPT_ERROR_FMT("Ssm[%d]: clCreateKernel(silu_mul) failed (%d)", layer_idx_, err);
        return false;
    }

    ssd_scan_kernel_ = clCreateKernel(scan_program_, "mamba2_ssd_scan", &err);
    if (err != CL_SUCCESS || !ssd_scan_kernel_) {
        NNOPT_ERROR_FMT("Ssm[%d]: clCreateKernel(mamba2_ssd_scan) failed (%d)", layer_idx_, err);
        return false;
    }

    // Lever E: transposed-state variant (default-on; opt-out via NNOPT_SSD_SCAN_T=0).
    ssd_scan_t_kernel_ = clCreateKernel(scan_program_, "mamba2_ssd_scan_t", &err);
    if (err != CL_SUCCESS || !ssd_scan_t_kernel_) {
        NNOPT_ERROR_FMT("Ssm[%d]: clCreateKernel(mamba2_ssd_scan_t) failed (%d) — Lever E disabled",
                        layer_idx_, err);
        ssd_scan_t_kernel_ = nullptr;
        err = CL_SUCCESS;
    }
    {
        const char* env_t = std::getenv("NNOPT_SSD_SCAN_T");
        const bool default_on = true;
        const bool t_opt_in = env_t ? (env_t[0] != '0') : default_on;
        ssd_scan_use_t_ = t_opt_in && (ssd_scan_t_kernel_ != nullptr);
    }

    // Cooperative variant: WG=128, 1 thread per state. Hardcoded for state_size==128.
    // Build it for ALL configs (program is shared across layers); only DISPATCH it
    // when d_state_==128. For other state sizes the runtime falls back to ssd_scan_kernel_.
    ssd_scan_coop_kernel_ = clCreateKernel(scan_program_, "mamba2_ssd_scan_coop", &err);
    if (err != CL_SUCCESS || !ssd_scan_coop_kernel_) {
        NNOPT_ERROR_FMT("Ssm[%d]: clCreateKernel(mamba2_ssd_scan_coop) failed (%d) — coop disabled, fall back to scalar+vec4",
                        layer_idx_, err);
        ssd_scan_coop_kernel_ = nullptr;
        // not fatal — old kernel still works
        err = CL_SUCCESS;
    }
    // Coop dispatch disabled by default — Step-A measurement showed both WG=128
    // (902 µs/call vs 436 µs scalar+vec4) and WG=32 (452 µs/call) regressed or
    // tied. The original 1-thread-per-(head,chan) scalar+vec4 kernel is already
    // close enough to BW-bound that L2 absorbs the cross-thread stride, and the
    // tree-reduce barriers in the coop layout cost more than the coalescing
    // gain. Kernel + binding kept for future re-evaluation under different
    // driver / device. Opt-in via env NNOPT_SSD_SCAN_COOP=1.
    const char* env_coop = std::getenv("NNOPT_SSD_SCAN_COOP");
    const bool coop_opt_in = env_coop && env_coop[0] != '0';
    ssd_scan_use_coop_ = coop_opt_in && (ssd_scan_coop_kernel_ != nullptr) && (d_state_ == 128);

    if (!utils_program_) {
        NNOPT_ERROR_FMT("Ssm[%d]: utils_program_ not set (call set_utils_program() before initialize)", layer_idx_);
        return false;
    }
    split_kernel_ = clCreateKernel(utils_program_, "split_last_dim_2", &err);
    if (err != CL_SUCCESS || !split_kernel_) {
        NNOPT_ERROR_FMT("Ssm[%d]: clCreateKernel(split_last_dim_2) failed (%d)", layer_idx_, err);
        return false;
    }

    return true;
}
