// AdainResBlk1d — the repeating StyleTTS2 residual block.
// See styletts_ops.h for the three variants and the norm1/upsample trap.
#include "styletts_ops.h"
#include "nnopt_error.h"
#include <vector>

namespace st {

cl_mem adain_resblk(OpenCLContext& ctx, cl_command_queue q, Weights& w,
                    cl_mem x, int C_in, int C_out, int T_in, int T_out,
                    const std::string& prefix,
                    cl_mem fc_norm1, cl_mem fc_norm2,
                    bool upsample, bool has_conv1x1) {
    const float ISQRT2 = 0.7071067690849304f;

    std::vector<cl_mem> owned;
    auto own = [&](cl_mem m) { if (m) owned.push_back(m); return m; };
    auto fail = [&]() -> cl_mem {
        for (cl_mem m : owned) if (m) clReleaseMemObject(m);
        return nullptr;
    };

    cl_mem n1w = w.get_buffer(prefix + ".norm1.norm.weight");
    cl_mem n1b = w.get_buffer(prefix + ".norm1.norm.bias");
    cl_mem n2w = w.get_buffer(prefix + ".norm2.norm.weight");
    cl_mem n2b = w.get_buffer(prefix + ".norm2.norm.bias");
    cl_mem c1w = w.get_buffer(prefix + ".conv1.weight_quantized");
    cl_mem c1b = w.get_buffer(prefix + ".conv1.bias");
    cl_mem c2w = w.get_buffer(prefix + ".conv2.weight_quantized");
    cl_mem c2b = w.get_buffer(prefix + ".conv2.bias");
    if (!n1w || !n1b || !n2w || !n2b || !c1w || !c1b || !c2w || !c2b) {
        NNOPT_ERROR_FMT("adain_resblk %s: missing core weights", prefix.c_str());
        return nullptr;
    }

    cl_mem params = own(alloc_f32(ctx, 2));
    if (!params) return fail();

    // ── shortcut ─────────────────────────────────────────────────────────
    cl_mem shortcut = nullptr;
    if (has_conv1x1) {
        cl_mem cxw = w.get_buffer(prefix + ".conv1x1.weight_quantized");   // NO bias
        if (!cxw) { NNOPT_ERROR_FMT("adain_resblk %s: missing conv1x1", prefix.c_str()); return fail(); }
        cl_mem src = x;
        int src_T = T_in;
        if (upsample) {
            // nearest 2x on the shortcut only
            cl_mem up = own(alloc(ctx, (size_t)C_in * T_out));
            if (!up || !resize_nearest_t(q, x, up, C_in, T_in, 2)) return fail();
            src = up; src_T = T_out;
        }
        cl_mem sq = own(alloc(ctx, (size_t)C_in * src_T));
        if (!sq || !dql(q, src, params, sq, C_in * src_T)) return fail();
        shortcut = own(alloc(ctx, (size_t)C_out * src_T));
        if (!shortcut || !conv1d(q, sq, cxw, nullptr, shortcut,
                                 C_in, C_out, src_T, src_T, 1, 1, 0, 1)) return fail();
    } else {
        shortcut = x;   // identity (C_in == C_out, T_in == T_out)
    }

    // ── residual path ────────────────────────────────────────────────────
    // norm1 operates at T_in, on the UN-upsampled x.
    cl_mem h1  = own(alloc(ctx, (size_t)C_in * T_in));
    cl_mem scr = own(alloc(ctx, (size_t)C_in * T_in));
    if (!h1 || !scr) return fail();
    if (!adain(q, x, n1w, n1b, fc_norm1, scr, h1, C_in, T_in)) return fail();
    if (!leaky_relu(q, h1, h1, C_in * T_in)) return fail();

    // pool: depthwise ConvTranspose1d, stride 2 -> exactly 2*T_in
    cl_mem pre_conv1 = h1;
    int pre_T = T_in;
    if (upsample) {
        cl_mem pw = w.get_buffer(prefix + ".pool.weight");
        cl_mem pb = w.get_buffer(prefix + ".pool.bias");
        if (!pw || !pb) { NNOPT_ERROR_FMT("adain_resblk %s: missing pool", prefix.c_str()); return fail(); }
        cl_mem up = own(alloc(ctx, (size_t)C_in * T_out));
        if (!up || !conv_transpose1d_dw(q, h1, pw, pb, up, C_in, T_in, T_out, 3, 2, 1)) return fail();
        pre_conv1 = up; pre_T = T_out;
    }

    cl_mem q1 = own(alloc(ctx, (size_t)C_in * pre_T));
    cl_mem h2 = own(alloc(ctx, (size_t)C_out * pre_T));
    if (!q1 || !h2) return fail();
    if (!dql(q, pre_conv1, params, q1, C_in * pre_T)) return fail();
    if (!conv1d(q, q1, c1w, c1b, h2, C_in, C_out, pre_T, pre_T, 3, 1, 1, 1)) return fail();

    cl_mem scr2 = own(alloc(ctx, (size_t)C_out * pre_T));
    cl_mem h3   = own(alloc(ctx, (size_t)C_out * pre_T));
    if (!scr2 || !h3) return fail();
    if (!adain(q, h2, n2w, n2b, fc_norm2, scr2, h3, C_out, pre_T)) return fail();
    if (!leaky_relu(q, h3, h3, C_out * pre_T)) return fail();

    cl_mem q2 = own(alloc(ctx, (size_t)C_out * pre_T));
    cl_mem h4 = own(alloc(ctx, (size_t)C_out * pre_T));
    if (!q2 || !h4) return fail();
    if (!dql(q, h3, params, q2, C_out * pre_T)) return fail();
    if (!conv1d(q, q2, c2w, c2b, h4, C_out, C_out, pre_T, pre_T, 3, 1, 1, 1)) return fail();

    // ── join ─────────────────────────────────────────────────────────────
    cl_mem out = alloc(ctx, (size_t)C_out * T_out);
    if (!out) return fail();
    if (!add_scaled(q, h4, shortcut, out, C_out * T_out, ISQRT2)) {
        clReleaseMemObject(out);
        return fail();
    }
    for (cl_mem m : owned) if (m) clReleaseMemObject(m);
    return out;
}

}  // namespace st
