// DPT neck (reassemble stage + fusion stage) + depth estimation head.
// Consumes (releases) the four backbone features; returns the final depth
// map as host fp32, already scaled by max_depth.
// Code moved verbatim from the former monolithic backbone.cpp.

#include "depth_common.h"

// ── PreActResidualLayer: relu conv1 relu conv2 + residual (channels=64) ──
static cl_mem preact_residual(Weights& W, const std::string& p, cl_mem in, int Ch, int Hh, int Ww) {
    cl_mem r1 = act_relu(in, (size_t)Ch*Hh*Ww);
    int oh, ow;
    cl_mem c1 = conv2d(r1, Ch, Hh, Ww, W.get_buffer(p+"convolution1.weight"), W.get_buffer(p+"convolution1.bias"), Ch, 3, 3, 1, 1, oh, ow);
    pool_release(r1);
    cl_mem r2 = act_relu(c1, (size_t)Ch*oh*ow);
    pool_release(c1);
    cl_mem c2 = conv2d(r2, Ch, oh, ow, W.get_buffer(p+"convolution2.weight"), W.get_buffer(p+"convolution2.bias"), Ch, 3, 3, 1, 1, oh, ow);
    pool_release(r2);
    cl_mem out = elt_add(c2, in, (size_t)Ch*Hh*Ww);
    pool_release(c2);
    return out;
}

std::vector<float> dpt_neck_head(Weights& Wt, cl_mem feats[4],
                                 int M, int D, int Hp, int Wp, int PS) {
    const int Np = Hp * Wp;
    // ── Reassemble: drop CLS, tokens->CHW [D,37,49], proj(1x1 -> C_i), resize ──
    const int neckC[4] = {48, 96, 192, 384};
    cl_mem reass[4];
    int rH[4], rW[4];
    for (int i = 0; i < 4; i++) {
        // tokens_to_chw: [M,D] -> [D, Hp, Wp]
        cl_mem chw = alloc_buf((size_t)D*Np);
        { cl_kernel kk=g_k.tokens_to_chw; int a=0;
          sa(kk,a++,sizeof(cl_mem),&feats[i]); sa(kk,a++,sizeof(cl_mem),&chw);
          sa(kk,a++,sizeof(int),&D); sa(kk,a++,sizeof(int),&Hp); sa(kk,a++,sizeof(int),&Wp);
          run1d(kk, (size_t)D*Np); }
        pool_release(feats[i]);
        std::string rp = "neck.reassemble_stage.layers." + std::to_string(i) + ".";
        int oh, ow;
        cl_mem proj = conv2d(chw, D, Hp, Wp, Wt.get_buffer(rp+"projection.weight"), Wt.get_buffer(rp+"projection.bias"), neckC[i], 1, 1, 1, 0, oh, ow);
        pool_release(chw);
        {
            char nm[96]; snprintf(nm, sizeof(nm), "neck_reassemble_stage_layers_%d_projection", i);
            DCHECK(nm, proj, (size_t)neckC[i]*oh*ow);
        }
        cl_mem resized;
        int roh, row;
        if (i == 0) { // factor 4: convT k4 s4
            resized = conv_transpose2d(proj, neckC[i], oh, ow, Wt.get_buffer(rp+"resize.weight"), Wt.get_buffer(rp+"resize.bias"), neckC[i], 4, 4, 4, roh, row);
        } else if (i == 1) { // factor 2: convT k2 s2
            resized = conv_transpose2d(proj, neckC[i], oh, ow, Wt.get_buffer(rp+"resize.weight"), Wt.get_buffer(rp+"resize.bias"), neckC[i], 2, 2, 2, roh, row);
        } else if (i == 2) { // factor 1: Identity
            resized = proj; roh = oh; row = ow; proj = nullptr;
        } else { // i==3, factor 0.5: conv k3 s2 pad1
            resized = conv2d(proj, neckC[i], oh, ow, Wt.get_buffer(rp+"resize.weight"), Wt.get_buffer(rp+"resize.bias"), neckC[i], 3, 3, 2, 1, roh, row);
        }
        if (proj) pool_release(proj);
        reass[i] = resized; rH[i] = roh; rW[i] = row;
        {
            // DepthAnythingReassembleLayer.forward output == resize output.
            char nm[96];
            snprintf(nm, sizeof(nm), "neck_reassemble_stage_layers_%d_resize", i);
            DCHECK(nm, resized, (size_t)neckC[i]*roh*row);
            snprintf(nm, sizeof(nm), "neck_reassemble_stage_layers_%d", i);
            DCHECK(nm, resized, (size_t)neckC[i]*roh*row);
        }
    }
    // DepthAnythingReassembleStage.forward returns the LIST of reassembled
    // features; the reference hook captured element[0]. Dump reass[0].
    DCHECK("neck_reassemble_stage", reass[0], (size_t)neckC[0]*rH[0]*rW[0]);

    // ── Neck convs: 3x3 pad1 C_i->64, NO bias ──
    cl_mem convd[4];
    for (int i = 0; i < 4; i++) {
        std::string cp = "neck.convs." + std::to_string(i) + ".weight";
        int oh, ow;
        cl_mem c = conv2d(reass[i], neckC[i], rH[i], rW[i], Wt.get_buffer(cp), nullptr, 64, 3, 3, 1, 1, oh, ow);
        pool_release(reass[i]);
        convd[i] = c; rH[i] = oh; rW[i] = ow;
    }
    DCHECK("neck_convs_0", convd[0], (size_t)64*rH[0]*rW[0]);
    DCHECK("neck_convs_1", convd[1], (size_t)64*rH[1]*rW[1]);
    DCHECK("neck_convs_2", convd[2], (size_t)64*rH[2]*rW[2]);
    DCHECK("neck_convs_3", convd[3], (size_t)64*rH[3]*rW[3]);

    // ── Fusion stage (reversed, start from idx 3) ──
    // for idx: size = shape of hidden_states[idx+1] (in reversed order) else None
    // reversed order: process feature 3, then 2, then 1, then 0.
    cl_mem fused = nullptr;
    int fH = 0, fW = 0;
    // Keep a clone of the coarsest (idx==0) fused output — both
    // DepthAnythingFeatureFusionStage AND DepthAnythingNeck reference hooks
    // captured this SAME coarse tensor (116032 elems), not the finest.
    cl_mem g_neck_coarse = nullptr;
    int g_neck_coarse_hw = 0;
    for (int idx = 0; idx < 4; idx++) {
        int fi = 3 - idx; // feature index in original order
        std::string lp = "neck.fusion_stage.layers." + std::to_string(idx) + ".";
        int Ch = 64;
        cl_mem hs = convd[fi];
        int hH = rH[fi], hW = rW[fi];
        // determine size = shape of next (reversed) hidden_state, else None (last)
        int size_h = -1, size_w = -1;
        if (idx != 3) { int nfi = 3 - (idx+1); size_h = rH[nfi]; size_w = rW[nfi]; }

        cl_mem cur;
        if (fused == nullptr) {
            // first layer: layer(hidden_state, size=size)
            cur = hs; // residual_layer2 applied below
        } else {
            // layer(fused, hidden_state, size=size): residual = hidden_state
            // if fused.shape != residual.shape -> interpolate residual to fused size
            cl_mem residual = hs;
            if (hH != fH || hW != fW) {
                cl_mem ri = bilinear(hs, Ch, hH, hW, fH, fW, 0); // align_corners=False
                residual = ri;
            }
            cl_mem rl1 = preact_residual(Wt, lp+"residual_layer1.", residual, Ch, fH, fW);
            {
                char nm[96];
                snprintf(nm, sizeof(nm), "neck_fusion_stage_layers_%d_residual_layer1", idx);
                DCHECK(nm, rl1, (size_t)Ch*fH*fW);
            }
            if (residual != hs) pool_release(residual);
            cl_mem added = elt_add(fused, rl1, (size_t)Ch*fH*fW);
            pool_release(rl1); pool_release(fused);
            cur = added;
            hH = fH; hW = fW;
        }
        // residual_layer2
        // (residual_layer1 already applied above when fused!=null; dump it by
        //  re-tagging: the added tensor IS residual_layer1's contribution site.
        //  For alignment we emit residual_layer1 dump inside the else-branch.)
        cl_mem rl2 = preact_residual(Wt, lp+"residual_layer2.", cur, Ch, hH, hW);
        {
            char nm[96];
            snprintf(nm, sizeof(nm), "neck_fusion_stage_layers_%d_residual_layer2", idx);
            DCHECK(nm, rl2, (size_t)Ch*hH*hW);
        }
        if (cur != hs) pool_release(cur);
        pool_release(hs); convd[fi] = nullptr;
        // interpolate: size or scale_factor 2, align_corners=True
        int oH, oW;
        if (size_h > 0) { oH = size_h; oW = size_w; }
        else { oH = hH * 2; oW = hW * 2; }
        cl_mem up = bilinear(rl2, Ch, hH, hW, oH, oW, 1);
        pool_release(rl2);
        // projection 1x1
        int poh, pow;
        cl_mem prj = conv2d(up, Ch, oH, oW, Wt.get_buffer(lp+"projection.weight"), Wt.get_buffer(lp+"projection.bias"), Ch, 1, 1, 1, 0, poh, pow);
        {
            char nm[96];
            snprintf(nm, sizeof(nm), "neck_fusion_stage_layers_%d_projection", idx);
            DCHECK(nm, prj, (size_t)Ch*poh*pow);
        }
        pool_release(up);
        fused = prj; fH = poh; fW = pow;
        {
            // DepthAnythingFeatureFusionLayer.forward output == projection output.
            char nm[96];
            snprintf(nm, sizeof(nm), "neck_fusion_stage_layers_%d", idx);
            DCHECK(nm, fused, (size_t)Ch*fH*fW);
        }
        // Reference: DepthAnythingFeatureFusionStage.forward returns a LIST of
        // fused_hidden_states; the reference hook captured element [0] = the
        // FIRST (coarsest) fused output [1,64,37,49]. Our idx==0 iteration
        // produces exactly that tensor, so dump it under "neck_fusion_stage"
        // to match the reference-captured element (SxS alignment).
        if (idx == 0) {
            DCHECK("neck_fusion_stage", fused, (size_t)64*fH*fW);
            g_neck_coarse = alloc_buf((size_t)64*fH*fW);
            clEnqueueCopyBuffer(g_q, fused, g_neck_coarse, 0, 0,
                                (size_t)64*fH*fW*sizeof(nnopt_storage_t), 0, nullptr, nullptr);
            g_neck_coarse_hw = fH*fW;
        }
    }
    // DepthAnythingNeck.forward's captured hook output == the FIRST (coarsest)
    // fused element [1,64,37,49], NOT the finest. Verified numerically: reference
    // neck_output == neck_fusion_stage_output (116032 elems, identical values).
    // We dumped it above at idx==0 under "neck_fusion_stage"; re-dump the SAME
    // coarse tensor here under "neck" so SxS pairs matching shapes. (The finest
    // `fused` feeds the head, which is verified end-to-end.)
    DCHECK("neck", g_neck_coarse, (size_t)64*g_neck_coarse_hw);
    if (g_neck_coarse) pool_release(g_neck_coarse);

    // ── Head: conv1(64->32,3x3p1); interp to (37*14,49*14) align_corners=True;
    //          conv2(32->32,3x3p1); relu; conv3(32->1,1x1); relu; *max_depth ──
    int oh, ow;
    cl_mem c1 = conv2d(fused, 64, fH, fW, Wt.get_buffer("head.conv1.weight"), Wt.get_buffer("head.conv1.bias"), 32, 3, 3, 1, 1, oh, ow);
    pool_release(fused);
    DCHECK("head_conv1", c1, (size_t)32*oh*ow);
    int tH = Hp * PS, tW = Wp * PS; // 518, 686
    cl_mem up = bilinear(c1, 32, oh, ow, tH, tW, 1);
    pool_release(c1);
    int c2h, c2w;
    cl_mem c2 = conv2d(up, 32, tH, tW, Wt.get_buffer("head.conv2.weight"), Wt.get_buffer("head.conv2.bias"), 32, 3, 3, 1, 1, c2h, c2w);
    pool_release(up);
    DCHECK("head_conv2", c2, (size_t)32*c2h*c2w);
    cl_mem r = act_relu(c2, (size_t)32*c2h*c2w);
    pool_release(c2);
    DCHECK("head_activation1", r, (size_t)32*c2h*c2w);
    int c3h, c3w;
    cl_mem c3 = conv2d(r, 32, c2h, c2w, Wt.get_buffer("head.conv3.weight"), Wt.get_buffer("head.conv3.bias"), 1, 1, 1, 1, 0, c3h, c3w);
    pool_release(r);
    DCHECK("head_conv3", c3, (size_t)c3h*c3w);
    cl_mem depth = act_relu(c3, (size_t)c3h*c3w);
    pool_release(c3);
    DCHECK("head_activation2", depth, (size_t)c3h*c3w);
    DCHECK("head", depth, (size_t)c3h*c3w);

    // download; *max_depth (1.0 -> no-op but apply anyway)
    std::vector<float> out = download_f32(depth, (size_t)c3h*c3w);
    pool_release(depth);
    float md = MODEL_CONFIG::MAX_DEPTH;
    for (float& x : out) x *= md;
    return out;
}
