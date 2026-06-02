// forward_dispatch_audio.cpp
// Reference: docs/MODALITY_ASR.md (ForwardDispatch::set_input_features)

#include "forward_dispatch_audio.h"

namespace ForwardDispatch {

static cl_mem g_input_features = nullptr;

void set_input_features(cl_mem feats) {
    g_input_features = feats;
}

cl_mem get_input_features() {
    return g_input_features;
}

}  // namespace ForwardDispatch
