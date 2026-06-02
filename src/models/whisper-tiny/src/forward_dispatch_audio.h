#pragma once
// Forward-dispatch audio fixture plumbing.
// Reference: docs/MODALITY_ASR.md (ForwardDispatch::set_input_features)

#include <CL/cl.h>

namespace ForwardDispatch {

// Global (process-local) handle to the encoder input_features buffer.
// main.cpp sets this once (or per-step), Model::forward reads it.
void set_input_features(cl_mem feats);
cl_mem get_input_features();

}  // namespace ForwardDispatch
