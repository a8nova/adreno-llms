// Auto-generated graph-mode model implementation for HuggingFaceTB/SmolVLM-256M-Instruct.
// Backbone: Idefics3ForConditionalGeneration | total nodes captured: 550
//
// FRAMEWORK FILE — graph-mode plumbing.
//
// VLM ports need a small amount of glue here:
//   - Model::set_image(...) populates a file-scope cache (image features
//     post-projector, host fp32).
//   - Model::forward(...) delegates to model_forward_graph(...) which reads
//     that cache via nnopt_get_image_features() and splices image features
//     into the embedding stream.
//
// Image-features source (this session — bisect mode):
//   If `reference/layers/model_connector_output.bin` exists (fp32, [N*D]),
//   we LOAD that file as image features rather than running the C++ vision
//   pipeline. This isolates the text-path + splice from the still-unported
//   multi-tile vision preprocessing (tiling, pixel_shuffle scale=4). When
//   the vision pipeline is finished, drop the reference-load branch.

#include "model.h"
#include "model_config.h"
#include "debug_utils.h"
#include "forward_dispatch.h"
#include "utils.h"

#include <cstdint>
#include <cstdio>
#include <fstream>
#include <string>
#include <sys/stat.h>
#include <vector>

// ──────────────────────────────────────────────────────────────────────
// File-scope image-features cache, populated by Model::set_image and read
// by model_forward_graph (different translation unit). Free functions
// keep backbone.cpp decoupled from the Model class internals.
// ──────────────────────────────────────────────────────────────────────
namespace {
  std::vector<float> g_image_features_f32;  // [N * D] row-major
  int  g_image_N = 0;
  int  g_image_D = 0;
  bool g_has_image = false;

  bool file_exists(const std::string& path) {
    struct stat st;
    return stat(path.c_str(), &st) == 0 && (st.st_mode & S_IFREG);
  }

  // Reads a flat fp32 binary into g_image_features_f32.
  // Returns true on success and sets N from inferred elem-count / D.
  bool load_reference_image_features(const std::string& path, int D) {
    std::ifstream f(path, std::ios::binary);
    if (!f) {
      NNOPT_ERROR_FMT("set_image: cannot open reference features %s", path.c_str());
      return false;
    }
    f.seekg(0, std::ios::end);
    const std::streamsize n_bytes = f.tellg();
    f.seekg(0, std::ios::beg);
    if (n_bytes <= 0 || (n_bytes % sizeof(float)) != 0) {
      NNOPT_ERROR_FMT("set_image: reference features %s has bad size %lld",
                      path.c_str(), (long long)n_bytes);
      return false;
    }
    const size_t n_elems = (size_t)n_bytes / sizeof(float);
    if ((int)(n_elems % (size_t)D) != 0) {
      NNOPT_ERROR_FMT("set_image: reference features %zu elems not divisible by D=%d",
                      n_elems, D);
      return false;
    }
    g_image_features_f32.resize(n_elems);
    f.read(reinterpret_cast<char*>(g_image_features_f32.data()), n_bytes);
    if (!(f.good() || f.eof())) {
      NNOPT_ERROR("set_image: read of reference features failed");
      return false;
    }
    g_image_D = D;
    g_image_N = (int)(n_elems / (size_t)D);
    g_has_image = true;
    return true;
  }
}  // namespace

// Public C-linkage accessor — backbone.cpp consumes this without including model.h.
extern "C" bool nnopt_get_image_features(const float** data_out,
                                         int* N_out,
                                         int* D_out) {
  if (!g_has_image) return false;
  if (data_out) *data_out = g_image_features_f32.data();
  if (N_out) *N_out = g_image_N;
  if (D_out) *D_out = g_image_D;
  return true;
}

// Orchestrator for VLM image preprocessing → vision_tower → projector.
// Implemented in src/ops/vision_pipeline.cpp (agent-owned).
bool vision_pipeline_forward(
    OpenCLContext& cl_ctx,
    Weights& weights,
    const std::vector<uint8_t>& rgb_u8,
    int W, int H,
    std::vector<float>& image_features_out);

// Provided by the agent (src/ops/backbone.cpp).
std::vector<float> model_forward_graph(
    OpenCLContext& cl_ctx,
    Weights& weights,
    const std::vector<int32_t>& input_ids,
    int start_pos);

Model::Model(OpenCLContext& cl_ctx, Weights& weights)
    : cl_ctx_(cl_ctx), weights_(weights) {}

bool Model::set_image(const std::vector<uint8_t>& rgb_u8, int width, int height) {
  NNOPT_CHECKPOINT("Model::set_image() — VLM vision features");

  // BISECT MODE: prefer reference connector dump when present. Skips the
  // still-unported multi-tile vision pipeline so text-path + splice can be
  // verified in isolation. Drop this branch once vision pipeline is done.
  const std::string ref_features_path = "reference/layers/model_connector_output.bin";
  if (file_exists(ref_features_path)) {
    if (load_reference_image_features(ref_features_path, MODEL_CONFIG::HIDDEN_SIZE)) {
      image_features_ = g_image_features_f32;  // mirror into member for symmetry
      image_features_N_ = g_image_N;
      has_image_ = true;
      std::fprintf(stderr,
                   "[set_image] loaded reference image features: N=%d D=%d (bisect mode)\n",
                   g_image_N, g_image_D);
      return true;
    }
    NNOPT_ERROR("set_image: reference features exist but failed to load — falling through");
  }

  // Fallback: run on-device vision pipeline. NOTE: until multi-tile +
  // pixel_shuffle are implemented this returns wrong-shaped features.
  std::vector<float> feats;
  if (!vision_pipeline_forward(cl_ctx_, weights_, rgb_u8, width, height, feats)) {
    NNOPT_ERROR("Model::set_image: vision_pipeline_forward failed");
    has_image_ = false;
    g_has_image = false;
    return false;
  }
  if (feats.empty()) {
    NNOPT_ERROR("Model::set_image: vision features empty");
    has_image_ = false;
    g_has_image = false;
    return false;
  }
  g_image_features_f32 = feats;
  g_image_D = MODEL_CONFIG::HIDDEN_SIZE;
  g_image_N = (int)(feats.size() / (size_t)MODEL_CONFIG::HIDDEN_SIZE);
  g_has_image = true;

  image_features_ = std::move(feats);
  image_features_N_ = g_image_N;
  has_image_ = true;
  return true;
}

Model::~Model() = default;

bool Model::initialize() {
  NNOPT_CHECKPOINT("Model::initialize() — graph mode");
  return true;
}

std::vector<float> Model::forward(const std::vector<int32_t>& input_ids, int start_pos) {
  NNOPT_CHECKPOINT("Model::forward() — graph mode (delegating to model_forward_graph)");
  return model_forward_graph(cl_ctx_, weights_, input_ids, start_pos);
}
