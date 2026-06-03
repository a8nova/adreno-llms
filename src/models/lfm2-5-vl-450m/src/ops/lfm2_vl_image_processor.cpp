// Reference: image_processing_lfm2_vl_fast.py (transformers, Lfm2VlImageProcessorFast).
// See header for the high-level algorithm description.
//
// Implementation notes:
//   * Bilinear resize uses stb_image_resize2.h (stbir_resize) with
//     STBIR_FILTER_TRIANGLE, which matches PyTorch
//     F.resize(mode='bilinear', antialias=True) for both up- and down-sampling.
//     (Triangle filter == bilinear texture; combined with stb's box-prefilter
//     during downsampling it produces antialiased output that matches
//     PyTorch's reference within a few LSB.)
//   * smart_resize() returns (w_bar, h_bar) like the python.
//   * find_closest_aspect_ratio() iterates the canonical sorted ratio list
//     and picks the min |aspect - rw/rh|. On equal distance the second-found
//     pair wins only when the smaller area would lose >50% pixels (matches
//     the reference's `area > 0.5 * target_area` tie-break).
//   * Main tiles are emitted row-major (row outer, col inner); thumbnail
//     appended last when use_thumbnail and grid != 1x1.

#include "ops/lfm2_vl_image_processor.h"

#include "debug_utils.h"
#include "model_config.h"

#include "stb_image_resize2.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>
#include <set>
#include <utility>
#include <vector>

namespace {

// PyTorch reference: round(number / factor) * factor  (banker's rounding).
// Python's round() uses banker's; C's std::round() is half-away-from-zero.
// For our use we only ever call this on positive integers/2 boundaries where
// the two agree, but match python explicitly to avoid future surprises.
inline int round_by_factor(int number, int factor) {
    // half-to-even rounding for positives
    double q = static_cast<double>(number) / factor;
    double floor_q = std::floor(q);
    double frac = q - floor_q;
    long long r;
    if (frac < 0.5) {
        r = static_cast<long long>(floor_q);
    } else if (frac > 0.5) {
        r = static_cast<long long>(floor_q) + 1;
    } else {
        // tie: round to even
        long long fi = static_cast<long long>(floor_q);
        r = (fi % 2 == 0) ? fi : (fi + 1);
    }
    return static_cast<int>(r * factor);
}

// smart_resize from the reference. Returns (w_bar, h_bar) — width first to
// mirror the python signature.
std::pair<int, int> smart_resize(int height, int width,
                                 int downsample_factor,
                                 int min_image_tokens,
                                 int max_image_tokens,
                                 int encoder_patch_size) {
    const int total_factor = encoder_patch_size * downsample_factor;
    const long long smart_min = static_cast<long long>(min_image_tokens) *
                                encoder_patch_size * encoder_patch_size *
                                downsample_factor * downsample_factor;
    const long long smart_max = static_cast<long long>(max_image_tokens) *
                                encoder_patch_size * encoder_patch_size *
                                downsample_factor * downsample_factor;

    int h_bar = std::max(total_factor, round_by_factor(height, total_factor));
    int w_bar = std::max(total_factor, round_by_factor(width, total_factor));

    const long long area = static_cast<long long>(h_bar) * w_bar;
    if (area > smart_max) {
        double beta = std::sqrt(
            static_cast<double>(height) * width / static_cast<double>(smart_max));
        int hb = static_cast<int>(std::floor(height / beta / total_factor)) * total_factor;
        int wb = static_cast<int>(std::floor(width / beta / total_factor)) * total_factor;
        h_bar = std::max(total_factor, hb);
        w_bar = std::max(total_factor, wb);
    } else if (area < smart_min) {
        double beta = std::sqrt(
            static_cast<double>(smart_min) / (static_cast<double>(height) * width));
        h_bar = static_cast<int>(std::ceil(height * beta / total_factor)) * total_factor;
        w_bar = static_cast<int>(std::ceil(width * beta / total_factor)) * total_factor;
    }

    return {w_bar, h_bar};  // (width, height) to match python smart_resize
}

bool is_image_too_large(int height, int width,
                        int max_image_tokens,
                        int encoder_patch_size,
                        int downsample_factor,
                        double max_pixels_tolerance) {
    const int total_factor = encoder_patch_size * downsample_factor;
    const int h_bar = std::max(encoder_patch_size, round_by_factor(height, total_factor));
    const int w_bar = std::max(encoder_patch_size, round_by_factor(width, total_factor));
    const double limit = static_cast<double>(max_image_tokens) *
                         encoder_patch_size * encoder_patch_size *
                         downsample_factor * downsample_factor *
                         max_pixels_tolerance;
    return static_cast<double>(h_bar) * w_bar > limit;
}

// Canonical, area-sorted unique (w, h) ratios with min<=w*h<=max. Matches
// the python's _target_ratios cache (which uses sorted(set(...), key=area)).
std::vector<std::pair<int, int>> target_ratios(int min_tiles, int max_tiles) {
    std::set<std::pair<int, int>> seen;
    std::vector<std::pair<int, int>> out;
    for (int n = min_tiles; n <= max_tiles; ++n) {
        for (int w = 1; w <= n; ++w) {
            for (int h = 1; h <= n; ++h) {
                if (w * h >= min_tiles && w * h <= max_tiles) {
                    if (seen.insert({w, h}).second) {
                        out.push_back({w, h});
                    }
                }
            }
        }
    }
    std::stable_sort(out.begin(), out.end(),
                     [](const std::pair<int, int>& a, const std::pair<int, int>& b) {
                         return a.first * a.second < b.first * b.second;
                     });
    return out;
}

// find_closest_aspect_ratio — matches python exactly. Iterates in area-sorted
// order and tracks the running best. On ratio_diff strictly less than the
// current best, replace. On equal diff (tie), replace only when
// area > 0.5 * tile_size^2 * w * h (i.e., the smaller current pick would lose
// too much fidelity).
std::pair<int, int> find_closest_aspect_ratio(
    double aspect_ratio,
    const std::vector<std::pair<int, int>>& ratios,
    int width, int height, int tile_size) {

    double best_diff = std::numeric_limits<double>::infinity();
    std::pair<int, int> best = {1, 1};
    const long long area = static_cast<long long>(width) * height;

    for (const auto& r : ratios) {
        double tar = static_cast<double>(r.first) / r.second;
        double diff = std::abs(aspect_ratio - tar);
        if (diff < best_diff) {
            best_diff = diff;
            best = r;
        } else if (diff == best_diff) {
            long long target_area = static_cast<long long>(tile_size) * tile_size *
                                    r.first * r.second;
            if (static_cast<double>(area) > 0.5 * target_area) {
                best = r;
            }
        }
    }
    return best;
}

// CPU bilinear resize with antialiasing (stb_image_resize2). Matches PyTorch's
// F.resize(..., antialias=True) within small numerical tolerance.
bool resize_rgb_u8(const uint8_t* in, int H_in, int W_in,
                   int H_out, int W_out,
                   std::vector<uint8_t>& out) {
    out.assign(static_cast<size_t>(H_out) * W_out * 3, 0);
    void* res = stbir_resize(
        in, W_in, H_in, /*input_stride*/ 0,
        out.data(), W_out, H_out, /*output_stride*/ 0,
        STBIR_RGB, STBIR_TYPE_UINT8,
        STBIR_EDGE_CLAMP, STBIR_FILTER_TRIANGLE);
    if (!res) {
        NNOPT_ERROR_FMT(
            "lfm2_vl_image_processor: stbir_resize failed (in=%dx%d → out=%dx%d)",
            W_in, H_in, W_out, H_out);
        return false;
    }
    return true;
}

}  // namespace

bool lfm2_vl_preprocess_image(
    const std::vector<uint8_t>& rgb_in,
    int H_in, int W_in,
    Lfm2VlImageProcessorOutput& out) {

    NNOPT_CHECKPOINT_FMT("lfm2_vl_preprocess_image: H=%d W=%d", H_in, W_in);
    out.tiles.clear();
    out.grid_h = 0;
    out.grid_w = 0;
    out.thumbnail_h = 0;
    out.thumbnail_w = 0;

    if (rgb_in.empty() || H_in <= 0 || W_in <= 0) {
        NNOPT_ERROR("lfm2_vl_preprocess_image: invalid input");
        return false;
    }
    if (rgb_in.size() != static_cast<size_t>(H_in) * W_in * 3u) {
        NNOPT_ERROR_FMT(
            "lfm2_vl_preprocess_image: rgb size (%zu) != H*W*3 (%d)",
            rgb_in.size(), H_in * W_in * 3);
        return false;
    }

    const int tile_size = MODEL_CONFIG::TILE_SIZE;
    const int patch = MODEL_CONFIG::ENCODER_PATCH_SIZE;
    const int downsample = MODEL_CONFIG::DOWNSAMPLE_FACTOR;
    const int min_tiles = MODEL_CONFIG::MIN_TILES;
    const int max_tiles = MODEL_CONFIG::MAX_TILES;
    const int min_image_tokens = MODEL_CONFIG::MIN_IMAGE_TOKENS;
    const int max_image_tokens = MODEL_CONFIG::MAX_IMAGE_TOKENS;
    const double max_pixels_tol = static_cast<double>(MODEL_CONFIG::MAX_PIXELS_TOLERANCE);
    const bool use_thumbnail = MODEL_CONFIG::USE_THUMBNAIL;
    const bool do_splitting = MODEL_CONFIG::DO_IMAGE_SPLITTING;

    // 1. Compute thumbnail target via smart_resize.
    auto wh = smart_resize(H_in, W_in, downsample,
                           min_image_tokens, max_image_tokens, patch);
    const int new_width = wh.first;
    const int new_height = wh.second;

    // 2. Is the image large enough to tile?
    const bool too_large = is_image_too_large(
        H_in, W_in, max_image_tokens, patch, downsample, max_pixels_tol);

    if (too_large && do_splitting) {
        // 3. Pick grid layout.
        double aspect = static_cast<double>(W_in) / H_in;
        auto ratios = target_ratios(min_tiles, max_tiles);
        auto grid = find_closest_aspect_ratio(aspect, ratios, W_in, H_in, tile_size);
        const int grid_w = grid.first;
        const int grid_h = grid.second;
        const int target_w = tile_size * grid_w;
        const int target_h = tile_size * grid_h;

        // Resize whole image to grid_h*tile x grid_w*tile.
        std::vector<uint8_t> resized;
        if (!resize_rgb_u8(rgb_in.data(), H_in, W_in,
                           target_h, target_w, resized)) {
            return false;
        }

        out.grid_h = grid_h;
        out.grid_w = grid_w;

        // Unfold into row-major tiles.
        out.tiles.reserve(static_cast<size_t>(grid_h) * grid_w +
                          (use_thumbnail && grid_h * grid_w != 1 ? 1u : 0u));
        for (int row = 0; row < grid_h; ++row) {
            for (int col = 0; col < grid_w; ++col) {
                Lfm2VlTile t;
                t.H_px = tile_size;
                t.W_px = tile_size;
                t.spatial_h = tile_size / patch;
                t.spatial_w = tile_size / patch;
                t.is_thumbnail = false;
                t.rgb.resize(static_cast<size_t>(tile_size) * tile_size * 3);
                // Copy a tile_size x tile_size sub-rect from resized at
                // (row*tile_size, col*tile_size). RGB stride = target_w*3.
                const int y0 = row * tile_size;
                const int x0 = col * tile_size;
                for (int y = 0; y < tile_size; ++y) {
                    const uint8_t* src_row =
                        resized.data() +
                        (static_cast<size_t>(y0 + y) * target_w + x0) * 3;
                    uint8_t* dst_row =
                        t.rgb.data() + static_cast<size_t>(y) * tile_size * 3;
                    std::memcpy(dst_row, src_row, static_cast<size_t>(tile_size) * 3);
                }
                out.tiles.push_back(std::move(t));
            }
        }

        // 4. Optional thumbnail.
        if (use_thumbnail && grid_h * grid_w != 1) {
            Lfm2VlTile th;
            th.H_px = new_height;
            th.W_px = new_width;
            th.spatial_h = new_height / patch;
            th.spatial_w = new_width / patch;
            th.is_thumbnail = true;
            if (!resize_rgb_u8(rgb_in.data(), H_in, W_in,
                               new_height, new_width, th.rgb)) {
                return false;
            }
            out.thumbnail_h = new_height;
            out.thumbnail_w = new_width;
            out.tiles.push_back(std::move(th));
        }
    } else {
        // Single resized tile, no splitting.
        Lfm2VlTile t;
        t.H_px = new_height;
        t.W_px = new_width;
        t.spatial_h = new_height / patch;
        t.spatial_w = new_width / patch;
        t.is_thumbnail = false;
        if (!resize_rgb_u8(rgb_in.data(), H_in, W_in,
                           new_height, new_width, t.rgb)) {
            return false;
        }
        out.grid_h = 1;
        out.grid_w = 1;
        out.tiles.push_back(std::move(t));
    }

    NNOPT_CHECKPOINT_FMT(
        "lfm2_vl_preprocess_image: %zu tiles (grid %dx%d, thumb %dx%d)",
        out.tiles.size(), out.grid_h, out.grid_w, out.thumbnail_h, out.thumbnail_w);
    return true;
}

// ─────────────────────────────────────────────────────────────────────
// Standalone CPU-only validation test (no OpenCL, no weights).
// Compile with:
//   clang++ -std=c++17 -DLFM2VL_PREPROCESS_TEST -I src \
//       -I build/fp16/_deps/stb-src \
//       src/ops/lfm2_vl_image_processor.cpp \
//       src/third_party/stb_image_impl.cpp \
//       -o /tmp/preprocess_test && /tmp/preprocess_test fixtures/sample.jpg
// ─────────────────────────────────────────────────────────────────────
#ifdef LFM2VL_PREPROCESS_TEST

#include "load_image.h"

#include <cstdio>
#include <cstdlib>
#include <string>

int main(int argc, char** argv) {
    const char* path = argc >= 2 ? argv[1] : "fixtures/sample.jpg";
    ImageBufferU8 img;
    if (!load_image_rgb_u8(path, img)) {
        std::fprintf(stderr, "test: failed to load %s: %s\n", path, img.error.c_str());
        return 1;
    }
    std::printf("loaded %s: H=%d W=%d C=%d\n", path, img.H, img.W, img.C);

    Lfm2VlImageProcessorOutput out;
    if (!lfm2_vl_preprocess_image(img.data, img.H, img.W, out)) {
        std::fprintf(stderr, "test: preprocess failed\n");
        return 1;
    }
    std::printf("grid: %dx%d, num_tiles=%zu, thumbnail=%dx%d\n",
                out.grid_h, out.grid_w, out.tiles.size(),
                out.thumbnail_h, out.thumbnail_w);
    for (size_t i = 0; i < out.tiles.size(); ++i) {
        const auto& t = out.tiles[i];
        std::printf("  tile[%zu]: H_px=%d W_px=%d sh=%d sw=%d thumb=%d\n",
                    i, t.H_px, t.W_px, t.spatial_h, t.spatial_w, t.is_thumbnail ? 1 : 0);
    }

    // Expected for fixtures/sample.jpg (1024x704):
    //   7 tiles, grid 3x2, first 6 are 512x512, last is 416x608, thumbnail flag on last.
    int rc = 0;
    if (out.tiles.size() != 7) {
        std::fprintf(stderr, "FAIL: expected 7 tiles, got %zu\n", out.tiles.size());
        rc = 2;
    }
    if (out.grid_h != 2 || out.grid_w != 3) {
        std::fprintf(stderr, "FAIL: expected grid 3x2 (w x h), got %dx%d\n",
                     out.grid_w, out.grid_h);
        rc = 2;
    }
    for (size_t i = 0; i < 6 && i < out.tiles.size(); ++i) {
        if (out.tiles[i].H_px != 512 || out.tiles[i].W_px != 512) {
            std::fprintf(stderr, "FAIL: tile %zu expected 512x512, got %dx%d\n",
                         i, out.tiles[i].H_px, out.tiles[i].W_px);
            rc = 2;
        }
    }
    if (out.tiles.size() >= 7) {
        const auto& th = out.tiles[6];
        if (th.H_px != 416 || th.W_px != 608 || !th.is_thumbnail) {
            std::fprintf(stderr,
                         "FAIL: thumbnail expected 416x608 is_thumbnail=1, "
                         "got %dx%d is_thumbnail=%d\n",
                         th.H_px, th.W_px, th.is_thumbnail ? 1 : 0);
            rc = 2;
        }
    }
    if (rc == 0) std::printf("PASS\n");
    return rc;
}

#endif  // LFM2VL_PREPROCESS_TEST
