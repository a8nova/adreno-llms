// Reference: image_processing_lfm2_vl_fast.py (transformers, Lfm2VlImageProcessorFast).
//
// Host-only CPU multi-tile image preprocessor for LFM2-VL.
//
// Pipeline (per the reference implementation):
//   1. is_image_too_large(H, W) decides whether tiling activates.
//   2. smart_resize(H, W)        computes the thumbnail target (h_bar, w_bar).
//   3. If too large AND do_image_splitting:
//        a. _get_grid_layout() picks (grid_w, grid_h) minimizing
//           |aspect - grid_w/grid_h| over rectangles with
//           MIN_TILES <= grid_w*grid_h <= MAX_TILES (sorted by area, with
//           tie-break favoring the smaller area unless it loses >50% pixels).
//        b. Resize to (grid_h*TILE_SIZE, grid_w*TILE_SIZE) and unfold into
//           grid_h*grid_w non-overlapping TILE_SIZE x TILE_SIZE tiles,
//           row-major (row 0 cols left→right, then row 1, ...).
//        c. If USE_THUMBNAIL and grid != 1x1, also resize original to
//           (h_bar, w_bar) and append as the last tile.
//   4. Otherwise: single tile resized to (h_bar, w_bar).
//
// The siglip vision tile forward (op_siglip_encoder.cpp) handles normalize +
// patchify on the GPU, so this preprocessor returns raw uint8 HWC RGB bytes
// for each tile.

#pragma once

#include <cstdint>
#include <vector>

struct Lfm2VlTile {
    std::vector<uint8_t> rgb;  // resized + cropped tile RGB bytes (H_px * W_px * 3, channels-last)
    int H_px = 0;              // pixel height (multiple of 16)
    int W_px = 0;              // pixel width  (multiple of 16)
    int spatial_h = 0;         // = H_px / 16
    int spatial_w = 0;         // = W_px / 16
    bool is_thumbnail = false; // true for the last tile if use_thumbnail and grid != 1x1
};

struct Lfm2VlImageProcessorOutput {
    std::vector<Lfm2VlTile> tiles;  // PyTorch order: row-major main tiles, then thumbnail (if any)
    int grid_h = 0;                  // tile grid height (rows)
    int grid_w = 0;                  // tile grid width  (cols)
    int thumbnail_h = 0;             // thumbnail H_px (0 if no thumbnail)
    int thumbnail_w = 0;             // thumbnail W_px (0 if no thumbnail)
};

// Preprocess a single RGB input image into a list of tiles matching
// Lfm2VlImageProcessorFast. Returns false on invalid input. Output tiles are
// emitted in the same order the PyTorch processor produces them, which is the
// order the tokenizer's <|img_row_r_col_c|> separators expect.
bool lfm2_vl_preprocess_image(
    const std::vector<uint8_t>& rgb_in,  // HWC, channels-last, uint8
    int H_in, int W_in,                  // input dimensions
    Lfm2VlImageProcessorOutput& out);
