// midi.cpp — live MIDI conditioning: wire format in, conditioning-block values out.
//
// No OpenCL, no weights, no engine state. See midi.h for the encoding and for why this is its own
// translation unit; tests/midi_oracle_test.cpp compiles it against upstream's own implementation.

#include <cmath>
#include "midi.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>

// Its own error printer rather than debug_utils.h: that header reaches for <CL/cl.h>, and the whole
// reason this file is a separate translation unit is that the correctness gate compiles it on a host
// with no OpenCL at all. Same destination and same shape as NNOPT_ERROR_FMT.
#define MIDI_ERROR_FMT(fmt, ...) do { \
    std::fprintf(stderr, "ERROR: " fmt " (midi.cpp:%d)\n", __VA_ARGS__, __LINE__); \
    std::fflush(stderr); \
} while (0)

// One pitch's app-reported state → the value the MODEL is given.
// Mirrors calculate_token() in upstream's live engine (core/src/mlx_engine.cpp:1430) at its
// DEFAULT onset_mode of 0 ("mask onsets"): a held pitch is always 3, i.e. the model is told the
// note is sounding but is left free to voice it as an onset or as a continuation. The 1/2
// distinction the app sends is therefore deliberately discarded here — it is carried on the wire
// so that enabling onset_mode 1 later is an engine change, not an app release.
int midi_note_token(int8_t state) {
    if (state < 0) return -1;                 // masked — the model decides everything
    if (state == 0) return 0;                 // explicitly off
    return 3;                                 // held: 1 (continuation) and 2 (onset) both fold here
}

bool midi_parse(const std::string& spec, MidiState& out) {
    MidiState st;
    st.active = true;
    // "60:2,64:1|drum:1" — the drum suffix is optional and always last.
    std::string notes_part = spec;
    const size_t bar = spec.find('|');
    if (bar != std::string::npos) {
        const std::string drum_part = spec.substr(bar + 1);
        notes_part = spec.substr(0, bar);
        static const std::string kDrumKey = "drum:";
        if (drum_part.rfind(kDrumKey, 0) != 0) {
            MIDI_ERROR_FMT("midi_parse: expected \"drum:<v>\" after '|', got \"%s\"", drum_part.c_str());
            return false;
        }
        const std::string v = drum_part.substr(kDrumKey.size());
        char* end = nullptr;
        const long dv = std::strtol(v.c_str(), &end, 10);
        if (end == v.c_str() || *end != '\0' || dv < -1 || dv > 1) {
            MIDI_ERROR_FMT("midi_parse: drum value must be -1, 0 or 1 (got \"%s\")", v.c_str());
            return false;
        }
        st.drum = (int8_t)dv;
    }
    // Empty note list is legal and meaningful: "MIDI is live, nothing held" — every pitch masked.
    size_t i = 0;
    while (i < notes_part.size()) {
        size_t j = notes_part.find(',', i);
        if (j == std::string::npos) j = notes_part.size();
        const std::string item = notes_part.substr(i, j - i);
        i = j + 1;
        if (item.empty()) continue;           // tolerate a trailing comma
        const size_t colon = item.find(':');
        if (colon == std::string::npos) {
            MIDI_ERROR_FMT("midi_parse: expected pitch:state, got \"%s\"", item.c_str());
            return false;
        }
        const std::string ps = item.substr(0, colon), vs = item.substr(colon + 1);
        char* pe = nullptr; const long pitch = std::strtol(ps.c_str(), &pe, 10);
        char* ve = nullptr; const long val   = std::strtol(vs.c_str(), &ve, 10);
        if (pe == ps.c_str() || *pe != '\0' || ve == vs.c_str() || *ve != '\0') {
            MIDI_ERROR_FMT("midi_parse: unparsable element \"%s\"", item.c_str());
            return false;
        }
        if (pitch < 0 || pitch >= kMidiNotes) {
            MIDI_ERROR_FMT("midi_parse: pitch %ld out of range 0..%d", pitch, kMidiNotes - 1);
            return false;
        }
        if (val < -1 || val > 3) {
            MIDI_ERROR_FMT("midi_parse: state %ld out of range -1..3 (pitch %ld)", val, pitch);
            return false;
        }
        st.notes[pitch] = (int8_t)val;
    }
    out = st;
    return true;
}

bool midi_apply_to_conditioning(const MidiState& midi, std::vector<int32_t>& tokens) {
    if ((int)tokens.size() != kMidiConditioningTokens) {
        MIDI_ERROR_FMT("midi_apply_to_conditioning: block is %zu tokens, expected %d",
                        tokens.size(), kMidiConditioningTokens);
        return false;
    }
    if (!midi.active) return true;   // no MIDI this request: leave the block exactly as it was
    for (int p = 0; p < kMidiNotes; ++p)
        tokens[kMidiNotesAt + p] = midi_note_token(midi.notes[p]) + kMidiOffset;
    tokens[kMidiDrumAt] = (int)midi.drum + kMidiOffset;
    return true;
}


// CFG scales, discretized the way magenta_rt does it: CFG = -1.0 + step * code, so 3.0 is code 20
// at the musiccoca/notes step of 0.2 and 1.0 is code 2 at the drums step of 1.0. These three
// channels are the model's own "how hard should I follow each conditioning stream" input — the
// notes one is the MIDI-following strength.
//
// {3.0, 1.0, 1.0}, matching the reference oracle the port is A/B'd against
// (~/Downloads/magenta-ref-mac/settings.json, and MagentaRT2System's own cfg_scales default in
// magenta_rt/mlx/system.py:256). This was 3.0/3.0/3.0 until 2026-09-01. Only musiccoca was ever
// right: the common render leaves all 128 note channels at -1 and then asked the model to follow
// that empty conditioning at 3.0, which no upstream path does.
static const int kCfg[3] = {20, 10, 2};
// Quantisation step per channel, from magenta_rt/config.py: musiccoca and notes share
// CFG_CONDITIONING_MUSICCOCA_NOTES (step 0.2), drums has CFG_CONDITIONING_DRUMS (step 1.0).
static const float kCfgStep[3] = {0.2f, 0.2f, 1.0f};

// The SAME three scales as floats, for the logit-space guidance. Derived from kCfg rather than
// written out again: the conditioning channels tell the model how hard to follow each stream and
// the guidance actually does it, so if those two ever disagreed the model would be told one thing
// and given another, silently.
// Live override of the three scales (request key cfgtok=) and of how many fine MusicCoCa levels are
// masked (mctail=). Unset, both reproduce the constants above exactly, so every render that does
// not ask keeps its conditioning byte-for-byte.
static float g_cfg_scale[3] = {-1.0f + kCfgStep[0] * (float)kCfg[0],
                               -1.0f + kCfgStep[1] * (float)kCfg[1],
                               -1.0f + kCfgStep[2] * (float)kCfg[2]};
static int g_mc_tail = 0;

void midi_set_cfg_scales(float musiccoca, float notes, float drums) {
    g_cfg_scale[0] = musiccoca; g_cfg_scale[1] = notes; g_cfg_scale[2] = drums;
}
void midi_set_musiccoca_tail(int levels) { g_mc_tail = levels < 0 ? 0 : (levels > kMidiStyleTokens ? kMidiStyleTokens : levels); }

// magenta_rt/mlx/export.py _discretize_cfg_token, in the same float32 arithmetic: clamp to
// [-1, 7], bin = round((v + 1) / step) (mx.round is round-half-to-even, as is nearbyintf in the
// default rounding mode), clip to [0, max_bin]. The +kMidiOffset is added by the caller.
static int cfg_bin(float v, float step, int max_bin) {
    const float c = v < -1.0f ? -1.0f : (v > 7.0f ? 7.0f : v);
    float b = std::nearbyintf((c - (-1.0f)) / step);
    if (b < 0.0f) b = 0.0f;
    if (b > (float)max_bin) b = (float)max_bin;
    return (int)b;
}

// Applied to a finished 144-token block, last, like MIDI: the block may come out of the per-prompt
// cache. Upstream's live engine (core/src/mlx_engine.cpp generate_frame) keeps only the coarsest
// 12 - kMusicCoCaMaskedTailLevels style levels and pins the rest to the mask id; the CFG channels
// carry whatever scales the player set.
bool midi_apply_live_policy(std::vector<int32_t>& tokens) {
    if ((int)tokens.size() != kMidiConditioningTokens) return false;
    for (int i = kMidiStyleTokens - g_mc_tail; i < kMidiStyleTokens; ++i) tokens[i] = kMaskedCondToken;
    static const int kMaxBin[3] = {40, 40, 8};
    for (int c = 0; c < 3; ++c)
        tokens[kMidiCfgAt + c] = cfg_bin(g_cfg_scale[c], kCfgStep[c], kMaxBin[c]) + kMidiOffset;
    return true;
}

void midi_cfg_scales(float* musiccoca, float* notes, float* drums) {
    if (musiccoca) *musiccoca = g_cfg_scale[0];
    if (notes)     *notes     = g_cfg_scale[1];
    if (drums)     *drums     = g_cfg_scale[2];
}

bool midi_build_conditioning(const std::vector<int32_t>& style_tokens, const MidiState* midi,
                             std::vector<int32_t>& out) {
    if ((int)style_tokens.size() != kMidiStyleTokens) {
        MIDI_ERROR_FMT("midi_build_conditioning: expected %d style tokens, got %zu",
                       kMidiStyleTokens, style_tokens.size());
        return false;
    }
    out.clear();
    out.reserve(kMidiConditioningTokens);
    for (int t : style_tokens)               out.push_back(t + kMidiOffset);
    for (int i = 0; i < kMidiNotes; ++i)     out.push_back(-1 + kMidiOffset);
    for (int i = 0; i < kMidiDrums; ++i)     out.push_back(-1 + kMidiOffset);
    // +7 like every other channel: export.py:234 redefines NUM_RESERVED_TOKENS as 6 + 1 before
    // _discretize_cfg_token uses it. Verified 2026-09-29: the published mrt2_small.mlxfn is an
    // 8-bit, num_cfgs=0 export and reproduces it 600/600 codes.
    for (int c : kCfg)                       out.push_back(c + kMidiOffset);
    // Applied as a patch rather than woven into the loops above so there is ONE implementation of
    // "where do the MIDI channels live", shared with the serve loop's cached-block path.
    if (midi && !midi_apply_to_conditioning(*midi, out)) return false;
    return true;
}
