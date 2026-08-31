// midi.cpp — live MIDI conditioning: wire format in, conditioning-block values out.
//
// No OpenCL, no weights, no engine state. See midi.h for the encoding and for why this is its own
// translation unit; tests/midi_oracle_test.cpp compiles it against upstream's own implementation.

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
// at the musiccoca/notes step of 0.2 and code 4 at the drums step of 1.0. These three channels are
// the model's own "how hard should I follow each conditioning stream" input — the notes one is the
// MIDI-following strength. Left at the port's long-standing 3.0/3.0/3.0 on purpose: changing it
// would break the byte-identity gate, and there is no on-device objective to auto-tune it against,
// so it must not become a flag either (HANDOFF-recordable-queues.md §6).
static const int kCfg[3] = {20, 20, 4};

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
    for (int c : kCfg)                       out.push_back(c + kMidiOffset);
    // Applied as a patch rather than woven into the loops above so there is ONE implementation of
    // "where do the MIDI channels live", shared with the serve loop's cached-block path.
    if (midi && !midi_apply_to_conditioning(*midi, out)) return false;
    return true;
}
