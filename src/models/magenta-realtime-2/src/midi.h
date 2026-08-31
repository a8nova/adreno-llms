#pragma once
// ─────────────────────────────────────────────────────────────────────────────
// Live MIDI conditioning for Magenta RT-2.
//
// Deliberately free of OpenCL and of every other engine dependency: this is pure integer policy —
// "what does the model get told about each pitch" — and keeping it in its own translation unit is
// what lets the correctness gate compile it on the HOST and diff it against upstream's own
// implementation, instead of only being able to observe it through a 2 GB model on a phone.
//
// Reference: magenta_rt/config.py:150-176 for the encoding, and upstream's live C++ engine
// (core/src/mlx_engine.cpp: calculate_token / populate_condition_tokens, core/src/midi_note_tracker.*)
// for the policy that turns a player's hands into those values.
// ─────────────────────────────────────────────────────────────────────────────

#include <cstdint>
#include <string>
#include <vector>

// ── the conditioning block's layout ──────────────────────────────────────────
// 12 MusicCoCa style codes ++ 128 note channels ++ 1 drum channel ++ 3 CFG channels.
constexpr int kMidiStyleTokens = 12;
constexpr int kMidiNotes       = 128;
constexpr int kMidiDrums       = 1;
constexpr int kMidiCfgChannels = 3;
constexpr int kMidiConditioningTokens =
    kMidiStyleTokens + kMidiNotes + kMidiDrums + kMidiCfgChannels;   // 144

// Every conditioning value is shifted by NUM_RESERVED_TOKENS (6) + 1 for the dropout token, so the
// model's "-1 = unconstrained" lands on 6. magenta_rt/mlx/system.py:364 applies exactly this, once,
// to the whole concatenated block — the per-CHANNEL offsets (each channel owns its own slice of the
// embedding table) are applied inside the model, by the encoder's MultiChannelEmbedding, which this
// port already reproduces in src/ops/Encoder.cpp.
constexpr int kMidiOffset = 7;

// Index of each section within the 144.
constexpr int kMidiNotesAt = kMidiStyleTokens;                 // 12
constexpr int kMidiDrumAt  = kMidiNotesAt + kMidiNotes;        // 140
constexpr int kMidiCfgAt   = kMidiDrumAt + kMidiDrums;         // 141

// ── one sample of the player's state ─────────────────────────────────────────
// Values are magenta_rt's PIANOROLL_WITH_ONSETS / DRUM_PIANOROLL encoding, PRE-offset:
//
//   notes[p]  -1 masked (unconstrained)   0 off   1 on/continuation   2 onset
//             3 on, model free to voice it as onset OR continuation
//   drum      -1 masked                   0 no drum                   1 play drum
//
// This is HELD STATE, not a stream of events. The model is conditioned on what is sounding at the
// instant the request is issued, so a request arriving mid-chord needs the whole chord — not the
// one note-on that happened to be newest.
//
// `active` distinguishes "no MIDI in this request" from "MIDI is live and nothing is held". Both
// produce identical tokens, which is the point: plugging in a controller and touching nothing must
// not change a single sample of the output.
struct MidiState {
    int8_t notes[kMidiNotes];
    int8_t drum;
    bool   active;

    MidiState() : drum(-1), active(false) {
        for (int i = 0; i < kMidiNotes; ++i) notes[i] = -1;
    }
};

// One pitch's app-reported state → the value the MODEL is given.
//
// Mirrors calculate_token() in upstream's live engine at its DEFAULT onset_mode of 0 ("mask
// onsets"): a held pitch is always 3 — the model is told the note is sounding and left free to
// voice it as an onset or as a continuation. The 1-vs-2 distinction the app sends is therefore
// discarded HERE rather than never being sent, so that enabling onset_mode 1 is an engine change
// and not an app release.
int midi_note_token(int8_t state);

// "60:2,64:1,67:1|drum:1" → MidiState. Pitches absent from the list stay masked; the `|drum:N`
// suffix is optional. An EMPTY string is valid and means "MIDI is live, nothing held".
// Returns false and leaves `out` untouched on any malformed element, so a garbled request
// conditions the model on nothing rather than on half a chord.
bool midi_parse(const std::string& spec, MidiState& out);

// 12 MusicCoCa style codes (+ optional MIDI) → the whole 144-token conditioning block.
//
// This is magenta_rt's _build_conditioning (mlx/system.py:314): style ++ notes ++ drums ++ cfg,
// every value shifted by kMidiOffset. It lives here rather than next to the MusicCoCa tower so the
// correctness gate can call the SHIPPING builder instead of a copy of it — a gate that tests its own
// re-implementation is a gate that passes while the product is wrong.
//
// `midi == nullptr` (or an inactive state) leaves notes and drums unconstrained, which is what this
// port sent before live MIDI existed and is therefore the byte-identity baseline.
// Returns false if `style_tokens` is not kMidiStyleTokens long.
bool midi_build_conditioning(const std::vector<int32_t>& style_tokens, const MidiState* midi,
                             std::vector<int32_t>& out);

// Overwrite the 129 MIDI channels of an ALREADY-BUILT 144-token block, in place.
//
// The serve loop caches the whole block per prompt/blend so a held puck costs no MusicCoCa work.
// MIDI changes every chunk by nature, so it is applied AFTER that cache is consulted rather than
// baked into it — otherwise every note-on would look like a prompt change and pay for a text-tower
// pass nothing asked for. A state with active=false is a no-op.
// Returns false if `tokens` is not 144 long.
bool midi_apply_to_conditioning(const MidiState& midi, std::vector<int32_t>& tokens);
