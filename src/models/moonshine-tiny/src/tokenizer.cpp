// Reference: .nnport/tokenizer.json:model.vocab
// Reference: .nnport/tokenizer.json:decoder
//
// Starting-point tokenizer implementation emitted by PortTokenizer. Iterates
// per the contract; agent extends as needed. The decoder_chain below is baked
// from .nnport/tokenizer_contract.json::decoder_chain at scaffold-emit time
// and covers ByteLevel, Metaspace, Replace, Strip, ByteFallback, and Fuse —
// the most common HuggingFace decoder steps. Add cases below if the contract
// emits a step type not handled here.

#include "tokenizer.h"

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <sstream>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

namespace {

// ── Step record baked from contract.decoder_chain ─────────────────────────
struct DecoderStep {
    std::string type;
    std::string from;
    std::string to;
    std::string replacement;
};

static const std::vector<DecoderStep> kDecoderChain = {
    DecoderStep{ /*type=*/"Replace", /*from=*/"▁", /*to=*/" ", /*replacement=*/"" },
    DecoderStep{ /*type=*/"ByteFallback", /*from=*/"", /*to=*/"", /*replacement=*/"" },
    DecoderStep{ /*type=*/"Fuse", /*from=*/"", /*to=*/"", /*replacement=*/"" },
    DecoderStep{ /*type=*/"Strip", /*from=*/"", /*to=*/"", /*replacement=*/"" },
};

static const std::unordered_set<int32_t> kSkipOnDecodeIds = { 0, 1, 2, 32000, 32001, 32002, 32003, 32004, 32005, 32006, 32007, 32008, 32009, 32010, 32011, 32012, 32013, 32014, 32015, 32016, 32017, 32018, 32019, 32020, 32021, 32022, 32023, 32024, 32025, 32026, 32027, 32028, 32029, 32030, 32031, 32032, 32033, 32034, 32035, 32036, 32037, 32038, 32039, 32040, 32041, 32042, 32043, 32044, 32045, 32046, 32047, 32048, 32049, 32050, 32051, 32052, 32053, 32054, 32055, 32056, 32057, 32058, 32059, 32060, 32061, 32062, 32063, 32064, 32065, 32066, 32067, 32068, 32069, 32070, 32071, 32072, 32073, 32074, 32075, 32076, 32077, 32078, 32079, 32080, 32081, 32082, 32083, 32084, 32085, 32086, 32087, 32088, 32089, 32090, 32091, 32092, 32093, 32094, 32095, 32096, 32097, 32098, 32099, 32100, 32101, 32102, 32103, 32104, 32105, 32106, 32107, 32108, 32109, 32110, 32111, 32112, 32113, 32114, 32115, 32116, 32117, 32118, 32119, 32120, 32121, 32122, 32123, 32124, 32125, 32126, 32127, 32128, 32129, 32130, 32131, 32132, 32133, 32134, 32135, 32136, 32137, 32138, 32139, 32140, 32141, 32142, 32143, 32144, 32145, 32146, 32147, 32148, 32149, 32150, 32151, 32152, 32153, 32154, 32155, 32156, 32157, 32158, 32159, 32160, 32161, 32162, 32163, 32164, 32165, 32166, 32167, 32168, 32169, 32170, 32171, 32172, 32173, 32174, 32175, 32176, 32177, 32178, 32179, 32180, 32181, 32182, 32183, 32184, 32185, 32186, 32187, 32188, 32189, 32190, 32191, 32192, 32193, 32194, 32195, 32196, 32197, 32198, 32199, 32200, 32201, 32202, 32203, 32204, 32205, 32206, 32207, 32208, 32209, 32210, 32211, 32212, 32213, 32214, 32215, 32216, 32217, 32218, 32219, 32220, 32221, 32222, 32223, 32224, 32225, 32226, 32227, 32228, 32229, 32230, 32231, 32232, 32233, 32234, 32235, 32236, 32237, 32238, 32239, 32240, 32241, 32242, 32243, 32244, 32245, 32246, 32247, 32248, 32249, 32250, 32251, 32252, 32253, 32254, 32255, 32256, 32257, 32258, 32259, 32260, 32261, 32262, 32263, 32264, 32265, 32266, 32267, 32268, 32269, 32270, 32271, 32272, 32273, 32274, 32275, 32276, 32277, 32278, 32279, 32280, 32281, 32282, 32283, 32284, 32285, 32286, 32287, 32288, 32289, 32290, 32291, 32292, 32293, 32294, 32295, 32296, 32297, 32298, 32299, 32300, 32301, 32302, 32303, 32304, 32305, 32306, 32307, 32308, 32309, 32310, 32311, 32312, 32313, 32314, 32315, 32316, 32317, 32318, 32319, 32320, 32321, 32322, 32323, 32324, 32325, 32326, 32327, 32328, 32329, 32330, 32331, 32332, 32333, 32334, 32335, 32336, 32337, 32338, 32339, 32340, 32341, 32342, 32343, 32344, 32345, 32346, 32347, 32348, 32349, 32350, 32351, 32352, 32353, 32354, 32355, 32356, 32357, 32358, 32359, 32360, 32361, 32362, 32363, 32364, 32365, 32366, 32367, 32368, 32369, 32370, 32371, 32372, 32373, 32374, 32375, 32376, 32377, 32378, 32379, 32380, 32381, 32382, 32383, 32384, 32385, 32386, 32387, 32388, 32389, 32390, 32391, 32392, 32393, 32394, 32395, 32396, 32397, 32398, 32399, 32400, 32401, 32402, 32403, 32404, 32405, 32406, 32407, 32408, 32409, 32410, 32411, 32412, 32413, 32414, 32415, 32416, 32417, 32418, 32419, 32420, 32421, 32422, 32423, 32424, 32425, 32426, 32427, 32428, 32429, 32430, 32431, 32432, 32433, 32434, 32435, 32436, 32437, 32438, 32439, 32440, 32441, 32442, 32443, 32444, 32445, 32446, 32447, 32448, 32449, 32450, 32451, 32452, 32453, 32454, 32455, 32456, 32457, 32458, 32459, 32460, 32461, 32462, 32463, 32464, 32465, 32466, 32467, 32468, 32469, 32470, 32471, 32472, 32473, 32474, 32475, 32476, 32477, 32478, 32479, 32480, 32481, 32482, 32483, 32484, 32485, 32486, 32487, 32488, 32489, 32490, 32491, 32492, 32493, 32494, 32495, 32496, 32497, 32498, 32499, 32500, 32501, 32502, 32503, 32504, 32505, 32506, 32507, 32508, 32509, 32510, 32511, 32512, 32513, 32514, 32515, 32516, 32517, 32518, 32519, 32520, 32521, 32522, 32523, 32524, 32525, 32526, 32527, 32528, 32529, 32530, 32531, 32532, 32533, 32534, 32535, 32536, 32537, 32538, 32539, 32540, 32541, 32542, 32543, 32544, 32545, 32546, 32547, 32548, 32549, 32550, 32551, 32552, 32553, 32554, 32555, 32556, 32557, 32558, 32559, 32560, 32561, 32562, 32563, 32564, 32565, 32566, 32567, 32568, 32569, 32570, 32571, 32572, 32573, 32574, 32575, 32576, 32577, 32578, 32579, 32580, 32581, 32582, 32583, 32584, 32585, 32586, 32587, 32588, 32589, 32590, 32591, 32592, 32593, 32594, 32595, 32596, 32597, 32598, 32599, 32600, 32601, 32602, 32603, 32604, 32605, 32606, 32607, 32608, 32609, 32610, 32611, 32612, 32613, 32614, 32615, 32616, 32617, 32618, 32619, 32620, 32621, 32622, 32623, 32624, 32625, 32626, 32627, 32628, 32629, 32630, 32631, 32632, 32633, 32634, 32635, 32636, 32637, 32638, 32639, 32640, 32641, 32642, 32643, 32644, 32645, 32646, 32647, 32648, 32649, 32650, 32651, 32652, 32653, 32654, 32655, 32656, 32657, 32658, 32659, 32660, 32661, 32662, 32663, 32664, 32665, 32666, 32667, 32668, 32669, 32670, 32671, 32672, 32673, 32674, 32675, 32676, 32677, 32678, 32679, 32680, 32681, 32682, 32683, 32684, 32685, 32686, 32687, 32688, 32689, 32690, 32691, 32692, 32693, 32694, 32695, 32696, 32697, 32698, 32699, 32700, 32701, 32702, 32703, 32704, 32705, 32706, 32707, 32708, 32709, 32710, 32711, 32712, 32713, 32714, 32715, 32716, 32717, 32718, 32719, 32720, 32721, 32722, 32723, 32724, 32725, 32726, 32727, 32728, 32729, 32730, 32731, 32732, 32733, 32734, 32735, 32736, 32737, 32738, 32739, 32740, 32741, 32742, 32743, 32744, 32745, 32746, 32747, 32748, 32749, 32750, 32751, 32752, 32753, 32754, 32755, 32756, 32757, 32758, 32759, 32760, 32761, 32762, 32763, 32764, 32765, 32766, 32767 };

// ── Stream-helper: read a little-endian uint32 ────────────────────────────
static bool read_u32_le(std::ifstream& in, uint32_t& v) {
    unsigned char b[4];
    in.read(reinterpret_cast<char*>(b), 4);
    if (!in) return false;
    v = static_cast<uint32_t>(b[0])
      | (static_cast<uint32_t>(b[1]) << 8)
      | (static_cast<uint32_t>(b[2]) << 16)
      | (static_cast<uint32_t>(b[3]) << 24);
    return true;
}

// ── ByteLevel decoder map (HuggingFace bytes_to_unicode inverted) ─────────
// HF maps each byte 0..255 to a unique Unicode code point so that all bytes
// can round-trip through string handling. To decode, we reverse: each Unicode
// codepoint in the chain output maps back to a byte. This is the standard
// GPT-2 / RoBERTa / Llama-BPE table.
static const std::unordered_map<uint32_t, uint8_t>& bytelevel_unicode_to_byte() {
    static const std::unordered_map<uint32_t, uint8_t> kMap = []() {
        std::unordered_map<uint32_t, uint8_t> m;
        // Bytes that are printable ASCII in HF's table map to themselves.
        std::vector<int> bs;
        for (int b = 33; b <= 126; b++) bs.push_back(b);
        for (int b = 161; b <= 172; b++) bs.push_back(b);
        for (int b = 174; b <= 255; b++) bs.push_back(b);
        std::vector<int> cs = bs;
        int n = 0;
        for (int b = 0; b < 256; b++) {
            if (std::find(bs.begin(), bs.end(), b) == bs.end()) {
                bs.push_back(b);
                cs.push_back(256 + n);
                n++;
            }
        }
        for (size_t i = 0; i < bs.size(); i++) {
            m[static_cast<uint32_t>(cs[i])] = static_cast<uint8_t>(bs[i]);
        }
        return m;
    }();
    return kMap;
}

// Decode a UTF-8 string into a sequence of Unicode codepoints.
static std::vector<uint32_t> utf8_to_codepoints(const std::string& s) {
    std::vector<uint32_t> out;
    out.reserve(s.size());
    size_t i = 0;
    while (i < s.size()) {
        unsigned char c = static_cast<unsigned char>(s[i]);
        uint32_t cp = 0;
        size_t len = 1;
        if (c < 0x80) { cp = c; len = 1; }
        else if ((c & 0xE0) == 0xC0) { cp = c & 0x1F; len = 2; }
        else if ((c & 0xF0) == 0xE0) { cp = c & 0x0F; len = 3; }
        else if ((c & 0xF8) == 0xF0) { cp = c & 0x07; len = 4; }
        else { i++; continue; }
        if (i + len > s.size()) break;
        for (size_t k = 1; k < len; k++) {
            unsigned char cc = static_cast<unsigned char>(s[i + k]);
            cp = (cp << 6) | (cc & 0x3F);
        }
        out.push_back(cp);
        i += len;
    }
    return out;
}

// Apply HuggingFace ByteLevel decode: codepoint sequence → byte sequence string.
static std::string apply_bytelevel(const std::string& concatenated) {
    const auto& m = bytelevel_unicode_to_byte();
    std::vector<uint32_t> cps = utf8_to_codepoints(concatenated);
    std::string out;
    out.reserve(cps.size());
    for (uint32_t cp : cps) {
        auto it = m.find(cp);
        if (it != m.end()) out.push_back(static_cast<char>(it->second));
        // Otherwise drop the codepoint silently — keeps output ASCII-clean
        // when the chain is misconfigured. Agent should diagnose if needed.
    }
    return out;
}

// Apply Metaspace decode: replace each occurrence of the replacement token
// (typically "▁", U+2581) with a space. Emitted by SentencePiece-family
// tokenizers (LLaMA, T5, etc.).
static std::string apply_metaspace(const std::string& s, const std::string& replacement) {
    if (replacement.empty()) return s;
    std::string out;
    out.reserve(s.size());
    size_t i = 0;
    while (i < s.size()) {
        if (s.compare(i, replacement.size(), replacement) == 0) {
            out.push_back(' ');
            i += replacement.size();
        } else {
            out.push_back(s[i]);
            i++;
        }
    }
    return out;
}

// Apply Replace decode: simple string replacement.
static std::string apply_replace(const std::string& s, const std::string& from, const std::string& to) {
    if (from.empty()) return s;
    std::string out;
    out.reserve(s.size());
    size_t i = 0;
    while (i < s.size()) {
        if (s.compare(i, from.size(), from) == 0) {
            out.append(to);
            i += from.size();
        } else {
            out.push_back(s[i]);
            i++;
        }
    }
    return out;
}

// Apply Strip decode: trim whitespace from one or both sides.
static std::string apply_strip(const std::string& s, const std::string& direction) {
    auto is_ws = [](unsigned char c) { return c == ' ' || c == '\t' || c == '\n' || c == '\r'; };
    size_t start = 0, end = s.size();
    if (direction != "right") {
        while (start < end && is_ws(static_cast<unsigned char>(s[start]))) start++;
    }
    if (direction != "left") {
        while (end > start && is_ws(static_cast<unsigned char>(s[end - 1]))) end--;
    }
    return s.substr(start, end - start);
}

// Apply ByteFallback decode: replace each occurrence of "<0xHH>" with the byte HH.
static std::string apply_byte_fallback(const std::string& s) {
    std::string out;
    out.reserve(s.size());
    size_t i = 0;
    while (i < s.size()) {
        if (i + 6 <= s.size() && s[i] == '<' && s[i + 1] == '0' && s[i + 2] == 'x' && s[i + 5] == '>') {
            char h1 = s[i + 3], h2 = s[i + 4];
            auto hex = [](char c) -> int {
                if (c >= '0' && c <= '9') return c - '0';
                if (c >= 'a' && c <= 'f') return 10 + (c - 'a');
                if (c >= 'A' && c <= 'F') return 10 + (c - 'A');
                return -1;
            };
            int v1 = hex(h1), v2 = hex(h2);
            if (v1 >= 0 && v2 >= 0) {
                out.push_back(static_cast<char>((v1 << 4) | v2));
                i += 6;
                continue;
            }
        }
        out.push_back(s[i]);
        i++;
    }
    return out;
}

static std::string apply_decoder_chain(const std::string& concatenated) {
    std::string s = concatenated;
    for (const auto& step : kDecoderChain) {
        if (step.type == "ByteLevel") {
            s = apply_bytelevel(s);
        } else if (step.type == "Metaspace") {
            s = apply_metaspace(s, step.replacement);
        } else if (step.type == "Replace") {
            s = apply_replace(s, step.from, step.to);
        } else if (step.type == "Strip") {
            // step.from used as direction by some HF dumps; fallback to "both".
            s = apply_strip(s, step.from);
        } else if (step.type == "ByteFallback") {
            s = apply_byte_fallback(s);
        } else if (step.type == "Fuse") {
            // No-op for plain string concatenation.
        }
        // Unknown step types fall through. Add a case above if your contract
        // emits a step type not handled here.
    }
    return s;
}

}  // namespace

bool Tokenizer::load(const std::string& vocab_path) {
    std::ifstream in(vocab_path, std::ios::binary);
    if (!in) return false;

    uint32_t vocab_count = 0, merges_count = 0;
    if (!read_u32_le(in, vocab_count) || !read_u32_le(in, merges_count)) return false;

    id_to_token_.assign(vocab_count, std::string());
    token_to_id_.clear();
    token_to_id_.reserve(vocab_count);

    for (uint32_t k = 0; k < vocab_count; k++) {
        uint32_t id = 0, byte_len = 0;
        if (!read_u32_le(in, id)) return false;
        if (!read_u32_le(in, byte_len)) return false;
        std::string tok(byte_len, '\0');
        if (byte_len > 0) {
            in.read(&tok[0], byte_len);
            if (!in) return false;
        }
        if (id >= id_to_token_.size()) {
            id_to_token_.resize(id + 1, std::string());
        }
        id_to_token_[id] = tok;
        token_to_id_[tok] = static_cast<int32_t>(id);
    }

    merges_.clear();
    merges_.reserve(merges_count);
    for (uint32_t k = 0; k < merges_count; k++) {
        uint32_t a_len = 0, b_len = 0;
        if (!read_u32_le(in, a_len)) return false;
        std::string a(a_len, '\0');
        if (a_len > 0) { in.read(&a[0], a_len); if (!in) return false; }
        if (!read_u32_le(in, b_len)) return false;
        std::string b(b_len, '\0');
        if (b_len > 0) { in.read(&b[0], b_len); if (!in) return false; }
        merges_.emplace_back(std::move(a), std::move(b));
    }

    bos_id_ = 1;
    eos_id_ = 2;
    pad_id_ = 0;
    return true;
}

// Greedy longest-prefix tokenization. This is intentionally simple — it works
// well enough for the convergence loop's text-prompt mode in many models, but
// will diverge from HuggingFace's BPE / SentencePiece / WordPiece tokenizers
// on edge cases. Iterate on this method as the model family requires; the
// reference signal is reference/reference_tokens.json::input_ids.
// Detect whether the baked decoder chain treats "▁" (U+2581) as a space
// marker. If so, the matching ENCODE side must pre-process the input by
// prepending "▁" and converting interior spaces to "▁" — otherwise vocab
// lookups for tokens like "▁The", "▁teacher" will miss and the prompt
// round-trips back as "Theteacherworkedatthe" with no spaces. This is the
// standard SentencePiece convention shared by Llama / Mistral / OpenELM /
// Gemma / Phi.
static bool sentencepiece_metaspace_active() {
    for (const auto& step : kDecoderChain) {
        if ((step.type == "Metaspace") || (step.type == "Replace" && step.from == "â")) {
            return true;
        }
    }
    return false;
}

static std::string sentencepiece_preprocess(const std::string& text) {
    // Convention: prepend "▁" to the start of the string, and replace every
    // ASCII space with "▁". Internal whitespace runs collapse to one "▁"
    // each. The token vocab carries entries like "▁The"; this transform
    // makes greedy-longest-match find them.
    static const char kSpaceMarker[] = "â"; // UTF-8 for U+2581
    std::string out;
    out.reserve(text.size() + 4);
    out.append(kSpaceMarker);
    for (size_t i = 0; i < text.size(); i++) {
        if (text[i] == ' ') {
            out.append(kSpaceMarker);
        } else {
            out.push_back(text[i]);
        }
    }
    return out;
}

std::vector<int32_t> Tokenizer::encode(const std::string& text) const {
    std::vector<int32_t> out;
    out.reserve(text.size());
    // Apply SentencePiece-family preprocessing when the decoder chain
    // expects "▁" markers — without this, prompt encode → decode loses
    // every space because the vocab stores "▁word", not "word".
    const std::string staged = sentencepiece_metaspace_active()
        ? sentencepiece_preprocess(text)
        : text;
    size_t i = 0;
    while (i < staged.size()) {
        // Greedy longest match against vocab.
        size_t best_len = 0;
        int32_t best_id = -1;
        for (size_t span = std::min<size_t>(staged.size() - i, 32); span >= 1; span--) {
            auto it = token_to_id_.find(staged.substr(i, span));
            if (it != token_to_id_.end()) {
                best_len = span;
                best_id = it->second;
                break;
            }
        }
        if (best_id < 0) {
            // Unknown byte — emit no token and advance one byte. Agent should
            // extend this with the family-appropriate fallback (ByteFallback,
            // <unk>, byte-pair merges, etc.) for production correctness.
            i++;
            continue;
        }
        out.push_back(best_id);
        i += best_len;
    }
    return out;
}

std::string Tokenizer::decode(const std::vector<int32_t>& token_ids) const {
    std::string concatenated;
    concatenated.reserve(token_ids.size() * 4);
    for (int32_t id : token_ids) {
        if (kSkipOnDecodeIds.find(id) != kSkipOnDecodeIds.end()) continue;
        if (id < 0 || static_cast<size_t>(id) >= id_to_token_.size()) continue;
        concatenated.append(id_to_token_[static_cast<size_t>(id)]);
    }
    return apply_decoder_chain(concatenated);
}
