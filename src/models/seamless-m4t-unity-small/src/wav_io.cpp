#include "wav_io.h"
#include "debug_utils.h"
#include <cstdio>
#include <cstring>

static uint32_t rd_u32(const uint8_t* p) { return p[0] | (p[1] << 8) | (p[2] << 16) | ((uint32_t)p[3] << 24); }
static uint16_t rd_u16(const uint8_t* p) { return (uint16_t)(p[0] | (p[1] << 8)); }

bool read_wav(const std::string& path, WavData& out) {
    FILE* f = fopen(path.c_str(), "rb");
    if (!f) { NNOPT_ERROR_FMT("read_wav: cannot open %s", path.c_str()); return false; }
    fseek(f, 0, SEEK_END); long sz = ftell(f); fseek(f, 0, SEEK_SET);
    std::vector<uint8_t> buf(sz);
    if (fread(buf.data(), 1, sz, f) != (size_t)sz) { fclose(f); return false; }
    fclose(f);
    if (sz < 44 || memcmp(buf.data(), "RIFF", 4) != 0 || memcmp(buf.data() + 8, "WAVE", 4) != 0) {
        NNOPT_ERROR_FMT("read_wav: %s is not a RIFF/WAVE file", path.c_str()); return false;
    }
    int fmt = 0, channels = 1, rate = 16000, bits = 16;
    size_t pos = 12;
    const uint8_t* data_ptr = nullptr; uint32_t data_len = 0;
    while (pos + 8 <= (size_t)sz) {
        const uint8_t* c = buf.data() + pos;
        uint32_t clen = rd_u32(c + 4);
        if (memcmp(c, "fmt ", 4) == 0) {
            fmt = rd_u16(c + 8); channels = rd_u16(c + 10); rate = rd_u32(c + 12); bits = rd_u16(c + 22);
        } else if (memcmp(c, "data", 4) == 0) {
            data_ptr = c + 8; data_len = clen; break;
        }
        pos += 8 + clen + (clen & 1);
    }
    if (!data_ptr) { NNOPT_ERROR("read_wav: no data chunk"); return false; }
    if (fmt != 1 || bits != 16) {
        NNOPT_ERROR_FMT("read_wav: unsupported format (fmt=%d bits=%d); need PCM 16-bit", fmt, bits);
        return false;
    }
    if ((size_t)(data_ptr - buf.data()) + data_len > (size_t)sz) data_len = (uint32_t)(sz - (data_ptr - buf.data()));
    int nsamp = data_len / 2;
    out.samples.resize(nsamp);
    memcpy(out.samples.data(), data_ptr, (size_t)nsamp * 2);
    out.channels = channels;
    out.sample_rate = rate;
    fprintf(stderr, "read_wav: %s — %d ch, %d Hz, %d frames\n", path.c_str(), channels, rate, nsamp / channels);
    return true;
}

bool write_wav(const std::string& path, const std::vector<int16_t>& mono, int sample_rate) {
    FILE* f = fopen(path.c_str(), "wb");
    if (!f) { NNOPT_ERROR_FMT("write_wav: cannot open %s", path.c_str()); return false; }
    uint32_t data_bytes = (uint32_t)(mono.size() * 2);
    uint32_t byte_rate = (uint32_t)sample_rate * 2;  // mono, 16-bit
    uint8_t hdr[44];
    memcpy(hdr, "RIFF", 4);
    uint32_t riff = 36 + data_bytes;
    hdr[4] = riff; hdr[5] = riff >> 8; hdr[6] = riff >> 16; hdr[7] = riff >> 24;
    memcpy(hdr + 8, "WAVE", 4);
    memcpy(hdr + 12, "fmt ", 4);
    hdr[16] = 16; hdr[17] = hdr[18] = hdr[19] = 0;   // fmt chunk size
    hdr[20] = 1; hdr[21] = 0;                        // PCM
    hdr[22] = 1; hdr[23] = 0;                        // mono
    hdr[24] = sample_rate; hdr[25] = sample_rate >> 8; hdr[26] = sample_rate >> 16; hdr[27] = sample_rate >> 24;
    hdr[28] = byte_rate; hdr[29] = byte_rate >> 8; hdr[30] = byte_rate >> 16; hdr[31] = byte_rate >> 24;
    hdr[32] = 2; hdr[33] = 0;                        // block align
    hdr[34] = 16; hdr[35] = 0;                       // bits per sample
    memcpy(hdr + 36, "data", 4);
    hdr[40] = data_bytes; hdr[41] = data_bytes >> 8; hdr[42] = data_bytes >> 16; hdr[43] = data_bytes >> 24;
    fwrite(hdr, 1, 44, f);
    fwrite(mono.data(), 2, mono.size(), f);
    fclose(f);
    fprintf(stderr, "write_wav: %s — %zu samples @ %d Hz\n", path.c_str(), mono.size(), sample_rate);
    return true;
}
