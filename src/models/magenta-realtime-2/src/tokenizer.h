#pragma once
// Minimal tokenizer interface to satisfy scaffolded main.cpp.
// This audio model likely does not use a text tokenizer in the C++ runtime.

#include <cstdint>
#include <string>
#include <vector>

class Tokenizer {
 public:
  bool load(const std::string& /*vocab_path*/) { return false; }
  std::vector<int> encode(const std::string& /*text*/) { return {}; }
  std::string decode(const std::vector<int32_t>& /*ids*/) { return std::string(); }
  int eos_token_id() const { return -1; }
};
