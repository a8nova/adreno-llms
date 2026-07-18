// Minimal JSON parser for the .nnb header (objects/arrays/strings/numbers/
// bool/null; UTF-8 passthrough; \uXXXX decoded). Not a general-purpose lib —
// just enough for the header this port's converter emits.
#pragma once
#include <cstdint>
#include <map>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

namespace jmini {

struct Value;
using ValuePtr = std::shared_ptr<Value>;

struct Value {
    enum Kind { OBJ, ARR, STR, NUM, BOOL, NUL } kind = NUL;
    std::map<std::string, ValuePtr> obj;
    std::vector<ValuePtr> arr;
    std::string str;
    double num = 0;
    bool b = false;

    const Value& at(const std::string& k) const {
        auto it = obj.find(k);
        if (it == obj.end()) throw std::runtime_error("json: missing key " + k);
        return *it->second;
    }
    bool has(const std::string& k) const { return obj.count(k) != 0; }
    double d() const { return num; }
    int64_t i() const { return (int64_t)num; }
    const std::string& s() const { return str; }
};

struct Parser {
    const char* p;
    const char* end;
    explicit Parser(const std::string& src) : p(src.data()), end(src.data() + src.size()) {}

    void ws() { while (p < end && (*p == ' ' || *p == '\t' || *p == '\n' || *p == '\r')) ++p; }
    char peek() { ws(); if (p >= end) throw std::runtime_error("json: eof"); return *p; }
    void expect(char c) { if (peek() != c) throw std::runtime_error(std::string("json: expected ") + c); ++p; }

    ValuePtr parse() {
        char c = peek();
        auto v = std::make_shared<Value>();
        if (c == '{') {
            ++p; v->kind = Value::OBJ;
            if (peek() == '}') { ++p; return v; }
            while (true) {
                std::string key = parse_string();
                expect(':');
                v->obj[key] = parse();
                char n = peek();
                if (n == ',') { ++p; continue; }
                expect('}'); break;
            }
        } else if (c == '[') {
            ++p; v->kind = Value::ARR;
            if (peek() == ']') { ++p; return v; }
            while (true) {
                v->arr.push_back(parse());
                char n = peek();
                if (n == ',') { ++p; continue; }
                expect(']'); break;
            }
        } else if (c == '"') {
            v->kind = Value::STR; v->str = parse_string();
        } else if (c == 't') { v->kind = Value::BOOL; v->b = true; p += 4; }
        else if (c == 'f') { v->kind = Value::BOOL; v->b = false; p += 5; }
        else if (c == 'n') { v->kind = Value::NUL; p += 4; }
        else {
            v->kind = Value::NUM;
            char* q = nullptr;
            v->num = strtod(p, &q);
            if (q == p) throw std::runtime_error("json: bad number");
            p = q;
        }
        return v;
    }

    std::string parse_string() {
        expect('"');
        std::string out;
        while (p < end && *p != '"') {
            if (*p == '\\') {
                ++p;
                char c = *p++;
                switch (c) {
                    case 'n': out += '\n'; break;
                    case 't': out += '\t'; break;
                    case 'r': out += '\r'; break;
                    case 'b': out += '\b'; break;
                    case 'f': out += '\f'; break;
                    case '/': out += '/'; break;
                    case '\\': out += '\\'; break;
                    case '"': out += '"'; break;
                    case 'u': {
                        unsigned cp = (unsigned)strtoul(std::string(p, p + 4).c_str(), nullptr, 16);
                        p += 4;
                        // surrogate pair
                        if (cp >= 0xD800 && cp <= 0xDBFF && p + 6 <= end && p[0] == '\\' && p[1] == 'u') {
                            unsigned lo = (unsigned)strtoul(std::string(p + 2, p + 6).c_str(), nullptr, 16);
                            if (lo >= 0xDC00 && lo <= 0xDFFF) {
                                cp = 0x10000 + ((cp - 0xD800) << 10) + (lo - 0xDC00);
                                p += 6;
                            }
                        }
                        // encode UTF-8
                        if (cp < 0x80) out += (char)cp;
                        else if (cp < 0x800) {
                            out += (char)(0xC0 | (cp >> 6));
                            out += (char)(0x80 | (cp & 0x3F));
                        } else if (cp < 0x10000) {
                            out += (char)(0xE0 | (cp >> 12));
                            out += (char)(0x80 | ((cp >> 6) & 0x3F));
                            out += (char)(0x80 | (cp & 0x3F));
                        } else {
                            out += (char)(0xF0 | (cp >> 18));
                            out += (char)(0x80 | ((cp >> 12) & 0x3F));
                            out += (char)(0x80 | ((cp >> 6) & 0x3F));
                            out += (char)(0x80 | (cp & 0x3F));
                        }
                        break;
                    }
                    default: out += c;
                }
            } else {
                out += *p++;
            }
        }
        expect('"');
        return out;
    }
};

inline ValuePtr parse(const std::string& s) { Parser pr(s); return pr.parse(); }

}  // namespace jmini
