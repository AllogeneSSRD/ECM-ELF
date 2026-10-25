#pragma once
#include <array>
#include <cstdint>
#include <cstring>
#include <stdexcept>
#include <string>
#include <string_view>

// Portable little-endian primitives. Lengths are checked before allocating.
namespace ecm_stage1_binary {
inline void u32(std::string& out, uint32_t x) {
    for (unsigned i = 0; i < 4; ++i)
        out.push_back(char(x >> (8 * i)));
}

inline void u64(std::string& out, uint64_t x) {
    for (unsigned i = 0; i < 8; ++i)
        out.push_back(char(x >> (8 * i)));
}

inline uint32_t crc32c(std::string_view bytes) {
    static const auto table = [] {
        std::array<uint32_t, 256> t{};
        for (unsigned i = 0; i < 256; ++i) {
            uint32_t c = i;
            for (unsigned b = 0; b < 8; ++b)
                c = (c >> 1) ^ ((c & 1) ? 0x82f63b78u : 0);
            t[i] = c;
        }
        return t;
    }();
    uint32_t c = ~0u;
    for (unsigned char x : bytes)
        c = table[(c ^ x) & 255] ^ (c >> 8);
    return ~c;
}

struct Reader {
    std::string_view data;
    size_t at = 0;

    explicit Reader(std::string_view x) : data(x) {
    }

    size_t left() const {
        return data.size() - at;
    }

    std::string_view take(uint64_t size) {
        if (size > left())
            throw std::runtime_error("truncated Stage1 checkpoint");
        auto value = data.substr(at, size_t(size));
        at += size_t(size);
        return value;
    }

    uint32_t u32() {
        auto s = take(4);
        uint32_t x = 0;
        for (unsigned i = 0; i < 4; ++i)
            x |= uint32_t(uint8_t(s[i])) << (8 * i);
        return x;
    }

    uint64_t u64() {
        auto s = take(8);
        uint64_t x = 0;
        for (unsigned i = 0; i < 8; ++i)
            x |= uint64_t(uint8_t(s[i])) << (8 * i);
        return x;
    }
};

inline constexpr char batch_magic[8] = {'E', 'C', 'M', 'S', '1', 'B', '2', '\0'};

inline bool is_batch(std::string_view bytes) {
    return bytes.size() >= 8 && !std::memcmp(bytes.data(), batch_magic, 8);
}
} // namespace ecm_stage1_binary
