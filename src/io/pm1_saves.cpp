#include "pm1_saves.h"
#include "files.h"
#include "stage1_binary.h"
#include "ecm/stage1/ntt_length.h"

#include <algorithm>
#include <array>
#include <charconv>
#include <cmath>
#include <cstdio>
#include <fstream>
#include <iomanip>
#include <locale>
#include <sstream>

namespace ecm_pm1 {
namespace fs = std::filesystem;
namespace binary = ecm_stage1_binary;
namespace {
void check_p(unsigned p) {
    if (p < ecm_stage1::stage1_ntt_min_exponent || p > ecm_stage1::stage1_ntt_max_exponent)
        throw std::invalid_argument("unsupported P-1 save exponent");
}

bool canonical(unsigned p, const std::vector<uint32_t>& words) {
    if (words.size() != (p + 31) / 32) return false;
    const uint32_t top = UINT32_MAX >> ((32 - p % 32) % 32);
    if (words.back() > top || std::all_of(words.begin(), words.end(), [](uint32_t w) { return w == 0; }))
        return false;
    // Exclude M_p itself as well as numbers with nonzero high padding.
    return words.back() != top ||
           std::any_of(words.begin(), words.end() - 1, [](uint32_t w) { return w != UINT32_MAX; });
}

// OWL uses the zlib/IEEE CRC-32 of the little-endian word bytes.
uint32_t owl_crc(std::string_view bytes) {
    static const auto table = [] {
        std::array<uint32_t, 256> entries{};
        for (unsigned i = 0; i < entries.size(); ++i) {
            uint32_t v = i;
            for (unsigned j = 0; j < 8; ++j) v = (v >> 1) ^ ((v & 1) ? 0xedb88320u : 0);
            entries[i] = v;
        }
        return entries;
    }();
    uint32_t crc = UINT32_MAX;
    for (uint8_t byte : bytes) crc = table[(crc ^ byte) & 255] ^ (crc >> 8);
    return ~crc;
}

std::string padded(uint64_t n, size_t width) {
    const auto s = std::to_string(n);
    return std::string(width > s.size() ? width - s.size() : 0, '0') + s;
}

template<typename T>
bool integer(std::string_view field, T& number) {
    if (field.empty()) return false;
    const char* end = field.data() + field.size();
    const auto r = std::from_chars(field.data(), end, number);
    return r.ec == std::errc{} && r.ptr == end;
}

bool consume(std::string_view& text, std::string_view prefix) {
    if (text.substr(0, prefix.size()) != prefix) return false;
    text.remove_prefix(prefix.size());
    return true;
}

template<typename T>
bool number_field(std::string_view& text, std::string_view label, T& number) {
    if (!consume(text, label)) return false;
    const size_t space = text.find(' ');
    if (space == text.npos || !integer(text.substr(0, space), number)) return false;
    text.remove_prefix(space + 1);
    return true;
}

std::optional<OwlSave> read_owl(const fs::path& path, unsigned p, uint64_t b1,
                                uint64_t bits, uint64_t file_k) {
    const size_t nwords = (p + 31) / 32;
    const size_t body_size = 4 + 4 * nwords;
    // Bound the read before allocating, even for a malformed header with no newline.
    std::ifstream input(path, std::ios::binary | std::ios::ate);
    if (!input) return {};
    const auto end = input.tellg();
    if (end <= std::streamoff(body_size) || end > std::streamoff(body_size + 1024)) return {};
    std::string bytes(static_cast<size_t>(end), '\0');
    input.seekg(0);
    if (!input.read(bytes.data(), std::streamsize(bytes.size()))) return {};

    const size_t header_size = bytes.size() - body_size;
    if (bytes[header_size - 1] != '\n') return {};
    std::string_view header(bytes.data(), header_size - 1);
    if (!header.empty() && header.back() == '\r') header.remove_suffix(1);
    const bool v2 = consume(header, "OWL P1 2 ");
    if (!v2 && !consume(header, "OWL P1 1 ")) return {};
    const std::string modulus = "N=1*2^" + std::to_string(p) + "-1 ";
    uint64_t saved_b1, saved_bits;
    OwlSave state;
    if (!consume(header, modulus) || !number_field(header, "B1=", saved_b1) || saved_b1 != b1 ||
        (v2 && !number_field(header, "B2=", state.b2)) || !number_field(header, "k=", state.k) ||
        state.k != file_k || state.k < 1 || !number_field(header, "bits=", saved_bits) ||
        state.k>saved_bits || (bits && saved_bits != bits) || !number_field(header, "err=", state.errors) || !consume(header, "time="))
        return {};
    state.bits=saved_bits;
    std::istringstream seconds{std::string(header)};
    seconds.imbue(std::locale::classic());
    if (!(seconds >> state.seconds) || !seconds.eof() || !std::isfinite(state.seconds) || state.seconds < 0)
        return {};

    binary::Reader body(std::string_view(bytes).substr(header_size));
    const uint32_t saved_crc = body.u32();
    if (saved_crc != owl_crc(body.data.substr(body.at))) return {};
    state.residue.reserve(nwords);
    for (size_t i = 0; i < nwords; ++i) state.residue.push_back(body.u32());
    if (!canonical(p, state.residue)) return {};
    return state;
}
std::vector<std::pair<uint64_t, fs::path>> owl_files(const fs::path& dir, unsigned p, uint64_t max_k) {
    const auto prefix = std::to_string(p) + "-";
    std::vector<std::pair<uint64_t, fs::path>> candidates;
    for (const auto& entry : fs::directory_iterator(dir)) {
        const auto path = entry.path();
        if (path.extension() != ".p1" || !fs::is_regular_file(entry.symlink_status())) continue;
        const auto stem = path.stem().u8string();
        std::string_view id(stem);
        uint64_t k;
        if (consume(id, prefix) && integer(id, k) && k > 0 && k <= max_k)
            candidates.emplace_back(k, path);
    }
    std::sort(candidates.rbegin(), candidates.rend());
    return candidates;
}
} // namespace

std::optional<OwlSave> load_owl(const fs::path& root, unsigned p, uint64_t b1, uint64_t bits) {
    check_p(p);
    const auto dir = root / ("p1-" + std::to_string(p));
    if (!fs::exists(dir)) return {};
    for (const auto& candidate : owl_files(dir, p, bits ? bits : UINT64_MAX)) {
        if (auto s = read_owl(candidate.second, p, b1, bits, candidate.first)) return s;
        std::fprintf(stderr, "P-1: skipped incompatible or damaged checkpoint %s\n",
                     candidate.second.u8string().c_str());
    }
    return {};
}

void write_owl(const fs::path& root, unsigned p, uint64_t b1, uint64_t b2,
               uint64_t bits, const OwlSave& save, unsigned backups) {
    check_p(p);
    if (b1 < 2 || save.k < 1 || save.k > bits || !std::isfinite(save.seconds) ||
        save.seconds < 0 || !canonical(p, save.residue))
        throw std::invalid_argument("invalid OWL P-1 checkpoint state");
    std::ostringstream header;
    header.imbue(std::locale::classic());
    header << "OWL P1 2 N=1*2^" << p << "-1 B1=" << b1 << " B2=" << b2 << " k=" << save.k
           << " bits=" << bits << " err=" << save.errors << " time=" << std::fixed
           << std::setprecision(6) << save.seconds << '\n';
    std::string words;
    words.reserve(save.residue.size() * 4);
    for (uint32_t w : save.residue) binary::u32(words, w);
    auto bytes = header.str();
    binary::u32(bytes, owl_crc(words));
    bytes += words;
    const auto dir = root / ("p1-" + std::to_string(p));
    fs::create_directories(dir);
    const auto path = dir / (std::to_string(p) + "-" + padded(save.k, 9) + ".p1");
    ecm_io::FileLock lock((dir / "checkpoints").u8string());
    ecm_io::write_atomic(path, bytes);
    // Keep only matching, verified older states in the rotation. Imported states
    // for a different assignment and damaged files are left available for inspection.
    const auto older = owl_files(dir, p, save.k - 1);
    if (older.size() <= backups) return;
    unsigned retained = 0;
    for (const auto& [k, prior] : older) {
        if (read_owl(prior, p, b1, bits, k) && retained++ >= backups) fs::remove(prior);
    }
}

std::string prime95_name(unsigned p) {
    check_p(p);
    // Every supported exponent is below 10 million.
    return "m" + padded(p, 7);
}

std::optional<Prime95Save> load_prime95(const fs::path& path, unsigned p) {
    check_p(p);
    if (!fs::exists(path))
        return {};
    const size_t limbs = (p + 31) / 32;
    if (fs::file_size(path) > 128 + 12 * limbs + (1u << 20))
        throw std::runtime_error("oversized Prime95 save: " + path.u8string());
    const auto bytes = ecm_io::read_file(path);
    binary::Reader r(bytes);
    const auto invalid = [&] {
        return std::runtime_error("incompatible or damaged Prime95 P-1 endpoint: " +
                                  path.u8string());
    };
    if (r.u32() != 0x317a394b)
        throw invalid();
    const auto version = r.u32();
    if ((version != 7 && version != 8) || r.u64() != 0x3ff0000000000000ULL || r.u32() != 2 ||
        r.u32() != p || r.u32() != UINT32_MAX)
        throw invalid();
    r.take(20); // Stage label and display percentage are not arithmetic state.
    const auto checksum = r.u32();
    const auto start = r.at;
    const auto state = r.u32();
    if (state != 2 && state != 5)
        throw invalid(); // MIDSTAGE or DONE
    Prime95Save result{r.u64(), ecm_io::fingerprint(bytes), {}, state == 5};
    r.u64(); // Stage2 coverage does not change the stage1 base.
    if (result.b1 < 2 || result.b1 > 5000000000ull || r.u32() != 1)
        throw invalid();
    uint32_t sum = 0;
    const auto giant = [&] {
        const auto used = r.u32();
        if (!used || used > limbs)
            throw invalid();
        sum += used; // Prime95 includes each giant's length twice.
        std::vector<uint32_t> words(limbs);
        for (unsigned i = 0; i < used; ++i)
            words[i] = r.u32();
        if (!words[used - 1] || !canonical(p, words))
            throw invalid();
        return words;
    };
    result.residue = giant();
    if (state == 2) {
        giant(); // invx
        const auto have_gg = r.u32();
        if (have_gg > 1)
            throw invalid();
        if (have_gg)
            giant();
    }
    binary::Reader body(std::string_view(bytes).substr(start, r.at - start));
    while (body.left())
        sum += body.u32();
    if (sum != checksum)
        throw invalid();
    if (r.left()) {
        if (r.take(16) != "MOREINFOJSONDATA")
            throw invalid();
        const auto size = r.u32();
        if (size != r.left() || size < 8 || r.u32() != 1 || r.u32() != 0)
            throw invalid();
    }
    return result;
}

std::optional<ExtensionSave> load_extension(const fs::path& root, unsigned p, uint64_t b1) {
    check_p(p);
    const auto path = root / "extension.pm1";
    if (!fs::exists(path))
        return {};
    const size_t limbs = (p + 31) / 32;
    const auto invalid = [&] {
        return std::runtime_error("invalid P-1 extension checkpoint: " + path.u8string());
    };
    if (fs::file_size(path) != 52 + 8 * limbs)
        throw invalid();
    const auto bytes = ecm_io::read_file(path);
    binary::Reader r(bytes);
    if (r.take(8) != "ECMP1EX1" || r.u32() != p || r.u64() != b1)
        throw invalid();
    ExtensionSave s;
    s.from = r.u64();
    s.bits = r.u64();
    s.k = r.u64();
    s.errors = r.u32();
    if (s.from < 2 || s.from >= b1 || !s.k || s.k > s.bits)
        throw invalid();
    for (auto* value : {&s.base, &s.residue}) {
        value->resize(limbs);
        for (auto& word : *value)
            word = r.u32();
        if (!canonical(p, *value))
            throw invalid();
    }
    if (r.u32() != binary::crc32c(std::string_view(bytes).substr(0, bytes.size() - 4)))
        throw invalid();
    return s;
}

void write_extension(const fs::path& root, unsigned p, uint64_t b1, const ExtensionSave& s) {
    std::string bytes = "ECMP1EX1";
    binary::u32(bytes, p);
    binary::u64(bytes, b1);
    binary::u64(bytes, s.from);
    binary::u64(bytes, s.bits);
    binary::u64(bytes, s.k);
    binary::u32(bytes, s.errors);
    for (const auto* words : {&s.base, &s.residue})
        for (auto word : *words)
            binary::u32(bytes, word);
    binary::u32(bytes, binary::crc32c(bytes));
    ecm_io::write_atomic(root / "extension.pm1", bytes);
}

fs::path write_prime95(const fs::path& dir, unsigned p, uint64_t b1,
                       const std::vector<uint32_t>& residue, uint64_t* fingerprint,
                       const Prime95Save* previous) {
    const auto path = dir / prime95_name(p);
    if (b1 < 2 || !canonical(p, residue))
        throw std::invalid_argument("invalid Prime95 P-1 completion state");
    size_t used = residue.size();
    while (residue[used - 1] == 0)
        --used;
    std::string body;
    binary::u32(body, 5);  // PM1_STATE_DONE
    binary::u64(body, b1); // B_done
    binary::u64(body, b1); // C_done
    binary::u32(body, 1);  // have_x
    binary::u32(body, uint32_t(used));
    for (size_t i = 0; i < used; ++i)
        binary::u32(body, residue[i]);
    // Prime95 counts the giant word count twice. Header bytes are excluded.
    uint32_t checksum = uint32_t(used);
    binary::Reader reader(body);
    while (reader.left())
        checksum += reader.u32();
    std::string bytes;
    binary::u32(bytes, 0x317a394b);
    binary::u32(bytes, 8);
    constexpr uint64_t ieee754_one = 0x3ff0000000000000ULL;
    binary::u64(bytes, ieee754_one); // K=1 stored as a double
    binary::u32(bytes, 2);
    binary::u32(bytes, p);
    binary::u32(bytes, UINT32_MAX); // C=-1
    bytes.append("S5", 2);
    bytes.append(10, '\0');
    binary::u64(bytes, ieee754_one); // percent complete
    binary::u32(bytes, checksum);
    bytes += body;
    if (fingerprint)
        *fingerprint = ecm_io::fingerprint(bytes);

    if (!dir.empty())
        fs::create_directories(dir);
    ecm_io::FileLock lock(path.u8string());
    if (fs::exists(path)) {
        if (fs::file_size(path) == bytes.size() && ecm_io::read_file(path) == bytes)
            return path;
        if (!previous || previous->b1 > b1 ||
            (previous->b1 == b1 && previous->residue != residue) ||
            fs::file_size(path) > 12ull * residue.size() + (1u << 20) + 128 ||
            ecm_io::fingerprint(ecm_io::read_file(path)) != previous->fingerprint)
            throw std::runtime_error("different Prime95 save already occupies " + path.u8string());
        if (previous->done && previous->b1 == b1) {
            if (fingerprint)
                *fingerprint = previous->fingerprint;
            return path;
        }
    } else if (previous) {
        throw std::runtime_error("Prime95 save disappeared during extension: " + path.u8string());
    }
    ecm_io::write_atomic(path, bytes);
    return path;
}
} // namespace ecm_pm1
