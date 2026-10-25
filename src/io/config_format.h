#pragma once

// Shared TOML reader. The supported value types are the scalar and integer-array
// types used by the generated ECM schema. GUI edits retain untouched source lines.
#include "io/json.h"
#include "../ecm/application/configuration/generated/ecm_toml_schema.h"
#include <algorithm>
#include <array>
#include <cctype>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <limits>
#include <regex>
#include <set>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace ecm_config {
inline std::string trim(std::string s) {
    size_t first = 0, last = s.size();
    while (first < last && std::isspace(static_cast<unsigned char>(s[first])))
        ++first;
    while (last > first && std::isspace(static_cast<unsigned char>(s[last - 1])))
        --last;
    return s.substr(first, last - first);
}

inline std::string lower(std::string s) {
    for (char& c : s)
        c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
    return s;
}

// Shared input value grammar, used before precedence by CLI and TOML.
inline std::vector<unsigned> affinity_indices(const std::string& raw) {
    const auto text = trim(raw);
    std::vector<unsigned> out;
    if (text.empty() || text == "none" || text == "auto")
        return out;
    const auto bad = [&]() {
        return std::runtime_error("key=affinity value='" + raw +
                                  "'; expected comma-separated CPU indices/ranges in [0,1023], at "
                                  "most 1024 entries, e.g. 0-7,10");
    };
    const auto index = [&](std::string token) {
        token = trim(token);
        if (token.empty() || token.find_first_not_of("0123456789") != token.npos)
            throw bad();
        const auto value = std::stoul(token);
        if (value > 1023)
            throw bad();
        return static_cast<unsigned>(value);
    };
    size_t at = 0;
    while (at <= text.size()) {
        const auto comma = text.find(',', at);
        const auto token = trim(text.substr(at, comma == text.npos ? text.npos : comma - at));
        const auto dash = token.find('-');
        unsigned lo = 0, hi = 0;
        try {
            lo = index(token.substr(0, dash));
            hi = dash == token.npos ? lo : index(token.substr(dash + 1));
        } catch (...) {
            throw bad();
        }
        if (hi < lo || hi - lo + 1 > 1024 - out.size())
            throw bad();
        for (auto value = lo; value <= hi; ++value)
            out.push_back(value);
        if (comma == text.npos)
            break;
        at = comma + 1;
    }
    return out;
}

inline int worker_header(const std::string& line, bool* bracket = nullptr) {
    if (bracket)
        *bracket = false;
    if (line.size() < 3 || line.front() != '[' || line.back() != ']')
        return 0;
    if (bracket)
        *bracket = true;
    const auto body = trim(line.substr(1, line.size() - 2));
    if (body.size() < 6 || lower(body.substr(0, 6)) != "worker")
        return 0;
    auto tail = trim(body.substr(6));
    if (tail.empty() || tail[0] != '#')
        return 0;
    tail = trim(tail.substr(1));
    if (tail.empty())
        return 0;
    int n = 0;
    for (char c : tail) {
        if (c < '0' || c > '9')
            return 0;
        n = n * 10 + c - '0';
        if (n > 1000000)
            return 0;
    }
    return n;
}

struct TomlLine {
    enum class Kind { Blank, Comment, Section, KeyValue, Other };
    Kind kind = Kind::Blank;
    std::string raw, section, key, value;
    int worker_scope = 0;
    size_t line_number = 0;
};

inline int toml_worker(const std::string& section) {
    if (section.compare(0, 8, "workers."))
        return 0;
    const auto end = section.find('.', 8);
    const auto number = section.substr(8, end == section.npos ? section.npos : end - 8);
    if (number.empty() || number[0] == '0')
        return 0;
    int value = 0;
    for (char c : number) {
        if (c < '0' || c > '9')
            return 0;
        value = value * 10 + c - '0';
        if (value > 1000000)
            return 0;
    }
    if (end != section.npos) {
        const auto tail = section.substr(end + 1);
        if (tail != "shared" && tail != "stage1" && tail != "stage2")
            return 0;
    }
    return value;
}

inline bool toml_section(const std::string& section) {
    return section == "shared" || section == "stage1" || section == "stage2" || section == "gui" ||
           toml_worker(section) > 0;
}

inline const TomlOptionMeta* toml_option(const std::string& section, const std::string& key) {
    std::string table = section;
    if (toml_worker(section) > 0) {
        const auto dot = section.find('.', 8);
        table = dot == section.npos ? "workers.1" : section.substr(dot + 1);
    }
    for (const auto& option : toml_options)
        if (table == option.table && key == option.key)
            return &option;
    return nullptr;
}

inline std::string toml_value(const std::string& source, char& kind) {
    bool basic = false, literal = false, escape = false;
    size_t comment = source.npos;
    for (size_t i = 0; i < source.size(); ++i) {
        const char c = source[i];
        if (escape) {
            escape = false;
            continue;
        }
        if (basic && c == '\\') {
            escape = true;
            continue;
        }
        if (c == '"' && !literal) {
            basic = !basic;
            continue;
        }
        if (c == '\'' && !basic) {
            literal = !literal;
            continue;
        }
        if (c == '#' && !basic && !literal) {
            comment = i;
            break;
        }
    }
    if (basic || literal)
        throw std::runtime_error("unterminated TOML string");
    const auto value = trim(source.substr(0, comment));
    if (value.empty())
        throw std::runtime_error("missing TOML value");
    if (value.front() == '"') {
        const auto parsed = ecm_results::json::parse(value);
        if (parsed.type != 's')
            throw std::runtime_error("invalid TOML string");
        kind = 's';
        return parsed.scalar;
    }
    if (value.front() == '\'') {
        if (value.size() < 2 || value.back() != '\'' || value.find('\'', 1) != value.size() - 1)
            throw std::runtime_error("invalid TOML literal string");
        kind = 's';
        return value.substr(1, value.size() - 2);
    }
    if (value.front() == '[') {
        const auto parsed = ecm_results::json::parse(value);
        if (parsed.type != 'a')
            throw std::runtime_error("invalid TOML integer array");
        std::string result;
        for (const auto& item : parsed.items) {
            if (item.type != 'd')
                throw std::runtime_error("TOML array requires integers");
            if (!result.empty())
                result += ',';
            result += item.scalar;
        }
        kind = 'a';
        return result;
    }
    if (value == "true" || value == "false") {
        kind = 'b';
        return value;
    }
    static const std::regex number(
        R"(^[+-]?(?:0|[1-9](?:_?[0-9])*)(?:\.[0-9](?:_?[0-9])*)?(?:[eE][+-]?[0-9](?:_?[0-9])*)?$)");
    if (!std::regex_match(value, number))
        throw std::runtime_error("unsupported TOML value");
    kind = value.find_first_of(".eE") == value.npos ? 'i' : 'f';
    std::string result;
    for (char c : value)
        if (c != '_')
            result += c;
    return result;
}
enum class TomlReadMode { Strict, Computation };

inline bool read_toml(const std::string& path, std::vector<TomlLine>& lines,
                      TomlReadMode mode = TomlReadMode::Strict,
                      std::vector<std::string>* ignored_gui = nullptr) {
    std::ifstream in(path, std::ios::binary);
    if (!in)
        return false;
    lines.clear();
    std::string raw, section;
    bool first = true;
    size_t number = 0;
    std::set<std::string> tables, keys;
    while (std::getline(in, raw)) {
        ++number;
        if (!raw.empty() && raw.back() == '\r')
            raw.pop_back();
        if (first && raw.compare(0, 3, "\xef\xbb\xbf") == 0)
            raw.erase(0, 3);
        first = false;
        TomlLine l;
        l.raw = raw;
        l.section = section;
        l.worker_scope = toml_worker(section);
        l.line_number = number;
        const auto t = trim(raw);
        try {
            if (t.empty())
                l.kind = TomlLine::Kind::Blank;
            else if (t[0] == '#')
                l.kind = TomlLine::Kind::Comment;
            else if (t.front() == '[') {
                const auto close = t.find(']');
                if (close == t.npos ||
                    !trim(t.substr(close + 1)).empty() && trim(t.substr(close + 1)).front() != '#')
                    throw std::runtime_error("invalid TOML table header");
                const auto name = trim(t.substr(1, close - 1));
                if (!toml_section(name) || !tables.insert(name).second)
                    throw std::runtime_error("unknown or duplicate TOML table: " + name);
                section = name;
                l.kind = TomlLine::Kind::Section;
                l.section = name;
                l.worker_scope = toml_worker(name);
            } else {
                const auto eq = t.find('=');
                if (eq == t.npos || section.empty())
                    throw std::runtime_error("TOML key needs a table and value");
                l.key = trim(t.substr(0, eq));
                if (l.key.empty() ||
                    !(std::isalpha(static_cast<unsigned char>(l.key.front())) ||
                      l.key.front() == '_') ||
                    std::any_of(l.key.begin(), l.key.end(),
                                [](unsigned char c) { return !(std::isalnum(c) || c == '_'); }))
                    throw std::runtime_error("invalid TOML key");
                if (!keys.insert(section + "." + l.key).second)
                    throw std::runtime_error("duplicate TOML key: " + l.key);
                const auto* option = toml_option(section, l.key);
                // GUI settings are outside the computation projection, except
                // NumWorkers: Stage1 routing uses it when the destination queue
                // has no worker sections. Keep its schema/type validation.
                const bool gui_table =
                    section == "gui" ||
                    (toml_worker(section) > 0 && section.find('.', 8) == std::string::npos);
                if (mode == TomlReadMode::Computation && gui_table &&
                    !(section == "gui" && l.key == "NumWorkers")) {
                    char kind = 0;
                    (void)toml_value(t.substr(eq + 1), kind);
                    if (!option && ignored_gui)
                        ignored_gui->push_back(path + ":" + std::to_string(number) + ": " +
                                               section + "." + l.key);
                    l.kind = TomlLine::Kind::Other;
                    lines.push_back(std::move(l));
                    continue;
                }
                if (!option)
                    throw std::runtime_error("unknown TOML key or invalid table: " + l.key);
                char kind = 0;
                l.value = toml_value(t.substr(eq + 1), kind);
                const std::string type = option->type;
                const bool allowed =
                    (type == "string" && kind == 's') || (type == "bool" && kind == 'b') ||
                    (type == "rect" && kind == 'a') ||
                    ((type == "int" || type == "u32" || type == "u64" || type == "i64") &&
                     kind == 'i') ||
                    ((type == "float" || type == "double") && (kind == 'i' || kind == 'f')) ||
                    ((type == "int|string" || type == "float|string") &&
                     (kind == 's' || kind == 'i' || kind == 'f'));
                if (!allowed)
                    throw std::runtime_error("wrong TOML value type for " + l.key);
                l.kind = TomlLine::Kind::KeyValue;
            }
        } catch (const std::exception& e) {
            const auto* meta = toml_option(section, l.key);
            throw std::runtime_error(
                "INPUT[config] " + path + ":" + std::to_string(number) +
                " worker=" + std::to_string(l.worker_scope) + " key=" + l.key + " value='" + raw +
                "'; " + e.what() +
                (meta ? std::string("; expected ") + meta->type + " " + meta->domain
                      : "; use a documented key in this table"));
        }
        lines.push_back(std::move(l));
    }
    return !in.bad();
}

using Entries = std::vector<std::pair<std::string, std::string>>;

inline void replace_entry(Entries& values, std::string key, const std::string& value) {
    values.erase(
        std::remove_if(values.begin(), values.end(), [&](const auto& p) { return p.first == key; }),
        values.end());
    values.emplace_back(std::move(key), value);
}

inline Entries cli_entries(const std::vector<TomlLine>& lines, int worker,
                           bool insensitive = false) {
    Entries values;
    (void)insensitive;
    // The worker table follows its matching global table. Other tables cannot
    // leak values into this projection.
    for (int pass = 0; pass < 2; ++pass)
        for (const auto& l : lines) {
            if (l.kind != TomlLine::Kind::KeyValue)
                continue;
            const auto allowed = [&](const std::string& section) {
                return section == "shared" || section == "stage1" || section == "stage2";
            };
            if (pass == 0 && !allowed(l.section))
                continue;
            if (pass == 1 && (worker <= 0 || l.worker_scope != worker ||
                              l.section == "workers." + std::to_string(worker)))
                continue;
            replace_entry(values, l.key, l.value);
        }
    return values;
}

inline Entries section_entries(const std::vector<TomlLine>& lines, const std::string& section) {
    Entries values;
    for (const auto& l : lines)
        if (l.kind == TomlLine::Kind::KeyValue && l.section == section)
            replace_entry(values, l.key, l.value);
    return values;
}

// Exact decimal/scientific uint64 input, shared by Stage2 TOML and CLI. Decimal
// string scaling avoids a floating-point round trip and requires no GMP linkage.
inline uint64_t unsigned_integer(std::string s, const char* label) {
    s = trim(s);
    const auto bad = [&]() {
        return std::runtime_error(std::string("invalid ") + label + ": " + s);
    };
    if (s.empty() || s.size() > 128 || s[0] == '-')
        throw bad();
    if (s[0] == '+')
        s.erase(0, 1);
    int exponent = 0;
    const auto e = s.find_first_of("eE");
    if (e != s.npos) {
        const auto es = s.substr(e + 1);
        size_t used = 0;
        try {
            exponent = std::stoi(es, &used);
        } catch (...) {
            throw bad();
        }
        if (used != es.size() || exponent < -1000 || exponent > 1000)
            throw bad();
        s.resize(e);
    }
    std::string digits;
    bool dot = false;
    int fractional = 0;
    for (char c : s) {
        if (c == '.' && !dot) {
            dot = true;
            continue;
        }
        if (c < '0' || c > '9')
            throw bad();
        digits += c;
        if (dot)
            ++fractional;
    }
    if (digits.empty())
        throw bad();
    const auto nonzero = digits.find_first_not_of('0');
    if (nonzero == digits.npos)
        return 0;
    digits.erase(0, nonzero);
    const int power = exponent - fractional;
    if (power < 0) {
        const auto count = static_cast<size_t>(-power);
        if (count >= digits.size() ||
            digits.find_first_not_of('0', digits.size() - count) != digits.npos)
            throw bad();
        digits.resize(digits.size() - count);
    } else {
        if (digits.size() + static_cast<size_t>(power) > 20)
            throw bad();
        digits.append(static_cast<size_t>(power), '0');
    }
    try {
        size_t used = 0;
        const auto n = std::stoull(digits, &used);
        if (used != digits.size())
            throw bad();
        return n;
    } catch (...) {
        throw bad();
    }
}

inline double positive(const std::string& s, const char* key) {
    size_t used = 0;
    const double n = std::stod(s, &used);
    if (used != s.size() || !std::isfinite(n) || n <= 0)
        throw std::runtime_error(std::string(key) + " must be finite and positive");
    return n;
}

inline bool boolean(const std::string& s, const char* key) {
    const auto v = lower(s);
    if (v == "1" || v == "true" || v == "yes" || v == "on")
        return true;
    if (v == "0" || v == "false" || v == "no" || v == "off")
        return false;
    throw std::runtime_error(std::string(key) + " must be true or false");
}

inline uint64_t bounded_integer(const std::string& s, const char* key, uint64_t low,
                                uint64_t high) {
    const auto n = unsigned_integer(s, key);
    if (n < low || n > high)
        throw std::runtime_error(std::string(key) + " outside [" + std::to_string(low) + "," +
                                 std::to_string(high) + "]");
    return n;
}

inline int signed_integer(const std::string& s, const char* key, int low, int high) {
    size_t used = 0;
    long long n = 0;
    try {
        n = std::stoll(s, &used);
    } catch (...) {
        throw std::runtime_error(std::string("invalid ") + key);
    }
    if (used != s.size() || n < low || n > high)
        throw std::runtime_error(std::string("invalid ") + key);
    return static_cast<int>(n);
}

inline int log_level(const std::string& s) {
    if (s == "-1")
        return -1;
    const auto v = lower(s);
    const char* names[] = {"quiet", "curve", "phases", "batches", "debug"};
    for (int i = 0; i < 5; ++i)
        if (v == names[i] || v == std::to_string(i))
            return i;
    throw std::runtime_error("log level must be quiet|curve|phases|batches|debug or 0..4");
}

inline void assign_value(std::string& out, const std::string& v) {
    out = v;
}

inline void assign_value(int& out, const std::string& v) {
    size_t used = 0;
    long long n = 0;
    try {
        n = std::stoll(v, &used);
    } catch (...) {
        throw std::runtime_error("invalid TOML integer");
    }
    if (used != v.size() || n < INT32_MIN || n > INT32_MAX)
        throw std::runtime_error("invalid TOML integer");
    out = static_cast<int>(n);
}

inline void assign_value(uint32_t& out, const std::string& v) {
    const auto n = unsigned_integer(v, "TOML unsigned integer");
    if (n > UINT32_MAX)
        throw std::runtime_error("TOML unsigned integer exceeds uint32");
    out = static_cast<uint32_t>(n);
}

inline void assign_value(uint64_t& out, const std::string& v) {
    out = unsigned_integer(v, "TOML unsigned integer");
}

inline void assign_value(double& out, const std::string& v) {
    size_t used = 0;
    double n = 0;
    try {
        n = std::stod(v, &used);
    } catch (...) {
        throw std::runtime_error("invalid TOML number");
    }
    if (used != v.size() || !std::isfinite(n))
        throw std::runtime_error("invalid TOML number");
    out = n;
}

inline void assign_value(bool& out, const std::string& v) {
    out = boolean(v, "TOML boolean");
}

inline int gui_integer(const std::string& v, int fallback, int low, int high) {
    if (v.empty())
        return fallback;
    try {
        size_t used = 0;
        const auto n = std::stoll(v, &used);
        if (used != v.size())
            return fallback;
        return static_cast<int>(std::max<int64_t>(low, std::min<int64_t>(high, n)));
    } catch (...) {
        return fallback;
    }
}

inline int64_t gui_long(const std::string& v, int64_t fallback, int64_t low, int64_t high) {
    try {
        return (std::max)(low, (std::min)(high, static_cast<int64_t>(std::stoll(v))));
    } catch (...) {
        return fallback;
    }
}

inline float gui_font(const std::string& v, float low, float high) {
    if (v.empty() || v == "auto")
        return 0;
    try {
        const auto n = std::stof(v);
        return !std::isfinite(n) || n < low ? 0 : (std::min)(n, high);
    } catch (...) {
        return 0;
    }
}

inline void gui_rect(std::array<int, 4>& out, const std::string& v) {
    std::array<int, 4> parsed{};
    size_t pos = 0;
    for (size_t i = 0; i < parsed.size(); ++i) {
        const auto comma = v.find(',', pos);
        const auto token = v.substr(pos, comma == v.npos ? v.npos : comma - pos);
        char* end = nullptr;
        const long n = std::strtol(token.c_str(), &end, 10);
        if (token.empty() || !end || *end)
            return;
        parsed[i] = static_cast<int>(n);
        if (comma == v.npos) {
            if (i + 1 == parsed.size())
                out = parsed;
            return;
        }
        pos = comma + 1;
    }
}

template <class C> struct Binding {
    const char* key;
    void (*assign)(C&, const std::string&);
};

template <class C, size_t N>
inline void apply(C& cfg, const Entries& entries, const Binding<C> (&bindings)[N]) {
    for (const auto& kv : entries)
        for (const auto& b : bindings)
            if (kv.first == b.key) {
                b.assign(cfg, kv.second);
                break;
            }
}

inline std::string worker_file(const std::string& name, int worker) {
    if (worker == 1)
        return name;
    const auto dot = name.find_last_of('.');
    return name.substr(0, dot) + "_" + std::to_string(worker) +
           (dot == name.npos ? std::string() : name.substr(dot));
}
} // namespace ecm_config
