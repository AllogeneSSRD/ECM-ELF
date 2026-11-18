#pragma once

// Shared TOML reader. The supported value types are the scalar and array
// types used by the generated ECM schema. GUI edits retain untouched source lines.
#include "../ecm/application/configuration/generated/ecm_toml_schema.h"
#include "io/json.h"
#include <algorithm>
#include <array>
#include <cctype>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <initializer_list>
#include <limits>
#include <map>
#include <regex>
#include <set>
#include <sstream>
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
    enum class Kind { Blank, Comment, Section, KeyValue, Other, Version };
    Kind kind = Kind::Blank;
    std::string raw, section, key, value;
    // Source syntax is separate from the stable runtime binding identity.
    std::string source_section, source_key, source_value;
    int worker_scope = 0;
    int config_version = 1;
    size_t line_number = 0;
    bool deferred = false; // Owned by another executable; retain the source for editing/routing.
    bool recognized = true;
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

// Keep syntax/type and semantic errors in the same readable format. A source
// value is the original RHS (including quotes), not the decoded runtime value.
inline std::string config_diagnostic(const std::string& path, size_t line, int worker,
                                     const std::string& key, const std::string& value,
                                     const std::string& reason,
                                     const std::string& expected = {}) {
    auto message = "INPUT[config]\nfile=" + path;
    if (line)
        message += ":" + std::to_string(line);
    message += "\nworker=" + std::to_string(worker) + "\nkey=" + key + "\nvalue=" + value +
               "\n" + reason;
    if (!expected.empty())
        message += "\nexpected " + expected;
    return message;
}

inline size_t toml_unquoted_offset(const std::string& source, char delimiter, size_t start = 0) {
    bool basic = false, literal = false, escape = false;
    for (size_t i = start; i < source.size(); ++i) {
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
        if (c == delimiter && !basic && !literal)
            return i;
    }
    if (basic || literal)
        throw std::runtime_error("unterminated TOML string");
    return source.npos;
}

inline size_t toml_comment_offset(const std::string& source) {
    return toml_unquoted_offset(source, '#');
}

inline size_t toml_assignment_offset(const std::string& source) {
    return toml_unquoted_offset(source, '=');
}

inline std::string toml_value(const std::string& source, char& kind) {
    const auto value = trim(source.substr(0, toml_comment_offset(source)));
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
inline std::vector<std::string> toml_string_array(const std::string& source) {
    std::istringstream lines(source);
    std::string line, value;
    while (std::getline(lines, line)) {
        value += line.substr(0, toml_comment_offset(line));
        value += ' ';
    }
    value = trim(std::move(value));
    if (value.empty() || value.front() != '[' || toml_unquoted_offset(value, ']', 1) != value.size() - 1)
        throw std::runtime_error("expected a TOML array of strings");
    const auto end = value.size() - 1;
    std::vector<std::string> items;
    for (size_t pos = 1; pos < end;) {
        const auto comma = toml_unquoted_offset(value, ',', pos);
        const auto stop = std::min(comma, end);
        const auto item = trim(value.substr(pos, stop - pos));
        if (item.empty() && stop == end) break;
        char kind = 0;
        auto decoded = toml_value(item, kind);
        if (kind != 's') throw std::runtime_error("TOML array requires string items");
        items.push_back(std::move(decoded));
        pos = stop + 1;
    }
    return items;
}

inline std::string encode_toml_string_array(const std::vector<std::string>& items) {
    std::string value = "[";
    for (const auto& item : items) {
        if (value.size() > 1) value += ", ";
        value += ecm_results::json::quote(item);
    }
    return value + "]";
}

// Only the consuming executable enforces a key's schema type.
inline std::string checked_toml_value(const std::string& source, const TomlOptionMeta& option) {
    if (std::string_view(option.type) == "string_array")
        return encode_toml_string_array(toml_string_array(source));
    char kind = 0;
    auto value = toml_value(source, kind);
    const std::string type = option.type;
    // Numeric aliases apply only to schema-declared booleans.
    // Keep the source for editing/diagnostics; consumers receive the
    // same canonical value as a literal true/false setting.
    if ((type == "bool" || type == "bool|string") && kind == 'i') {
        if (value == "1" || value == "+1")
            value = "true";
        else if (value == "0" || value == "+0" || value == "-0")
            value = "false";
        else
            throw std::runtime_error(std::string("integer boolean must be 0 or 1 for ") + option.key);
        kind = 'b';
    }
    const bool allowed =
        (type == "string" && kind == 's') || (type == "bool" && kind == 'b') ||
        (type == "bool|string" && (kind == 'b' || kind == 's')) ||
        (type == "rect" && kind == 'a') ||
        ((type == "int" || type == "u32" || type == "u64" || type == "i64") &&
         kind == 'i') ||
        ((type == "float" || type == "double") && (kind == 'i' || kind == 'f')) ||
        ((type == "int|string" || type == "float|string") &&
         (kind == 's' || kind == 'i' || kind == 'f'));
    if (!allowed)
        throw std::runtime_error(std::string("wrong TOML value type for ") + option.key);
    return value;
}

inline std::string toml_source_rhs(const TomlLine& line) {
    if (!line.source_value.empty())
        return line.source_value;
    const auto offset = toml_assignment_offset(line.raw);
    return offset == line.raw.npos ? std::string{} : trim(line.raw.substr(offset + 1));
}

inline std::vector<std::string> toml_enum_names(const std::string& conversion) {
    std::vector<std::string> names;
    size_t start = 5; // enum:
    for (;;) {
        const auto stop = conversion.find('|', start);
        names.push_back(conversion.substr(start, stop == conversion.npos ? stop : stop - start));
        if (stop == conversion.npos)
            return names;
        start = stop + 1;
    }
}

inline std::string toml_binding_value(const TomlLine& line, const TomlOptionMeta& option) {
    const auto source = toml_source_rhs(line);
    const std::string conversion = option.conversion;
    if (conversion.empty()) {
        auto value = checked_toml_value(source, option);
        if (std::string_view(option.key) == "shared_cpu_affinity")
            (void)affinity_indices(value);
        return value;
    }
    char kind = 0;
    const auto value = toml_value(source, kind);
    if (conversion == "bool01" || conversion == "nonzero_bool") {
        if (kind != 'b')
            throw std::runtime_error("expected TOML boolean true or false");
        return value == "true" ? "1" : "0";
    }
    if (conversion.compare(0, 5, "enum:") == 0) {
        const auto names = toml_enum_names(conversion);
        const auto found = std::find(names.begin(), names.end(), value);
        if (kind != 's' || found == names.end())
            throw std::runtime_error("expected one of " + conversion.substr(5));
        return std::to_string(found - names.begin());
    }
    throw std::runtime_error("unsupported configuration conversion " + conversion);
}

// Convert a decoded runtime binding value to a lexical TOML value.
inline std::string toml_binding_text(const std::string& value, const TomlOptionMeta& option) {
    const std::string conversion = option.conversion;
    if (!conversion.empty()) {
        long long index = 0;
        if (!value.empty() && value != "auto") {
            size_t used = 0;
            index = std::stoll(value, &used);
            if (used != value.size())
                throw std::runtime_error("invalid integer: " + value);
        }
        if (conversion == "nonzero_bool")
            return index ? "true" : "false";
        if (conversion == "bool01" && (index == 0 || index == 1))
            return index ? "true" : "false";
        if (conversion.compare(0, 5, "enum:") == 0) {
            const auto names = toml_enum_names(conversion);
            if (index >= 0 && static_cast<size_t>(index) < names.size())
                return ecm_results::json::quote(names[static_cast<size_t>(index)]);
        }
        throw std::runtime_error("value outside configuration conversion range: " + value);
    }

    const std::string type = option.type;
    if (type == "string" || type == "int|string" || type == "float|string" ||
        (type == "bool|string" && value == "auto"))
        return ecm_results::json::quote(value);
    if (type == "rect")
        return "[" + value + "]";
    if (type == "bool" || type == "bool|string") {
        if (value == "true" || value == "1") return "true";
        if (value == "false" || value == "0") return "false";
        throw std::runtime_error("invalid boolean: " + value);
    }
    return value;
}

using TomlPath = std::vector<std::string>;

inline bool toml_bare_key(const std::string& key) {
    return !key.empty() && key.find_first_not_of(
        "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789_-") == key.npos;
}

inline TomlPath toml_path(const std::string& source) {
    TomlPath result;
    size_t start = 0;
    for (;;) {
        const auto stop = toml_unquoted_offset(source, '.', start);
        auto part = trim(source.substr(start, stop == source.npos ? stop : stop - start));
        if (!toml_bare_key(part)) {
            char kind = 0;
            if (toml_comment_offset(part) != part.npos)
                throw std::runtime_error("invalid TOML key");
            part = toml_value(part, kind);
            if (kind != 's')
                throw std::runtime_error("invalid TOML key");
        }
        result.push_back(std::move(part));
        if (stop == source.npos)
            return result;
        start = stop + 1;
    }
}

struct TomlAddress {
    std::string section, key;
    int worker_scope = 0;
    const TomlOptionMeta* option = nullptr;
};

inline int toml_table_worker(const std::string& table) {
    if (table == "shared" || table == "stage1" || table == "stage2" || table == "gui")
        return 0;
    if (table.compare(0, 7, "worker.") == 0) {
        const int worker = toml_worker("workers." + table.substr(7));
        if (worker && table == "worker." + std::to_string(worker))
            return worker;
    }
    throw std::runtime_error("unknown TOML table: " + table);
}

inline std::string toml_binding_section(const std::string& table, int worker) {
    if (!worker)
        return table;
    auto section = "workers." + std::to_string(worker);
    if (table != "workers.1")
        section += "." + table;
    return section;
}

inline TomlAddress toml_address(const std::string& table, const std::string& source_key) {
    const auto path = toml_path(source_key);
    if (path.size() != 1 || path.front().find('.') != std::string::npos)
        throw std::runtime_error("TOML keys must be flat; use the documented key with underscores");
    const int worker = toml_table_worker(table);
    TomlAddress address{toml_binding_section(worker ? "workers.1" : table, worker), path.front(), worker};
    for (const auto& option : toml_options) {
        if (address.key != option.toml_key)
            continue;
        const std::string_view owner = option.toml_table;
        if (worker) {
            if (option.global_only || owner == "gui")
                throw std::runtime_error(std::string("option belongs in [") + option.toml_table + "]");
        } else if (table != owner) {
            throw std::runtime_error(std::string("option belongs in [") + option.toml_table + "]");
        }
        address.option = &option;
        address.key = option.key;
        address.section = toml_binding_section(option.table, worker);
        break;
    }
    return address;
}

enum class TomlReadMode { Strict, Computation, Stage1, Stage2, Gui };
inline bool deferred_toml(TomlReadMode mode, const std::string& section) {
    auto table = section;
    if(toml_worker(section)>0) {
        const auto dot=section.find('.',8);
        table=dot==section.npos?"workers":section.substr(dot+1);
    }
    if(mode==TomlReadMode::Stage1)
        return table!="shared" && table!="stage1";
    if(mode==TomlReadMode::Stage2)return table!="shared" && table!="stage2";
    if(mode==TomlReadMode::Gui)return table!="shared" && table!="gui" && table!="workers";
    return false;
}


inline bool read_toml(std::istream& in, const std::string& path, std::vector<TomlLine>& lines,
                      TomlReadMode mode = TomlReadMode::Strict,
                      std::vector<std::string>* ignored_gui = nullptr,
                      int* config_version = nullptr, int active_worker = 0) {
    if (!in)
        return false;
    lines.clear();
    std::string raw, section, source_section;
    bool first = true;
    constexpr int version = business_config_version;
    size_t number = 0, version_line = 0;
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
        l.source_section = source_section;
        l.worker_scope = toml_worker(section);
        l.config_version = version;
        l.line_number = number;
        const auto t = trim(raw);
        std::string diagnostic_value = t;
        const TomlOptionMeta* option = nullptr;
        try {
            if (t.empty())
                l.kind = TomlLine::Kind::Blank;
            else if (t[0] == '#')
                l.kind = TomlLine::Kind::Comment;
            else if (t.front() == '[') {
                if (!version_line)
                    throw std::runtime_error("config_version = 4 must be declared before the first table");
                const auto close = toml_unquoted_offset(t, ']', 1);
                if (close == t.npos ||
                    (!trim(t.substr(close + 1)).empty() && trim(t.substr(close + 1)).front() != '#'))
                    throw std::runtime_error("invalid TOML table header");
                source_section = trim(t.substr(1, close - 1));
                const int worker = toml_table_worker(source_section);
                if (!tables.insert(source_section).second)
                    throw std::runtime_error("duplicate TOML table: " + source_section);
                section = toml_binding_section(worker ? "workers.1" : source_section, worker);
                l.kind = TomlLine::Kind::Section;
                l.source_section = source_section;
                l.section = section;
                l.worker_scope = toml_worker(section);
            } else {
                const auto eq = toml_assignment_offset(t);
                if (eq == t.npos)
                    throw std::runtime_error("TOML key needs a table and value");
                l.source_key = l.key = trim(t.substr(0, eq));
                l.source_value = diagnostic_value = trim(t.substr(eq + 1));
                const auto parsed_key = toml_path(l.source_key);
                if (parsed_key.size() == 1 && parsed_key.front() == "config_version") {
                    if (!section.empty())
                        throw std::runtime_error("config_version must be declared before the first table");
                    if (version_line)
                        throw std::runtime_error("duplicate config_version (first defined at line " +
                                                 std::to_string(version_line) + ")");
                    char kind = 0;
                    const auto value = toml_value(l.source_value, kind);
                    if (kind != 'i')
                        throw std::runtime_error("config_version must be an integer");
                    const auto decimal = value.front() == '+' ? value.substr(1) : value;
                    if (decimal != std::to_string(version))
                        throw std::runtime_error("unsupported config_version: " + value);
                    l.config_version = version;
                    l.kind = TomlLine::Kind::Version;
                    l.value = std::to_string(version);
                    version_line = number;
                    lines.push_back(std::move(l));
                    continue;
                }
                if (section.empty())
                    throw std::runtime_error("TOML key needs a table and value");
                const auto address = toml_address(source_section, l.source_key);
                if (!keys.insert(source_section + "." + parsed_key.front()).second)
                    throw std::runtime_error("duplicate TOML key: " + l.source_key);
                l.section = address.section;
                l.key = address.key;
                l.worker_scope = address.worker_scope;
                option = address.option;
                if (!option && l.worker_scope > 0)
                    throw std::runtime_error("unknown worker option: " + l.source_key);
                l.recognized = option != nullptr;
                const bool string_array = option && std::string_view(option->type) == "string_array";
                if (string_array && !l.source_value.empty() && l.source_value.front() == '[') {
                    auto part = l.source_value;
                    while (toml_unquoted_offset(part.substr(0, toml_comment_offset(part)), ']') == part.npos) {
                        if (!std::getline(in, part)) throw std::runtime_error("unterminated TOML array");
                        ++number;
                        if (!part.empty() && part.back() == '\r') part.pop_back();
                        l.raw += '\n' + part;
                        l.source_value += '\n' + part;
                    }
                    diagnostic_value = l.source_value;
                }
                const bool gui_table =
                    l.section == "gui" ||
                    (l.worker_scope > 0 && l.section.find('.', 8) == std::string::npos);
                if (mode == TomlReadMode::Computation && gui_table) {
                    char kind = 0;
                    if (!string_array) (void)toml_value(t.substr(eq + 1), kind);
                    if (!option && ignored_gui)
                        ignored_gui->push_back(path + ":" + std::to_string(number) + ": " +
                                               l.source_section + "." + l.source_key);
                    l.kind = TomlLine::Kind::Other;
                    lines.push_back(std::move(l));
                    continue;
                }
                if ((active_worker > 0 && l.worker_scope > 0 && l.worker_scope != active_worker) ||
                    (l.key == "b2_ratio" && mode == TomlReadMode::Stage1) ||
                    deferred_toml(mode, l.section)) {
                    // Foreign enum/type extensions must not gate this program.
                    // Decode supported scalars for optional previews; opaque values
                    // remain verbatim. Missing values/unclosed strings still fail.
                    l.deferred = true;
                    char kind = 0;
                    try { l.value = option ? toml_binding_value(l,*option) : toml_value(l.source_value,kind); }
                    catch(const std::exception&) {
                        try { l.value = toml_value(l.source_value,kind); }
                        catch(const std::exception& e) {
                            const std::string reason=e.what();
                            if(reason=="unterminated TOML string" || reason=="missing TOML value")throw;
                            l.value = diagnostic_value;
                        }
                    }
                    l.kind = TomlLine::Kind::KeyValue;
                    lines.push_back(std::move(l));
                    continue;
                }
                if (!option)
                    throw std::runtime_error("unknown TOML key or invalid table: " + l.source_key);
                l.value = toml_binding_value(l, *option);
                l.kind = TomlLine::Kind::KeyValue;
            }
        } catch (const std::exception& e) {
            throw std::runtime_error(config_diagnostic(
                path, l.line_number, l.worker_scope, l.source_key, diagnostic_value, e.what(),
                option ? (*option->conversion ?
                    "the documented form of " + std::string(option->toml_key) :
                    std::string(option->type) + " " + option->domain) :
                l.source_key == "config_version" ? "config_version = 4 at the file root" :
                "a documented key in this table"));
        }
        lines.push_back(std::move(l));
    }
    if (!version_line)
        throw std::runtime_error(config_diagnostic(path, 0, 0, "config_version", "",
            "missing config_version", "config_version = 4 at the file root"));
    if (config_version) *config_version = version;
    return !in.bad();
}

inline bool read_toml(const std::string& path, std::vector<TomlLine>& lines,
                      TomlReadMode mode = TomlReadMode::Strict,
                      std::vector<std::string>* ignored_gui = nullptr,
                     int* config_version = nullptr, int active_worker = 0) {
    std::ifstream in(path, std::ios::binary);
    return read_toml(in, path, lines, mode, ignored_gui, config_version, active_worker);
}

using Entries = std::vector<std::pair<std::string, std::string>>;

inline void replace_entry(Entries& values, std::string key, const std::string& value) {
    values.erase(
        std::remove_if(values.begin(), values.end(), [&](const auto& p) { return p.first == key; }),
        values.end());
    values.emplace_back(std::move(key), value);
}

inline Entries cli_entries(const std::vector<TomlLine>& lines, int worker,
                           bool insensitive = false, bool preview = false) {
    Entries values;
    (void)insensitive;
    // The worker table follows its matching global table. Other tables cannot
    // leak values into this projection.
    for (int pass = 0; pass < 2; ++pass)
        for (const auto& l : lines) {
            if (l.kind != TomlLine::Kind::KeyValue || !l.recognized)
                continue;
            const auto allowed = [&](const std::string& section) {
                return section == "shared" || section == "stage1" || section == "stage2";
            };
            if (pass == 0 && !allowed(l.section))
                continue;
            if (pass == 1 && (worker <= 0 || l.worker_scope != worker ||
                              l.section == "workers." + std::to_string(worker)))
                continue;
            if(l.deferred) {
                if(!preview)continue;
                const auto* meta=toml_option(l.section,l.key);
                if(!meta)continue;
                try { replace_entry(values,l.key,toml_binding_value(l,*meta)); }
                catch(const std::exception&) {} // Display defaults; the worker owns rejection.
            } else replace_entry(values, l.key, l.value);
        }
    return values;
}

// A GUI preview is advisory. It neither validates nor rewrites computation input.
inline Entries preview_entries(const std::vector<TomlLine>& lines, int worker, bool = false) {
    return cli_entries(lines, worker, false, true);
}

inline int toml_document_version(const std::vector<TomlLine>& lines) {
    for (const auto& line : lines)
        if (line.kind == TomlLine::Kind::Version) return line.config_version;
    return business_config_version;
}

// Stage1 handoff consumes only these requested receiver fields, not Stage2 arithmetic.
inline Entries selected_entries(const std::vector<TomlLine>& lines, int worker,
                                const std::vector<const char*>& keys, const std::string& path) {
    std::vector<const TomlLine*> selected;
    for(int pass=0;pass<2;++pass)for(const auto& l:lines) {
        if(l.kind!=TomlLine::Kind::KeyValue || !l.recognized ||
           !std::any_of(keys.begin(),keys.end(),[&](const char* key){return l.key==key;}))continue;
        if(pass==0 && l.section!="shared" && l.section!="stage1" && l.section!="stage2")continue;
        if(pass==1 && (worker<=0 || l.worker_scope!=worker || l.section=="workers."+std::to_string(worker)))continue;
        if(!toml_option(l.section,l.key))continue;
        selected.erase(std::remove_if(selected.begin(),selected.end(),[&](const auto* prior){return prior->key==l.key;}),selected.end());
        selected.push_back(&l);
    }
    Entries result;
    for(const auto* line:selected) {
        const auto& l=*line;const auto* meta=toml_option(l.section,l.key);
        try { replace_entry(result,l.key,toml_binding_value(l,*meta)); }
        catch(const std::exception& e) {
            throw std::runtime_error(config_diagnostic(path,l.line_number,l.worker_scope,l.key,
                toml_source_rhs(l),e.what(),std::string(meta->type)+" "+meta->domain));
        }
    }
    return result;
}

inline Entries section_entries(const std::vector<TomlLine>& lines, const std::string& section) {
    Entries values;
    for (const auto& l : lines)
        if (l.kind == TomlLine::Kind::KeyValue && l.recognized && l.section == section)
            replace_entry(values, l.key, l.value);
    return values;
}

// Shared worker count bounds both GUI workers and automatic handoff.
inline int configured_workflow_worker_count(const std::vector<TomlLine>& lines,
                                            const std::string& path) {
    const TomlLine* selected = nullptr;
    for (const auto& line : lines) {
        if (line.kind != TomlLine::Kind::KeyValue || !line.recognized)
            continue;
        if (line.section == "shared" && line.key == "workflow_worker_count")
            selected = &line;
    }
    if (!selected)
        return 0;
    const auto& line = *selected;
    const auto& meta = *toml_option(line.section, line.key);
    try {
        const auto value = toml_binding_value(line, meta);
        const auto count = std::stoi(value);
        if (count < 1 || count > 64)
            throw std::runtime_error("worker count outside 1..64");
        return count;
    } catch (const std::exception& e) {
        throw std::runtime_error(config_diagnostic(path, line.line_number, line.worker_scope,
            line.source_key, toml_source_rhs(line), e.what(), std::string(meta.type) + " " + meta.domain));
    }
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
    throw std::runtime_error(std::string(key) + " must be true, false, 0 or 1");
}

// TOML normalizes integer aliases before projection; GUI/CLI callers may also
// pass the numeric spelling directly. "auto" remains a distinct string value.
inline int automatic_boolean(const std::string& s, const char* key) {
    if (s == "auto") return -1;
    if (s == "true" || s == "1") return 1;
    if (s == "false" || s == "0") return 0;
    throw std::runtime_error(std::string(key) + " must be auto, true, false, 0 or 1");
}
inline std::string automatic_boolean_text(const std::string& s, const char* key) {
    const auto mode = automatic_boolean(s, key);
    return mode < 0 ? "auto" : mode ? "true" : "false";
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

inline void assign_value(std::string& out, const std::string& v) { out = v; }

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

inline void assign_value(bool& out, const std::string& v) { out = boolean(v, "TOML boolean"); }

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
    // Project a previously parsed value without converting the source text again.
    void (*copy)(C&, const C&);
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
