#pragma once
#include <algorithm>
#include <cctype>
#include <iomanip>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

// Small ordered JSON reader for owned result records. Numbers remain lexical:
// ECM moduli must never pass through a floating point representation.
namespace ecm_results {
namespace json {
inline std::string quote(const std::string& s) {
    std::ostringstream out;
    out << '"';
    for (unsigned char c : s) {
        if (c == '"' || c == '\\')
            out << '\\' << c;
        else if (c < 32)
            out << "\\u" << std::hex << std::setw(4) << std::setfill('0') << unsigned(c)
                << std::dec;
        else
            out << c;
    }
    out << '"';
    return out.str();
}

struct Value {
    char type = 'n';
    std::string scalar = "null";
    std::vector<std::pair<std::string, Value>> members;
    std::vector<Value> items;

    static Value string(std::string s) {
        Value v;
        v.type = 's';
        v.scalar = std::move(s);
        return v;
    }

    static Value number(std::string s) {
        Value v;
        v.type = 'd';
        v.scalar = std::move(s);
        return v;
    }

    const Value& get(const std::string& key) const {
        for (const auto& p : members)
            if (p.first == key)
                return p.second;
        static const Value empty;
        return empty;
    }

    bool has(const std::string& key) const {
        for (const auto& p : members)
            if (p.first == key)
                return true;
        return false;
    }

    std::string text(const std::string& fallback = "") const {
        return type == 's' || type == 'd' ? scalar : fallback;
    }

    void set(const std::string& key, Value value) {
        for (auto& p : members)
            if (p.first == key) {
                p.second = std::move(value);
                return;
            }
        members.emplace_back(key, std::move(value));
    }

    std::string dump() const {
        if (type == 's')
            return quote(scalar);
        if (type == 'o') {
            std::string s = "{";
            for (const auto& p : members) {
                if (s.size() > 1)
                    s += ',';
                s += quote(p.first) + ':' + p.second.dump();
            }
            return s + '}';
        }
        if (type == 'a') {
            std::string s = "[";
            for (const auto& v : items) {
                if (s.size() > 1)
                    s += ',';
                s += v.dump();
            }
            return s + ']';
        }
        return scalar;
    }
};

class Reader {
    const std::string& s;
    size_t p = 0;

    [[noreturn]] void fail() const {
        throw std::runtime_error("invalid result JSON");
    }

    void ws() {
        while (p < s.size() && (s[p] == ' ' || s[p] == '\t' || s[p] == '\r' || s[p] == '\n'))
            ++p;
    }

    unsigned hex4() {
        unsigned v = 0;
        for (int i = 0; i < 4; ++i) {
            if (p == s.size())
                fail();
            const char c = s[p++];
            v <<= 4;
            if (c >= '0' && c <= '9')
                v += c - '0';
            else if (c >= 'a' && c <= 'f')
                v += c - 'a' + 10;
            else if (c >= 'A' && c <= 'F')
                v += c - 'A' + 10;
            else
                fail();
        }
        return v;
    }

    void utf8(std::string& out, unsigned v) {
        if (v < 128)
            out += char(v);
        else if (v < 2048) {
            out += char(0xc0 | (v >> 6));
            out += char(0x80 | (v & 63));
        } else if (v < 65536) {
            out += char(0xe0 | (v >> 12));
            out += char(0x80 | ((v >> 6) & 63));
            out += char(0x80 | (v & 63));
        } else {
            out += char(0xf0 | (v >> 18));
            out += char(0x80 | ((v >> 12) & 63));
            out += char(0x80 | ((v >> 6) & 63));
            out += char(0x80 | (v & 63));
        }
    }

    std::string string() {
        if (p == s.size() || s[p++] != '"')
            fail();
        std::string out;
        while (p < s.size()) {
            unsigned char c = s[p++];
            if (c == '"')
                return out;
            if (c < 32)
                fail();
            if (c != '\\') {
                out += char(c);
                continue;
            }
            if (p == s.size())
                fail();
            c = s[p++];
            switch (c) {
            case '"':
            case '\\':
            case '/':
                out += char(c);
                break;
            case 'b':
                out += '\b';
                break;
            case 'f':
                out += '\f';
                break;
            case 'n':
                out += '\n';
                break;
            case 'r':
                out += '\r';
                break;
            case 't':
                out += '\t';
                break;
            case 'u': {
                auto v = hex4();
                if (v >= 0xd800 && v <= 0xdbff) {
                    if (p + 2 > s.size() || s[p++] != '\\' || s[p++] != 'u')
                        fail();
                    auto low = hex4();
                    if (low < 0xdc00 || low > 0xdfff)
                        fail();
                    v = 0x10000 + ((v - 0xd800) << 10) + (low - 0xdc00);
                } else if (v >= 0xdc00 && v <= 0xdfff)
                    fail();
                utf8(out, v);
                break;
            }
            default:
                fail();
            }
        }
        fail();
    }

    Value value(unsigned depth) {
        if (depth > 64)
            fail();
        ws();
        if (p == s.size())
            fail();
        Value v;
        if (s[p] == '"')
            return Value::string(string());
        if (s[p] == '{' || s[p] == '[') {
            const bool object = s[p++] == '{';
            v.type = object ? 'o' : 'a';
            ws();
            const char end = object ? '}' : ']';
            if (p < s.size() && s[p] == end) {
                ++p;
                return v;
            }
            for (;;) {
                ws();
                std::string key;
                if (object) {
                    key = string();
                    if (v.has(key))
                        fail();
                    ws();
                    if (p == s.size() || s[p++] != ':')
                        fail();
                }
                auto child = value(depth + 1);
                if (object)
                    v.members.emplace_back(key, std::move(child));
                else
                    v.items.push_back(std::move(child));
                ws();
                if (p == s.size())
                    fail();
                if (s[p++] == end)
                    return v;
                if (s[p - 1] != ',')
                    fail();
            }
        }
        const auto begin = p;
        if (s[p] == '-' || std::isdigit(static_cast<unsigned char>(s[p]))) {
            if (s[p] == '-')
                ++p;
            if (p == s.size())
                fail();
            if (s[p] == '0')
                ++p;
            else {
                if (s[p] < '1' || s[p] > '9')
                    fail();
                while (p < s.size() && std::isdigit(static_cast<unsigned char>(s[p])))
                    ++p;
            }
            if (p < s.size() && s[p] == '.') {
                ++p;
                auto start = p;
                while (p < s.size() && std::isdigit(static_cast<unsigned char>(s[p])))
                    ++p;
                if (start == p)
                    fail();
            }
            if (p < s.size() && (s[p] == 'e' || s[p] == 'E')) {
                ++p;
                if (p < s.size() && (s[p] == '+' || s[p] == '-'))
                    ++p;
                auto start = p;
                while (p < s.size() && std::isdigit(static_cast<unsigned char>(s[p])))
                    ++p;
                if (start == p)
                    fail();
            }
            return Value::number(s.substr(begin, p - begin));
        }
        for (const auto& literal : {"true", "false", "null"}) {
            const std::string token = literal;
            if (s.compare(p, token.size(), token) == 0) {
                p += token.size();
                v.scalar = token;
                return v;
            }
        }
        fail();
    }

public:
    explicit Reader(const std::string& text) : s(text) {
    }

    Value read() {
        auto v = value(0);
        ws();
        if (p != s.size())
            fail();
        return v;
    }
};

inline Value parse(const std::string& s) {
    return Reader(s).read();
}

inline std::string joined(const Value& v) {
    std::string s;
    for (const auto& i : v.items) {
        if (!s.empty())
            s += ',';
        s += i.text();
    }
    return s;
}
} // namespace json
} // namespace ecm_results
