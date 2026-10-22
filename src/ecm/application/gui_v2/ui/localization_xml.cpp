// Adapted from gui/localization.cpp: same Panel/Item grammar, Unicode filesystem paths.
#include "../../gui/localization.h"
#include <filesystem>

#include <algorithm>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <set>
#include <sstream>

#ifdef _WIN32
#include <windows.h>
#endif

namespace ecmgui {
namespace {

bool valid_utf8(const std::string &s) {
    std::size_t i = 0;
    while (i < s.size()) {
        const unsigned char c = static_cast<unsigned char>(s[i]);
        std::size_t extra = 0;
        if (c < 0x80) extra = 0;
        else if ((c & 0xE0) == 0xC0) extra = 1;
        else if ((c & 0xF0) == 0xE0) extra = 2;
        else if ((c & 0xF8) == 0xF0) extra = 3;
        else return false;
        if (i + extra >= s.size()) return false;
        for (std::size_t k = 1; k <= extra; ++k) {
            if ((static_cast<unsigned char>(s[i + k]) & 0xC0) != 0x80) return false;
        }
        i += extra + 1;
    }
    return true;
}

// XML entity decoding (the files use &amp; &lt; &gt; &quot; &apos; and &#NN;).
std::string decode_entities(const std::string &s) {
    std::string out;
    out.reserve(s.size());
    for (std::size_t i = 0; i < s.size(); ++i) {
        if (s[i] != '&') {
            out.push_back(s[i]);
            continue;
        }
        const std::size_t semi = s.find(';', i);
        if (semi == std::string::npos || semi - i > 10) {
            out.push_back(s[i]);
            continue;
        }
        const std::string ent = s.substr(i + 1, semi - i - 1);
        if (ent == "amp") out.push_back('&');
        else if (ent == "lt") out.push_back('<');
        else if (ent == "gt") out.push_back('>');
        else if (ent == "quot") out.push_back('"');
        else if (ent == "apos") out.push_back('\'');
        else if (!ent.empty() && ent[0] == '#') {
            const long code = std::strtol(ent.c_str() + 1, nullptr, 10);
            // Only the BMP subset we can store as UTF-8 without extra work.
            if (code > 0 && code < 0x80) {
                out.push_back(static_cast<char>(code));
            } else if (code >= 0x80 && code <= 0x7FF) {
                out.push_back(static_cast<char>(0xC0 | (code >> 6)));
                out.push_back(static_cast<char>(0x80 | (code & 0x3F)));
            } else if (code > 0x7FF && code <= 0xFFFF) {
                out.push_back(static_cast<char>(0xE0 | (code >> 12)));
                out.push_back(static_cast<char>(0x80 | ((code >> 6) & 0x3F)));
                out.push_back(static_cast<char>(0x80 | (code & 0x3F)));
            } else {
                out.push_back('?');
            }
        } else {
            out.push_back('&');
            continue;
        }
        i = semi;
    }
    return out;
}

// Prefix test written with an explicit length (std::string::compare(pos, len, lit)
// compares only `len` characters -- mixing that up silently matches nothing, which
// is exactly how the first version failed to find the <EcmGui> root).
bool starts_with(const std::string &s, const char *prefix) {
    const std::size_t n = std::strlen(prefix);
    return s.size() >= n && s.compare(0, n, prefix) == 0;
}

std::string attr(const std::string &tag, const std::string &name) {
    const std::string needle = name + "=\"";
    std::size_t p = tag.find(needle);
    if (p == std::string::npos) {
        const std::string needle2 = name + "='";
        p = tag.find(needle2);
        if (p == std::string::npos) return std::string();
        const std::size_t end = tag.find('\'', p + needle2.size());
        if (end == std::string::npos) return std::string();
        return decode_entities(tag.substr(p + needle2.size(), end - p - needle2.size()));
    }
    const std::size_t end = tag.find('"', p + needle.size());
    if (end == std::string::npos) return std::string();
    return decode_entities(tag.substr(p + needle.size(), end - p - needle.size()));
}

} // namespace

bool parse_localization_xml(const std::string &path, std::vector<Localization::Entry> &out,
                            std::string &err) {
    out.clear();
    err.clear();

    std::ifstream in(std::filesystem::u8path(path), std::ios::in | std::ios::binary);
    if (!in.is_open()) {
        err = "cannot open " + path;
        return false;
    }
    std::ostringstream ss;
    ss << in.rdbuf();
    std::string text = ss.str();

    std::size_t start = 0;
    if (text.size() >= 3 && static_cast<unsigned char>(text[0]) == 0xEF &&
        static_cast<unsigned char>(text[1]) == 0xBB && static_cast<unsigned char>(text[2]) == 0xBF) {
        start = 3;
    }
    if (!valid_utf8(text.substr(start))) {
        err = path + " is not valid UTF-8 (localization files must be UTF-8, with or without BOM)";
        return false;
    }

    // Drop comments so a commented-out <Item> never becomes a string.
    std::string clean;
    clean.reserve(text.size());
    for (std::size_t i = 0; i < text.size();) {
        if (text.compare(i, 4, "<!--") == 0) {
            const std::size_t end = text.find("-->", i + 4);
            if (end == std::string::npos) break;
            i = end + 3;
            continue;
        }
        clean.push_back(text[i]);
        ++i;
    }

    bool saw_root = false;
    std::string panel;
    panel.clear();
    std::size_t i = 0;
    while (i < clean.size()) {
        const std::size_t lt = clean.find('<', i);
        if (lt == std::string::npos) break;
        const std::size_t gt = clean.find('>', lt);
        if (gt == std::string::npos) break;
        const std::string tag = clean.substr(lt, gt - lt + 1);
        i = gt + 1;

        if (starts_with(tag, "<?") || starts_with(tag, "<!")) continue;

        if (tag[1] == '/') {
            const std::string name = tag.substr(2, tag.size() - 3);
            if (name == "Panel") panel.clear();
            continue;
        }
        if (starts_with(tag, "<EcmGui") || starts_with(tag, "<Native-Langue")) {
            saw_root = true;
            continue;
        }
        if (starts_with(tag, "<Panel")) {
            panel = attr(tag, "id");
            continue;
        }
        if (starts_with(tag, "<Item")) {
            Localization::Entry e;
            e.panel = panel;
            e.id = attr(tag, "id");
            e.text = attr(tag, "name");
            if (e.id.empty()) {
                err = path + ": an <Item> has no id attribute";
                return false;
            }
            out.push_back(e);
            continue;
        }
    }

    if (!saw_root) {
        err = path + ": not a localization file (no <EcmGui>/<Native-Langue> element)";
        return false;
    }
    if (clean.find("</EcmGui>") == std::string::npos || clean.find("</Native-Langue>") == std::string::npos) {
        err = path + ": incomplete localization XML"; return false;
    }
    if (out.empty()) {
        err = path + ": no <Item> entries found";
        return false;
    }
    return true;
}


} // namespace ecmgui
