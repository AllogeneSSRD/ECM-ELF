#include "ini_file.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <sstream>

#ifdef _WIN32
#include <windows.h>
#else
#include <unistd.h>
#endif

namespace ecmgui {
namespace {

bool replace_file_atomic(const std::string &tmp, const std::string &dst, std::string &err) {
#ifdef _WIN32
    if (!MoveFileExA(tmp.c_str(), dst.c_str(), MOVEFILE_REPLACE_EXISTING)) {
        err = "MoveFileEx failed (" + std::to_string(GetLastError()) + ")";
        return false;
    }
    return true;
#else
    if (std::rename(tmp.c_str(), dst.c_str()) != 0) {
        err = std::string("rename failed: ") + std::strerror(errno);
        return false;
    }
    return true;
#endif
}

} // namespace

bool IniFile::load(const std::string &path, std::string &err) {
    path_=path;lines_.clear();loaded_=false;dirty_=false;
    if(!ecm_config::read_ini(path,lines_)){err="cannot open "+path;return false;}
    loaded_=true;return true;
}

void IniFile::reset(const std::string &path) {
    path_ = path;
    lines_.clear();
    loaded_ = true;
    dirty_ = false;
}

std::string IniFile::get(const std::string &section, const std::string &key,
                         const std::string &def) const {
    // Later occurrences win, matching the driver's "last one wins" behaviour.
    const std::string *found = nullptr;
    for (const Line &l : lines_) {
        if (l.kind == Line::Kind::KeyValue && l.section == section && l.key == key) {
            found = &l.value;
        }
    }
    return found ? *found : def;
}

bool IniFile::has(const std::string &section, const std::string &key) const {
    for (const Line &l : lines_) {
        if (l.kind == Line::Kind::KeyValue && l.section == section && l.key == key) {
            return true;
        }
    }
    return false;
}

void IniFile::set(const std::string &section, const std::string &key,
                  const std::string &value) {
    // Replace the LAST occurrence (that is the effective one) and drop nothing else.
    for (std::size_t i = lines_.size(); i-- > 0;) {
        Line &l = lines_[i];
        if (l.kind == Line::Kind::KeyValue && l.section == section && l.key == key) {
            if (l.value == value) return;                 // no change -> not dirty
            l.value = value;
            l.raw = key + " = " + value;
            dirty_ = true;
            return;
        }
    }

    Line nl;
    nl.kind = Line::Kind::KeyValue;
    nl.section = section;
    nl.key = key;
    nl.value = value;
    nl.raw = key + " = " + value;

    if (section.empty()) {
        // Global keys live before the first section header.
        std::size_t pos = lines_.size();
        for (std::size_t i = 0; i < lines_.size(); ++i) {
            if (lines_[i].kind == Line::Kind::Section) { pos = i; break; }
        }
        lines_.insert(lines_.begin() + static_cast<std::ptrdiff_t>(pos), nl);
        dirty_ = true;
        return;
    }

    // Append at the end of that section (before the next header), or create it.
    std::size_t sec_start = lines_.size();
    for (std::size_t i = 0; i < lines_.size(); ++i) {
        if (lines_[i].kind == Line::Kind::Section && lines_[i].section == section) {
            sec_start = i;
            break;
        }
    }
    if (sec_start == lines_.size()) {
        if (!lines_.empty() && lines_.back().kind != Line::Kind::Blank) {
            Line blank;
            blank.kind = Line::Kind::Blank;
            lines_.push_back(blank);
        }
        Line hdr;
        hdr.kind = Line::Kind::Section;
        hdr.section = section;
        hdr.raw = "[" + section + "]";
        lines_.push_back(hdr);
        lines_.push_back(nl);
        dirty_ = true;
        return;
    }
    std::size_t pos = lines_.size();
    for (std::size_t i = sec_start + 1; i < lines_.size(); ++i) {
        if (lines_[i].kind == Line::Kind::Section) { pos = i; break; }
    }
    // Step back over trailing blank lines so the new key stays inside the section.
    while (pos > sec_start + 1 && lines_[pos - 1].kind == Line::Kind::Blank) --pos;
    lines_.insert(lines_.begin() + static_cast<std::ptrdiff_t>(pos), nl);
    dirty_ = true;
}

int IniFile::get_int(const std::string &section, const std::string &key, int def) const {
    const std::string v = get(section, key);
    if (v.empty()) return def;
    char *end = nullptr;
    const long n = std::strtol(v.c_str(), &end, 10);
    return (end && *end == '\0') ? static_cast<int>(n) : def;
}

double IniFile::get_double(const std::string &section, const std::string &key,
                           double def) const {
    const std::string v = get(section, key);
    if (v.empty()) return def;
    char *end = nullptr;
    const double d = std::strtod(v.c_str(), &end);
    return (end && *end == '\0') ? d : def;
}

void IniFile::set_int(const std::string &section, const std::string &key, int value) {
    set(section, key, std::to_string(value));
}

void IniFile::set_double(const std::string &section, const std::string &key, double value) {
    char buf[64];
    std::snprintf(buf, sizeof(buf), "%.6g", value);
    set(section, key, buf);
}

std::vector<std::string> IniFile::sections() const {
    std::vector<std::string> out;
    bool have_global = false;
    for (const Line &l : lines_) {
        if (l.kind == Line::Kind::Section) {
            bool seen = false;
            for (const std::string &s : out) {
                if (s == l.section) { seen = true; break; }
            }
            if (!seen) out.push_back(l.section);
        } else if (l.kind == Line::Kind::KeyValue && l.section.empty() && !have_global) {
            have_global = true;
            out.insert(out.begin(), std::string());
        }
    }
    return out;
}

bool IniFile::save(std::string &err) {
    err.clear();
    if (path_.empty()) {
        err = "no path";
        return false;
    }
    const std::string tmp = path_ + ".tmp";

    {
        std::ofstream out(tmp, std::ios::out | std::ios::trunc | std::ios::binary);
        if (!out.is_open()) {
            err = "cannot write " + tmp;
            return false;
        }
        for (const Line &l : lines_) {
            out << l.raw << "\r\n";
        }
        out.close();
        if (out.fail()) {
            err = "write failed: " + tmp;
            std::remove(tmp.c_str());
            return false;
        }
    }

    // One .bak generation: copy the current file before replacing it.
    {
        std::ifstream src(path_, std::ios::in | std::ios::binary);
        if (src.is_open()) {
            std::ofstream bak(path_ + ".bak", std::ios::out | std::ios::trunc | std::ios::binary);
            if (bak.is_open()) {
                bak << src.rdbuf();
                bak.close();
            }
        }
    }

    if (!replace_file_atomic(tmp, path_, err)) {
        std::remove(tmp.c_str());
        return false;
    }
    dirty_ = false;
    return true;
}

} // namespace ecmgui
