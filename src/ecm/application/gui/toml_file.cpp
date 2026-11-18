#include "toml_file.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <sstream>
#include <string_view>

#ifdef _WIN32
#include <windows.h>
#else
#include <unistd.h>
#endif

namespace ecmgui {
namespace {

bool replace_file_atomic(const std::filesystem::path &tmp, const std::filesystem::path &dst, std::string &err) {
#ifdef _WIN32
    if (!MoveFileExW(tmp.c_str(), dst.c_str(), MOVEFILE_REPLACE_EXISTING)) {
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

void replace_source_value(TomlFile::Line &line, const std::string &encoded) {
    const auto eq = ecm_config::toml_assignment_offset(line.raw);
    const auto start = line.raw.find_first_not_of(" \t", eq + 1);
    const auto comment = ecm_config::toml_comment_offset(line.raw.substr(start));
    const auto value_end = line.raw.find_last_not_of(" \t",
        comment == std::string::npos ? std::string::npos : start + comment - 1);
    line.raw.replace(start, value_end - start + 1, encoded);
    line.source_value = ecm_config::trim(line.raw.substr(eq + 1));
}

void replace_string_array(TomlFile::Line& line, const std::string& encoded) {
    const auto eq = ecm_config::toml_assignment_offset(line.raw);
    const auto start = line.raw.find_first_not_of(" \t", eq + 1);
    if (line.raw.find('\n') == line.raw.npos || line.raw[start] != '[') {
        replace_source_value(line, encoded);
        return;
    }

    // Keep comments from array rows when the editor replaces the list as a whole.
    std::string comments;
    for (auto pos = start; pos < line.raw.size();) {
        const auto next = line.raw.find('\n', pos);
        const auto row = line.raw.substr(pos, next - pos);
        const auto comment = ecm_config::toml_comment_offset(row);
        const auto close = ecm_config::toml_unquoted_offset(row.substr(0, comment), ']');
        if (close != row.npos) {
            line.raw = line.raw.substr(0, start) + encoded + line.raw.substr(pos + close + 1) + comments;
            line.source_value = encoded;
            return;
        }
        if (comment != row.npos) comments += '\n' + row.substr(comment);
        if (next == line.raw.npos) break;
        pos = next + 1;
    }
    throw std::runtime_error("unterminated TOML array");
}

} // namespace

bool TomlFile::load(const std::string &path, std::string &err) {
    path_=path;lines_.clear();loaded_=false;dirty_=false;
    config_version_ = ecm_config::business_config_version;
    try {
        std::ifstream input(std::filesystem::u8path(path),std::ios::binary);
        if(!ecm_config::read_toml(input,path,lines_,ecm_config::TomlReadMode::Gui,nullptr,&config_version_)){
            err="cannot open "+path;
            return false;
        }
        // Only foreign computation is advisory. Keep the generated validation
        // and normalization for owned fields, especially shared queue settings.
        for(const auto& line:lines_)if(line.kind==Line::Kind::KeyValue && !line.deferred) {
            try {
                const ecm_config::Entries entry{{line.key,line.value}};
                if(line.section=="gui")ecm_config::read_gui(entry);
                else if(line.worker_scope>0 && line.section=="workers."+std::to_string(line.worker_scope))
                    ecm_config::read_worker(entry);
                else { ecm_config::read_stage1(entry);ecm_config::read_stage2(entry); }
            } catch(const std::exception& e) {
                const auto* meta=ecm_config::toml_option(line.section,line.key);
                throw std::runtime_error(ecm_config::config_diagnostic(path,line.line_number,line.worker_scope,
                    line.source_key,line.source_value,e.what(),
                    std::string(meta->type)+" "+meta->domain));
            }
        }
        ecm_config::configured_workflow_worker_count(lines_, path);
    } catch(const std::exception &e) {err=e.what();return false;}
    loaded_=true;return true;
}

void TomlFile::reset(const std::string &path) {
    path_ = path;
    lines_.clear();
    config_version_ = ecm_config::business_config_version;
    Line version;
    version.kind = Line::Kind::Version;
    version.config_version = config_version_;
    version.raw = "config_version = " + std::to_string(config_version_);
    lines_.push_back(std::move(version));
    loaded_ = true;
    dirty_ = false;
}

std::string TomlFile::get(const std::string &section, const std::string &key,
                         const std::string &def) const {
    for (const Line &l : lines_) {
        if (l.kind == Line::Kind::KeyValue && l.recognized && l.section == section && l.key == key) {
            return l.value;
        }
    }
    return def;
}

bool TomlFile::has(const std::string &section, const std::string &key) const {
    for (const Line &l : lines_) {
        if (l.kind == Line::Kind::KeyValue && l.recognized && l.section == section && l.key == key) {
            return true;
        }
    }
    return false;
}

std::pair<std::string, std::string> TomlFile::display_path(const std::string &section,
                                                           const std::string &key) const {
    const auto* option = ecm_config::toml_option(section, key);
    if (!option)
        return {};
    const int worker = ecm_config::toml_worker(section);
    if (!worker)
        return {option->toml_table, option->toml_key};
    if (option->global_only)
        return {};
    return {"worker." + std::to_string(worker), option->toml_key};
}

std::string TomlFile::display_value(const std::string &section, const std::string &key) const {
    const auto value = get(section, key);
    const auto* option = ecm_config::toml_option(section, key);
    if (!option || !*option->conversion || value.empty())
        return value;
    try {
        char kind = 0;
        return ecm_config::toml_value(ecm_config::toml_binding_text(value, *option), kind);
    } catch (const std::exception&) {
        return value; // The worker will report unsupported compute settings.
    }
}

std::string TomlFile::binding_value(const std::string &section, const std::string &key,
                                    const std::string &value) const {
    const auto* option = ecm_config::toml_option(section, key);
    if (!option || !*option->conversion)
        return value;
    Line line;
    line.config_version = config_version_;
    line.source_value = std::string_view(option->conversion).substr(0, 5) == "enum:"
                            ? ecm_results::json::quote(value)
                            : value;
    return ecm_config::toml_binding_value(line, *option);
}

void TomlFile::set(const std::string &section, const std::string &key,
                  const std::string &value) {
    const auto *option=ecm_config::toml_option(section,key);
    if(!option)throw std::runtime_error("unknown TOML option: "+section+"."+key);
    auto [table, field] = display_path(section, key);
    if (table.empty() || field.empty())
        throw std::runtime_error("option unavailable in this configuration version: " + key);
    const auto encoded = ecm_config::toml_binding_text(value, *option);
    Line nl;
    nl.kind = Line::Kind::KeyValue;
    nl.config_version = config_version_;
    nl.section = section;
    nl.worker_scope = ecm_config::toml_worker(section);
    nl.key = key;
    nl.source_value = encoded;
    nl.value = ecm_config::toml_binding_value(nl, *option);
    nl.recognized = true;
    for (auto &l : lines_) {
        if (l.kind == Line::Kind::KeyValue && l.recognized && l.section == section && l.key == key) {
            if (l.value == nl.value) return;
            if(std::string_view(option->type)=="string_array") replace_string_array(l, encoded);
            else replace_source_value(l, encoded);
            l.value = nl.value;
            dirty_ = true;
            return;
        }
    }

    std::size_t sec_start = lines_.size();
    for (std::size_t i = 0; i < lines_.size(); ++i) {
        if (lines_[i].kind == Line::Kind::Section && lines_[i].source_section == table) {
            sec_start = i;
            break;
        }
    }
    nl.source_section = table;
    nl.source_key = field;
    nl.raw = field + " = " + encoded;
    if (sec_start == lines_.size()) {
        if (!lines_.empty() && lines_.back().kind != Line::Kind::Blank) {
            Line blank;
            blank.kind = Line::Kind::Blank;
            lines_.push_back(blank);
        }
        Line hdr;
        hdr.kind = Line::Kind::Section;
        hdr.section = section;
        hdr.source_section = table;
        hdr.worker_scope = nl.worker_scope;
        hdr.config_version = config_version_;
        hdr.raw = "[" + table + "]";
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

int TomlFile::get_int(const std::string &section, const std::string &key, int def) const {
    const std::string v = get(section, key);
    if (v.empty()) return def;
    char *end = nullptr;
    const long n = std::strtol(v.c_str(), &end, 10);
    return (end && *end == '\0') ? static_cast<int>(n) : def;
}

double TomlFile::get_double(const std::string &section, const std::string &key,
                           double def) const {
    const std::string v = get(section, key);
    if (v.empty()) return def;
    char *end = nullptr;
    const double d = std::strtod(v.c_str(), &end);
    return (end && *end == '\0') ? d : def;
}

void TomlFile::set_int(const std::string &section, const std::string &key, int value) {
    set(section, key, std::to_string(value));
}

void TomlFile::set_double(const std::string &section, const std::string &key, double value) {
    char buf[64];
    std::snprintf(buf, sizeof(buf), "%.6g", value);
    set(section, key, buf);
}

std::vector<std::string> TomlFile::sections() const {
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

bool TomlFile::save(std::string &err) {
    err.clear();
    if (path_.empty()) {
        err = "no path";
        return false;
    }
    try {
        std::stringstream source;
        for (const Line &line : lines_) source << line.raw << '\n';
        std::vector<Line> verified;
        if (!ecm_config::read_toml(source, path_, verified, ecm_config::TomlReadMode::Gui)) {
            err = "cannot validate " + path_;
            return false;
        }
    } catch (const std::exception &error) {
        err = error.what();
        return false;
    }
    const auto path = std::filesystem::u8path(path_);
    auto tmp = path;tmp += ".tmp";

    {
        std::ofstream out(tmp, std::ios::out | std::ios::trunc | std::ios::binary);
        if (!out.is_open()) {
            err = "cannot write " + tmp.u8string();
            return false;
        }
        for (const Line &l : lines_) {
            out << l.raw << "\r\n";
        }
        out.close();
        if (out.fail()) {
            err = "write failed: " + tmp.u8string();
            std::error_code ignored;
            std::filesystem::remove(tmp,ignored);
            return false;
        }
    }

    // One .bak generation: copy the current file before replacing it.
    {
        std::ifstream src(path, std::ios::in | std::ios::binary);
        if (src.is_open()) {
            auto backup=path;backup += ".bak";
            std::ofstream bak(backup, std::ios::out | std::ios::trunc | std::ios::binary);
            if (bak.is_open()) {
                bak << src.rdbuf();
                bak.close();
            }
        }
    }

    if (!replace_file_atomic(tmp, path, err)) {
        std::error_code ignored;
        std::filesystem::remove(tmp,ignored);
        return false;
    }
    dirty_ = false;
    return true;
}

} // namespace ecmgui
