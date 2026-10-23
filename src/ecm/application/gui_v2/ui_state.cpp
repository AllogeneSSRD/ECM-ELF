#include "ui_state.h"
#include "../gui/toml_file.h"
#include "../../storage/results/ecm_result_json.h"
#include <windows.h>
#include <shlobj.h>
#include <fstream>
#include <sstream>
#include <iomanip>
#include <limits>
#include <map>
#include <set>
#include <type_traits>

namespace ecmgui_v2 {
namespace {
void parse(const std::string& text, std::string& value) { value = text; }
void parse(const std::string& text, bool& value) {
    if (text != "true" && text != "false") throw std::runtime_error("UI boolean must be true or false");
    value = text == "true";
}
void parse(const std::string& text, int& value) {
    size_t used = 0; const auto number = std::stoll(text, &used);
    if (used != text.size() || number < INT_MIN || number > INT_MAX) throw std::runtime_error("UI integer out of range");
    value = static_cast<int>(number);
}
void parse(const std::string& text, float& value) {
    size_t used = 0; value = std::stof(text, &used);
    if (used != text.size() || !std::isfinite(value)) throw std::runtime_error("invalid UI scale/position");
}
void emit(std::ostream& out, const std::string& value) { out << ecm_results::json::quote(value); }
void emit(std::ostream& out, bool value) { out << (value ? "true" : "false"); }
void emit(std::ostream& out, int value) { out << value; }
void emit(std::ostream& out, float value) { out << std::setprecision(std::numeric_limits<float>::max_digits10) << value; }
}
std::filesystem::path UiState::default_path(const std::string& config) {
    wchar_t* local = nullptr;
    if (FAILED(SHGetKnownFolderPath(FOLDERID_LocalAppData, 0, nullptr, &local))) throw std::runtime_error("Cannot locate local application data");
    std::filesystem::path base(local); CoTaskMemFree(local);
    // Stable FNV identity only names UI preferences; it is not an ownership lock.
    unsigned long long hash = 14695981039346656037ull;
    auto identity = config.empty() ? std::string("default") : std::filesystem::weakly_canonical(std::filesystem::u8path(config)).u8string();
    for (unsigned char c : identity) { hash ^= static_cast<unsigned char>(std::tolower(c)); hash *= 1099511628211ull; }
    std::ostringstream name; name << std::hex << hash << ".toml";
    return base / L"ECM-ELF" / L"gui-v2" / name.str();
}
bool UiState::load() {
    if (path.empty() || !std::filesystem::exists(path)) return true;
    try {
        if (std::filesystem::file_size(path) > 1048576) throw std::runtime_error("UI state exceeds 1 MiB limit");
        // Reuse value decoding, never the business parser's table/key registry.
        std::ifstream input(path); if (!input) throw std::runtime_error("cannot open UI state");
        std::map<std::string,std::pair<std::string,char>> values;
        std::set<std::string> tables; std::string section, line;
        while (std::getline(input, line)) {
            auto text = ecm_config::trim(line);
            if (text.compare(0,3,"\xef\xbb\xbf") == 0) text.erase(0,3);
            if (text.empty() || text[0] == '#') continue;
            if (text[0] == '[') {
                const auto close = text.find(']');
                if (close == text.npos || !ecm_config::trim(text.substr(close+1)).empty()) throw std::runtime_error("invalid UI table");
                section = text.substr(1,close-1);
                if (!tables.insert(section).second) throw std::runtime_error("duplicate UI table");
            } else {
                const auto equal = text.find('=');
                if (equal == text.npos || section.empty()) throw std::runtime_error("invalid UI assignment");
                const auto key = section + '.' + ecm_config::trim(text.substr(0,equal));
                char kind = 0; const auto decoded = ecm_config::toml_value(text.substr(equal+1),kind);
                if (!values.emplace(key,std::make_pair(decoded,kind)).second) throw std::runtime_error("duplicate UI key");
            }
        }
        if (input.bad()) throw std::runtime_error("cannot read UI state");
        const std::array<bool,3> saved_tabs{values.count("ui.nav_hide_tab_bar")!=0,values.count("ui.workspace_hide_tab_bar")!=0,values.count("ui.right_hide_tab_bar")!=0};
        UiValues candidate;
        std::set<std::string> known_tables;
        visit_ui_fields(candidate, [&](const char* table, const char* key, auto& field) {
            known_tables.insert(table);
            const auto found = values.find(std::string(table) + '.' + key);
            if (found == values.end()) return;
            const auto kind = found->second.second; using Type = std::decay_t<decltype(field)>;
            const bool valid = (std::is_same_v<Type,std::string> && kind == 's') || (std::is_same_v<Type,bool> && kind == 'b') ||
                (std::is_same_v<Type,int> && kind == 'i') || (std::is_same_v<Type,float> && (kind == 'i' || kind == 'f'));
            if (!valid) throw std::runtime_error("wrong UI value type");
            parse(found->second.first, field); values.erase(found);
        });
        if (!values.empty()) throw std::runtime_error("unknown UI key: " + values.begin()->first);
        for (const auto& table : tables) if (!known_tables.count(table)) throw std::runtime_error("unknown UI table: " + table);
        if(candidate.ui_nav_order.size()==7){auto ids=candidate.ui_nav_order;std::sort(ids.begin(),ids.end());if(ids=="0123456")candidate.ui_nav_order+='7';}
        validate_ui(candidate);
        if(candidate.ui_dock.size()%2 || candidate.ui_dock.find_first_not_of("0123456789abcdef")!=std::string::npos)
            throw std::runtime_error("invalid UI layout hex");
        static_cast<UiValues&>(*this) = std::move(candidate);dock_tab_saved=saved_tabs;
        return true;
    } catch (const std::exception& error) { writable = false; notice = "UI state not loaded; original retained: " + std::string(error.what()); return false; }
}
bool UiState::save() {
    if (!writable) return false;
    std::filesystem::path temporary;
    try {
        if (path.empty()) return true;
        validate_ui(*this);
        if (!path.parent_path().empty()) std::filesystem::create_directories(path.parent_path());
        temporary = path; temporary += ".tmp." + std::to_string(GetCurrentProcessId());
        std::ofstream out(temporary, std::ios::binary | std::ios::trunc);
        if (!out) throw std::runtime_error("cannot prepare UI state");
        out << "# ECM-ELF GUI v2 UI state; business TOML is independent.\n";
        std::string current;
        visit_ui_fields(*this, [&](const char* table, const char* key, const auto& field) {
            if (current != table) { current = table; out << "\n[" << current << "]\n"; }
            out << key << " = "; emit(out, field); out << '\n';
        });
        out.close(); if (!out) throw std::runtime_error("cannot finish UI state");
        if (!MoveFileExW(temporary.c_str(), path.c_str(), MOVEFILE_REPLACE_EXISTING | MOVEFILE_WRITE_THROUGH)) throw std::runtime_error("cannot replace UI state");
        return true;
    } catch (const std::exception& error) { notice = "UI state not saved: " + std::string(error.what()); return false; }
}
std::string UiState::layout() const {
    if (ui_dock.size() % 2) throw std::runtime_error("odd UI layout hex length");
    std::string text; text.reserve(ui_dock.size() / 2);
    const auto nibble = [](char c) { if (c >= '0' && c <= '9') return c - '0'; if (c >= 'a' && c <= 'f') return c - 'a' + 10; throw std::runtime_error("invalid UI layout hex"); };
    for (size_t i = 0; i < ui_dock.size(); i += 2) text.push_back(static_cast<char>((nibble(ui_dock[i]) << 4) | nibble(ui_dock[i + 1])));
    return text;
}
void UiState::layout(const std::string& text) {
    if (text.size() > 262144) throw std::runtime_error("UI layout exceeds 256 KiB");
    const char digits[] = "0123456789abcdef"; ui_dock.clear(); ui_dock.reserve(text.size() * 2);
    for (unsigned char c : text) { ui_dock.push_back(digits[c >> 4]); ui_dock.push_back(digits[c & 15]); }
}
}
