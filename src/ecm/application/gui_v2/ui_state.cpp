#include "ui_state.h"
#include "../gui/toml_file.h"
#include <charconv>
#include <iomanip>
#include <limits>
#include <locale>
#include <sstream>

namespace ecmgui_v2 {
namespace {
void parse(const std::string& text, std::string& value) { value = text; }
void parse(const std::string& text, bool& value) { value = ecm_config::boolean(text, "GUI preference"); }
void parse(const std::string& text, int& value) {
    size_t used = 0;
    const auto number = std::stoll(text, &used);
    if (used != text.size() || number < INT_MIN || number > INT_MAX)
        throw std::runtime_error("invalid GUI integer");
    value = static_cast<int>(number);
}
void parse(const std::string& text, float& value) {
    const auto end = text.data() + text.size();
    const auto result = std::from_chars(text.data(), end, value);
    if (result.ptr != end || result.ec != std::errc{} || !std::isfinite(value))
        throw std::runtime_error("invalid GUI decimal");
}

std::string encode(const std::string& value) { return value; }
std::string encode(bool value) { return value ? "true" : "false"; }
std::string encode(int value) { return std::to_string(value); }
std::string encode(float value) {
    std::ostringstream out;
    out.imbue(std::locale::classic());
    out << std::setprecision(std::numeric_limits<float>::max_digits10) << value;
    return out.str();
}
}

bool UiState::load(const ecmgui::TomlFile& config) {
    try {
        UiValues candidate;
        visit_ui_fields([&](const char* key, auto member) {
            if (config.has("gui", key)) parse(config.get("gui", key), candidate.*member);
        });
        validate_ui(candidate);
        if (candidate.ui_dock.size() % 2 ||
            candidate.ui_dock.find_first_not_of("0123456789abcdef") != std::string::npos)
            throw std::runtime_error("invalid GUI layout hex");
        static_cast<UiValues&>(*this) = std::move(candidate);
        mark_saved();
        dock_tab_saved = {config.has("gui", "nav_hide_tab_bar"),
                          config.has("gui", "workspace_hide_tab_bar"),
                          config.has("gui", "right_hide_tab_bar")};
        notice.clear();
        writable = true;
        return true;
    } catch (const std::exception& error) {
        notice = "GUI preferences could not be loaded: " + std::string(error.what());
        writable = false;
        return false;
    }
}

void UiState::apply_changes(ecmgui::TomlFile& config) const {
    if (!writable) throw std::runtime_error(notice);
    validate_ui(*this);
    visit_ui_fields([&](const char* key, auto member) {
        if (this->*member == baseline_.*member) return;
        const auto value = encode(this->*member);
        if (!config.has("gui", key) || config.get("gui", key) != value)
            config.set("gui", key, value);
    });
}

void UiState::mark_saved() { baseline_ = *this; }

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
