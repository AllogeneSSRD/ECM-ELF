#pragma once
#include <string>
namespace ecmgui_v2::ui {
// External UTF-8 XML resources; pointers remain valid until explicit reload.
bool load_languages(const std::string& directory, std::string& error);
const char* text(const char* key, bool chinese);
std::string message(const std::string& value, bool chinese);
const std::string& language_directory();
}
