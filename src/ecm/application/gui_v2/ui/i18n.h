#pragma once
#include <string>
#include <vector>
namespace ecmgui_v2::ui {
// External UTF-8 XML resources; pointers remain valid until explicit reload.
bool load_languages(const std::string& directory, std::string& error,const std::string& file="chineseSimplified.xml");
void refresh_language_files();
const std::vector<std::string>& language_files();
const std::string& selected_language_file();
std::string language_label(const std::string& filename);
const char* text(const char* key, bool chinese);
std::string message(const std::string& value, bool chinese);
const std::string& language_directory();
}
