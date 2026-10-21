#include "../../../src/ecm/application/gui_v2/ui_state.h"
#include "../../../src/ecm/application/gui_v2/pet/window.h"
#include <cassert>
#include <fstream>
#include <cstdio>
int main(int argc, char** argv) {
    if (argc != 2) return 2;
    namespace fs = std::filesystem;
    const auto dir = fs::u8path(argv[1]); fs::create_directories(dir);
    ecmgui_v2::UiState state; state.path = dir / "state.toml";
    state.ui_theme = "light"; state.ui_language = "zh"; state.pet_enabled = true; state.pet_left = 126.5f;
    state.worker_stage1_exe = "C:\\中文\\worker.exe"; state.layout("[Window][Test]\nDockId=0x123\n");
    assert(state.save());
    ecmgui_v2::UiState copy; copy.path = state.path; assert(copy.load());
    assert(copy.ui_theme == "light" && copy.ui_language == "zh" && copy.pet_enabled && copy.pet_left == 126.5f);
    assert(copy.worker_stage1_exe == state.worker_stage1_exe && copy.layout() == state.layout());
    const auto invalid = dir / "invalid.toml";
    { std::ofstream out(invalid); out << "[state]\nschema=999\n"; }
    std::ifstream original(invalid,std::ios::binary); const std::string before((std::istreambuf_iterator<char>(original)),{}); original.close();
    ecmgui_v2::UiState bad; bad.path = invalid; assert(!bad.load() && !bad.save());
    std::ifstream retained(invalid,std::ios::binary); const std::string after((std::istreambuf_iterator<char>(retained)),{});
    assert(after == before);
    using ecmgui_v2::pet::place;
    const auto top = place({100,400,1300,1200}, {0,0,1920,1440}, 1, 72);
    assert(!top.bottom && top.x == 172 && top.y == 276 && top.width == 112 && top.height == 124);
    const auto bottom = place({-8,-8,1928,1448}, {0,0,1920,1400}, 2, 72);
    assert(bottom.bottom && bottom.y == 1152 && bottom.width == 224 && bottom.x >= 0);
    const auto negative = place({-1900,300,-700,1100}, {-1920,0,0,1400}, 1.5f, 99999);
    assert(negative.x + negative.width <= 0 && negative.width == 168 && negative.height == 186);
    std::puts("PASS: sidecar roundtrip, invalid-state retention, Unicode paths, pet work-area/DPI geometry");
}
