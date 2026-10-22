#include "../../../src/ecm/application/gui_v2/ui_state.h"
#include "../../../src/ecm/application/gui_v2/pet/window.h"
#include <cassert>
#include <fstream>
#include <cstdio>
int main(int argc, char** argv) {
    if (argc != 2) return 2;
    namespace fs = std::filesystem;
    ULONG_PTR gdiplus;Gdiplus::GdiplusStartupInput input;assert(Gdiplus::GdiplusStartup(&gdiplus,&input,nullptr)==Gdiplus::Ok);
    {Gdiplus::Bitmap atlas(16,16,PixelFormat32bppARGB);Gdiplus::Graphics g(&atlas);g.Clear(Gdiplus::Color(255,0,255,0));
    Gdiplus::SolidBrush red(Gdiplus::Color(255,255,0,0));g.FillRectangle(&red,4,4,4,4);g.Flush(Gdiplus::FlushIntentionSync);
    std::vector<unsigned> pixels(144);ecmgui_v2::pet::raster_frame(atlas,{4,4,4,4},12,12,pixels.data());
    int contaminated=0;for(const auto pixel:pixels)if(((pixel>>8)&255)>0)++contaminated;
    std::printf("pet adjacent-tile contaminated pixels: %d\n",contaminated);std::fflush(stdout);assert(contaminated==0);}
    Gdiplus::GdiplusShutdown(gdiplus);
    const auto owner=CreateWindowExW(0,L"STATIC",L"pet anchor fixture",WS_OVERLAPPEDWINDOW,100,0,800,600,nullptr,nullptr,GetModuleHandleW(nullptr),nullptr);assert(owner);
    const auto client=ecmgui_v2::pet::client_bounds(owner);RECT outer{};assert(GetWindowRect(owner,&outer));
    assert(client.bottom<outer.bottom);const auto client_bottom=ecmgui_v2::pet::place(client,{0,0,1920,1440},1,72);
    assert(client_bottom.bottom&&client_bottom.y+client_bottom.height==client.bottom);DestroyWindow(owner);
    const auto dir = fs::u8path(argv[1]); fs::create_directories(dir);
    ecmgui_v2::UiState state; state.path = dir / "state.toml";
    assert(!state.ui_external_windows);state.ui_external_windows=true;
    state.ui_theme = "light"; state.ui_language = "zh"; state.pet_enabled = true; state.pet_left = 126.5f;
    state.worker_stage1_exe = "C:\\中文\\worker.exe"; state.layout("[Window][Test]\nDockId=0x123\n");
    state.ui_right_pages=(1<<3)|(1<<4);state.ui_right_page=3;state.ui_right_visible=false;state.ui_smooth_scroll=false;state.ui_toasts=false;
    assert(state.save());
    ecmgui_v2::UiState copy; copy.path = state.path; assert(copy.load());
    assert(copy.ui_theme == "light" && copy.ui_language == "zh" && copy.pet_enabled && copy.pet_left == 126.5f);
    assert(copy.worker_stage1_exe == state.worker_stage1_exe && copy.layout() == state.layout());
    assert(copy.ui_external_windows);
    assert(copy.ui_right_pages==24&&copy.ui_right_page==3&&!copy.ui_right_visible&&!copy.ui_smooth_scroll&&!copy.ui_toasts);
    const auto legacy=dir/"old-sidecar.toml";{std::ofstream out(legacy);out<<"[ui]\nlanguage=\"zh\"\n";}
    ecmgui_v2::UiState migrated;migrated.path=legacy;assert(migrated.load()&&migrated.ui_right_pages==0&&migrated.ui_smooth_scroll&&migrated.ui_toasts&&!migrated.ui_external_windows);
    const auto invalid = dir / "invalid.toml";
    { std::ofstream out(invalid); out << "[state]\nschema=999\n"; }
    std::ifstream original(invalid,std::ios::binary); const std::string before((std::istreambuf_iterator<char>(original)),{}); original.close();
    ecmgui_v2::UiState bad; bad.path = invalid; assert(!bad.load() && !bad.save());
    std::ifstream retained(invalid,std::ios::binary); const std::string after((std::istreambuf_iterator<char>(retained)),{});
    assert(after == before);
    const auto invalid_layout=dir/"invalid-layout.toml";
    {std::ofstream out(invalid_layout);out<<"[ui]\ndock=\"0\"\n";}
    ecmgui_v2::UiState bad_layout;bad_layout.path=invalid_layout;assert(!bad_layout.load()&&!bad_layout.save());
    using ecmgui_v2::pet::place;
    const auto top = place({100,400,1300,1200}, {0,0,1920,1440}, 1, 72);
    assert(!top.bottom && top.x == 172 && top.y == 276 && top.width == 112 && top.height == 124);
    const auto bottom = place({-8,-8,1928,1448}, {0,0,1920,1400}, 2, 72);
    assert(bottom.bottom && bottom.y == 1152 && bottom.width == 224 && bottom.x >= 0);
    const auto negative = place({-1900,300,-700,1100}, {-1920,0,0,1400}, 1.5f, 99999);
    assert(negative.x + negative.width <= 0 && negative.width == 168 && negative.height == 186);
    std::puts("PASS: sidecar roundtrip, invalid-state retention, Unicode paths, pet work-area/DPI geometry");
}
