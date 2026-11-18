#include "../../../src/ecm/application/gui_v2/ui_state.h"
#include "../../../src/ecm/application/gui/toml_file.h"
#include "../../../src/ecm/application/gui_v2/pet/window.h"
#include "../../../src/ecm/application/gui_v2/generated/pet_skins.h"
#include <cassert>
#include <fstream>
#include <filesystem>
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
    const auto path=dir/fs::u8path("配置.toml");
    {std::ofstream out(path);out<<"config_version = 4\n[stage2]\nb2 = 40000\n[gui]\nstage1_exe = 'C:/中文/worker.exe' # preserve executable\n";}
    ecmgui::TomlFile config;std::string error;assert(config.load(path.u8string(),error));
    ecmgui_v2::UiState state;assert(state.load(config));
    assert(state.pet_skin=="classic");state.pet_skin="hi_love_elf";
    assert(ecmgui_v2::pet::skins.size()==4&&ecmgui_v2::pet::find_skin(state.pet_skin));
    assert(!ecmgui_v2::pet::find_skin("../../arbitrary-image"));
    assert(!state.ui_external_windows);state.ui_external_windows=true;
    assert(!state.window_title_bar);state.window_title_bar=true;
    assert(state.ui_nav_order=="01237456");state.ui_nav_order="40123567";
    assert(state.ui_font_size==15);state.ui_font_size=24;
    assert(state.ui_primary_font_offset==1&&state.ui_secondary_font_offset==-2);state.ui_primary_font_offset=4;state.ui_secondary_font_offset=-4;
    assert(state.ui_gpu_history_seconds==60);state.ui_gpu_history_seconds=120;
    assert(state.ui_stage2_timing_source=="auto"&&state.ui_stage2_tune_file.empty());
    state.ui_stage2_timing_source="tune";state.ui_stage2_tune_file="D:/tune 测试/tune.toml";
    assert(state.ui_language_file.empty());state.ui_language_file="custom-中文.xml";
    assert(state.ui_worker_chart_columns==1);state.ui_worker_chart_columns=2;
    state.ui_theme = "light"; state.ui_language = "zh"; state.pet_enabled = true; state.pet_left = 126.5f;
    state.layout("[Window][Test]\nDockId=0x123\n");
    state.ui_right_pages=(1<<3)|(1<<4);state.ui_right_page=3;state.ui_right_visible=false;state.ui_smooth_scroll=false;state.ui_toasts=false;
    state.ui_logs_keep_visible=true;state.ui_worker_chart_auto_y=true;state.ui_nav_hide_tab_bar=false;state.ui_workspace_hide_tab_bar=false;state.ui_right_hide_tab_bar=true;
    state.apply_changes(config);assert(config.save(error));state.mark_saved();
    assert(config.load(path.u8string(),error));
    ecmgui_v2::UiState copy;assert(copy.load(config));
    assert(copy.ui_theme == "light" && copy.ui_language == "zh" && copy.pet_enabled && copy.pet_left == 126.5f);
    assert(config.get("gui","exe")=="C:/中文/worker.exe"&&copy.layout()==state.layout());
    assert(copy.ui_external_windows);
    assert(copy.window_title_bar&&copy.ui_nav_order=="40123567");
    assert(copy.ui_font_size==24);
    assert(copy.ui_primary_font_offset==4&&copy.ui_secondary_font_offset==-4);
    assert(copy.ui_gpu_history_seconds==120);
    assert(copy.ui_stage2_timing_source=="tune"&&copy.ui_stage2_tune_file==state.ui_stage2_tune_file);
    assert(copy.ui_language_file=="custom-中文.xml");
    assert(copy.ui_worker_chart_columns==2);
    assert(copy.ui_logs_keep_visible&&copy.ui_worker_chart_auto_y&&!copy.ui_nav_hide_tab_bar&&!copy.ui_workspace_hide_tab_bar&&copy.ui_right_hide_tab_bar);
    assert(copy.dock_tab_saved[0]&&copy.dock_tab_saved[1]&&copy.dock_tab_saved[2]);
    assert(copy.pet_skin=="hi_love_elf");
    assert(copy.ui_right_pages==24&&copy.ui_right_page==3&&!copy.ui_right_visible&&!copy.ui_smooth_scroll&&!copy.ui_toasts);
    const auto invalid = dir / "invalid.toml";
    for(const auto option:{"nav_order='0012345'","nav_order='012345'","nav_order='0123457'",
                           "gpu_history_seconds=61","worker_chart_columns=0","worker_chart_columns=3",
                           "body_font_size=31","primary_font_offset=7","workspace_layout='0'",
                           "pet_skin='../../outside'"}){
        {std::ofstream out(invalid);out<<"config_version=4\n[gui]\n"<<option<<'\n';}
        assert(config.load(invalid.u8string(),error));
        ecmgui_v2::UiState bad;assert(!bad.load(config)&&!bad.writable);
        assert(!config.dirty());
    }
    using ecmgui_v2::pet::place;
    const auto top = place({100,400,1300,1200}, {0,0,1920,1440}, 1, 72);
    assert(!top.bottom && top.x == 172 && top.y == 276 && top.width == 112 && top.height == 124);
    const auto bottom = place({-8,-8,1928,1448}, {0,0,1920,1400}, 2, 72);
    assert(bottom.bottom && bottom.y == 1152 && bottom.width == 224 && bottom.x >= 0);
    const auto negative = place({-1900,300,-700,1100}, {-1920,0,0,1400}, 1.5f, 99999);
    assert(negative.x + negative.width <= 0 && negative.width == 168 && negative.height == 186);
    // Both normal animation and native move reuse the same signed edge offset.
    for(float scale:{1.f,1.25f,1.5f,2.f})for(float rise:{0.f,1.5f,3.f}){
        const RECT work{-1920,-300,0,1400},owner{-1800,400,-700,1200};
        const auto outside=place(owner,work,scale,72);
        const auto inside=place({-1800,-300,-700,1200},work,scale,72);
        assert(!outside.bottom&&inside.bottom);
        for(int raster_height:{outside.height,124,248}){
            const auto y=ecmgui_v2::pet::floating_y(outside,work.top,scale,rise,raster_height);
            assert(y+raster_height==owner.top+std::lround(rise*scale));
        }
        const auto y=ecmgui_v2::pet::floating_y(inside,work.top,scale,rise,inside.height);
        assert(y+inside.height==1200-std::lround(rise*scale));
    }
    std::puts("PASS: main TOML preference roundtrip, invalid-value retention, Unicode paths, pet work-area/DPI geometry");
}
