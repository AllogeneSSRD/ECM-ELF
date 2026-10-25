#include "../../../src/ecm/application/gui_v2/ui_state.h"
#include "../../../src/ecm/application/gui_v2/pet/window.h"
#include "../../../src/ecm/application/gui_v2/generated/pet_skins.h"
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
    state.worker_stage1_exe = "C:\\中文\\worker.exe"; state.layout("[Window][Test]\nDockId=0x123\n");
    state.ui_right_pages=(1<<3)|(1<<4);state.ui_right_page=3;state.ui_right_visible=false;state.ui_smooth_scroll=false;state.ui_toasts=false;
    state.ui_logs_keep_visible=true;state.ui_worker_chart_auto_y=true;state.ui_nav_hide_tab_bar=true;state.ui_workspace_hide_tab_bar=true;state.ui_right_hide_tab_bar=true;
    assert(state.save());
    ecmgui_v2::UiState copy; copy.path = state.path; assert(copy.load());
    assert(copy.ui_theme == "light" && copy.ui_language == "zh" && copy.pet_enabled && copy.pet_left == 126.5f);
    assert(copy.worker_stage1_exe == state.worker_stage1_exe && copy.layout() == state.layout());
    assert(copy.ui_external_windows);
    assert(copy.window_title_bar&&copy.ui_nav_order=="40123567");
    assert(copy.ui_font_size==24);
    assert(copy.ui_primary_font_offset==4&&copy.ui_secondary_font_offset==-4);
    assert(copy.ui_gpu_history_seconds==120);
    assert(copy.ui_stage2_timing_source=="tune"&&copy.ui_stage2_tune_file==state.ui_stage2_tune_file);
    assert(copy.ui_language_file=="custom-中文.xml");
    assert(copy.ui_worker_chart_columns==2);
    assert(copy.ui_logs_keep_visible&&copy.ui_worker_chart_auto_y&&copy.ui_nav_hide_tab_bar&&copy.ui_workspace_hide_tab_bar&&copy.ui_right_hide_tab_bar);
    assert(copy.dock_tab_saved[0]&&copy.dock_tab_saved[1]&&copy.dock_tab_saved[2]);
    assert(copy.pet_skin=="hi_love_elf");
    assert(copy.ui_right_pages==24&&copy.ui_right_page==3&&!copy.ui_right_visible&&!copy.ui_smooth_scroll&&!copy.ui_toasts);
    const auto legacy=dir/"old-sidecar.toml";{std::ofstream out(legacy);out<<"[ui]\nlanguage=\"zh\"\n";}
    ecmgui_v2::UiState migrated;migrated.path=legacy;assert(migrated.load()&&migrated.ui_right_pages==0&&migrated.ui_smooth_scroll&&migrated.ui_toasts&&!migrated.ui_external_windows&&!migrated.window_title_bar&&migrated.ui_font_size==15&&migrated.ui_gpu_history_seconds==60&&migrated.ui_nav_order=="01237456");
    {const auto old=dir/"seven-pages.toml";{std::ofstream out(old);out<<"[ui]\nnav_order=\"4012356\"\nright_pages=16\n";}ecmgui_v2::UiState previous;previous.path=old;assert(previous.load()&&previous.ui_nav_order=="40123567"&&previous.ui_right_pages==16);}
    for(const auto order:{"0012345","012345","0123457"}){
        const auto invalid_order=dir/(std::string("invalid-order-")+order+".toml");{std::ofstream out(invalid_order);out<<"[ui]\nnav_order=\""<<order<<"\"\n";}
        ecmgui_v2::UiState bad_order;bad_order.path=invalid_order;assert(!bad_order.load()&&!bad_order.save());
    }
    const auto invalid_range=dir/"invalid-gpu-range.toml";{std::ofstream out(invalid_range);out<<"[ui]\ngpu_history_seconds=61\n";}
    ecmgui_v2::UiState bad_range;bad_range.path=invalid_range;assert(!bad_range.load()&&!bad_range.save());
    for(int columns:{0,3}){
        const auto file=dir/("invalid-chart-columns-"+std::to_string(columns)+".toml");{std::ofstream out(file);out<<"[ui]\nworker_chart_columns="<<columns<<"\n";}
        ecmgui_v2::UiState bad_columns;bad_columns.path=file;assert(!bad_columns.load()&&!bad_columns.save());
    }
    assert(migrated.ui_worker_chart_columns==1&&!migrated.ui_logs_keep_visible&&!migrated.ui_worker_chart_auto_y&&!migrated.dock_tab_saved[0]);
    const auto invalid_font=dir/"invalid-font.toml";{std::ofstream out(invalid_font);out<<"[ui]\nfont_size=31\n";}
    ecmgui_v2::UiState bad_font;bad_font.path=invalid_font;assert(!bad_font.load()&&!bad_font.save());
    const auto invalid_offset=dir/"invalid-offset.toml";{std::ofstream out(invalid_offset);out<<"[ui]\nprimary_font_offset=7\n";}
    ecmgui_v2::UiState bad_offset;bad_offset.path=invalid_offset;assert(!bad_offset.load()&&!bad_offset.save());
    const auto invalid = dir / "invalid.toml";
    { std::ofstream out(invalid); out << "[state]\nschema=999\n"; }
    std::ifstream original(invalid,std::ios::binary); const std::string before((std::istreambuf_iterator<char>(original)),{}); original.close();
    ecmgui_v2::UiState bad; bad.path = invalid; assert(!bad.load() && !bad.save());
    std::ifstream retained(invalid,std::ios::binary); const std::string after((std::istreambuf_iterator<char>(retained)),{});
    assert(after == before);
    const auto invalid_layout=dir/"invalid-layout.toml";
    {std::ofstream out(invalid_layout);out<<"[ui]\ndock=\"0\"\n";}
    ecmgui_v2::UiState bad_layout;bad_layout.path=invalid_layout;assert(!bad_layout.load()&&!bad_layout.save());
    const auto invalid_skin=dir/"invalid-skin.toml";
    {std::ofstream out(invalid_skin);out<<"[pet]\nskin=\"../../outside\"\n";}
    ecmgui_v2::UiState bad_skin;bad_skin.path=invalid_skin;assert(!bad_skin.load()&&!bad_skin.save());
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
    std::puts("PASS: sidecar roundtrip, invalid-state retention, Unicode paths, pet work-area/DPI geometry");
}
