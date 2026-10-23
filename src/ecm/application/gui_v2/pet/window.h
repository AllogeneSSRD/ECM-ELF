#pragma once
#include "../ui_state.h"
#include "behavior.h"
#include <windows.h>
#include <objidl.h>
#include <gdiplus.h>
#include <memory>
#include <vector>
#include <array>

namespace ecmgui_v2::pet {
struct Placement { int x, y, width, height; bool bottom; };
Placement place(RECT main, RECT work, float scale, float left);
RECT client_bounds(HWND owner);
void raster_frame(Gdiplus::Bitmap& atlas,const Gdiplus::Rect& source,int width,int height,unsigned* pixels);
class Window {
public:
    Window(HWND owner, UiState& state, FILE* log) : owner_(owner), state_(state), log_(log) {}
    ~Window();
    void tick(Mood mood);
    void follow_owner();
    HWND handle() const { return hwnd_; }
    std::string error;
private:
    HWND owner_ = nullptr, hwnd_ = nullptr;
    UiState& state_;
    FILE* log_ = nullptr;
    ULONG_PTR gdiplus_ = 0;
    std::unique_ptr<Gdiplus::Bitmap> atlas_;
    std::string loaded_skin_;
    HDC dc_ = nullptr;
    HBITMAP bitmap_ = nullptr, previous_ = nullptr;
    unsigned* pixels_ = nullptr;
    std::vector<unsigned> cached_frames_;
    int cache_width_=0,cache_height_=0;
    std::array<bool,7> cached_rows_{};
    Placement previous_place_{};
    Placement previous_anchor_{};
    Behavior behavior_{static_cast<uint32_t>(GetTickCount64())};
    float rise_=0;
    int sway_=0;
    bool dragging_ = false, moved_ = false;
    POINT drag_start_{};
    float drag_left_ = 0;
    int frame_ = -1, row_ = -1;
    bool open();
    void close();
    void clear_bitmap();
    void paint(const Placement& spot, int row, int frame, int sway=0);
    static LRESULT CALLBACK procedure(HWND, UINT, WPARAM, LPARAM);
};
}
