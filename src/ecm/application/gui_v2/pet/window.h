#pragma once
#include "../ui_state.h"
#include <windows.h>
#include <objidl.h>
#include <gdiplus.h>
#include <memory>
#include <vector>
#include <array>

namespace ecmgui_v2::pet {
enum class Mood { Calm, Working, Waiting, Error, Done, Doze, Wink };
struct Placement { int x, y, width, height; bool bottom; };
Placement place(RECT main, RECT work, float scale, float left);
class Window {
public:
    Window(HWND owner, UiState& state, FILE* log) : owner_(owner), state_(state), log_(log) {}
    ~Window();
    void tick(Mood mood);
    HWND handle() const { return hwnd_; }
    std::string error;
private:
    HWND owner_ = nullptr, hwnd_ = nullptr;
    UiState& state_;
    FILE* log_ = nullptr;
    ULONG_PTR gdiplus_ = 0;
    std::unique_ptr<Gdiplus::Bitmap> atlas_;
    HDC dc_ = nullptr;
    HBITMAP bitmap_ = nullptr, previous_ = nullptr;
    unsigned* pixels_ = nullptr;
    std::vector<unsigned> cached_frames_;
    int cache_width_=0,cache_height_=0;
    std::array<bool,7> cached_rows_{};
    Placement previous_place_{};
    bool dragging_ = false, moved_ = false;
    POINT drag_start_{};
    float drag_left_ = 0;
    unsigned long long since_ = 0, petted_until_ = 0;
    Mood source_ = Mood::Calm;
    int frame_ = -1, row_ = -1;
    bool open();
    void close();
    void clear_bitmap();
    void paint(const Placement& spot, int row, int frame);
    static LRESULT CALLBACK procedure(HWND, UINT, WPARAM, LPARAM);
};
}
