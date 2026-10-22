#include "window.h"
#include "../../gui/platform.h"
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>

namespace ecmgui_v2::pet {
RECT client_bounds(HWND owner) {
    RECT bounds{};if(!GetClientRect(owner,&bounds))throw std::runtime_error("Cannot read pet owner client bounds");
    POINT origin{};if(!ClientToScreen(owner,&origin))throw std::runtime_error("Cannot locate pet owner client bounds");
    OffsetRect(&bounds,origin.x,origin.y);return bounds;
}
void raster_frame(Gdiplus::Bitmap& atlas,const Gdiplus::Rect& source,int width,int height,unsigned* pixels) {
    std::unique_ptr<Gdiplus::Bitmap> isolated(atlas.Clone(source,PixelFormat32bppPARGB));
    if(!isolated||isolated->GetLastStatus()!=Gdiplus::Ok)throw std::runtime_error("Cannot isolate pet frame");
    Gdiplus::Bitmap target(width,height,width*4,PixelFormat32bppPARGB,reinterpret_cast<BYTE*>(pixels));
    Gdiplus::Graphics graphics(&target);graphics.SetCompositingMode(Gdiplus::CompositingModeSourceCopy);
    graphics.Clear(Gdiplus::Color(0,0,0,0));graphics.SetInterpolationMode(Gdiplus::InterpolationModeHighQualityBicubic);
    Gdiplus::ImageAttributes attributes;attributes.SetWrapMode(Gdiplus::WrapModeTileFlipXY);
    if(graphics.DrawImage(isolated.get(),Gdiplus::Rect(0,0,width,height),0,0,source.Width,source.Height,Gdiplus::UnitPixel,&attributes)!=Gdiplus::Ok)throw std::runtime_error("Cannot cache pet atlas");
}
Placement place(RECT main, RECT work, float scale, float left) {
    const int width = std::max(1, static_cast<int>(std::lround(112 * scale)));
    const int height = std::max(1, static_cast<int>(std::lround(124 * scale)));
    const float max_left = std::max(0.f, (main.right - main.left) / scale - 112);
    int x = main.left + static_cast<int>(std::lround(std::clamp(left, 0.f, max_left) * scale));
    x = std::clamp(x, static_cast<int>(work.left), static_cast<int>(std::max(work.left, work.right - width)));
    const bool bottom = main.top - work.top < height;
    const int y = bottom ? std::max(work.top, std::min(main.bottom, work.bottom) - height) : main.top - height;
    return {x, y, width, height, bottom};
}
bool Window::open() {
    if (hwnd_) return true;
    if (!gdiplus_) { Gdiplus::GdiplusStartupInput input; if (Gdiplus::GdiplusStartup(&gdiplus_, &input, nullptr) != Gdiplus::Ok) { error = "Pet: Windows graphics unavailable"; return false; } }
    const auto path = std::filesystem::u8path(ecmgui::exe_dir()) / L"assets" / L"pet_ely.png";
    if (!std::filesystem::exists(path) || std::filesystem::file_size(path) > 8388608) { error = "Pet atlas missing or too large: " + path.u8string(); return false; }
    atlas_.reset(Gdiplus::Bitmap::FromFile(path.c_str(), FALSE));
    if (!atlas_ || atlas_->GetLastStatus() != Gdiplus::Ok || atlas_->GetWidth() != 1792 || atlas_->GetHeight() != 1736) {
        atlas_.reset(); error = "Pet atlas invalid; expected 1792 x 1736 RGBA"; return false;
    }
    WNDCLASSEXW wc{sizeof(wc)}; wc.lpfnWndProc = procedure; wc.hInstance = GetModuleHandleW(nullptr);
    wc.lpszClassName = L"ecm_gui_v2_pet"; wc.hCursor = LoadCursorW(nullptr, IDC_HAND);
    if (!RegisterClassExW(&wc) && GetLastError() != ERROR_CLASS_ALREADY_EXISTS) { error = "Cannot register pet window"; return false; }
    hwnd_ = CreateWindowExW(WS_EX_LAYERED | WS_EX_TOOLWINDOW | WS_EX_NOACTIVATE, wc.lpszClassName, L"Ely companion", WS_POPUP,
        0, 0, 112, 124, owner_, nullptr, wc.hInstance, this);
    if (!hwnd_) { error = "Cannot create pet window"; return false; }
    since_ = GetTickCount64(); frame_ = row_ = -1; error.clear(); return true;
}
void Window::clear_bitmap() {
    if (dc_ && previous_) SelectObject(dc_, previous_);
    if (bitmap_) DeleteObject(bitmap_); if (dc_) DeleteDC(dc_);
    dc_ = nullptr; bitmap_ = previous_ = nullptr; pixels_ = nullptr;
}
void Window::close() {
    if (hwnd_) { DestroyWindow(hwnd_); hwnd_ = nullptr; }
    clear_bitmap(); atlas_.reset(); cached_frames_.clear();cached_frames_.shrink_to_fit();cached_rows_.fill(false);cache_width_=cache_height_=0;frame_ = row_ = -1;
}
Window::~Window() {
    close(); UnregisterClassW(L"ecm_gui_v2_pet", GetModuleHandleW(nullptr));
    if (gdiplus_) Gdiplus::GdiplusShutdown(gdiplus_);
}
void Window::paint(const Placement& spot, int row, int frame) {
    if (!bitmap_ || spot.width != previous_place_.width || spot.height != previous_place_.height) {
        clear_bitmap(); dc_ = CreateCompatibleDC(nullptr);
        BITMAPINFO info{}; info.bmiHeader.biSize = sizeof(BITMAPINFOHEADER);
        info.bmiHeader.biWidth = spot.width; info.bmiHeader.biHeight = -spot.height;
        info.bmiHeader.biPlanes = 1; info.bmiHeader.biBitCount = 32; info.bmiHeader.biCompression = BI_RGB;
        bitmap_ = CreateDIBSection(dc_, &info, DIB_RGB_COLORS, reinterpret_cast<void**>(&pixels_), nullptr, 0);
        if (!dc_ || !bitmap_) throw std::runtime_error("Cannot allocate pet surface");
        previous_ = static_cast<HBITMAP>(SelectObject(dc_, bitmap_));
    }
    const auto count=static_cast<size_t>(spot.width)*spot.height;
    if(cache_width_!=spot.width||cache_height_!=spot.height){
        cached_frames_.assign(count*56,0);
        cached_rows_.fill(false);cache_width_=spot.width;cache_height_=spot.height;
    }
    if(!cached_rows_[row]){
        for(int index=0;index<8;++index){
            auto* pixels=cached_frames_.data()+count*(row*8+index);
            raster_frame(*atlas_,Gdiplus::Rect(index*224,row*248,224,248),spot.width,spot.height,pixels);
        }
        cached_rows_[row]=true;
    }
    std::memcpy(pixels_,cached_frames_.data()+count*(row*8+frame),count*sizeof(unsigned));
    POINT destination{spot.x, spot.y}, origin{}; SIZE size{spot.width, spot.height};
    BLENDFUNCTION blend{AC_SRC_OVER, 0, 255, AC_SRC_ALPHA};
    if (!UpdateLayeredWindow(hwnd_, nullptr, &destination, &size, dc_, &origin, 0, &blend, ULW_ALPHA)) throw std::runtime_error("Cannot composite pet window");
    previous_place_ = spot; frame_ = frame; row_ = row;
}
void Window::tick(Mood mood) {
    if (!state_.pet_enabled) { if (hwnd_ || atlas_) close(); return; }
    if (IsIconic(owner_) || !IsWindowVisible(owner_)) { if (hwnd_) ShowWindow(hwnd_, SW_HIDE); return; }
    try {
        if (!open()) { state_.pet_enabled = false; std::fprintf(log_, "pet_error: %s\n", error.c_str()); return; }
        const auto now = GetTickCount64();
        if (mood != source_) { source_ = mood; since_ = now; }
        auto shown = source_;
        if (shown == Mood::Done && now - since_ >= 4000) shown = Mood::Calm;
        if (shown == Mood::Calm && now - since_ >= 300000) shown = Mood::Doze;
        if (petted_until_ > now) shown = Mood::Wink;
        const RECT main=client_bounds(owner_);
        MONITORINFO monitor{sizeof(monitor)};
        if (!GetMonitorInfoW(MonitorFromWindow(owner_, MONITOR_DEFAULTTONEAREST), &monitor)) throw std::runtime_error("Cannot locate pet work area");
        const float scale = std::max(1u, GetDpiForWindow(owner_)) / 96.f;
        const auto spot = place(main, monitor.rcWork, scale, state_.pet_left);
        const int frame = state_.pet_animate ? static_cast<int>((now / 150) % 8) : 0;
        const int row = static_cast<int>(shown);
        const bool relocated = spot.x != previous_place_.x || spot.y != previous_place_.y || spot.width != previous_place_.width || spot.height != previous_place_.height;
        if (row != row_ || relocated) {
            std::fprintf(log_, "pet: mood=%d edge=%s dpi=%.0f x=%d y=%d w=%d h=%d\n", row, spot.bottom ? "bottom" : "top", scale * 96, spot.x, spot.y, spot.width, spot.height); std::fflush(log_);
        }
        if (frame != frame_ || row != row_ || relocated) paint(spot, row, frame);
        if (!IsWindowVisible(hwnd_)) ShowWindow(hwnd_, SW_SHOWNOACTIVATE);
    } catch (const std::exception& failure) { error = failure.what(); state_.pet_enabled = false; close(); std::fprintf(log_, "pet_error: %s\n", error.c_str()); std::fflush(log_); }
}
LRESULT CALLBACK Window::procedure(HWND hwnd, UINT message, WPARAM w, LPARAM l) {
    auto self = reinterpret_cast<Window*>(GetWindowLongPtrW(hwnd, GWLP_USERDATA));
    if (message == WM_NCCREATE) { self = static_cast<Window*>(reinterpret_cast<CREATESTRUCTW*>(l)->lpCreateParams); SetWindowLongPtrW(hwnd, GWLP_USERDATA, reinterpret_cast<LONG_PTR>(self)); }
    if (!self) return DefWindowProcW(hwnd, message, w, l);
    if (message == WM_MOUSEACTIVATE) return MA_NOACTIVATE;
    if (message == WM_LBUTTONDOWN) {
        GetCursorPos(&self->drag_start_); self->drag_left_ = self->state_.pet_left; self->dragging_ = true; self->moved_ = false; SetCapture(hwnd); return 0;
    }
    if (message == WM_MOUSEMOVE && self->dragging_) {
        POINT point{}; GetCursorPos(&point); const float scale = GetDpiForWindow(self->owner_) / 96.f;
        if (std::abs(point.x - self->drag_start_.x) + std::abs(point.y - self->drag_start_.y) > 4 * scale) self->moved_ = true;
        if (self->moved_) {
            const RECT owner=client_bounds(self->owner_);
            self->state_.pet_left = std::clamp(self->drag_left_ + (point.x - self->drag_start_.x) / scale, 0.f, std::max(0.f, (owner.right - owner.left) / scale - 112));
        }
        return 0;
    }
    if (message == WM_LBUTTONUP && self->dragging_) {
        self->dragging_ = false; ReleaseCapture();
        if (!self->moved_) { self->petted_until_ = GetTickCount64() + 1500; self->since_ = GetTickCount64(); }
        return 0;
    }
    if (message == WM_CAPTURECHANGED) { self->dragging_ = false; return 0; }
    if (message == WM_CLOSE) { self->state_.pet_enabled = false; ShowWindow(hwnd, SW_HIDE); return 0; }
    if (message == WM_DPICHANGED) { self->frame_ = -1; return 0; }
    return DefWindowProcW(hwnd, message, w, l);
}
}
