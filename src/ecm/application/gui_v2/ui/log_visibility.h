#pragma once
namespace ecmgui_v2::ui {
// Manual visibility on a non-worker view lasts only until that view changes.
struct LogVisibility {
    int context=-1;
    bool manual=false;
    void change_view(int current){if(context!=current){context=current;manual=false;}}
    bool visible(bool enabled,bool keep,bool workers)const{return manual||(enabled&&(keep||workers));}
};
}
