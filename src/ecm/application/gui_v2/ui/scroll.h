#pragma once
#include <cmath>
#include <algorithm>
namespace ecmgui_v2::ui {
// Conserve wheel distance while easing it over a short, bounded burst.
struct SmoothScroll {
    double x=0,y=0;
    void add(double horizontal,double vertical){x+=horizontal;y+=vertical;}
    bool active() const{return std::abs(x)+std::abs(y)>.0001;}
    void cancel(){x=y=0;}
    void step(double dt,double& horizontal,double& vertical){
        const double fraction=1-std::exp(-std::clamp(dt,.001,.1)/.065);
        horizontal=x*fraction;vertical=y*fraction;x-=horizontal;y-=vertical;
        if(std::abs(x)<.001){horizontal+=x;x=0;}if(std::abs(y)<.001){vertical+=y;y=0;}
    }
};
}
