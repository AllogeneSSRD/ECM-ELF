#pragma once
#include "theme.h"
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <limits>
#include <vector>

namespace ecmgui_v2::ui {
struct ChartPoint { double x,y; };
struct ChartSeries { const char* label; ImVec4 color; std::vector<ChartPoint> points; bool visible=true; };
struct ChartBounds {double xmin=0,xmax=1,ymin=0,ymax=1;};
struct ChartFormat {
    bool integer_x=false,percent_y=false;
    const char* x_format="%.6g";
    const char* y_format="%.6g";
    double xmin=std::numeric_limits<double>::quiet_NaN(),xmax=std::numeric_limits<double>::quiet_NaN();
};
inline double integer_step(double range,int ticks){
    const double wanted=std::max(1.,range/std::max(1,ticks));
    const double decade=std::pow(10.,std::floor(std::log10(wanted)));
    for(double scale:{1.,2.,5.,10.})if(scale*decade>=wanted)return std::max(1.,scale*decade);
    return std::ceil(wanted);
}
inline ChartBounds chart_bounds(const std::vector<ChartSeries>& series,double ymax=0) {
    ChartBounds b;bool first=true;double peak=0;
    for(const auto& s:series)if(s.visible)for(const auto& p:s.points)if(std::isfinite(p.x)&&std::isfinite(p.y)){
        if(first){b.xmin=b.xmax=p.x;first=false;}else{b.xmin=std::min(b.xmin,p.x);b.xmax=std::max(b.xmax,p.x);}
        peak=std::max(peak,p.y);
    }
    if(b.xmax<=b.xmin)b.xmax=b.xmin+1;
    b.ymax=ymax>0?ymax:std::max(1.,peak*1.12);return b;
}
inline int chart_nearest(const std::vector<ChartPoint>& points,double x) {
    int found=-1;double distance=std::numeric_limits<double>::infinity();
    for(size_t i=0;i<points.size();++i)if(std::isfinite(points[i].x)&&std::isfinite(points[i].y)&&std::abs(points[i].x-x)<distance){found=static_cast<int>(i);distance=std::abs(points[i].x-x);}
    return found;
}
// Finite adjacent samples only: unavailable telemetry creates a visible gap.
inline void chart(const char* id,std::vector<ChartSeries>& series,const char* xlabel,const char* ylabel,bool zh,double ymax=0,float logical_height=180,ChartFormat format={}) {
    ImGui::PushID(id); const float scale=ImGui::GetFontSize()/15.f;
    const float legend_right=ImGui::GetCursorScreenPos().x+ImGui::GetContentRegionAvail().x;
    for(size_t i=0;i<series.size();++i){auto& s=series[i];ImGui::PushID(static_cast<int>(i));
        const float legend_width=ImGui::CalcTextSize(s.label).x+ImGui::GetFontSize()+ImGui::GetStyle().FramePadding.y*2+ImGui::GetStyle().ItemInnerSpacing.x;
        if(i&&ImGui::GetItemRectMax().x+legend_width+ImGui::GetStyle().ItemSpacing.x<legend_right)ImGui::SameLine();
        const auto key=ImGui::GetID("visible");s.visible=ImGui::GetStateStorage()->GetBool(key,true);
        ImGui::PushStyleColor(ImGuiCol_Text,s.color);if(ImGui::Checkbox(s.label,&s.visible))ImGui::GetStateStorage()->SetBool(key,s.visible);ImGui::PopStyleColor();ImGui::PopID();}
    auto bounds=chart_bounds(series,ymax);
    if(std::isfinite(format.xmin)&&std::isfinite(format.xmax)&&format.xmin<format.xmax){bounds.xmin=format.xmin;bounds.xmax=format.xmax;}
    const auto origin=ImGui::GetCursorScreenPos();
    const ImVec2 size{std::max(140.f*scale,ImGui::GetContentRegionAvail().x),logical_height*scale};
    ImGui::InvisibleButton("plot",size);const bool hovered=ImGui::IsItemHovered();auto* draw=ImGui::GetWindowDrawList();
    const auto p=palette(ImGui::GetStyleColorVec4(ImGuiCol_WindowBg).x>.5f);
    const ImVec2 lo{origin.x+48*scale,origin.y+12*scale},hi{origin.x+size.x-12*scale,origin.y+size.y-38*scale};
    draw->PushClipRect(origin,{origin.x+size.x,origin.y+size.y},true);
    draw->AddRectFilled(origin,{origin.x+size.x,origin.y+size.y},ImGui::GetColorU32(p.sunken),6*scale);
    const auto point=[&](double x,double y){return ImVec2{lo.x+static_cast<float>((x-bounds.xmin)/(bounds.xmax-bounds.xmin))*(hi.x-lo.x),hi.y-static_cast<float>((y-bounds.ymin)/(bounds.ymax-bounds.ymin))*(hi.y-lo.y)};};
    char label[64];
    for(int i=0;i<=4;++i){double y=bounds.ymax*i/4.;auto pos=point(bounds.xmin,y);
        draw->AddLine(pos,{hi.x,pos.y},ImGui::GetColorU32(p.border));std::snprintf(label,sizeof(label),format.percent_y?"%.0f%%":"%.3g",y);draw->AddText({origin.x+5*scale,pos.y-6*scale},ImGui::GetColorU32(p.muted),label);
    }
    const int ticks=std::clamp(static_cast<int>((hi.x-lo.x)/(80*scale)),1,4);
    const double step=format.integer_x?integer_step(bounds.xmax-bounds.xmin,ticks):(bounds.xmax-bounds.xmin)/ticks;
    const double first=format.integer_x?std::ceil(bounds.xmin/step)*step:bounds.xmin;
    for(int i=0;i<=ticks+1;++i){const double x=first+step*i;if(x>bounds.xmax+step*.001)break;auto xp=point(x,0);draw->AddLine(xp,{xp.x,lo.y},ImGui::GetColorU32(p.border));
        std::snprintf(label,sizeof(label),format.integer_x?"%.0f":"%.3g",x==0?0:x);draw->AddText({std::clamp(xp.x-ImGui::CalcTextSize(label).x/2,lo.x,hi.x-ImGui::CalcTextSize(label).x),hi.y+4*scale},ImGui::GetColorU32(p.muted),label);
    }
    draw->PushClipRect(lo,hi,true);
    for(const auto& s:series)if(s.visible)for(size_t i=1;i<s.points.size();++i){const auto a=s.points[i-1],b=s.points[i];
        if(!std::isfinite(a.x)||!std::isfinite(a.y)||!std::isfinite(b.x)||!std::isfinite(b.y))continue;
        const auto ap=point(a.x,a.y),bp=point(b.x,b.y);auto fill=s.color;fill.w=.09f;
        draw->AddQuadFilled(ap,bp,{bp.x,hi.y},{ap.x,hi.y},ImGui::GetColorU32(fill));draw->AddLine(ap,bp,ImGui::GetColorU32(s.color),2*scale);
    }
    draw->PopClipRect();
    draw->AddText({lo.x,origin.y+size.y-17*scale},ImGui::GetColorU32(p.muted),xlabel);
    draw->AddText({hi.x-ImGui::CalcTextSize(ylabel).x,origin.y+2*scale},ImGui::GetColorU32(p.muted),ylabel);
    draw->PopClipRect();
    if(hovered&&ImGui::GetIO().MousePos.x>=lo.x&&ImGui::GetIO().MousePos.x<=hi.x){
        const auto mx=ImGui::GetIO().MousePos.x;const double x=bounds.xmin+(mx-lo.x)/(hi.x-lo.x)*(bounds.xmax-bounds.xmin);
        double sample_x=x;bool any=false;
        for(const auto& s:series)if(s.visible){const int n=chart_nearest(s.points,x);if(n>=0){sample_x=s.points[static_cast<size_t>(n)].x;any=true;break;}}
        const auto cross=point(sample_x,0);draw->AddLine({cross.x,lo.y},{cross.x,hi.y},ImGui::GetColorU32(p.focus));ImGui::BeginTooltip();
        if(any){std::snprintf(label,sizeof(label),format.x_format,sample_x);ImGui::Text("%s: %s",xlabel,label);}
        for(const auto& s:series)if(s.visible){const auto found=std::find_if(s.points.begin(),s.points.end(),[sample_x](const ChartPoint& v){return v.x==sample_x;});
            if(found!=s.points.end()&&std::isfinite(found->y)){std::snprintf(label,sizeof(label),format.y_format,found->y);ImGui::TextColored(s.color,"%s: %s",s.label,label);}}
        if(!any)ImGui::TextUnformatted(zh?"无有效采样":"No valid samples");ImGui::EndTooltip();
    }
    ImGui::PopID();
}
}
