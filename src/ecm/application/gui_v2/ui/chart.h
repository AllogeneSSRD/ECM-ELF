#include "i18n.h"
#pragma once
#include "theme.h"
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <limits>
#include <vector>
#include <string>

namespace ecmgui_v2::ui {
struct ChartPoint { double x,y; };
struct ChartSeries { const char* label; ImVec4 color; std::vector<ChartPoint> points; bool visible=true; std::string readout,readout_capacity; };
struct ChartBounds {double xmin=0,xmax=1,ymin=0,ymax=1;};
struct ChartMetrics {float height=0,plot_height=0,stroke=0,label_size=0,value_size=0;int header_rows=0;bool x_grid=true,x_ticks=true;};
struct ChartFormat {
    bool integer_x=false,percent_y=false;
    const char* x_format="%.6g";
    const char* y_format="%.6g";
    double xmin=std::numeric_limits<double>::quiet_NaN(),xmax=std::numeric_limits<double>::quiet_NaN();
    bool show_y_ticks=true,show_axis_titles=true;
    bool legend_controls=true;
    const char* corner_text=nullptr;
    bool compact=false,show_x_ticks=true,show_x_grid=true;
    float stroke=1.15f;
    ChartMetrics* metrics=nullptr;
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
    for(size_t i=0;format.legend_controls&&i<series.size();++i){auto& s=series[i];ImGui::PushID(static_cast<int>(i));
        const float legend_width=ImGui::CalcTextSize(s.label).x+ImGui::GetFontSize()+ImGui::GetStyle().FramePadding.y*2+ImGui::GetStyle().ItemInnerSpacing.x;
        if(i&&ImGui::GetItemRectMax().x+legend_width+ImGui::GetStyle().ItemSpacing.x<legend_right)ImGui::SameLine();
        const auto key=ImGui::GetID("visible");s.visible=ImGui::GetStateStorage()->GetBool(key,true);
        ImGui::PushStyleColor(ImGuiCol_Text,s.color);if(ImGui::Checkbox(s.label,&s.visible))ImGui::GetStateStorage()->SetBool(key,s.visible);ImGui::PopStyleColor();ImGui::PopID();}
    if(!format.legend_controls)for(auto& s:series)s.visible=true;
    auto bounds=chart_bounds(series,ymax);
    if(std::isfinite(format.xmin)&&std::isfinite(format.xmax)&&format.xmin<format.xmax){bounds.xmin=format.xmin;bounds.xmax=format.xmax;}
    const auto origin=ImGui::GetCursorScreenPos();
    const float width=std::max(140.f*scale,ImGui::GetContentRegionAvail().x);
    const auto font=ImGui::GetFont();const float body_px=ImGui::GetFontSize();
    const float caption_px=body_px*secondary_font_size(ImGui::GetStyle().FontSizeBase)/ImGui::GetStyle().FontSizeBase;
    const auto caption_width=[&](const char* text){return font->CalcTextSizeA(caption_px,std::numeric_limits<float>::max(),0,text).x;};
    // Layout from a capacity pattern, independently of the changing live value.
    // Proportional fonts may give '1' and '8' different widths: reserve the widest digit.
    char widest='0';float digit_width=0;
    for(char c='0';c<='9';++c){const char text[]{c,0};const float w=ImGui::CalcTextSize(text).x;if(w>digit_width){digit_width=w;widest=c;}}
    const auto capacity=[&](const ChartSeries& s){
        std::string value=s.readout_capacity.empty()?(format.percent_y?"100.0 / 100%":s.readout):s.readout_capacity;
        for(char& c:value)if(c>='0'&&c<='9')c=widest;return value;
    };
    struct LegendCell{const ChartSeries* series;ImVec2 label,value;float label_wrap,value_wrap;};std::vector<LegendCell> cells;
    float compact_header=0;int header_rows=0;
    if(format.compact&&!series.empty()){
        const float available=width-20*scale,gap=16*scale;
        const float column=(available-gap*(series.size()-1))/series.size();bool columns=true;
        for(const auto& s:series)if(caption_width(s.label)+ImGui::CalcTextSize(capacity(s).c_str()).x+24*scale>column)columns=false;
        float y=origin.y+5*scale;size_t i=0;
        for(const auto& s:series){
            const float left=origin.x+10*scale+(columns?i*(column+gap):0),right=columns?left+column:origin.x+width-10*scale;
            const float value_width=ImGui::CalcTextSize(s.readout.c_str()).x;
            const auto reserved=capacity(s);
            const bool stacked=caption_width(s.label)+ImGui::CalcTextSize(reserved.c_str()).x+24*scale>right-left;
            const float label_wrap=stacked?std::max(1.f,right-left-12*scale):0,value_wrap=stacked?std::max(1.f,right-left):0;
            const auto label_size=font->CalcTextSizeA(caption_px,std::numeric_limits<float>::max(),label_wrap,s.label);
            const auto value_size=font->CalcTextSizeA(body_px,std::numeric_limits<float>::max(),value_wrap,reserved.c_str());
            const float h=stacked?label_size.y+3*scale+value_size.y:std::max(caption_px,body_px);
            cells.push_back({&s,{left+12*scale,y+(stacked?0:(h-caption_px)*.5f)},{stacked?left:std::max(left,right-value_width),y+(stacked?label_size.y+3*scale:0)},label_wrap,value_wrap});
            if(!columns){y+=h+4*scale;header_rows+=stacked?2:1;}else header_rows=1;
            ++i;
        }
        compact_header=(columns?body_px+10*scale:y-origin.y+2*scale);
    }
    float legend_width=0;for(const auto& s:series)legend_width+=ImGui::CalcTextSize(s.label).x+22*scale;
    const bool split_header=!format.compact&&!format.legend_controls&&format.corner_text&&legend_width+ImGui::CalcTextSize(format.corner_text).x+24*scale>width;
    const float extra=split_header?ImGui::GetTextLineHeight()+4*scale:0;
    const float footer=format.compact?(format.show_x_ticks?caption_px+9*scale:6*scale):(format.show_axis_titles?38:22)*scale;
    const ImVec2 size{width,format.compact?std::max(logical_height*scale,compact_header+28*scale+footer):logical_height*scale+extra};
    ImGui::InvisibleButton("plot",size);const bool hovered=ImGui::IsItemHovered();auto* draw=ImGui::GetWindowDrawList();
    const auto p=palette(ImGui::GetStyleColorVec4(ImGuiCol_WindowBg).x>.5f);
    const float header=format.legend_controls?0:ImGui::GetTextLineHeight()+8*scale+extra;
    const ImVec2 lo{origin.x+(format.show_y_ticks?48:10)*scale,origin.y+(format.compact?compact_header:12*scale+header)},hi{origin.x+size.x-12*scale,origin.y+size.y-footer};
    if(format.metrics)*format.metrics={size.y,hi.y-lo.y,format.stroke*scale,format.compact?caption_px:body_px,body_px,header_rows,format.show_x_grid,format.show_x_ticks};
    draw->PushClipRect(origin,{origin.x+size.x,origin.y+size.y},true);
    draw->AddRectFilled(origin,{origin.x+size.x,origin.y+size.y},ImGui::GetColorU32(p.sunken),6*scale);
    if(format.compact){
        for(const auto& cell:cells){const auto& s=*cell.series;
            draw->AddLine({cell.label.x-12*scale,cell.label.y+caption_px*.5f},{cell.label.x-5*scale,cell.label.y+caption_px*.5f},ImGui::GetColorU32(s.color),format.stroke*scale);
            draw->AddText(font,caption_px,cell.label,ImGui::GetColorU32(p.muted),s.label,nullptr,cell.label_wrap);
            draw->AddText(font,body_px,cell.value,ImGui::GetColorU32(s.color),s.readout.c_str(),nullptr,cell.value_wrap);
        }
    }else if(!format.legend_controls){
        float x=origin.x+12*scale;const float y=origin.y+7*scale;
        for(const auto& s:series){draw->AddLine({x,y+ImGui::GetFontSize()*.5f},{x+9*scale,y+ImGui::GetFontSize()*.5f},ImGui::GetColorU32(s.color),format.stroke*scale);x+=14*scale;
            draw->AddText({x,y},ImGui::GetColorU32(s.color),s.label);x+=ImGui::CalcTextSize(s.label).x+8*scale;}
        if(format.corner_text)draw->AddText({std::max(origin.x+12*scale,origin.x+width-12*scale-ImGui::CalcTextSize(format.corner_text).x),y+(split_header?ImGui::GetTextLineHeight()+4*scale:0)},ImGui::GetColorU32(p.fg),format.corner_text);
    }
    const auto point=[&](double x,double y){return ImVec2{lo.x+static_cast<float>((x-bounds.xmin)/(bounds.xmax-bounds.xmin))*(hi.x-lo.x),hi.y-static_cast<float>((y-bounds.ymin)/(bounds.ymax-bounds.ymin))*(hi.y-lo.y)};};
    char label[64];
    auto grid=p.border;grid.w*=.55f;
    for(int i=0;i<=4;++i){double y=bounds.ymax*i/4.;auto pos=point(bounds.xmin,y);
        draw->AddLine(pos,{hi.x,pos.y},ImGui::GetColorU32(grid));
        if(format.show_y_ticks){std::snprintf(label,sizeof(label),format.percent_y?"%.0f%%":"%.3g",y);draw->AddText({origin.x+5*scale,pos.y-6*scale},ImGui::GetColorU32(p.muted),label);}
    }
    const int ticks=std::clamp(static_cast<int>((hi.x-lo.x)/(80*scale)),1,4);
    const double step=format.integer_x?integer_step(bounds.xmax-bounds.xmin,ticks):(bounds.xmax-bounds.xmin)/ticks;
    const double first=format.integer_x?std::ceil(bounds.xmin/step)*step:bounds.xmin;
    for(int i=0;i<=ticks+1;++i){const double x=first+step*i;if(x>bounds.xmax+step*.001)break;auto xp=point(x,0);if(format.show_x_grid)draw->AddLine(xp,{xp.x,lo.y},ImGui::GetColorU32(grid));
        if(format.show_x_ticks){std::snprintf(label,sizeof(label),format.integer_x?"%.0f":"%.3g",x==0?0:x);
            const float text_width=format.compact?caption_width(label):ImGui::CalcTextSize(label).x;
            draw->AddText(font,format.compact?caption_px:body_px,{std::clamp(xp.x-text_width/2,lo.x,hi.x-text_width),hi.y+4*scale},ImGui::GetColorU32(p.muted),label);}
    }
    draw->PushClipRect(lo,hi,true);
    for(const auto& s:series)if(s.visible)for(size_t i=1;i<s.points.size();++i){const auto a=s.points[i-1],b=s.points[i];
        if(!std::isfinite(a.x)||!std::isfinite(a.y)||!std::isfinite(b.x)||!std::isfinite(b.y))continue;
        const auto ap=point(a.x,a.y),bp=point(b.x,b.y);auto fill=s.color;fill.w=.12f;
        draw->AddQuadFilled(ap,bp,{bp.x,hi.y},{ap.x,hi.y},ImGui::GetColorU32(fill));draw->AddLine(ap,bp,ImGui::GetColorU32(s.color),format.stroke*scale);
    }
    for(const auto& s:series)if(s.visible&&s.points.size()==1&&std::isfinite(s.points[0].x)&&std::isfinite(s.points[0].y))
        draw->AddCircleFilled(point(s.points[0].x,s.points[0].y),2*scale,ImGui::GetColorU32(s.color));
    draw->PopClipRect();
    if(format.show_axis_titles){
        draw->AddText({lo.x,origin.y+size.y-17*scale},ImGui::GetColorU32(p.muted),xlabel);
        draw->AddText({hi.x-ImGui::CalcTextSize(ylabel).x,origin.y+2*scale},ImGui::GetColorU32(p.muted),ylabel);
    }
    draw->PopClipRect();
    if(hovered&&ImGui::GetIO().MousePos.x>=lo.x&&ImGui::GetIO().MousePos.x<=hi.x){
        const auto mx=ImGui::GetIO().MousePos.x;const double x=bounds.xmin+(mx-lo.x)/(hi.x-lo.x)*(bounds.xmax-bounds.xmin);
        double sample_x=x;bool any=false;
        for(const auto& s:series)if(s.visible){const int n=chart_nearest(s.points,x);if(n>=0){sample_x=s.points[static_cast<size_t>(n)].x;any=true;break;}}
        const auto cross=point(sample_x,0);draw->AddLine({cross.x,lo.y},{cross.x,hi.y},ImGui::GetColorU32(p.focus));ImGui::BeginTooltip();
        if(any){std::snprintf(label,sizeof(label),format.x_format,sample_x);ImGui::Text("%s: %s",xlabel,label);}
        for(const auto& s:series)if(s.visible){const auto found=std::find_if(s.points.begin(),s.points.end(),[sample_x](const ChartPoint& v){return v.x==sample_x;});
            if(found!=s.points.end()&&std::isfinite(found->y)){std::snprintf(label,sizeof(label),format.y_format,found->y);ImGui::TextColored(s.color,"%s: %s",s.label,label);}}
        if(!any)ImGui::TextUnformatted(ecmgui_v2::ui::text("no_valid_samples",zh));ImGui::EndTooltip();
    }
    ImGui::PopID();
}
}
