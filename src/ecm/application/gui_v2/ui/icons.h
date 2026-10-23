#pragma once
#include "imgui.h"
#include <cmath>
#include <initializer_list>
namespace ecmgui_v2::ui {
// Application-drawn line icons: one view box / stroke, no font or image dependency.
enum class Icon { None, Play, Stop, Grid, List, Sliders, Check, Folder, Cpu, Results, Generate, Chart, Gpu, Settings, Palette, Pet, Layout };
inline Icon page_icon(int page){
    static const Icon icons[]{Icon::Cpu,Icon::Results,Icon::Generate,Icon::Chart,Icon::Gpu,Icon::Settings,Icon::Grid,Icon::Generate};
    return page>=0&&page<8?icons[page]:Icon::Grid;
}
inline void line_icon(ImDrawList* draw,Icon shape,ImVec2 pos,float size,ImU32 color){
    const float u=size/24,stroke=1.6f*u;
    const auto point=[&](float x,float y){return ImVec2{pos.x+x*u,pos.y+y*u};};
    const auto line=[&](float x,float y,float a,float b){draw->AddLine(point(x,y),point(a,b),color,stroke);};
    const auto rect=[&](float x,float y,float a,float b,float radius=2.f){draw->AddRect(point(x,y),point(a,b),color,radius*u,0,stroke);};
    const auto circle=[&](float x,float y,float r){draw->AddCircle(point(x,y),r*u,color,16,stroke);};
    switch(shape){
    case Icon::None:break;
    case Icon::Play:
        // Trim the three tips and join the edges with tangent rounded corners.
        draw->PathLineTo(point(6,5));
        draw->PathBezierCubicCurveTo(point(6,3.667f),point(6.572f,3.343f),point(7.715f,4.029f));
        draw->PathLineTo(point(19.285f,10.971f));
        draw->PathBezierCubicCurveTo(point(20.428f,11.657f),point(20.428f,12.343f),point(19.285f,13.029f));
        draw->PathLineTo(point(7.715f,19.971f));
        draw->PathBezierCubicCurveTo(point(6.572f,20.657f),point(6,20.333f),point(6,19));
        draw->PathStroke(color,stroke,ImDrawFlags_Closed);break;
    case Icon::Stop:rect(4,4,20,20,2);break;
    case Icon::Grid:for(int x:{3,14})for(int y:{3,14})rect(float(x),float(y),float(x+7),float(y+7),1);break;
    case Icon::List:for(float y:{3.f,11.f,19.f})rect(3,y,21,y+3,1);break;
    case Icon::Cpu:rect(6,6,18,18);rect(9,9,15,15,1);for(float a:{8.f,12.f,16.f}){line(a,2,a,6);line(a,18,a,22);line(2,a,6,a);line(18,a,22,a);}break;
    case Icon::Results:rect(5,3,19,21);line(8,7,15,7);line(8,11,13,11);line(9,16,11,18);line(11,18,16,13);break;
    case Icon::Generate:line(4,20,17,7);line(7,17,10,20);line(17,2,17,5);line(20,5,23,5);line(3,6,3,10);line(1,8,5,8);line(20,14,20,18);line(18,16,22,16);break;
    case Icon::Chart:line(3,3,3,21);line(3,21,22,21);line(6,17,10,11);line(10,11,14,14);line(14,14,20,5);break;
    case Icon::Gpu:rect(3,4,21,17);line(8,21,16,21);line(12,17,12,21);line(6,12,10,12);line(10,12,13,8);line(13,8,17,12);break;
    case Icon::Settings:{circle(12,12,3);for(int i=0;i<8;++i){const float a=float(i)*3.14159265f/4;line(12+7*std::cos(a),12+7*std::sin(a),12+10*std::cos(a),12+10*std::sin(a));}circle(12,12,7);break;}
    case Icon::Palette:draw->AddBezierCubic(point(21,12),point(21,0),point(2,0),point(3,13),color,stroke);draw->AddBezierCubic(point(3,13),point(4,24),point(18,23),point(14,16),color,stroke);draw->AddBezierCubic(point(14,16),point(12,12),point(21,17),point(21,12),color,stroke);for(auto c:{ImVec2{8,8},ImVec2{13,6},ImVec2{17,9}})circle(c.x,c.y,1);break;
    case Icon::Pet:draw->AddBezierCubic(point(4,15),point(1,5),point(8,1),point(9,7),color,stroke);draw->AddBezierCubic(point(15,7),point(16,1),point(23,5),point(20,15),color,stroke);draw->AddBezierCubic(point(4,15),point(4,24),point(20,24),point(20,15),color,stroke);circle(8,14,1);circle(16,14,1);line(10,18,14,18);break;
    case Icon::Layout:rect(3,3,21,21);line(9,3,9,21);line(9,15,21,15);break;
    case Icon::Folder:line(3,8,3,4);line(3,4,10,4);line(10,4,13,8);rect(3,8,21,20);break;
    case Icon::Check:line(4,12,10,18);line(10,18,21,5);break;
    case Icon::Sliders:for(int i=0;i<3;++i){const float y=5.f+i*7;const float x=i==1?15.f:8.f;line(3,y,x-3,y);line(x+3,y,21,y);circle(x,y,3);}break;
    }
}
}
