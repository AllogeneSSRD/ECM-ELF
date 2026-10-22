// Real ImGui chart layout, without a renderer, GPU load or telemetry overrides.
#include "../../../src/ecm/application/gui_v2/ui/chart.h"
#include <cstdio>
#include <cstdlib>
#include <cstring>
namespace ecmgui_v2::ui {
Palette palette(bool){return {};}
float secondary_font_size(float n){return n-2;}
float primary_font_size(float n){return n+1;}
const char* text(const char* key,bool){return key;}
}
int main(int argc,char** argv){
    const bool expect_jump=argc>1&&std::strcmp(argv[1],"--expect-jump")==0;
    ImGui::CreateContext();auto& io=ImGui::GetIO();io.IniFilename=nullptr;io.DisplaySize={1600,900};io.DeltaTime=1.f/60;
    io.Fonts->AddFontDefault();unsigned char* pixels;int w,h;io.Fonts->GetTexDataAsRGBA32(&pixels,&w,&h);
    int cases=0,jumps=0;
    for(float size:{15.f,24.f}){
        ImGui::GetStyle().FontSizeBase=size;
        for(int metric=0;metric<3;++metric){
        const std::vector<const char*> values=metric==0?std::vector<const char*>{"0 / 100%","9 / 100%","99 / 100%","100 / 100%","-- / 100%"}:metric==1?std::vector<const char*>{"0 MHz","999 MHz","9999 MHz","10000 MHz","-- MHz"}:std::vector<const char*>{"0.0 / 999 W","99.9 / 999 W","999.9 / 999 W","2997.0 / 999 W","-- / 999 W"};
        for(int width=190;width<=900;++width){
            ecmgui_v2::ui::ChartMetrics previous;bool first=true;
            for(const char* value:values){
                ImGui::NewFrame();ImGui::SetNextWindowPos({0,0});ImGui::SetNextWindowSize({float(width),600});
                ImGui::Begin("legend-test",nullptr,ImGuiWindowFlags_NoSavedSettings|ImGuiWindowFlags_NoScrollbar);
                std::vector<ecmgui_v2::ui::ChartSeries> series{{"GPU load",{},{{0,0},{1,100}}},{"Memory occupied",{},{{0,0},{1,100}}}};
                series[0].readout=value;series[1].readout="100.0 / 100%";
                if(metric){series[0].label=metric==1?"Core frequency":"Power";series[1].label=metric==1?"Memory frequency":"Power reserve";
                    series[0].readout_capacity=series[1].readout_capacity=metric==1?"99999 MHz":"2997.0 / 999 W";series[1].readout=value;}
                ecmgui_v2::ui::ChartMetrics m;ecmgui_v2::ui::ChartFormat format;
                format.compact=true;format.legend_controls=false;format.show_x_ticks=false;format.show_y_ticks=false;format.percent_y=metric==0;format.metrics=&m;
                ecmgui_v2::ui::chart("usage",series,"time","%",false,100,64,format);
                ImGui::End();ImGui::Render();++cases;
                if(!first&&(std::abs(previous.height-m.height)>.01||previous.header_rows!=m.header_rows)){
                    ++jumps;if(!expect_jump){std::fprintf(stderr,"Jump at font=%.0f width=%d value=%s: %.2f -> %.2f, rows %d -> %d\n",size,width,value,previous.height,m.height,previous.header_rows,m.header_rows);return 1;}
                }
                previous=m;first=false;
            }
        }
        }
    }
    ImGui::DestroyContext();std::printf("{\"cases\":%d,\"jumps\":%d,\"expected_failure\":%s}\n",cases,jumps,expect_jump?"true":"false");
    return expect_jump?(jumps?0:2):0;
}
