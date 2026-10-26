#include "probability_panel.h"
#include "localization.h"
#include "ecm/stage2/probability.h"
#include "imgui.h"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <limits>
#include <sstream>
#include <stdexcept>

namespace ecmgui {
namespace {
using namespace probability_ui;
constexpr ImU32 color_stage1=IM_COL32(55,184,160,255);
constexpr ImU32 color_combined=IM_COL32(174,142,239,255);
constexpr ImU32 color_conditional=IM_COL32(238,170,84,255);
constexpr const char *solve_keys[]={"forward","solve_curves","solve_size","solve_b1","solve_b2","solve_divisor"};
constexpr const char *scope_keys[]={"scope_stage1","scope_combined","scope_conditional"};
constexpr const char *preset_keys[]={"preset_suyama","preset_param1","preset_param2","preset_param3","preset_custom","preset_t4","preset_t8","preset_t12","preset_t16"};
std::string number(double value) {
    if(std::isnan(value))return "n/a";
    if(!std::isfinite(value))return "inf";
    char text[64];std::snprintf(text,sizeof(text),"%.7g",value);return text;
}
double parse(const char *text,const char *error) {
    try {std::size_t used=0;const auto value=std::stod(text,&used);
        while(text[used]==' '||text[used]=='\t')++used;
        if(text[used] || !std::isfinite(value))throw std::runtime_error(error);
        return value;
    }catch(...){throw std::runtime_error(error);}
}
template<std::size_t N>void set_number(std::array<char,N> &text,double value) {
    std::snprintf(text.data(),text.size(),"%.12g",value);
}
double preset_divisor(int preset) {
    const double baseline=std::exp(ecm_stage2::probability::suyama_delta);
    // Same theoretical constants as tools/ecm_prob/model.py and rho.py.
    if(preset==1)return baseline*.416384512396064;
    if(preset==3)return baseline*.330484606500389;
    static constexpr double torsion[]={4,8,12,16};
    if(preset>=5)return torsion[preset-5];
    return baseline;
}
bool choice(const Localization &loc,const char *id,int &value,const char *const *keys,int count) {
    bool changed=false;ImGui::SetNextItemWidth(-1);
    if(ImGui::BeginCombo(id,loc.t("prob",keys[value]).c_str())) {
        for(int i=0;i<count;++i)if(ImGui::Selectable(loc.t("prob",keys[i]).c_str(),value==i)){value=i;changed=true;}
        ImGui::EndCombo();
    }
    return changed;
}
void chart(const Localization &loc,const char *id,const char *title,const char *x_label,
           const std::vector<Point> &points,double marker,bool logarithmic,bool conditional) {
    if(points.size()<2)return;
    ImGui::PushID(id);
    ImGui::TextUnformatted(title);
    ImGui::TextColored(ImGui::ColorConvertU32ToFloat4(color_stage1),"%s",loc.t("prob","legend_stage1").c_str());
    if(ImGui::GetContentRegionAvail().x>ImGui::CalcTextSize(loc.t("prob","legend_combined").c_str()).x+40)ImGui::SameLine();
    ImGui::TextColored(ImGui::ColorConvertU32ToFloat4(color_combined),"%s",loc.t("prob","legend_combined").c_str());
    if(conditional)ImGui::TextColored(ImGui::ColorConvertU32ToFloat4(color_conditional),"%s",loc.t("prob","scope_conditional").c_str());
    const float font=ImGui::GetFontSize();
    const float width=std::max(1.0f,ImGui::GetContentRegionAvail().x);
    const float height=font*10.5f;
    ImGui::InvisibleButton("##plot",ImVec2(width,height));
    const auto min=ImGui::GetItemRectMin(),max=ImGui::GetItemRectMax();
    auto *draw=ImGui::GetWindowDrawList();
    draw->AddRectFilled(min,max,IM_COL32(24,30,42,255),6);
    if(width<font*12) {
        draw->PushClipRect(min,max,true);
        draw->AddText(ImVec2(min.x+8,min.y+8),ImGui::GetColorU32(ImGuiCol_TextDisabled),loc.t("prob","widen_plot").c_str());
        draw->PopClipRect();ImGui::PopID();return;
    }
    double floor=0;
    if(logarithmic) {
        double least=1;
        for(const auto &p:points){if(p.stage1>0)least=std::min(least,p.stage1);if(p.combined>0)least=std::min(least,p.combined);if(conditional && p.conditional>0)least=std::min(least,p.conditional);}
        floor=std::clamp(std::floor(std::log10(least)), -14.,-2.);
    }
    float widest=0;
    for(int i=0;i<=4;++i) {
        const double p=logarithmic?std::pow(10.,floor+i*.25*(-floor)):i*.25;
        widest=std::max(widest,ImGui::CalcTextSize((number(p*100)+"%").c_str()).x);
    }
    const ImVec2 a(min.x+widest+font*.7f,min.y+font*.5f);
    const ImVec2 b(max.x-font*.8f,max.y-font*2.4f);
    if(b.x-a.x<font*6) {
        draw->PushClipRect(min,max,true);
        draw->AddText(ImGui::GetFont(),font,ImVec2(min.x+8,min.y+8),ImGui::GetColorU32(ImGuiCol_TextDisabled),loc.t("prob","widen_plot").c_str(),nullptr,width-16);
        draw->PopClipRect();ImGui::PopID();return;
    }
    const double x0=points.front().x,x1=points.back().x;
    const auto xpos=[&](double x){return a.x+float((x-x0)/(x1-x0))*(b.x-a.x);};
    const auto ypos=[&](double p){const double t=logarithmic?(std::max(floor,p>0?std::log10(p):floor)-floor)/(-floor):std::clamp(p,0.,1.);return b.y-float(t)*(b.y-a.y);};
    draw->PushClipRect(min,max,true);
    for(int i=0;i<=4;++i) {
        const double t=i/4.;const float y=b.y-float(t)*(b.y-a.y),x=a.x+float(t)*(b.x-a.x);
        draw->AddLine(ImVec2(a.x,y),ImVec2(b.x,y),IM_COL32(255,255,255,24));
        draw->AddLine(ImVec2(x,a.y),ImVec2(x,b.y),IM_COL32(255,255,255,18));
        const double probability=logarithmic?std::pow(10.,floor+t*(-floor)):t;
        const auto label=number(probability*100)+"%";
        const auto text_size=ImGui::CalcTextSize(label.c_str());
        draw->AddText(ImVec2(a.x-font*.4f-text_size.x,y-text_size.y*.5f),ImGui::GetColorU32(ImGuiCol_TextDisabled),label.c_str());
        const int step=b.x-a.x<font*10?4:b.x-a.x<font*18?2:1;
        if(i%step==0) {
            const auto tick=number(x0+(x1-x0)*t);const auto tick_size=ImGui::CalcTextSize(tick.c_str());
            const float tx=std::clamp(x-tick_size.x*.5f,min.x,max.x-tick_size.x);
            draw->AddText(ImVec2(tx,b.y+font*.25f),ImGui::GetColorU32(ImGuiCol_TextDisabled),tick.c_str());
        }
    }
    draw->AddRect(a,b,IM_COL32(255,255,255,56));
    const auto label_size=ImGui::CalcTextSize(x_label);
    draw->AddText(ImVec2(a.x+(b.x-a.x-label_size.x)*.5f,max.y-font*1.1f),ImGui::GetColorU32(ImGuiCol_TextDisabled),x_label);
    draw->PushClipRect(a,b,true);
    for(std::size_t i=1;i<points.size();++i) {
        const auto &left=points[i-1],&right=points[i];
        if(!logarithmic || (left.stage1>0 && right.stage1>0))
            draw->AddLine(ImVec2(xpos(left.x),ypos(left.stage1)),ImVec2(xpos(right.x),ypos(right.stage1)),color_stage1,2);
        if(!logarithmic || (left.combined>0 && right.combined>0))
            draw->AddLine(ImVec2(xpos(left.x),ypos(left.combined)),ImVec2(xpos(right.x),ypos(right.combined)),color_combined,2);
        if(conditional && std::isfinite(left.conditional) && std::isfinite(right.conditional) &&
           (!logarithmic || (left.conditional>0 && right.conditional>0)))
            draw->AddLine(ImVec2(xpos(left.x),ypos(left.conditional)),ImVec2(xpos(right.x),ypos(right.conditional)),color_conditional,2);
    }
    if(marker>=x0 && marker<=x1) {
        const float x=xpos(marker);
        for(float y=a.y;y<b.y;y+=font*.45f)draw->AddLine(ImVec2(x,y),ImVec2(x,std::min(y+font*.2f,b.y)),IM_COL32(230,211,160,160));
    }
    draw->PopClipRect();draw->PopClipRect();
    const auto mouse=ImGui::GetMousePos();
    if(ImGui::IsItemHovered() && mouse.x>=a.x && mouse.x<=b.x && mouse.y>=a.y && mouse.y<=b.y) {
        const double x=x0+(mouse.x-a.x)/(b.x-a.x)*(x1-x0);
        const std::size_t index=std::min(points.size()-1,static_cast<std::size_t>(std::round((x-x0)/(x1-x0)*(points.size()-1))));
        const auto &p=points[index];
        draw->PushClipRect(a,b,true);
        draw->AddLine(ImVec2(xpos(p.x),a.y),ImVec2(xpos(p.x),b.y),IM_COL32(255,255,255,115));
        draw->PopClipRect();
        ImGui::BeginTooltip();ImGui::Text("%s: %s",x_label,number(p.x).c_str());
        ImGui::Text("%s: %s%%",loc.t("prob","legend_stage1").c_str(),number(p.stage1*100).c_str());
        ImGui::Text("%s: %s%%",loc.t("prob","legend_combined").c_str(),number(p.combined*100).c_str());
        if(conditional)ImGui::Text("%s: %s%%",loc.t("prob","scope_conditional").c_str(),number(p.conditional*100).c_str());ImGui::EndTooltip();
    }
    ImGui::PopID();
}
}
probability_ui::Request ProbabilityPanel::request() const {
    probability_ui::Request r;auto &p=r.parameters;
    r.decimal_digits=digits_;r.solve=static_cast<probability_ui::Solve>(solve_);
    p.factor_bits=solve_==2?2:probability_ui::bits_from_size(parse(size_.data(),"invalid_size"),digits_);
    p.b1=solve_==3?2:parse(b1_.data(),"invalid_b1");p.stage2=stage2_;p.ratio_mode=ratio_mode_;
    if(stage2_){if(solve_==4){p.ratio_mode=false;p.b2=p.b1;}else if(ratio_mode_)p.b2_ratio=parse(ratio_.data(),"invalid_ratio");else p.b2=parse(b2_.data(),"invalid_b2");}
    p.curves=solve_==1?0:parse(curves_.data(),"invalid_curves");
    p.effective_divisor=solve_==5?preset_divisor(0):preset_==4?parse(divisor_.data(),"invalid_divisor"):preset_divisor(preset_);
    p.scope=static_cast<probability_ui::Scope>(scope_);
    r.cumulative_target=cumulative_target_;if(solve_)r.target=parse(target_.data(),"invalid_target")/100.;
    r.size_min=parse(size_min_.data(),"invalid_plot_range");r.size_max=parse(size_max_.data(),"invalid_plot_range");
    r.curves_max=parse(curves_max_.data(),"invalid_plot_range");r.auto_curves_range=auto_curves_range_;return r;
}
void ProbabilityPanel::calculate() {
    if(pending_.valid())return;
    try {
        const auto input=request();error_.clear();dirty_=false;
        pending_=std::async(std::launch::async,[input]{return probability_ui::calculate(input);});
    } catch(const std::system_error &){error_="calculation_failed";dirty_=true;}
      catch(const std::exception &error){error_=error.what();dirty_=true;}
}
void ProbabilityPanel::apply_solution() {
    if(!result_ || !result_->error.empty())return;
    const auto &r=*result_;const auto &p=r.parameters;
    digits_=r.request.decimal_digits;stage2_=p.stage2;ratio_mode_=p.ratio_mode;
    scope_=static_cast<int>(p.scope);
    set_number(size_,probability_ui::size_from_bits(p.factor_bits,digits_));
    set_number(b1_,p.b1);set_number(b2_,probability_ui::effective_b2(p));
    set_number(ratio_,p.b2_ratio);set_number(curves_,p.curves);set_number(divisor_,p.effective_divisor);
    preset_=4;solve_=0;dirty_=true;
}
void ProbabilityPanel::draw(const Localization &loc,bool &open) {
    if(pending_.valid() && pending_.wait_for(std::chrono::seconds(0))==std::future_status::ready) {
        try {
            auto completed=pending_.get();error_=completed.error;
            if(completed.error.empty())result_=std::make_unique<probability_ui::Result>(std::move(completed));
        }catch(...){error_="calculation_failed";}
    }
    const bool visible=ImGui::Begin((loc.t("prob","title")+"###prob").c_str(),&open);
    if(!visible){ImGui::End();return;}
    if(!initialized_){initialized_=true;set_number(divisor_,preset_divisor(preset_));calculate();}
    const auto text=[&](const char *key){return loc.t("prob",key);};
    ImGui::TextWrapped("%s",text("intro").c_str());
    bool changed=false;
    const auto input=[&](const char *label,const char *id,Text &value) {
        ImGui::TextWrapped("%s",text(label).c_str());ImGui::SetNextItemWidth(-1);
        changed|=ImGui::InputText(id,value.data(),value.size());
    };
    if(ImGui::BeginTable("##params",2,ImGuiTableFlags_SizingStretchSame|ImGuiTableFlags_BordersInnerV|ImGuiTableFlags_PadOuterX)) {
        ImGui::TableNextRow();ImGui::TableSetColumnIndex(0);ImGui::BeginGroup();
        ImGui::TextColored(ImVec4(.48f,.84f,.75f,1.f),"%s",text("parameters_group").c_str());ImGui::Separator();
        ImGui::TextUnformatted(text("factor_size").c_str());
        bool unit=digits_;
        if(ImGui::RadioButton(text("bits").c_str(),!digits_))unit=false;ImGui::SameLine();
        if(ImGui::RadioButton(text("digits").c_str(),digits_))unit=true;
        if(unit!=digits_) {
            for(Text *field:{&size_,&size_min_,&size_max_})try {
                set_number(*field,probability_ui::size_from_bits(probability_ui::bits_from_size(parse(field->data(),"invalid_size"),digits_),unit));
            }catch(...){}
            digits_=unit;changed=true;
        }
        ImGui::BeginDisabled(solve_==2);ImGui::SetNextItemWidth(-1);changed|=ImGui::InputText("##size",size_.data(),size_.size());ImGui::EndDisabled();
        ImGui::BeginDisabled(solve_==1);input("curves","##curves",curves_);ImGui::EndDisabled();
        ImGui::BeginDisabled(solve_==3);input("b1","##b1",b1_);ImGui::EndDisabled();
        ImGui::Spacing();changed|=ImGui::Checkbox(text("use_stage2").c_str(),&stage2_);
        ImGui::BeginDisabled(!stage2_ || solve_==4);changed|=ImGui::Checkbox(text("use_ratio").c_str(),&ratio_mode_);
        if(ratio_mode_)input("ratio","##ratio",ratio_);else input("b2","##b2",b2_);
        ImGui::EndDisabled();
        ImGui::Spacing();ImGui::TextWrapped("%s",text("number_hint").c_str());ImGui::EndGroup();
        ImGui::TableSetColumnIndex(1);ImGui::BeginGroup();
        ImGui::TextColored(ImVec4(.82f,.72f,1.f,1.f),"%s",text("probability_group").c_str());ImGui::Separator();
        ImGui::TextUnformatted(text("mode").c_str());changed|=choice(loc,"##solve",solve_,solve_keys,6);
        if(solve_==1 && !cumulative_target_){cumulative_target_=true;changed=true;}
        ImGui::TextUnformatted(text("model").c_str());
        if(choice(loc,"##preset",preset_,preset_keys,9)){changed=true;if(preset_!=4)set_number(divisor_,preset_divisor(preset_));}
        ImGui::BeginDisabled(preset_!=4 || solve_==5);input("divisor","##divisor",divisor_);ImGui::EndDisabled();
        ImGui::TextUnformatted(text("scope").c_str());
        changed|=choice(loc,"##scope",scope_,scope_keys,stage2_?3:1);
        if(!stage2_ && scope_!=0){scope_=0;changed=true;}
        ImGui::Spacing();ImGui::BeginDisabled(!solve_ || solve_==1);
        changed|=ImGui::Checkbox(text("cumulative_target").c_str(),&cumulative_target_);ImGui::EndDisabled();
        ImGui::BeginDisabled(!solve_);ImGui::TextWrapped("%s",text(cumulative_target_?"target_total":"target_single").c_str());
        ImGui::SetNextItemWidth(-1);changed|=ImGui::InputText("##target",target_.data(),target_.size());
        if(ImGui::SmallButton("63.2121% (1-exp(-1))")){set_number(target_,(1-std::exp(-1.))*100);changed=true;}
        ImGui::EndDisabled();
        ImGui::Spacing();ImGui::TextWrapped("%s",text("scope_hint").c_str());ImGui::EndGroup();ImGui::EndTable();
    }
    if(ImGui::CollapsingHeader(text("plot_range").c_str())) {
        changed|=ImGui::Checkbox(text("auto_range").c_str(),&auto_curves_range_);
        if(ImGui::BeginTable("##ranges",3,ImGuiTableFlags_SizingStretchSame)) {
            ImGui::TableNextRow();ImGui::TableSetColumnIndex(0);input("size_min","##size_min",size_min_);
            ImGui::TableSetColumnIndex(1);input("size_max","##size_max",size_max_);
            ImGui::TableSetColumnIndex(2);input("curves_max","##curves_max",curves_max_);ImGui::EndTable();
        }
    }
    if(changed)dirty_=true;
    ImGui::BeginDisabled(pending_.valid());
    if(ImGui::Button(text("calculate").c_str()))calculate();ImGui::EndDisabled();
    ImGui::SameLine();
    if(pending_.valid())ImGui::TextDisabled("%s",text("calculating").c_str());
    else if(dirty_)ImGui::TextDisabled("%s",text("pending_edits").c_str());
    if(!error_.empty()){ImGui::PushStyleColor(ImGuiCol_Text,ImVec4(1.f,.67f,.43f,1.f));ImGui::TextWrapped("%s",text(error_.c_str()).c_str());ImGui::PopStyleColor();}
    if(result_) {
        const auto &r=*result_;const auto &p=r.parameters;const auto &prob=r.probability;
        ImGui::Separator();
        if(dirty_ || pending_.valid() || !error_.empty())ImGui::TextDisabled("%s",text("last_valid_result").c_str());
        ImGui::TextWrapped("%s: B1=%s  B2=%s  %s=%s  C=%s  D_eff=%s",
            text("result_parameters").c_str(),number(p.b1).c_str(),number(probability_ui::effective_b2(p)).c_str(),
            text(r.request.decimal_digits?"digits":"bits").c_str(),number(probability_ui::size_from_bits(p.factor_bits,r.request.decimal_digits)).c_str(),number(p.curves).c_str(),number(p.effective_divisor).c_str());
        if(r.request.solve!=probability_ui::Solve::Forward) {
            double value=r.solved_value;
            if(r.request.solve==probability_ui::Solve::FactorSize)value=probability_ui::size_from_bits(value,r.request.decimal_digits);
            ImGui::TextColored(ImVec4(.82f,.72f,1.f,1.f),"%s: %s",text(solve_keys[static_cast<int>(r.request.solve)]).c_str(),number(value).c_str());
            ImGui::TextWrapped("%s: %s%%",text("residual").c_str(),number(r.residual*100).c_str());
            if(r.lower_bound_sufficient)ImGui::TextWrapped("%s",text("lower_bound_sufficient").c_str());
            if(!r.lower_bound_sufficient && r.residual>1e-4)ImGui::TextWrapped("%s",text("residual_warning").c_str());
            ImGui::BeginDisabled(pending_.valid());if(ImGui::SmallButton(text("apply_solution").c_str()))apply_solution();ImGui::EndDisabled();
        }
        if(ImGui::BeginTable("##probabilities",3,ImGuiTableFlags_RowBg|ImGuiTableFlags_BordersInnerH)) {
            ImGui::TableSetupColumn(text("quantity").c_str());ImGui::TableSetupColumn(text("single_probability").c_str());ImGui::TableSetupColumn(text("total_probability").c_str());ImGui::TableHeadersRow();
            const double values[]={prob.stage1,prob.combined,prob.conditional};
            const char *labels[]={"legend_stage1","legend_combined","scope_conditional"};
            for(int i=0;i<3;++i){ImGui::TableNextRow();ImGui::TableSetColumnIndex(0);ImGui::TextWrapped("%s",text(labels[i]).c_str());
                ImGui::TableSetColumnIndex(1);ImGui::Text("%s%%",number(values[i]*100).c_str());
                ImGui::TableSetColumnIndex(2);ImGui::Text("%s%%",number(probability_ui::cumulative(values[i],p.curves)*100).c_str());}
            ImGui::EndTable();
        }
        const double mean=prob.selected>0?1/prob.selected:std::numeric_limits<double>::infinity();
        ImGui::TextWrapped("%s (%s): %s",text("expected_curves").c_str(),text(scope_keys[static_cast<int>(p.scope)]).c_str(),number(mean).c_str());
        ImGui::TextWrapped("%s: %s%%",text("miss_probability").c_str(),number(probability_ui::miss(prob.selected,p.curves)*100).c_str());
        if(prob.selected==0)ImGui::TextWrapped("%s",text("zero_tail").c_str());
        if(ImGui::SmallButton(text("copy_csv").c_str()))ImGui::SetClipboardText(probability_ui::curve_csv(r).c_str());
        ImGui::SameLine();ImGui::Checkbox(text("log_y").c_str(),&logarithmic_);
        ImGui::Checkbox(text("show_conditional").c_str(),&show_conditional_);
        if(logarithmic_)ImGui::TextWrapped("%s",text("log_hint").c_str());
        chart(loc,"size",text("plot_size").c_str(),text(r.request.decimal_digits?"digits":"bits").c_str(),r.sizes,probability_ui::size_from_bits(p.factor_bits,r.request.decimal_digits),logarithmic_,show_conditional_ || p.scope==probability_ui::Scope::Stage2Conditional);
        const auto cumulative_size_title=text("plot_size_cumulative")+" (C="+number(p.curves)+")";
        chart(loc,"size_totals",cumulative_size_title.c_str(),text(r.request.decimal_digits?"digits":"bits").c_str(),r.size_totals,probability_ui::size_from_bits(p.factor_bits,r.request.decimal_digits),logarithmic_,show_conditional_ || p.scope==probability_ui::Scope::Stage2Conditional);
        chart(loc,"counts",text("plot_curves").c_str(),text("curves").c_str(),r.curve_counts,p.curves,logarithmic_,show_conditional_ || p.scope==probability_ui::Scope::Stage2Conditional);
    }
    ImGui::Separator();ImGui::TextWrapped("%s",text("model_note").c_str());
    ImGui::TextWrapped("%s",text("size_note").c_str());ImGui::End();
}
}
