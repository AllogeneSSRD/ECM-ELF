#pragma once
#include <cmath>
#include <stdexcept>
#include <array>
#include <algorithm>
namespace ecmgui_v2::ui {
// Frozen stage1-only representative data, tools/ecm_prob/out/d_eff_representative.json.
// For a combined-model heuristic preserve the measured Edwards/Suyama ratio,
// rebased onto GMP-ECM's exp(3.134) Suyama constant. This is NOT a direct S2 fit.
inline constexpr double empirical_suyama=20.90211505168743;
inline constexpr double empirical_edwards_2x8=26.408122047632308;
inline constexpr double empirical_edwards_12=25.690791072125354;
inline constexpr int custom_curve_profile=8;
struct CurvePreset {int id;const char* label;const char* source;double empirical;double theoretical_factor;const char* generator;bool edwards;};
// Empirical values are the frozen representative output of tools/ecm_prob/model.py.
// Montgomery factors follow THEORETICAL_D; Edwards has no theoretical constant.
inline constexpr std::array<CurvePreset,8> curve_presets{{
    {0,"param 0 (Suyama)","suyama_s10",empirical_suyama,1,"m0",false},
    {1,"Z2 x Z8 (Prime95)","edwards_Z2xZ8",empirical_edwards_2x8,0,"e",true},
    {2,"Z12","edwards_Z12",empirical_edwards_12,0,nullptr,true},
    {3,"param 1","param1_s10",8.258986726111255,.416384512396064,nullptr,false},
    {4,"param 3","param3_s10",6.405914356403636,.330484606500389,"m3",false},
    {5,"param 2","param2_s10",20.15667357813588,1,"m2",false},
    {6,"Z2 x Z4","edwards_Z2xZ4",14.16200621274284,0,nullptr,true},
    {7,"Z4","edwards_Z4",8.510017237617587,0,nullptr,true},
}};
inline double curve_divisor(int profile,bool empirical=false){
    const double baseline=std::exp(3.134);
    if(profile<0||profile>=custom_curve_profile)throw std::runtime_error("Unknown curve probability profile");
    const auto& preset=curve_presets[profile];
    return empirical?preset.empirical:baseline*(preset.edwards?preset.empirical/empirical_suyama:preset.theoretical_factor);
}
inline std::array<int,4> edwards_profiles(){
    std::array<int,4> ids{1,2,6,7};
    std::sort(ids.begin(),ids.end(),[](int a,int b){return curve_presets[a].empirical>curve_presets[b].empirical;});
    return ids;
}
}
