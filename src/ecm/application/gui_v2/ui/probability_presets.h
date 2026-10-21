#pragma once
#include <cmath>
#include <stdexcept>
namespace ecmgui_v2::ui {
// Frozen stage1-only representative data, tools/ecm_prob/out/d_eff_representative.json.
// For a combined-model heuristic preserve the measured Edwards/Suyama ratio,
// rebased onto GMP-ECM's exp(3.134) Suyama constant. This is NOT a direct S2 fit.
inline constexpr double empirical_suyama=20.90211505168743;
inline constexpr double empirical_edwards_2x8=26.408122047632308;
inline constexpr double empirical_edwards_12=25.690791072125354;
inline double curve_divisor(int profile,bool empirical=false){
    const double baseline=std::exp(3.134);
    switch(profile){
    case 0:return empirical?empirical_suyama:baseline;
    case 1:return empirical?empirical_edwards_2x8:baseline*empirical_edwards_2x8/empirical_suyama;
    case 2:return empirical?empirical_edwards_12:baseline*empirical_edwards_12/empirical_suyama;
    case 3:return empirical?8.258986726111255:baseline*.416384512396064;
    case 4:return empirical?6.405914356403636:baseline*.330484606500389;
    default:throw std::runtime_error("Unknown curve probability profile");
    }
}
}
