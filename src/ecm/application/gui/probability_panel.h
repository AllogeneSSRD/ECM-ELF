#pragma once
#include "probability_model.h"
#include <array>
#include <future>
#include <memory>

namespace ecmgui {
class Localization;
class ProbabilityPanel {
public:
    void draw(const Localization &localization,bool &open);
private:
    using Text=std::array<char,64>;
    probability_ui::Request request() const;
    void calculate();
    void apply_solution();
    Text size_{{"130"}},b1_{{"110e6"}},b2_{{"11e9"}},ratio_{{"100"}},curves_{{"960"}};
    Text divisor_{{"22.965657"}},target_{{"95"}};
    Text size_min_{{"60"}},size_max_{{"200"}},curves_max_{{"1920"}};
    int preset_=0,scope_=1,solve_=0;
    bool digits_=false,stage2_=true,ratio_mode_=true,cumulative_target_=true;
    bool logarithmic_=false,auto_curves_range_=true,dirty_=true,initialized_=false;
    bool show_conditional_=false;
    std::string error_;
    std::future<probability_ui::Result> pending_;
    std::unique_ptr<probability_ui::Result> result_;
};
}
