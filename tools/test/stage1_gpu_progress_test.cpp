#ifdef _WIN32
#include <windows.h>
#endif
#include "ecm/stage1/progress.h"
#include <cstdlib>
#include <limits>
#include <stdexcept>

using namespace ecm_stage1_gpu_progress;
static void check(bool ok,const char *name) {
    if(!ok){std::fprintf(stderr,"FAIL: %s\n",name);std::exit(1);}
}
static bool close_to(double a,double b){return std::abs(a-b)<1e-9;}
int main() {
    SpeedWindow meter;
    check(meter.speed()==0 && meter.remaining(1000,900)==0,"no invented speed before work");
    meter.add(256,128);
    check(close_to(meter.speed(),2),"slice rate");
    check(close_to(meter.per_curve(200000,8),12.5),"complete-curve amortized estimate");
    check(close_to(meter.remaining(200000,160000),20),"remaining excludes restored work");
    check(meter.remaining(200000,200000)==0,"complete has zero remaining");
    meter.add(0,10);meter.add(50,0);meter.add(50,std::numeric_limits<double>::infinity());
    check(close_to(meter.speed(),2),"invalid/no-work samples ignored");
    meter.add(4,1);
    check(close_to(meter.speed(),4),"default zero disables averaging");
    SpeedWindow window(5);
    window.add(400,4000);window.add(300,1000);
    check(close_to(window.speed(),.14),"average is weighted by actual sample duration");
    window.add(400,2000);
    check(close_to(window.speed(),.18),"partial oldest slice expires at five seconds");
    window.add(600,6000);
    check(close_to(window.speed(),.1),"long slice replaces complete time window");
    SpeedWindow fractional(.5);
    fractional.add(.5,250);fractional.add(1,500);
    check(close_to(fractional.speed(),.002),"fractional work and subsecond windows");
    SpeedWindow maximum(60);
    for(int i=0;i<61;++i)maximum.add(i==0?10000:1,1000);
    check(close_to(maximum.speed(),.001),"sixty-second maximum expires old samples");
    for(double invalid:{-1.0,60.01,std::numeric_limits<double>::infinity(),std::numeric_limits<double>::quiet_NaN()}) {
        bool rejected=false;
        try{SpeedWindow bad(invalid);}catch(const std::invalid_argument&){rejected=true;}
        check(rejected,"invalid window rejected");
    }
    set_average_seconds(5);
    SpeedWindow configured;
    configured.add(1,1000);configured.add(3,1000);
    check(close_to(configured.speed(),.002),"resolved worker display policy is applied");
    set_average_seconds(0);
    const auto text=line(100,144350,222,"columns",.56,4.5,0);
    check(text=="GPU: [========================================] 100.0%  144350, +222 columns (~0.56 s/curve)  elapsed 4.5s  remaining 0.0s","shared text protocol");
    check(text.find('\r')==std::string::npos && text.find('\033')==std::string::npos,"plain text in shared formatter");
    check(line(-1,0,0,"bits",0,0,0).find("0.0%")!=std::string::npos,"lower clamp");
    check(line(101,1,1,"bits",1,1,0).find("100.0%")!=std::string::npos,"upper clamp");
    std::puts("Stage1 progress window and text protocol checks passed");
}
