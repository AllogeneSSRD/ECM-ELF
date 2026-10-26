#ifdef _WIN32
#include <windows.h>
#endif
#include "ecm/stage1/progress.h"
#include <cstdlib>
#include <limits>

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
    SpeedWindow window;
    for(uint64_t i=1;i<=60;++i)window.add(i,1);
    check(close_to(window.speed(),35.5),"only newest 50 slice speeds retained");
    const auto text=line(100,144350,222,"columns",.56,4.5,0);
    check(text=="GPU: [========================================] 100.0%  144350, +222 columns (~0.56 s/curve)  elapsed 4.5s  remaining 0.0s","shared text protocol");
    check(text.find('\r')==std::string::npos && text.find('\033')==std::string::npos,"plain text in shared formatter");
    check(line(-1,0,0,"bits",0,0,0).find("0.0%")!=std::string::npos,"lower clamp");
    check(line(101,1,1,"bits",1,1,0).find("100.0%")!=std::string::npos,"upper clamp");
    std::puts("GPU progress: 11 checks passed");
}
