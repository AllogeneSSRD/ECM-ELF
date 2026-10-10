#include "../../src/core/ecm_stage2_probability.h"
#include <iomanip>
#include <iostream>
int main() {
    try {
        double b1,b2,bits,delta;
        std::cout<<std::setprecision(17);
        while(std::cin>>b1>>b2>>bits>>delta)
            std::cout<<ecm_stage2::probability::success(b1,b2,bits,delta)<<'\n';
        return std::cin.eof()?0:2;
    } catch(const std::exception &e){std::cerr<<e.what()<<'\n';return 1;}
}
