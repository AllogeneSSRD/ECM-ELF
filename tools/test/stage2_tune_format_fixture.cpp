#include "../../src/core/ecm_stage2_tune_format.h"
#include <iostream>
int main(int argc,char **argv) {
    try {
        if(argc==2) {
            const auto e=ecm_stage2::tune::effort(std::stoi(argv[1]));
            std::cout<<e.first<<' '<<e.last<<' '<<e.repeats<<'\n';return 0;
        }
        std::string line;
        while(std::getline(std::cin,line))std::cout<<ecm_stage2::tune::table(line);
        return 0;
    } catch(const std::exception &e) {std::cerr<<e.what()<<'\n';return 2;}
}
