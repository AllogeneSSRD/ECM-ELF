#include <cstdio>
#include <cstring>
int main(int argc,char** argv){
    if(argc!=5||std::strcmp(argv[1],"-i")||std::strcmp(argv[3],"--query-supported-clocks=graphics")||std::strcmp(argv[4],"--format=csv,noheader,nounits"))return 2;
    if(std::strcmp(argv[2],"GPU-11111111-1111-1111-1111-111111111111")){std::puts("N/A");return 0;}
    std::puts("3105\n210\n2600");return 0;
}
